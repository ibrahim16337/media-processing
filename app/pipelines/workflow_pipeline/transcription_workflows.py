from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from app.config.paths import (
    TRANSCRIPT_DIR,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
)
from app.pipelines.export_pipeline.playlist_excel_exporter import generate_playlist_excel
from app.pipelines.media_pipeline.audio_standardizer import standardize_audio
from app.pipelines.media_pipeline.ingestion_runner import run_ingestion
from app.pipelines.media_pipeline.youtube_downloader import (
    download_youtube_audio,
    fetch_youtube_video_metadata,
)
from app.pipelines.playlist_pipeline.playlist_runner import run_playlist_download
from app.pipelines.transcription_pipeline.transcription_runner import build_transcription_cmd


ProgressCallback = Callable[[dict[str, Any]], None] | None


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _find_transcript_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted([p for p in folder.glob("*.txt") if p.is_file()])


def _build_transcript_path(audio_or_wav_file: Path, transcript_dir: Path) -> Path:
    return transcript_dir / f"{audio_or_wav_file.stem}.txt"


def _derive_video_type(width: int | None, height: int | None) -> str:
    if not width or not height:
        return ""
    return "Short" if height > width else "Long"


def _write_video_meta_sidecar(media_path: Path, metadata: dict[str, Any]) -> Path:
    sidecar_path = media_path.with_suffix(".video_meta.json")
    sidecar_payload = {
        "video_id": metadata.get("video_id"),
        "title": metadata.get("title"),
        "duration_seconds": metadata.get("duration_seconds"),
        "video_width": metadata.get("video_width"),
        "video_height": metadata.get("video_height"),
        "video_type": _derive_video_type(
            metadata.get("video_width"),
            metadata.get("video_height"),
        ),
        "webpage_url": metadata.get("webpage_url"),
    }
    sidecar_path.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sidecar_path


def _run_transcription_subprocess(
    input_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    progress_pattern = re.compile(r"(\d+\.\d+)%")
    output_lines: list[str] = []

    cmd = build_transcription_cmd(input_path, output_dir)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    start_time = time.time()

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )

    last_percent: float = 0.0

    while True:
        line = process.stdout.readline() if process.stdout else ""

        if not line and process.poll() is not None:
            break

        if line:
            clean_line = line.strip()
            if clean_line:
                output_lines.append(clean_line)

            match = progress_pattern.search(line)
            if match:
                try:
                    last_percent = float(match.group(1))
                except Exception:
                    pass

    process.wait()

    elapsed = time.time() - start_time
    success = process.returncode == 0

    return {
        "ok": success,
        "returncode": process.returncode,
        "elapsed_sec": elapsed,
        "progress_percent": last_percent if success else 0.0,
        "output_lines": output_lines,
        "cmd": cmd,
    }


def transcribe_single_youtube(
    youtube_url: str,
    transcript_output_dir: Path = TRANSCRIPT_DIR,
    audio_output_dir: Path = UPLOAD_AUDIO_DIR,
    download_progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    youtube_url = _safe_string(youtube_url)
    if not youtube_url:
        raise ValueError("YouTube URL is required.")

    transcript_output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    youtube_metadata: dict[str, Any] = {}
    try:
        youtube_metadata = fetch_youtube_video_metadata(youtube_url)
    except Exception:
        youtube_metadata = {}

    download_start = time.time()
    downloaded_audio = download_youtube_audio(
        youtube_url,
        progress_callback=download_progress_callback,
    )
    download_time = time.time() - download_start

    standardize_start = time.time()
    wav_file = standardize_audio(downloaded_audio, audio_output_dir)
    standardize_time = time.time() - standardize_start

    downloaded_sidecar = None
    wav_sidecar = None

    if youtube_metadata:
        try:
            downloaded_sidecar = _write_video_meta_sidecar(downloaded_audio, youtube_metadata)
        except Exception:
            downloaded_sidecar = None

        try:
            wav_sidecar = _write_video_meta_sidecar(wav_file, youtube_metadata)
        except Exception:
            wav_sidecar = None

    transcription_result = _run_transcription_subprocess(
        input_path=wav_file,
        output_dir=transcript_output_dir,
    )

    transcript_file = _build_transcript_path(wav_file, transcript_output_dir)

    return {
        "ok": transcription_result["ok"],
        "mode": "single_youtube",
        "youtube_url": youtube_url,
        "downloaded_audio": str(downloaded_audio),
        "wav_file": str(wav_file),
        "transcript_file": str(transcript_file),
        "transcript_exists": transcript_file.exists(),
        "download_time_sec": download_time,
        "standardize_time_sec": standardize_time,
        "transcription_time_sec": transcription_result["elapsed_sec"],
        "transcription": transcription_result,
        "youtube_metadata": youtube_metadata,
        "downloaded_audio_sidecar": str(downloaded_sidecar) if downloaded_sidecar else "",
        "wav_sidecar": str(wav_sidecar) if wav_sidecar else "",
    }


def transcribe_single_media_file(
    media_file: str | Path,
    transcript_output_dir: Path = TRANSCRIPT_DIR,
    audio_output_dir: Path = UPLOAD_AUDIO_DIR,
) -> dict[str, Any]:
    media_path = Path(media_file)
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")

    transcript_output_dir.mkdir(parents=True, exist_ok=True)
    audio_output_dir.mkdir(parents=True, exist_ok=True)

    standardize_start = time.time()
    wav_file = standardize_audio(media_path, audio_output_dir)
    standardize_time = time.time() - standardize_start

    transcription_result = _run_transcription_subprocess(
        input_path=wav_file,
        output_dir=transcript_output_dir,
    )

    transcript_file = _build_transcript_path(wav_file, transcript_output_dir)

    return {
        "ok": transcription_result["ok"],
        "mode": "single_media_file",
        "input_file": str(media_path),
        "wav_file": str(wav_file),
        "transcript_file": str(transcript_file),
        "transcript_exists": transcript_file.exists(),
        "standardize_time_sec": standardize_time,
        "transcription_time_sec": transcription_result["elapsed_sec"],
        "transcription": transcription_result,
    }


def transcribe_batch_media(
    audio_input_dir: Path = UPLOAD_AUDIO_DIR,
    video_input_dir: Path = UPLOAD_VIDEO_DIR,
    transcript_output_dir: Path = TRANSCRIPT_DIR,
) -> dict[str, Any]:
    audio_input_dir.mkdir(parents=True, exist_ok=True)
    video_input_dir.mkdir(parents=True, exist_ok=True)
    transcript_output_dir.mkdir(parents=True, exist_ok=True)

    initial_audio_files = sorted([p for p in audio_input_dir.glob("*") if p.is_file()])
    initial_video_files = sorted([p for p in video_input_dir.glob("*") if p.is_file()])

    standardize_start = time.time()
    standardized_files: list[Path] = []

    if initial_video_files:
        standardized_files = run_ingestion(video_input_dir, audio_input_dir)

    standardize_time = time.time() - standardize_start

    all_audio_files = sorted([p for p in audio_input_dir.glob("*") if p.is_file()])

    if not all_audio_files:
        return {
            "ok": False,
            "mode": "batch_media",
            "audio_input_dir": str(audio_input_dir),
            "video_input_dir": str(video_input_dir),
            "transcript_output_dir": str(transcript_output_dir),
            "audio_file_count": 0,
            "video_file_count": len(initial_video_files),
            "standardized_files": [],
            "transcript_files": [],
            "error": "No audio files available for transcription.",
        }

    transcription_result = _run_transcription_subprocess(
        input_path=audio_input_dir,
        output_dir=transcript_output_dir,
    )

    transcript_files = _find_transcript_files(transcript_output_dir)

    return {
        "ok": transcription_result["ok"],
        "mode": "batch_media",
        "audio_input_dir": str(audio_input_dir),
        "video_input_dir": str(video_input_dir),
        "transcript_output_dir": str(transcript_output_dir),
        "audio_file_count": len(all_audio_files),
        "video_file_count": len(initial_video_files),
        "standardized_files": [str(p) for p in standardized_files],
        "transcript_files": [str(p) for p in transcript_files],
        "standardize_time_sec": standardize_time,
        "transcription_time_sec": transcription_result["elapsed_sec"],
        "transcription": transcription_result,
    }


def transcribe_playlist(
    playlist_url: str,
    quality: str = "720p",
    download_progress_callback: ProgressCallback = None,
    generate_playlist_excel_file: bool = True,
) -> dict[str, Any]:
    playlist_url = _safe_string(playlist_url)
    if not playlist_url:
        raise ValueError("Playlist URL is required.")

    playlist_start = time.time()

    playlist_data = run_playlist_download(
        url=playlist_url,
        quality=quality,
        progress_callback=download_progress_callback,
    )

    download_time = time.time() - playlist_start

    result = playlist_data["result"]
    standardized_files = playlist_data["standardized_files"]
    paths = result["paths"]
    manifest = result["manifest"]

    transcription_result = _run_transcription_subprocess(
        input_path=paths.audio,
        output_dir=paths.transcripts,
    )

    transcript_files = _find_transcript_files(paths.transcripts)

    playlist_excel_path = None
    if generate_playlist_excel_file and transcription_result["ok"]:
        playlist_excel_path = generate_playlist_excel(paths, manifest)

    return {
        "ok": transcription_result["ok"],
        "mode": "playlist",
        "playlist_url": playlist_url,
        "quality": quality,
        "playlist_root": str(paths.root),
        "videos_dir": str(paths.videos),
        "audio_dir": str(paths.audio),
        "transcripts_dir": str(paths.transcripts),
        "metadata_dir": str(paths.metadata),
        "manifest": manifest,
        "standardized_files": [str(p) for p in standardized_files],
        "transcript_files": [str(p) for p in transcript_files],
        "playlist_excel_file": str(playlist_excel_path) if playlist_excel_path else "",
        "download_time_sec": download_time,
        "transcription_time_sec": transcription_result["elapsed_sec"],
        "transcription": transcription_result,
    }