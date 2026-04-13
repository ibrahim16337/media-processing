from __future__ import annotations

import io
import subprocess
import zipfile
from pathlib import Path
from typing import Any, Sequence

from app.config.paths import (
    FFPROBE_EXE,
    PLAYLISTS_DIR,
    TRANSCRIPT_DIR,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
)
from app.pipelines.export_pipeline.transcript_excel_exporter import (
    export_transcript_excel,
)


CANDIDATE_ENCODINGS = [
    "utf-8",
    "utf-8-sig",
    "utf-16",
    "utf-16-le",
    "utf-16-be",
    "cp1256",
    "iso-8859-6",
]


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _read_text_any(path: Path) -> str:
    last_error: Exception | None = None

    for enc in CANDIDATE_ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except Exception as e:
            last_error = e

    if last_error:
        raise last_error

    raise RuntimeError(f"Failed to read transcript file: {path}")


def _find_transcript_files(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []

    return sorted(
        [
            p for p in folder.glob("*.txt")
            if p.is_file()
        ],
        key=lambda p: p.name.lower(),
    )


def _format_duration(seconds: float | int | None) -> str:
    if seconds is None:
        return ""

    try:
        total = int(float(seconds))
    except Exception:
        return ""

    hours = total // 3600
    minutes = (total % 3600) // 60
    secs = total % 60

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"

    return f"{minutes}:{secs:02d}"


def _probe_media_duration(media_path: Path) -> str:
    if not media_path.exists() or not media_path.is_file():
        return ""

    try:
        cmd = [
            str(FFPROBE_EXE),
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )

        value = result.stdout.strip()
        if not value:
            return ""

        return _format_duration(float(value))

    except Exception:
        return ""


def _find_matching_media_file(
    transcript_file: Path,
    media_lookup_dirs: Sequence[Path] | None = None,
) -> Path | None:
    if not media_lookup_dirs:
        return None

    transcript_stem = transcript_file.stem

    for folder in media_lookup_dirs:
        if not folder.exists() or not folder.is_dir():
            continue

        exact_matches = sorted(
            [
                p for p in folder.glob("*")
                if p.is_file() and p.stem == transcript_stem
            ],
            key=lambda p: p.name.lower(),
        )

        if exact_matches:
            return exact_matches[0]

    return None


def build_transcript_rows(
    transcript_files: Sequence[str | Path],
    media_lookup_dirs: Sequence[str | Path] | None = None,
) -> list[dict[str, str]]:
    normalized_transcript_files = [Path(p) for p in transcript_files if Path(p).exists() and Path(p).is_file()]
    normalized_media_dirs = [Path(p) for p in (media_lookup_dirs or []) if Path(p).exists() and Path(p).is_dir()]

    rows: list[dict[str, str]] = []

    for transcript_file in normalized_transcript_files:
        try:
            transcription_text = _read_text_any(transcript_file).strip()
        except Exception:
            transcription_text = ""

        matched_media = _find_matching_media_file(
            transcript_file=transcript_file,
            media_lookup_dirs=normalized_media_dirs,
        )

        audio_length = _probe_media_duration(matched_media) if matched_media else ""

        rows.append(
            {
                "filename": transcript_file.name,
                "transcription": transcription_text,
                "audio length": audio_length,
            }
        )

    return rows


def export_transcript_files_to_excel(
    transcript_files: Sequence[str | Path],
    output_file: str | Path,
    media_lookup_dirs: Sequence[str | Path] | None = None,
    sheet_name: str = "Transcripts",
) -> dict[str, Any]:
    rows = build_transcript_rows(
        transcript_files=transcript_files,
        media_lookup_dirs=media_lookup_dirs,
    )

    output_path = export_transcript_excel(
        rows=rows,
        output_file=output_file,
        sheet_name=sheet_name,
    )

    return {
        "ok": True,
        "mode": "transcript_files_export",
        "row_count": len(rows),
        "rows": rows,
        "output_file": str(output_path),
    }


def export_transcript_folder_to_excel(
    transcript_folder: str | Path,
    output_file: str | Path,
    media_lookup_dirs: Sequence[str | Path] | None = None,
    sheet_name: str = "Transcripts",
) -> dict[str, Any]:
    folder = Path(transcript_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Transcript folder not found: {folder}")

    transcript_files = _find_transcript_files(folder)

    rows = build_transcript_rows(
        transcript_files=transcript_files,
        media_lookup_dirs=media_lookup_dirs,
    )

    output_path = export_transcript_excel(
        rows=rows,
        output_file=output_file,
        sheet_name=sheet_name,
    )

    return {
        "ok": True,
        "mode": "transcript_folder_export",
        "transcript_folder": str(folder),
        "transcript_count": len(transcript_files),
        "row_count": len(rows),
        "rows": rows,
        "output_file": str(output_path),
    }


def export_global_transcripts_to_excel(
    output_file: str | Path,
    transcript_dir: Path = TRANSCRIPT_DIR,
    audio_dir: Path = UPLOAD_AUDIO_DIR,
    video_dir: Path = UPLOAD_VIDEO_DIR,
    sheet_name: str = "Transcripts",
) -> dict[str, Any]:
    return export_transcript_folder_to_excel(
        transcript_folder=transcript_dir,
        output_file=output_file,
        media_lookup_dirs=[audio_dir, video_dir],
        sheet_name=sheet_name,
    )


def export_playlist_transcripts_to_excel(
    playlist_slug: str,
    output_file: str | Path | None = None,
    sheet_name: str = "Transcripts",
) -> dict[str, Any]:
    playlist_slug = _safe_string(playlist_slug)
    if not playlist_slug:
        raise ValueError("playlist_slug is required.")

    playlist_root = PLAYLISTS_DIR / playlist_slug
    transcripts_dir = playlist_root / "transcripts"
    audio_dir = playlist_root / "audio"
    videos_dir = playlist_root / "videos"
    metadata_dir = playlist_root / "metadata"

    if output_file is None:
        output_file = metadata_dir / f"{playlist_slug}_transcripts.xlsx"

    return export_transcript_folder_to_excel(
        transcript_folder=transcripts_dir,
        output_file=output_file,
        media_lookup_dirs=[audio_dir, videos_dir],
        sheet_name=sheet_name,
    )


def build_transcript_zip_bytes(
    transcript_files: Sequence[str | Path],
) -> bytes:
    valid_files = [Path(p) for p in transcript_files if Path(p).exists() and Path(p).is_file()]

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in valid_files:
            zip_file.write(file_path, arcname=file_path.name)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def build_transcript_folder_zip_bytes(
    transcript_folder: str | Path,
) -> dict[str, Any]:
    folder = Path(transcript_folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Transcript folder not found: {folder}")

    transcript_files = _find_transcript_files(folder)
    zip_bytes = build_transcript_zip_bytes(transcript_files)

    return {
        "ok": True,
        "mode": "transcript_folder_zip",
        "transcript_folder": str(folder),
        "transcript_count": len(transcript_files),
        "zip_bytes": zip_bytes,
    }


def build_global_transcripts_zip_bytes(
    transcript_dir: Path = TRANSCRIPT_DIR,
) -> dict[str, Any]:
    return build_transcript_folder_zip_bytes(transcript_dir)