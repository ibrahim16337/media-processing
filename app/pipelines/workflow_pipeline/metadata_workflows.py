from __future__ import annotations

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from app.config.paths import (
    METADATA_PLAYLIST_DIR,
    TEMP_DIR,
    slugify,
)
from app.pipelines.metadata_generation_pipeline.metadata_runner import (
    run_metadata_generation,
)
from app.pipelines.metadata_generation_pipeline.ollama_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP_MS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
)
from app.pipelines.workflow_pipeline.transcription_workflows import (
    transcribe_batch_media,
    transcribe_playlist,
    transcribe_single_media_file,
    transcribe_single_youtube,
)


ProgressCallback = Callable[[dict[str, Any]], None] | None


def _emit_progress(
    progress_callback: ProgressCallback,
    stage: str,
    percent: float,
    message: str,
    current: int | None = None,
    total: int | None = None,
) -> None:
    if progress_callback is None:
        return

    progress_callback(
        {
            "stage": stage,
            "percent": max(0.0, min(100.0, float(percent))),
            "message": message,
            "current": current,
            "total": total,
        }
    )


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _timestamp_str() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _build_temp_workspace(prefix: str, name: str | None = None) -> Path:
    safe_name = slugify(name or prefix)
    workspace = TEMP_DIR / "workflow_pipeline" / prefix / f"{_timestamp_str()}_{safe_name}"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _copy_files_to_folder(files: Sequence[str | Path], destination_dir: Path) -> list[Path]:
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_files: list[Path] = []
    for file_path in files:
        src = Path(file_path)
        if not src.exists() or not src.is_file():
            continue

        dest = destination_dir / src.name
        shutil.copy2(src, dest)
        copied_files.append(dest)

    return copied_files


def _filter_recent_files(
    files: list[str | Path],
    start_time: float,
    margin_sec: float = 3.0,
) -> list[Path]:
    selected: list[Path] = []

    for file_path in files:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            continue

        try:
            modified = path.stat().st_mtime
        except Exception:
            continue

        if modified >= (start_time - margin_sec):
            selected.append(path)

    return selected


def _run_metadata_only(
    source_type: str,
    source_path: str | Path,
    transcript_column: str | None = None,
    filename_column: str | None = None,
    sheet_name: str | None = None,
    output_name: str | None = None,
    output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    return run_metadata_generation(
        source_type=source_type,
        source_path=source_path,
        transcript_column=transcript_column,
        filename_column=filename_column,
        sheet_name=sheet_name,
        output_name=output_name,
        output_dir=output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=progress_callback,
    )


def generate_metadata_from_single_youtube(
    youtube_url: str,
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    download_progress_callback: ProgressCallback = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
    transcription_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate metadata from a single YouTube URL.

    Flow:
    YouTube URL -> download/standardize/transcribe -> transcript -> LLM metadata.

    transcription_settings is forwarded into the transcription workflow so the
    Metadata / SEO tab can use the same dynamic runtime settings as the main
    Transcription tab.
    """

    def transcription_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = (raw_percent / 100.0) * 60.0

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "transcription"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    transcription_result = transcribe_single_youtube(
        youtube_url=youtube_url,
        download_progress_callback=download_progress_callback,
        progress_callback=transcription_progress_bridge,
        transcription_settings=transcription_settings,
    )

    transcript_file = Path(transcription_result.get("transcript_file", ""))

    if not transcription_result.get("ok", False) or not transcript_file.exists():
        return {
            "ok": False,
            "mode": "single_youtube_metadata",
            "transcription": transcription_result,
            "metadata": None,
            "error": transcription_result.get(
                "error",
                "Transcription failed or transcript file was not created.",
            ),
        }

    def metadata_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = 60.0 + ((raw_percent / 100.0) * 40.0)

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "metadata"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    metadata_result = _run_metadata_only(
        source_type="single_file",
        source_path=transcript_file,
        output_name=output_name or transcript_file.stem,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=metadata_progress_bridge,
    )

    return {
        "ok": bool(transcription_result.get("ok", False) and metadata_result.get("ok", False)),
        "mode": "single_youtube_metadata",
        "transcription": transcription_result,
        "metadata": metadata_result,
        "transcript_file": str(transcript_file),
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_single_media_file(
    media_file: str | Path,
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
    transcription_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate metadata from a single local audio/video file.

    Flow:
    media file -> standardize/transcribe -> transcript -> LLM metadata.
    """

    def transcription_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = (raw_percent / 100.0) * 60.0

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "transcription"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    transcription_result = transcribe_single_media_file(
        media_file=media_file,
        progress_callback=transcription_progress_bridge,
        transcription_settings=transcription_settings,
    )

    transcript_file = Path(transcription_result.get("transcript_file", ""))

    if not transcription_result.get("ok", False) or not transcript_file.exists():
        return {
            "ok": False,
            "mode": "single_media_metadata",
            "transcription": transcription_result,
            "metadata": None,
            "error": transcription_result.get(
                "error",
                "Transcription failed or transcript file was not created.",
            ),
        }

    def metadata_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = 60.0 + ((raw_percent / 100.0) * 40.0)

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "metadata"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    metadata_result = _run_metadata_only(
        source_type="single_file",
        source_path=transcript_file,
        output_name=output_name or transcript_file.stem,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=metadata_progress_bridge,
    )

    return {
        "ok": bool(transcription_result.get("ok", False) and metadata_result.get("ok", False)),
        "mode": "single_media_metadata",
        "transcription": transcription_result,
        "metadata": metadata_result,
        "transcript_file": str(transcript_file),
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_batch_media(
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
    transcription_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate metadata from the current batch media input.

    Flow:
    batch media -> standardize/transcribe -> transcript folder -> LLM metadata.
    """

    start_time = time.time()

    def transcription_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = (raw_percent / 100.0) * 55.0

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "transcription"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    transcription_result = transcribe_batch_media(
        progress_callback=transcription_progress_bridge,
        transcription_settings=transcription_settings,
    )

    transcript_files = transcription_result.get("transcript_files", []) or []
    recent_transcript_files = _filter_recent_files(transcript_files, start_time)

    if not recent_transcript_files:
        recent_transcript_files = [Path(p) for p in transcript_files if Path(p).exists()]

    if not recent_transcript_files:
        return {
            "ok": False,
            "mode": "batch_media_metadata",
            "transcription": transcription_result,
            "metadata": None,
            "error": transcription_result.get(
                "error",
                "No transcript files were found for metadata generation.",
            ),
        }

    workspace = _build_temp_workspace("batch_media_metadata", output_name or "batch_media")
    staged_transcripts_dir = workspace / "transcripts"
    staged_files = _copy_files_to_folder(recent_transcript_files, staged_transcripts_dir)

    def metadata_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = 55.0 + ((raw_percent / 100.0) * 45.0)

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "metadata"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    metadata_result = _run_metadata_only(
        source_type="folder",
        source_path=staged_transcripts_dir,
        output_name=output_name or "batch_media_metadata",
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=metadata_progress_bridge,
    )

    return {
        "ok": bool(transcription_result.get("ok", False) and metadata_result.get("ok", False)),
        "mode": "batch_media_metadata",
        "transcription": transcription_result,
        "metadata": metadata_result,
        "staged_transcripts_dir": str(staged_transcripts_dir),
        "transcript_files_used": [str(p) for p in staged_files],
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_playlist(
    playlist_url: str,
    quality: str = "720p",
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = METADATA_PLAYLIST_DIR,
    download_progress_callback: ProgressCallback = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
    transcription_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate metadata from a YouTube playlist.

    Flow:
    playlist -> download/standardize/transcribe -> transcript folder -> LLM metadata.
    """

    def transcription_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = (raw_percent / 100.0) * 60.0

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "transcription"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    transcription_result = transcribe_playlist(
        playlist_url=playlist_url,
        quality=quality,
        download_progress_callback=download_progress_callback,
        generate_playlist_excel_file=True,
        progress_callback=transcription_progress_bridge,
        transcription_settings=transcription_settings,
    )

    transcripts_dir = Path(transcription_result.get("transcripts_dir", ""))

    if not transcription_result.get("ok", False) or not transcripts_dir.exists():
        return {
            "ok": False,
            "mode": "playlist_metadata",
            "transcription": transcription_result,
            "metadata": None,
            "error": transcription_result.get(
                "error",
                "Playlist transcription failed or transcript folder was not created.",
            ),
        }

    playlist_root = Path(transcription_result.get("playlist_root", "")).name or "playlist"

    def metadata_progress_bridge(event: dict[str, Any]) -> None:
        raw_percent = float(event.get("percent", 0))
        mapped_percent = 60.0 + ((raw_percent / 100.0) * 40.0)

        _emit_progress(
            progress_callback,
            stage=event.get("stage", "metadata"),
            percent=mapped_percent,
            message=_safe_string(event.get("message", "")),
            current=event.get("current"),
            total=event.get("total"),
        )

    metadata_result = _run_metadata_only(
        source_type="folder",
        source_path=transcripts_dir,
        output_name=output_name or playlist_root,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=metadata_progress_bridge,
    )

    return {
        "ok": bool(transcription_result.get("ok", False) and metadata_result.get("ok", False)),
        "mode": "playlist_metadata",
        "transcription": transcription_result,
        "metadata": metadata_result,
        "playlist_excel": transcription_result.get("playlist_excel_file", ""),
        "metadata_excel": metadata_result.get("excel_output", ""),
        "transcripts_dir": str(transcripts_dir),
    }


def generate_metadata_from_single_transcript_file(
    transcript_file: str | Path,
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    transcript_path = Path(transcript_file)
    if not transcript_path.exists():
        raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

    metadata_result = _run_metadata_only(
        source_type="single_file",
        source_path=transcript_path,
        output_name=output_name or transcript_path.stem,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "ok": metadata_result.get("ok", False),
        "mode": "single_transcript_metadata",
        "metadata": metadata_result,
        "source_path": str(transcript_path),
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_transcript_folder(
    transcript_folder: str | Path,
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    folder_path = Path(transcript_folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Transcript folder not found: {folder_path}")

    metadata_result = _run_metadata_only(
        source_type="folder",
        source_path=folder_path,
        output_name=output_name or folder_path.name,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "ok": metadata_result.get("ok", False),
        "mode": "transcript_folder_metadata",
        "metadata": metadata_result,
        "source_path": str(folder_path),
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_transcript_files(
    transcript_files: Sequence[str | Path],
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    valid_files = [Path(p) for p in transcript_files if Path(p).exists() and Path(p).is_file()]

    if not valid_files:
        raise ValueError("No valid transcript files were provided.")

    workspace = _build_temp_workspace("transcript_files_metadata", output_name or "transcript_files")
    staged_transcripts_dir = workspace / "transcripts"
    staged_files = _copy_files_to_folder(valid_files, staged_transcripts_dir)

    metadata_result = _run_metadata_only(
        source_type="folder",
        source_path=staged_transcripts_dir,
        output_name=output_name or "transcript_files_metadata",
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "ok": metadata_result.get("ok", False),
        "mode": "transcript_files_metadata",
        "metadata": metadata_result,
        "staged_transcripts_dir": str(staged_transcripts_dir),
        "source_files": [str(p) for p in staged_files],
        "metadata_excel": metadata_result.get("excel_output", ""),
    }


def generate_metadata_from_excel(
    excel_file: str | Path,
    transcript_column: str,
    filename_column: str | None = None,
    sheet_name: str | None = None,
    output_name: str | None = None,
    metadata_output_dir: str | Path | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    sleep_ms: int = DEFAULT_SLEEP_MS,
    temperature: float = DEFAULT_TEMPERATURE,
    num_ctx: int = DEFAULT_NUM_CTX,
    num_predict: int = DEFAULT_NUM_PREDICT,
    seed: int | None = None,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    excel_path = Path(excel_file)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    metadata_result = _run_metadata_only(
        source_type="excel",
        source_path=excel_path,
        transcript_column=transcript_column,
        filename_column=filename_column,
        sheet_name=sheet_name,
        output_name=output_name or excel_path.stem,
        output_dir=metadata_output_dir,
        model=model,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        sleep_ms=sleep_ms,
        temperature=temperature,
        num_ctx=num_ctx,
        num_predict=num_predict,
        seed=seed,
        progress_callback=progress_callback,
    )

    return {
        "ok": metadata_result.get("ok", False),
        "mode": "excel_metadata",
        "metadata": metadata_result,
        "source_path": str(excel_path),
        "metadata_excel": metadata_result.get("excel_output", ""),
    }
