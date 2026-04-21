from __future__ import annotations

import posixpath
from pathlib import Path
from typing import Any, Callable

from app.pipelines.server_upload_pipeline.server_client import (
    SFTPServerClient,
    VIDEO_EXTS,
)


ProgressCallback = Callable[[dict[str, Any]], None] | None


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


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


def upload_video_files(
    upload_items: list[dict[str, Any]],
    remote_dir: str,
    overwrite: bool = False,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    valid_items: list[dict[str, Any]] = []

    for item in upload_items:
        local_path = Path(item["local_path"])
        target_name = _safe_string(item.get("target_name")) or local_path.name

        if not local_path.exists():
            continue

        if local_path.suffix.lower() not in VIDEO_EXTS:
            continue

        if not Path(target_name).suffix:
            target_name = f"{target_name}{local_path.suffix}"

        valid_items.append(
            {
                "local_path": local_path,
                "target_name": target_name,
                "original_name": local_path.name,
            }
        )

    if not valid_items:
        raise ValueError("No valid video files were provided for upload.")

    total_files = len(valid_items)
    results: list[dict[str, Any]] = []

    _emit_progress(
        progress_callback,
        stage="upload",
        percent=0,
        message="Starting server upload...",
        current=0,
        total=total_files,
    )

    with SFTPServerClient() as client:
        remote_dir = client.normalize_remote_path(remote_dir)

        for index, item in enumerate(valid_items, start=1):
            local_path: Path = item["local_path"]
            target_name: str = item["target_name"]
            remote_path = posixpath.join(remote_dir, target_name)

            if not overwrite and client.path_exists(remote_path):
                results.append(
                    {
                        "original_name": item["original_name"],
                        "uploaded_name": target_name,
                        "remote_path": remote_path,
                        "status": "skipped",
                        "reason": "File already exists on server.",
                    }
                )

                done_percent = (index / total_files) * 100
                _emit_progress(
                    progress_callback,
                    stage="upload",
                    percent=done_percent,
                    message=f"Skipped existing file {index} of {total_files}: {target_name}",
                    current=index,
                    total=total_files,
                )
                continue

            def file_progress(sent_bytes: int, total_bytes: int) -> None:
                fraction = (sent_bytes / total_bytes) if total_bytes else 0.0
                overall_percent = ((index - 1) + fraction) / total_files * 100

                _emit_progress(
                    progress_callback,
                    stage="upload",
                    percent=overall_percent,
                    message=f"Uploading file {index} of {total_files}: {target_name}",
                    current=index,
                    total=total_files,
                )

            try:
                upload_result = client.upload_file(
                    local_path=local_path,
                    remote_dir=remote_dir,
                    remote_filename=target_name,
                    progress_callback=file_progress,
                )

                results.append(
                    {
                        "original_name": item["original_name"],
                        "uploaded_name": target_name,
                        "remote_path": upload_result["remote_path"],
                        "size_bytes": upload_result["size_bytes"],
                        "modified": upload_result["modified"],
                        "status": "uploaded",
                        "reason": "",
                    }
                )

            except Exception as e:
                results.append(
                    {
                        "original_name": item["original_name"],
                        "uploaded_name": target_name,
                        "remote_path": remote_path,
                        "status": "failed",
                        "reason": str(e),
                    }
                )

            done_percent = (index / total_files) * 100
            _emit_progress(
                progress_callback,
                stage="upload",
                percent=done_percent,
                message=f"Processed file {index} of {total_files}: {target_name}",
                current=index,
                total=total_files,
            )

    uploaded_count = len([r for r in results if r["status"] == "uploaded"])
    skipped_count = len([r for r in results if r["status"] == "skipped"])
    failed_count = len([r for r in results if r["status"] == "failed"])

    _emit_progress(
        progress_callback,
        stage="upload",
        percent=100,
        message="Server upload completed.",
        current=total_files,
        total=total_files,
    )

    return {
        "ok": failed_count == 0,
        "remote_dir": remote_dir,
        "total_files": total_files,
        "uploaded_count": uploaded_count,
        "skipped_count": skipped_count,
        "failed_count": failed_count,
        "results": results,
    }