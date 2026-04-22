from __future__ import annotations

import posixpath
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from app.config.paths import SERVER_UPLOAD_TEMP_DIR
from app.config.server_upload_config import (
    get_server_upload_config,
    mask_server_upload_config,
)
from app.pipelines.server_upload_pipeline.server_client import SFTPServerClient
from app.pipelines.server_upload_pipeline.upload_runner import upload_video_files


ProgressCallback = Callable[[dict[str, Any]], None] | None


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _normalize_remote_path(path: str) -> str:
    normalized = posixpath.normpath(_safe_string(path) or "/")

    if not normalized.startswith("/"):
        normalized = "/" + normalized

    return normalized


def _build_parent_path(current_path: str) -> str | None:
    current_path = _normalize_remote_path(current_path)

    if current_path == "/":
        return None

    parent = posixpath.dirname(current_path)
    parent = _normalize_remote_path(parent)

    return parent


def _build_breadcrumbs(current_path: str) -> list[dict[str, str]]:
    current_path = _normalize_remote_path(current_path)

    if current_path == "/":
        return [{"name": "/", "path": "/"}]

    parts = [part for part in current_path.split("/") if part]
    breadcrumbs: list[dict[str, str]] = [{"name": "/", "path": "/"}]

    running_path = ""
    for part in parts:
        running_path = f"{running_path}/{part}"
        breadcrumbs.append(
            {
                "name": part,
                "path": running_path,
            }
        )

    return breadcrumbs


def _validate_new_folder_name(folder_name: str) -> str:
    name = _safe_string(folder_name)

    if not name:
        raise ValueError("Folder name is required.")
    if name in {".", ".."}:
        raise ValueError("Folder name cannot be '.' or '..'.")
    if "/" in name or "\\" in name:
        raise ValueError("Folder name cannot contain slashes.")

    return name


def _filter_files(files: list[dict[str, Any]], query: str, videos_only: bool = False) -> list[dict[str, Any]]:
    normalized_query = _safe_string(query).lower()

    filtered = []
    for file in files:
        if videos_only and not bool(file.get("is_video", False)):
            continue

        filename = _safe_string(file.get("name", "")).lower()
        if normalized_query and normalized_query not in filename:
            continue

        filtered.append(file)

    return filtered


def browse_remote_directory(remote_path: str | None = None) -> dict[str, Any]:
    config = get_server_upload_config()
    requested_path = _safe_string(remote_path) or config.remote_root or "/"

    with SFTPServerClient() as client:
        snapshot = client.list_directory(requested_path)

    current_path = _normalize_remote_path(snapshot["path"])
    parent_path = _build_parent_path(current_path)
    breadcrumbs = _build_breadcrumbs(current_path)

    return {
        "ok": True,
        "connection": mask_server_upload_config(config),
        "current_path": current_path,
        "parent_path": parent_path,
        "breadcrumbs": breadcrumbs,
        "folder_count": snapshot["folder_count"],
        "file_count": snapshot["file_count"],
        "video_count": snapshot["video_count"],
        "folders": snapshot["folders"],
        "files": snapshot["files"],
        "video_files": [file for file in snapshot["files"] if file.get("is_video", False)],
    }


def browse_remote_root() -> dict[str, Any]:
    config = get_server_upload_config()
    return browse_remote_directory(config.remote_root)


def browse_remote_parent(current_path: str) -> dict[str, Any]:
    parent_path = _build_parent_path(current_path)

    if parent_path is None:
        return browse_remote_directory("/")

    return browse_remote_directory(parent_path)


def browse_remote_child(current_path: str, child_folder_name: str) -> dict[str, Any]:
    current_path = _normalize_remote_path(current_path)
    child_name = _safe_string(child_folder_name)

    if not child_name:
        raise ValueError("child_folder_name is required.")

    child_path = posixpath.join(current_path, child_name)
    child_path = _normalize_remote_path(child_path)

    return browse_remote_directory(child_path)


def browse_remote_path(remote_path: str) -> dict[str, Any]:
    return browse_remote_directory(remote_path)


def create_remote_folder(current_path: str, folder_name: str) -> dict[str, Any]:
    current_path = _normalize_remote_path(current_path)
    validated_name = _validate_new_folder_name(folder_name)
    new_folder_path = _normalize_remote_path(posixpath.join(current_path, validated_name))

    with SFTPServerClient() as client:
        create_result = client.create_directory(
            new_folder_path,
            exist_ok=False,
            recursive=True,
        )

    refreshed_snapshot = browse_remote_directory(current_path)

    return {
        "ok": True,
        "current_path": current_path,
        "created_folder_name": validated_name,
        "created_folder_path": create_result["path"],
        "snapshot": refreshed_snapshot,
    }


def delete_remote_file(remote_file_path: str, current_path: str | None = None) -> dict[str, Any]:
    normalized_file_path = _normalize_remote_path(remote_file_path)

    with SFTPServerClient() as client:
        delete_result = client.delete_file(normalized_file_path)

    refresh_path = _normalize_remote_path(current_path) if current_path else _normalize_remote_path(
        posixpath.dirname(normalized_file_path)
    )

    refreshed_snapshot = browse_remote_directory(refresh_path)

    return {
        "ok": True,
        "deleted_file_name": delete_result["filename"],
        "deleted_file_path": delete_result["remote_path"],
        "snapshot": refreshed_snapshot,
    }


def search_remote_videos(current_path: str, query: str) -> dict[str, Any]:
    snapshot = browse_remote_directory(current_path)
    matches = _filter_files(snapshot.get("files", []), query=query, videos_only=True)

    return {
        "ok": True,
        "current_path": snapshot["current_path"],
        "query": _safe_string(query),
        "match_count": len(matches),
        "matches": matches,
        "snapshot": snapshot,
    }


def filter_remote_files_in_snapshot(snapshot: dict[str, Any], query: str, videos_only: bool = False) -> list[dict[str, Any]]:
    files = snapshot.get("files", [])
    return _filter_files(files, query=query, videos_only=videos_only)


def upload_streamlit_video_files(
    uploaded_files: list[Any],
    remote_dir: str,
    rename_map: dict[str, str] | None = None,
    overwrite: bool = False,
    progress_callback: ProgressCallback = None,
) -> dict[str, Any]:
    if not uploaded_files:
        raise ValueError("No uploaded files were provided.")

    rename_map = rename_map or {}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = SERVER_UPLOAD_TEMP_DIR / f"upload_{timestamp}"
    workspace.mkdir(parents=True, exist_ok=True)

    staged_items: list[dict[str, Any]] = []

    try:
        for uploaded_file in uploaded_files:
            original_name = Path(uploaded_file.name).name
            local_path = workspace / original_name

            with open(local_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            target_name = _safe_string(rename_map.get(original_name)) or original_name

            staged_items.append(
                {
                    "local_path": local_path,
                    "target_name": target_name,
                    "original_name": original_name,
                }
            )

        upload_result = upload_video_files(
            upload_items=staged_items,
            remote_dir=remote_dir,
            overwrite=overwrite,
            progress_callback=progress_callback,
        )
        
        config = get_server_upload_config()
        web_url = config.web_url
        results_with_urls: list[dict[str, Any]] = []
        
        for row in upload_result["results"]:
            row_copy = dict(row)
            remote_path = _safe_string(row_copy.get("remote_path", ""))
            row_copy["public_url"] = _build_public_file_url(web_url, remote_path) if remote_path else ""
            results_with_urls.append(row_copy)
        
        
        return {
            "ok": upload_result["ok"],
            "remote_dir": upload_result["remote_dir"],
            "workspace": str(workspace),
            "total_files": upload_result["total_files"],
            "uploaded_count": upload_result["uploaded_count"],
            "skipped_count": upload_result["skipped_count"],
            "failed_count": upload_result["failed_count"],
            "results": results_with_urls,
        }

    finally:
        try:
            shutil.rmtree(workspace, ignore_errors=True)
        except Exception:
            pass


def _normalize_web_base_url(web_url: str) -> str:
    value = _safe_string(web_url)
    if not value:
        return ""
    return value.rstrip("/")


def _build_public_file_url(web_url: str, remote_path: str) -> str:
    base = _normalize_web_base_url(web_url)
    normalized_remote_path = _normalize_remote_path(remote_path)

    if not base:
        return ""

    encoded_path = quote(normalized_remote_path, safe="/-_.~")
    return f"{base}{encoded_path}"