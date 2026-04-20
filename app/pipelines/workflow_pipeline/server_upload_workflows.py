from __future__ import annotations

import posixpath
from typing import Any

from app.config.server_upload_config import (
    get_server_upload_config,
    mask_server_upload_config,
)
from app.pipelines.server_upload_pipeline.server_client import SFTPServerClient


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