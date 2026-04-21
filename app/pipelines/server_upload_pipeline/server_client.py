from __future__ import annotations

import posixpath
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import paramiko

from app.config.server_upload_config import get_server_upload_config


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
FileProgressCallback = Callable[[int, int], None] | None


@dataclass(frozen=True)
class RemoteFolderEntry:
    name: str
    path: str
    type: str = "folder"


@dataclass(frozen=True)
class RemoteFileEntry:
    name: str
    path: str
    type: str = "file"
    size_bytes: int = 0
    modified: str = ""
    is_video: bool = False


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_video_filename(filename: str) -> bool:
    lower_name = _safe_string(filename).lower()
    return any(lower_name.endswith(ext) for ext in VIDEO_EXTS)


def _format_modified_time(epoch_value: float | int | None) -> str:
    if epoch_value is None:
        return ""
    try:
        return datetime.fromtimestamp(float(epoch_value)).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return ""


class SFTPServerClient:
    def __init__(self) -> None:
        self.config = get_server_upload_config()
        self.transport: paramiko.Transport | None = None
        self.sftp: paramiko.SFTPClient | None = None

    def connect(self) -> None:
        if self.transport is not None and self.sftp is not None:
            return

        transport = paramiko.Transport((self.config.host, self.config.port))
        transport.connect(
            username=self.config.username,
            password=self.config.password,
        )

        self.transport = transport
        self.sftp = paramiko.SFTPClient.from_transport(transport)

    def close(self) -> None:
        if self.sftp is not None:
            try:
                self.sftp.close()
            except Exception:
                pass
            self.sftp = None

        if self.transport is not None:
            try:
                self.transport.close()
            except Exception:
                pass
            self.transport = None

    def __enter__(self) -> "SFTPServerClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_sftp(self) -> paramiko.SFTPClient:
        if self.sftp is None:
            raise RuntimeError("SFTP client is not connected.")
        return self.sftp

    def normalize_remote_path(self, remote_path: str | None) -> str:
        path = _safe_string(remote_path) or self.config.remote_root or "/"
        normalized = posixpath.normpath(path)

        if not normalized.startswith("/"):
            normalized = "/" + normalized

        return normalized

    def path_exists(self, remote_path: str) -> bool:
        sftp = self._require_sftp()
        path = self.normalize_remote_path(remote_path)

        try:
            sftp.stat(path)
            return True
        except FileNotFoundError:
            return False
        except IOError:
            return False

    def is_directory(self, remote_path: str) -> bool:
        sftp = self._require_sftp()
        path = self.normalize_remote_path(remote_path)

        try:
            attrs = sftp.stat(path)
            mode = int(getattr(attrs, "st_mode", 0) or 0)
            return stat.S_ISDIR(mode)
        except FileNotFoundError:
            return False
        except IOError:
            return False

    def list_directory(self, remote_path: str | None = None) -> dict[str, Any]:
        sftp = self._require_sftp()
        path = self.normalize_remote_path(remote_path)

        items = sftp.listdir_attr(path)

        folders: list[RemoteFolderEntry] = []
        files: list[RemoteFileEntry] = []

        for item in items:
            item_name = _safe_string(getattr(item, "filename", ""))
            if not item_name:
                continue

            item_path = posixpath.join(path, item_name)
            mode = getattr(item, "st_mode", 0)

            if stat.S_ISDIR(mode):
                folders.append(
                    RemoteFolderEntry(
                        name=item_name,
                        path=item_path,
                    )
                )
            else:
                files.append(
                    RemoteFileEntry(
                        name=item_name,
                        path=item_path,
                        size_bytes=int(getattr(item, "st_size", 0) or 0),
                        modified=_format_modified_time(getattr(item, "st_mtime", None)),
                        is_video=_is_video_filename(item_name),
                    )
                )

        folders = sorted(folders, key=lambda x: x.name.lower())
        files = sorted(files, key=lambda x: x.name.lower())

        return {
            "path": path,
            "folder_count": len(folders),
            "file_count": len(files),
            "video_count": len([f for f in files if f.is_video]),
            "folders": [folder.__dict__ for folder in folders],
            "files": [file.__dict__ for file in files],
        }

    def list_root(self) -> dict[str, Any]:
        return self.list_directory(self.config.remote_root)

    def create_directory(
        self,
        remote_path: str,
        exist_ok: bool = False,
        recursive: bool = True,
    ) -> dict[str, Any]:
        sftp = self._require_sftp()
        normalized_path = self.normalize_remote_path(remote_path)

        if normalized_path == "/":
            return {
                "created": False,
                "path": "/",
                "name": "/",
            }

        if self.path_exists(normalized_path):
            if not self.is_directory(normalized_path):
                raise NotADirectoryError(f"Remote path exists but is not a directory: {normalized_path}")
            if not exist_ok:
                raise FileExistsError(f"Remote directory already exists: {normalized_path}")
            return {
                "created": False,
                "path": normalized_path,
                "name": posixpath.basename(normalized_path),
            }

        parts = [part for part in normalized_path.split("/") if part]
        current = "/"
        created_any = False

        for index, part in enumerate(parts, start=1):
            if current == "/":
                current = f"/{part}"
            else:
                current = posixpath.join(current, part)

            if self.path_exists(current):
                if not self.is_directory(current):
                    raise NotADirectoryError(f"Remote path exists but is not a directory: {current}")
                continue

            if not recursive and index != len(parts):
                raise FileNotFoundError(
                    f"Parent directory does not exist for non-recursive create: {normalized_path}"
                )

            sftp.mkdir(current)
            created_any = True

        return {
            "created": created_any,
            "path": normalized_path,
            "name": posixpath.basename(normalized_path),
        }

    def delete_file(self, remote_file_path: str) -> dict[str, Any]:
        sftp = self._require_sftp()
        normalized_path = self.normalize_remote_path(remote_file_path)

        if not self.path_exists(normalized_path):
            raise FileNotFoundError(f"Remote file not found: {normalized_path}")

        attrs = sftp.stat(normalized_path)
        mode = int(getattr(attrs, "st_mode", 0) or 0)
        
        if stat.S_ISDIR(mode):
            raise IsADirectoryError(f"Remote path is a directory, not a file: {normalized_path}")

        sftp.remove(normalized_path)

        return {
            "deleted": True,
            "remote_path": normalized_path,
            "filename": posixpath.basename(normalized_path),
        }

    def upload_file(
        self,
        local_path: str | Path,
        remote_dir: str,
        remote_filename: str | None = None,
        progress_callback: FileProgressCallback = None,
    ) -> dict[str, Any]:
        sftp = self._require_sftp()

        local_file = Path(local_path)
        if not local_file.exists():
            raise FileNotFoundError(f"Local file not found: {local_file}")

        remote_dir = self.normalize_remote_path(remote_dir)
        remote_filename = _safe_string(remote_filename) or local_file.name
        remote_path = posixpath.join(remote_dir, remote_filename)

        sftp.put(
            str(local_file),
            remote_path,
            callback=progress_callback,
            confirm=True,
        )

        stat_result = sftp.stat(remote_path)

        return {
            "remote_path": remote_path,
            "remote_filename": remote_filename,
            "size_bytes": int(getattr(stat_result, "st_size", 0) or 0),
            "modified": _format_modified_time(getattr(stat_result, "st_mtime", None)),
        }