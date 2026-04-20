from __future__ import annotations

import posixpath
import stat
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import paramiko

from app.config.server_upload_config import get_server_upload_config


VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}


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