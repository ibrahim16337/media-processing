from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class ServerUploadConfig:
    protocol: str
    host: str
    port: int
    username: str
    password: str
    remote_root: str
    auto_accept_host_key: bool
    web_url: str


def _safe_string(value: str | None) -> str:
    return (value or "").strip()


def _to_bool(value: str | None, default: bool = False) -> bool:
    normalized = _safe_string(value).lower()

    if not normalized:
        return default

    return normalized in {"1", "true", "yes", "y", "on"}


def get_server_upload_config() -> ServerUploadConfig:
    protocol = _safe_string(os.getenv("SERVER_UPLOAD_PROTOCOL", "sftp")).lower()
    host = _safe_string(os.getenv("SERVER_UPLOAD_HOST"))
    port_raw = _safe_string(os.getenv("SERVER_UPLOAD_PORT", "22"))
    username = _safe_string(os.getenv("SERVER_UPLOAD_USERNAME"))
    password = _safe_string(os.getenv("SERVER_UPLOAD_PASSWORD"))
    remote_root = _safe_string(os.getenv("SERVER_UPLOAD_REMOTE_ROOT", "/"))
    web_url = _safe_string(os.getenv("SERVER_UPLOAD_WEB_URL"))
    auto_accept_host_key = _to_bool(
        os.getenv("SERVER_UPLOAD_AUTO_ACCEPT_HOST_KEY", "true"),
        default=True,
    )

    if protocol != "sftp":
        raise ValueError(
            f"Unsupported SERVER_UPLOAD_PROTOCOL='{protocol}'. "
            f"This pipeline is currently configured for SFTP only."
        )

    if not host:
        raise ValueError("Missing SERVER_UPLOAD_HOST in environment.")
    if not username:
        raise ValueError("Missing SERVER_UPLOAD_USERNAME in environment.")
    if not password:
        raise ValueError("Missing SERVER_UPLOAD_PASSWORD in environment.")

    try:
        port = int(port_raw)
    except ValueError as e:
        raise ValueError(f"Invalid SERVER_UPLOAD_PORT='{port_raw}'. Must be an integer.") from e

    if port <= 0:
        raise ValueError("SERVER_UPLOAD_PORT must be greater than 0.")

    if not remote_root:
        remote_root = "/"

    if not remote_root.startswith("/"):
        remote_root = "/" + remote_root

    return ServerUploadConfig(
        protocol=protocol,
        host=host,
        port=port,
        username=username,
        password=password,
        remote_root=remote_root,
        auto_accept_host_key=auto_accept_host_key,
        web_url=web_url,
    )


def mask_server_upload_config(config: ServerUploadConfig) -> dict[str, str | int | bool]:
    return {
        "protocol": config.protocol,
        "host": config.host,
        "port": config.port,
        "username": config.username,
        "password": "********",
        "remote_root": config.remote_root,
        "auto_accept_host_key": config.auto_accept_host_key,
        "web_url": config.web_url,
    }