from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import yt_dlp

from app.config.paths import UPLOAD_AUDIO_DIR


def download_youtube_audio(url, progress_callback=None):

    output_template = str(UPLOAD_AUDIO_DIR / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "progress_hooks": [progress_callback] if progress_callback else [],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

    return Path(filename)

def _as_dict(obj: Any, context: str = "yt-dlp result") -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    try:
        return dict(obj)
    except Exception as e:
        raise ValueError(f"{context} is not dictionary-like: {type(obj)!r}") from e

def fetch_youtube_video_metadata(url: str) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
        raw_info = ydl.extract_info(url, download=False)

    info = _as_dict(raw_info, "youtube metadata")

    width = info.get("width")
    height = info.get("height")

    if (not width or not height) and isinstance(info.get("formats"), list):
        for fmt in reversed(info["formats"]):
            if not isinstance(fmt, dict):
                continue

            fmt_width = fmt.get("width")
            fmt_height = fmt.get("height")
            vcodec = fmt.get("vcodec")

            if fmt_width and fmt_height and vcodec and vcodec != "none":
                width = fmt_width
                height = fmt_height
                break

    return {
        "video_id": info.get("id"),
        "title": info.get("title"),
        "duration_seconds": info.get("duration"),
        "video_width": width,
        "video_height": height,
        "webpage_url": info.get("webpage_url") or url,
    }