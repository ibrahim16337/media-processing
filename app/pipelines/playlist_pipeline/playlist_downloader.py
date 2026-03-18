from __future__ import annotations

import json
from typing import Any, Callable, cast

import yt_dlp

from app.config.paths import build_playlist_paths
from app.config.paths import FFMPEG_DIR

QUALITY_HEIGHTS = {
    "480p": 480,
    "720p": 720,
    "1080p": 1080,
}

def _as_dict(obj: Any, context: str = "yt-dlp result") -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj

    try:
        return dict(obj)
    except Exception as e:
        raise ValueError(f"{context} is not dictionary-like: {type(obj)!r}") from e


def inspect_playlist(url: str) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "quiet": True,
        "extract_flat": True,
        "noplaylist": False,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(cast(Any, opts)) as ydl:
        raw_info = ydl.extract_info(url, download=False)

    info = _as_dict(raw_info, "playlist preview")
    raw_entries = info.get("entries") or []

    entries: list[dict[str, Any]] = []

    if isinstance(raw_entries, list):
        for item in raw_entries:
            item_dict = _as_dict(item, "playlist entry") if item is not None else None

            if item_dict:
                entries.append(
                    {
                        "id": item_dict.get("id"),
                        "title": item_dict.get("title"),
                        "url": item_dict.get("webpage_url") or item_dict.get("url"),
                        "duration": item_dict.get("duration"),
                    }
                )

    return {
        "title": info.get("title"),
        "id": info.get("id"),
        "webpage_url": info.get("webpage_url"),
        "entry_count": len(entries),
        "entries": entries,
    }


def _build_ydl_opts(
    paths,
    max_height: int,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    return {
        "quiet": True,
        "noplaylist": False,
        "ignoreerrors": True,
        "ffmpeg_location": str(FFMPEG_DIR),
        "format": (
            f"bv*[ext=mp4][height<={max_height}]+ba[ext=m4a]/"
            f"b[ext=mp4][height<={max_height}]/"
            f"bv*[height<={max_height}]+ba/"
            f"b[height<={max_height}]/"
            f"best"
        ),
        "merge_output_format": "mp4",
        "outtmpl": str(
            paths.videos / "%(playlist_index)03d - %(title)s [%(id)s].%(ext)s"
        ),
        "download_archive": str(paths.archive_file),
        "progress_hooks": [progress_callback] if progress_callback else [],
    }


def download_playlist(
    url: str,
    quality: str = "720p",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    preview = inspect_playlist(url)

    playlist_name = preview.get("title") or preview.get("id") or "playlist"

    paths = build_playlist_paths(str(playlist_name))
    max_height = QUALITY_HEIGHTS.get(quality, 720)

    ydl_opts = _build_ydl_opts(
        paths=paths,
        max_height=max_height,
        progress_callback=progress_callback,
    )

    with yt_dlp.YoutubeDL(cast(Any, ydl_opts)) as ydl:
        raw_info = ydl.extract_info(url, download=True)

    info = _as_dict(raw_info, "playlist download result")
    raw_entries = info.get("entries") or []

    entries: list[dict[str, Any]] = []

    if isinstance(raw_entries, list):
        for item in raw_entries:
            if item is None:
                continue

            item_dict = _as_dict(item, "downloaded playlist entry")

            entries.append(
                {
                    "id": item_dict.get("id"),
                    "title": item_dict.get("title"),
                    "url": item_dict.get("webpage_url") or item_dict.get("url"),
                    "duration": item_dict.get("duration"),
                }
            )

    manifest = {
        "playlist_title": info.get("title"),
        "playlist_id": info.get("id"),
        "playlist_url": info.get("webpage_url"),
        "requested_quality": quality,
        "max_height": max_height,
        "entry_count": len(entries),
        "videos_dir": str(paths.videos),
        "audio_dir": str(paths.audio),
        "transcripts_dir": str(paths.transcripts),
        "archive_file": str(paths.archive_file),
        "entries": entries,
    }

    paths.manifest_file.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return {
        "playlist_name": str(playlist_name),
        "paths": paths,
        "manifest": manifest,
        "info": info,
    }