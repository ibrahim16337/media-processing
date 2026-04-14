from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def export_metadata_json(
    rows: list[dict[str, Any]],
    output_file: str | Path,
    pretty: bool = True,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    normalized_rows: list[dict[str, str]] = []

    for row in rows:
        normalized_rows.append(
            {
                "filename": _safe_string(row.get("filename", "")),
                "title": _safe_string(row.get("title", "")),
                "transcript": _safe_string(row.get("transcript", "")),
                "video_length": _safe_string(row.get("video_length", "")),
                "video_type": _safe_string(row.get("video_type", "")),
                "description": _safe_string(row.get("description", "")),
                "tags": _safe_string(row.get("tags", "")),
                "hashtags": _safe_string(row.get("hashtags", "")),
                "upload_link": _safe_string(row.get("upload_link", "")),
            }
        )

    json_text = json.dumps(
        normalized_rows,
        ensure_ascii=False,
        indent=2 if pretty else None,
    )

    output_path.write_text(json_text, encoding="utf-8")
    return output_path