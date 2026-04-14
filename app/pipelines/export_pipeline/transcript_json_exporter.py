from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _safe_value(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def export_transcript_json(
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
                "filename": _safe_value(row.get("filename", "")),
                "transcription": _safe_value(row.get("transcription", "")),
                "audio length": _safe_value(row.get("audio length", "")),
                "video type": _safe_value(row.get("video type", "")),
            }
        )

    json_text = json.dumps(
        normalized_rows,
        ensure_ascii=False,
        indent=2 if pretty else None,
    )

    output_path.write_text(json_text, encoding="utf-8")
    return output_path