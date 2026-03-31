from __future__ import annotations

import json
from typing import Any


REQUIRED_KEYS = ("title", "description", "tags", "hashtags")


def _normalize_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def parse_metadata_json(raw_text: str) -> dict[str, Any]:
    raw_text = (raw_text or "").strip()

    if not raw_text:
        return {
            "ok": False,
            "data": {
                "title": "",
                "description": "",
                "tags": "",
                "hashtags": "",
            },
            "error": "Empty LLM response.",
            "raw_text": raw_text,
        }

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {
            "ok": False,
            "data": {
                "title": "",
                "description": "",
                "tags": "",
                "hashtags": "",
            },
            "error": f"Invalid JSON response: {e}",
            "raw_text": raw_text,
        }

    if not isinstance(parsed, dict):
        return {
            "ok": False,
            "data": {
                "title": "",
                "description": "",
                "tags": "",
                "hashtags": "",
            },
            "error": "LLM response JSON is not an object.",
            "raw_text": raw_text,
        }

    missing_keys = [key for key in REQUIRED_KEYS if key not in parsed]
    if missing_keys:
        return {
            "ok": False,
            "data": {
                "title": _normalize_string(parsed.get("title", "")),
                "description": _normalize_string(parsed.get("description", "")),
                "tags": _normalize_string(parsed.get("tags", "")),
                "hashtags": _normalize_string(parsed.get("hashtags", "")),
            },
            "error": f"Missing required keys: {', '.join(missing_keys)}",
            "raw_text": raw_text,
        }

    normalized = {
        "title": _normalize_string(parsed.get("title", "")),
        "description": _normalize_string(parsed.get("description", "")),
        "tags": _normalize_string(parsed.get("tags", "")),
        "hashtags": _normalize_string(parsed.get("hashtags", "")),
    }

    return {
        "ok": True,
        "data": normalized,
        "error": "",
        "raw_text": raw_text,
    }


def build_result_row(filename: str, parsed_result: dict[str, Any]) -> dict[str, str]:
    data = parsed_result.get("data", {}) or {}

    return {
        "filename": (filename or "").strip(),
        "title": _normalize_string(data.get("title", "")),
        "description": _normalize_string(data.get("description", "")),
        "tags": _normalize_string(data.get("tags", "")),
        "hashtags": _normalize_string(data.get("hashtags", "")),
    }