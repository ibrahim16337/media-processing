from __future__ import annotations

from typing import Any

from app.pipelines.metadata_generation_pipeline.prompt_presets import (
    DEFAULT_PROMPT_PRESETS,
    build_simple_prompt_from_settings,
)


DEFAULT_CREATIVE_PROMPT = str(
    DEFAULT_PROMPT_PRESETS["Default SEO Metadata"].get("creative_prompt", "")
).strip()


LOCKED_JSON_OUTPUT_INSTRUCTIONS = """
LOCKED OUTPUT RULES:
These rules are mandatory and must be followed even if the editable prompt says otherwise.

- Output ONLY valid JSON.
- Do not use markdown.
- Do not add explanations before or after the JSON.
- JSON keys must be exactly: "title", "description", "tags", "hashtags".
- All values must be strings only.
- Do not return arrays or nested objects.
- The final response must match this structure exactly:

{"title":"...","description":"...","tags":"tag1, tag2, tag3","hashtags":"#Tag #Tag #Tag"}
""".strip()


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _replace_transcript_placeholder(prompt: str, transcript_text: str) -> str:
    if "{{text}}" in prompt:
        return prompt.replace("{{text}}", transcript_text)

    if "{{transcript}}" in prompt:
        return prompt.replace("{{transcript}}", transcript_text)

    return f"""{prompt}

Transcript:
<transcript>
{transcript_text}
</transcript>"""


def build_metadata_prompt(
    transcript_text: str,
    creative_prompt: str | None = None,
    prompt_settings: dict[str, Any] | None = None,
) -> str:
    """
    Build the final prompt for metadata generation.

    The user-editable prompt controls the creative/SEO style.
    The locked JSON output instructions are always appended so the parser can
    reliably parse the LLM response.
    """
    transcript_text = transcript_text or ""
    prompt_settings = prompt_settings or {}

    mode = _safe_string(prompt_settings.get("mode", "")).lower()

    if mode == "simple":
        editable_prompt = build_simple_prompt_from_settings(
            prompt_settings.get("simple_settings", {}) or {}
        )
    else:
        editable_prompt = _safe_string(
            prompt_settings.get("creative_prompt")
            or creative_prompt
            or DEFAULT_CREATIVE_PROMPT
        )

    if not editable_prompt:
        editable_prompt = DEFAULT_CREATIVE_PROMPT

    prompt_with_transcript = _replace_transcript_placeholder(
        editable_prompt,
        transcript_text,
    )

    return f"""{prompt_with_transcript}

{LOCKED_JSON_OUTPUT_INSTRUCTIONS}""".strip()
