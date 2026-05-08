from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config.paths import DATA_DIR, slugify


PROMPT_PRESETS_DIR = DATA_DIR / "prompt_presets"
PROMPT_PRESETS_FILE = PROMPT_PRESETS_DIR / "metadata_prompt_presets.json"


DEFAULT_PROMPT_PRESETS: dict[str, dict[str, Any]] = {
    "Default SEO Metadata": {
        "name": "Default SEO Metadata",
        "description": "General Urdu/Roman Urdu YouTube metadata prompt for Islamic lecture transcripts.",
        "mode": "advanced",
        "readonly": True,
        "creative_prompt": """ROLE: You are an expert YouTube SEO metadata editor.

TASK:
Read the transcript and generate metadata for a respectful Islamic educational video.

STYLE:
- Keep the tone respectful, clear, informative, and YouTube-friendly.
- Avoid clickbait.
- Avoid mentioning that this is a video, lecture, clip, talk, speaker, channel, episode, or session.
- Use the real subject matter from the transcript.
- If the transcript is Urdu, understand it fully before writing metadata.

TITLE RULES:
- Write one strong title.
- Use natural Roman Urdu.
- Start with a concrete topic, theme, concept, name, Surah, Ayah, event, or issue from the transcript.
- Do not start with generic words like "This", "Today", "Understanding", "Introduction", "Overview", or "Bayan".

DESCRIPTION RULES:
- Write a clear English description.
- Start directly with the topic, not with "This video..." or "In this lecture...".
- Explain the main theme, message, and context in a respectful way.
- Keep it useful for YouTube search and viewers.

TAGS RULES:
- Generate exactly 20 comma-separated tags.
- First 10 tags should be Roman Urdu.
- Last 10 tags should be English where possible.
- Tags must be specific to the transcript.

HASHTAGS RULES:
- Generate 3 to 5 relevant hashtags.
""",
        "simple_settings": {
            "video_type": "Islamic educational content",
            "platform": "YouTube",
            "language": "Mixed: Roman Urdu title/tags and English description",
            "tone": "Respectful, informative, and SEO-focused",
            "title_style": "Topic-first Roman Urdu",
            "description_style": "Detailed English encyclopedia-style description",
            "tag_count": 20,
            "hashtag_count": "3-5",
            "extra_instructions": "Avoid clickbait and avoid media words such as video, lecture, clip, talk, speaker, channel, episode, or session.",
        },
    },
    "Short Islamic Reminder": {
        "name": "Short Islamic Reminder",
        "description": "For short emotional reminder clips, reels, and short-form Islamic content.",
        "mode": "simple",
        "readonly": True,
        "creative_prompt": "",
        "simple_settings": {
            "video_type": "Short Islamic reminder",
            "platform": "YouTube Shorts / Instagram Reels / Facebook Reels",
            "language": "Roman Urdu title, English description, mixed tags",
            "tone": "Emotional, respectful, concise, and spiritually impactful",
            "title_style": "Short, powerful, topic-first Roman Urdu",
            "description_style": "Short but meaningful English description",
            "tag_count": 20,
            "hashtag_count": "3-5",
            "extra_instructions": "Make the title emotionally strong but not clickbait. Keep the metadata suitable for short-form content.",
        },
    },
    "Full Lecture Metadata": {
        "name": "Full Lecture Metadata",
        "description": "For longer lectures and detailed educational uploads.",
        "mode": "simple",
        "readonly": True,
        "creative_prompt": "",
        "simple_settings": {
            "video_type": "Full Islamic lecture",
            "platform": "YouTube",
            "language": "Roman Urdu title, English description, mixed tags",
            "tone": "Scholarly, respectful, clear, and educational",
            "title_style": "Detailed topic-first Roman Urdu",
            "description_style": "Longer structured English description covering major themes",
            "tag_count": 20,
            "hashtag_count": "3-5",
            "extra_instructions": "Focus on the main themes, named concepts, Surah/Ayah references, historical references, and key educational value.",
        },
    },
}


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _normalise_preset(preset: dict[str, Any]) -> dict[str, Any]:
    name = str(preset.get("name", "")).strip()
    if not name:
        name = "Untitled Prompt"

    mode = str(preset.get("mode", "advanced")).strip().lower()
    if mode not in {"simple", "advanced"}:
        mode = "advanced"

    simple_settings = preset.get("simple_settings")
    if not isinstance(simple_settings, dict):
        simple_settings = {}

    return {
        "name": name,
        "description": str(preset.get("description", "") or "").strip(),
        "mode": mode,
        "readonly": bool(preset.get("readonly", False)),
        "creative_prompt": str(preset.get("creative_prompt", "") or ""),
        "simple_settings": simple_settings,
        "created_at": str(preset.get("created_at", "") or _now_iso()),
        "updated_at": str(preset.get("updated_at", "") or _now_iso()),
    }


def _default_presets_copy() -> dict[str, dict[str, Any]]:
    now = _now_iso()
    result: dict[str, dict[str, Any]] = {}

    for name, preset in DEFAULT_PROMPT_PRESETS.items():
        item = _normalise_preset(preset)
        item["created_at"] = now
        item["updated_at"] = now
        item["readonly"] = True
        result[name] = item

    return result


def ensure_prompt_presets_file() -> Path:
    PROMPT_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    if not PROMPT_PRESETS_FILE.exists():
        save_prompt_presets(_default_presets_copy())

    return PROMPT_PRESETS_FILE


def load_prompt_presets() -> dict[str, dict[str, Any]]:
    PROMPT_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    presets = _default_presets_copy()

    if PROMPT_PRESETS_FILE.exists():
        try:
            loaded = json.loads(PROMPT_PRESETS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for name, preset in loaded.items():
                    if isinstance(preset, dict):
                        normalised = _normalise_preset(preset)
                        presets[normalised["name"]] = normalised
        except Exception:
            # Keep built-in presets available even if the user JSON file is broken.
            pass
    else:
        save_prompt_presets(presets)

    # Built-ins should always remain readonly and available.
    for name, preset in _default_presets_copy().items():
        presets[name] = preset

    return dict(sorted(presets.items(), key=lambda item: item[0].lower()))


def save_prompt_presets(presets: dict[str, dict[str, Any]]) -> None:
    PROMPT_PRESETS_DIR.mkdir(parents=True, exist_ok=True)

    normalised: dict[str, dict[str, Any]] = {}
    for _, preset in presets.items():
        item = _normalise_preset(preset)
        normalised[item["name"]] = item

    PROMPT_PRESETS_FILE.write_text(
        json.dumps(normalised, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_prompt_preset(name: str | None) -> dict[str, Any]:
    presets = load_prompt_presets()

    if name and name in presets:
        return presets[name]

    return presets.get("Default SEO Metadata") or next(iter(presets.values()))


def get_prompt_preset_names() -> list[str]:
    presets = load_prompt_presets()
    return list(presets.keys())


def save_prompt_preset(
    name: str,
    creative_prompt: str,
    description: str = "",
    mode: str = "advanced",
    simple_settings: dict[str, Any] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    clean_name = str(name or "").strip()
    if not clean_name:
        raise ValueError("Preset name is required.")

    presets = load_prompt_presets()

    if clean_name in presets and presets[clean_name].get("readonly"):
        raise ValueError("Built-in presets cannot be overwritten. Use 'Save as New Preset' with a different name.")

    if clean_name in presets and not overwrite:
        raise ValueError("A preset with this name already exists. Choose another name or update the existing preset.")

    now = _now_iso()
    existing = presets.get(clean_name, {})

    preset = _normalise_preset(
        {
            "name": clean_name,
            "description": description,
            "mode": mode,
            "readonly": False,
            "creative_prompt": creative_prompt,
            "simple_settings": simple_settings or {},
            "created_at": existing.get("created_at", now),
            "updated_at": now,
        }
    )

    presets[clean_name] = preset
    save_prompt_presets(presets)

    return preset


def delete_prompt_preset(name: str) -> None:
    presets = load_prompt_presets()

    if name not in presets:
        raise ValueError("Preset not found.")

    if presets[name].get("readonly"):
        raise ValueError("Built-in presets cannot be deleted.")

    del presets[name]
    save_prompt_presets(presets)


def build_simple_prompt_from_settings(settings: dict[str, Any] | None) -> str:
    settings = settings or {}

    video_type = str(settings.get("video_type", "") or "General educational video").strip()
    platform = str(settings.get("platform", "") or "YouTube").strip()
    language = str(settings.get("language", "") or "Roman Urdu title and English description").strip()
    tone = str(settings.get("tone", "") or "Respectful, clear, and SEO-focused").strip()
    title_style = str(settings.get("title_style", "") or "Topic-first, natural, and searchable").strip()
    description_style = str(settings.get("description_style", "") or "Clear, informative, and search-friendly").strip()
    tag_count = str(settings.get("tag_count", "") or "20").strip()
    hashtag_count = str(settings.get("hashtag_count", "") or "3-5").strip()
    extra_instructions = str(settings.get("extra_instructions", "") or "").strip()

    prompt = f"""ROLE:
You are an expert metadata and SEO editor.

CONTENT TYPE:
{video_type}

TARGET PLATFORM:
{platform}

LANGUAGE REQUIREMENTS:
{language}

TONE:
{tone}

TITLE STYLE:
{title_style}

DESCRIPTION STYLE:
{description_style}

TAGS:
Generate exactly {tag_count} comma-separated tags. Make tags specific to the transcript.

HASHTAGS:
Generate {hashtag_count} highly relevant hashtags.

CREATIVE INSTRUCTIONS:
- Read the transcript carefully and identify the real topic, subtopics, named concepts, places, people, religious terms, and main message.
- Write metadata that fits the content type and target platform.
- Avoid generic wording.
- Avoid clickbait.
- Avoid mentioning that this is a video, lecture, clip, talk, speaker, channel, episode, or session unless the user specifically requests it.
"""

    if extra_instructions:
        prompt += f"\nEXTRA USER INSTRUCTIONS:\n{extra_instructions}\n"

    return prompt.strip()
