from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from app.config.paths import (
    TRANSCRIBER_ENGINE_PATH,
    WHISPER_CACHE_DIR,
    TRANSCRIPTION_BATCH_SIZE,
    TRANSCRIPTION_NUM_WORKERS,
    TRANSCRIPTION_DEVICE,
    TRANSCRIPTION_COMPUTE_TYPE,
    TRANSCRIPTION_BEAM_SIZE,
    TRANSCRIPTION_DECODE_MODE,
)


DEFAULT_LECTURE_PROMPT = (
    "This is an Urdu lecture with frequent English, Arabic, and Persian words. "
    "Preserve English technical or quoted words in English as spoken. "
    "Preserve Arabic and Persian religious terms as spoken. "
    "Do not translate foreign words into Urdu. "
    "Transcribe naturally and accurately."
)

DEFAULT_LECTURE_HOTWORDS = (
    "Quran Qur'an Hadith Sunnah tafseer tafsir ayah aayat surah "
    "fiqh shariah deen iman Islam Islamic Arabic Persian English "
    "Allah Rasul Muhammad"
)


def _safe_string(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _model_cache_markers(model_name: str) -> list[str]:
    model = _safe_string(model_name).lower()
    return [
        model,
        f"faster-whisper-{model}",
        f"models--systran--faster-whisper-{model}",
        f"models--Systran--faster-whisper-{model}",
    ]


def _cache_dir_has_requested_model(cache_dir: Path, model_name: str) -> bool:
    if not cache_dir.exists() or not cache_dir.is_dir():
        return False

    model = _safe_string(model_name).lower()

    candidate_dirs = [
        cache_dir / f"models--systran--faster-whisper-{model}",
        cache_dir / f"models--Systran--faster-whisper-{model}",
    ]

    for model_dir in candidate_dirs:
        snapshots_dir = model_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        for snapshot in snapshots_dir.iterdir():
            if not snapshot.is_dir():
                continue

            model_bin = snapshot / "model.bin"
            if model_bin.exists() and model_bin.is_file():
                return True

    return False


def build_transcription_cmd(
    input_folder,
    output_dir,
    model="large-v3",
    device=TRANSCRIPTION_DEVICE,
    compute_type=TRANSCRIPTION_COMPUTE_TYPE,
    batch_size=TRANSCRIPTION_BATCH_SIZE,
    num_workers=TRANSCRIPTION_NUM_WORKERS,
    beam_size=TRANSCRIPTION_BEAM_SIZE,
    vad=True,
    recursive=True,
    local_only=True,
    language="ur",
    multilingual=False,
    overwrite=False,
    chunk_length=20,
    vad_min_silence_ms=500,
    cache_dir=WHISPER_CACHE_DIR,
    decode_mode=TRANSCRIPTION_DECODE_MODE,
    initial_prompt=DEFAULT_LECTURE_PROMPT,
    hotwords=DEFAULT_LECTURE_HOTWORDS,
):
    input_path = Path(input_folder)
    output_dir = Path(output_dir)

    effective_decode_mode = str(decode_mode)
    if input_path.is_file():
        effective_decode_mode = "single"

    cmd = [
        sys.executable,
        str(TRANSCRIBER_ENGINE_PATH),
        str(input_path),
        "-o",
        str(output_dir),
        "--device",
        str(device),
        "--model",
        str(model),
        "--compute_type",
        str(compute_type),
        "--num_workers",
        str(num_workers),
        "--batch_size",
        str(batch_size),
        "--beam_size",
        str(beam_size),
        "--language",
        str(language),
        "--vad_min_silence_ms",
        str(vad_min_silence_ms),
        "--decode_mode",
        str(effective_decode_mode),
    ]

    model_is_in_cache = False

    if cache_dir:
        cache_path = Path(cache_dir)
        model_is_in_cache = _cache_dir_has_requested_model(cache_path, model)
        if model_is_in_cache:
            cmd.extend(["--cache_dir", str(cache_path)])

    if vad:
        cmd.append("--vad")

    if recursive:
        cmd.append("--recursive")

    if local_only and model_is_in_cache:
        cmd.append("--local_only")

    if multilingual:
        cmd.append("--multilingual")

    if overwrite:
        cmd.append("--overwrite")

    if chunk_length is not None:
        cmd.extend(["--chunk_length", str(chunk_length)])

    if initial_prompt:
        cmd.extend(["--initial_prompt", str(initial_prompt)])

    if hotwords:
        cmd.extend(["--hotwords", str(hotwords)])

    return cmd