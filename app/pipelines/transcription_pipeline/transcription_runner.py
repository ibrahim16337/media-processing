import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
ENGINE_PATH = ROOT / "app" / "pipelines" / "transcription_pipeline" / "transcriber_engine.py"

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


def build_transcription_cmd(
    input_folder,
    output_dir,
    model="large-v3",
    device="cuda",
    compute_type="float16",
    batch_size=8,
    beam_size=5,
    vad=True,
    recursive=True,
    local_only=True,
    language="ur",
    multilingual=False,
    overwrite=False,
    chunk_length=20,
    vad_min_silence_ms=500,
    cache_dir=r"D:\AI_MODELS\whisper\faster_whisper_cache",
    decode_mode="single",
    initial_prompt=DEFAULT_LECTURE_PROMPT,
    hotwords=DEFAULT_LECTURE_HOTWORDS,
):
    cmd = [
        sys.executable,
        str(ENGINE_PATH),
        str(input_folder),
        "-o",
        str(output_dir),
        "--device",
        str(device),
        "--model",
        str(model),
        "--compute_type",
        str(compute_type),
        "--batch_size",
        str(batch_size),
        "--beam_size",
        str(beam_size),
        "--language",
        str(language),
        "--vad_min_silence_ms",
        str(vad_min_silence_ms),
        "--cache_dir",
        str(cache_dir),
        "--decode_mode",
        str(decode_mode),
    ]

    if vad:
        cmd.append("--vad")

    if recursive:
        cmd.append("--recursive")

    if local_only:
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