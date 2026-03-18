import subprocess
from pathlib import Path
from app.config.paths import FFMPEG_EXE


def standardize_audio(input_file: Path, output_dir: Path):
    """
    Convert any media file into Whisper-ready WAV
    Format:
        WAV
        16kHz
        Mono
    """
    output_file = output_dir / f"{input_file.stem}.wav"

    cmd = [
        str(FFMPEG_EXE),
        "-y",
        "-i",
        str(input_file),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_file),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_file