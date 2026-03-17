import subprocess
from pathlib import Path

# CHANGE THIS PATH IF YOUR FFMPEG LOCATION IS DIFFERENT
FFMPEG_PATH = r"D:\tools\ffmpeg\bin\ffmpeg.exe"

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
        FFMPEG_PATH,
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