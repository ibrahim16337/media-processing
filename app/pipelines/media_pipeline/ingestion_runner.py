from pathlib import Path

from app.config.paths import UPLOAD_AUDIO_DIR
from app.pipelines.media_pipeline.audio_standardizer import standardize_audio

SUPPORTED_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi"
}

def run_ingestion(input_folder):
    input_folder = Path(input_folder)
    files = [
        f for f in input_folder.glob("*")
        if f.suffix.lower() in SUPPORTED_EXTS
    ]
    print(f"Found {len(files)} media files")
    converted = []
    for f in files:
        # Skip already standardized wav files
        if f.suffix.lower() == ".wav":
            print(f"Skipping WAV file: {f.name}")
            converted.append(f)
            continue
        print(f"Standardizing: {f.name}")
        wav_file = standardize_audio(f, UPLOAD_AUDIO_DIR)
        converted.append(wav_file)
        print(f"→ Created {wav_file.name}")
    return converted