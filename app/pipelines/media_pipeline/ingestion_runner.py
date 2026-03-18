from pathlib import Path

from app.pipelines.media_pipeline.audio_standardizer import standardize_audio

SUPPORTED_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".avi", ".webm"
}


def run_ingestion(input_folder: Path, output_dir: Path):
    input_folder = Path(input_folder)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    files = [
        f for f in input_folder.glob("*")
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTS
    ]

    print(f"Found {len(files)} media files")

    converted = []

    for f in files:
        if f.suffix.lower() == ".wav":
            print(f"Skipping WAV file: {f.name}")
            converted.append(f)
            continue

        print(f"Standardizing: {f.name}")
        wav_file = standardize_audio(f, output_dir)
        converted.append(wav_file)
        print(f"→ Created {wav_file.name}")

    return converted