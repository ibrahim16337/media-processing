import subprocess
import sys
from pathlib import Path
from app.config.paths import UPLOAD_AUDIO_DIR, TRANSCRIPT_DIR


def run_transcription():

    script_path = Path("app/pipelines/transcription_pipeline/transcriber_engine.py")

    cmd = [
        sys.executable,  # ensures same Python (venv) is used
        str(script_path),
        str(UPLOAD_AUDIO_DIR),
        "-o",
        str(TRANSCRIPT_DIR),
        "--device",
        "cuda",
        "--vad",
        "--recursive"
    ]

    print("Starting transcription pipeline...")

    subprocess.run(cmd)

    print("Transcription finished.")