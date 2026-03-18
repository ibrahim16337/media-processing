import subprocess
import sys
from pathlib import Path


def build_transcription_cmd(input_path: Path, output_dir: Path):
    script_path = Path("app/pipelines/transcription_pipeline/transcriber_engine.py")

    cmd = [
        sys.executable,
        str(script_path),
        str(input_path),
        "-o",
        str(output_dir),
        "--device",
        "cuda",
        "--model",
        "large-v3",
        "--compute_type",
        "float16",
        "--batch_size",
        "8",
        "--beam_size",
        "2",
        "--vad",
        "--recursive",
        "--local_only",
    ]
    return cmd


def run_transcription_blocking(input_path: Path, output_dir: Path):
    cmd = build_transcription_cmd(input_path, output_dir)
    subprocess.run(cmd, check=True)