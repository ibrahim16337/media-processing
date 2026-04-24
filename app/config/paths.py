from pathlib import Path
from dataclasses import dataclass
import re

# ---------------------------------------------------
# Base project directory
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

# ---------------------------------------------------
# Data directories
# ---------------------------------------------------

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_AUDIO_DIR = UPLOAD_DIR / "audio"
UPLOAD_VIDEO_DIR = UPLOAD_DIR / "video"

TRANSCRIPT_DIR = DATA_DIR / "transcripts"
PLAYLISTS_DIR = DATA_DIR / "playlists"

# ---------------------------------------------------
# Metadata output directories
# ---------------------------------------------------

METADATA_OUTPUT_DIR = DATA_DIR / "metadata_outputs"
METADATA_SINGLE_DIR = METADATA_OUTPUT_DIR / "single"
METADATA_BATCH_DIR = METADATA_OUTPUT_DIR / "batch"
METADATA_PLAYLIST_DIR = METADATA_OUTPUT_DIR / "playlists"
METADATA_EXCEL_IMPORT_DIR = METADATA_OUTPUT_DIR / "excel_imports"

# ---------------------------------------------------
# Model directories
# ---------------------------------------------------

MODEL_DIR = BASE_DIR / "models"   # keep for backward compatibility

AI_MODELS_DIR = Path(r"D:\AI_Models")
WHISPER_BASE_DIR = AI_MODELS_DIR / "whisper"
WHISPER_CACHE_DIR = WHISPER_BASE_DIR / "faster_whisper_cache"

# ---------------------------------------------------
# Logs
# ---------------------------------------------------

LOG_DIR = BASE_DIR / "logs"

# ---------------------------------------------------
# Temporary processing files
# ---------------------------------------------------

TEMP_DIR = DATA_DIR / "temp"
TEMP_SINGLE_BATCH_DIR = TEMP_DIR / "single_batch"
SERVER_UPLOAD_TEMP_DIR = TEMP_DIR / "server_upload"

# ---------------------------------------------------
# Dataset export
# ---------------------------------------------------

DATASET_DIR = DATA_DIR / "dataset"

# ---------------------------------------------------
# External tools / machine-specific paths
# Change only these when moving to another PC
# ---------------------------------------------------

FFMPEG_DIR = Path(r"D:\tools\ffmpeg\bin")
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "ffprobe.exe"

CUDA_BIN_DIR = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")

# ---------------------------------------------------
# Internal important file paths
# ---------------------------------------------------

TRANSCRIBER_ENGINE_PATH = (
    BASE_DIR / "app" / "pipelines" / "transcription_pipeline" / "transcriber_engine.py"
)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s-]", "", value)
    value = re.sub(r"[-\s]+", "-", value)
    return value or "playlist"


@dataclass
class PlaylistPaths:
    root: Path
    videos: Path
    audio: Path
    transcripts: Path
    metadata: Path
    archive_file: Path
    manifest_file: Path


def build_playlist_paths(playlist_name: str) -> PlaylistPaths:
    slug = slugify(playlist_name)
    root = PLAYLISTS_DIR / slug

    paths = PlaylistPaths(
        root=root,
        videos=root / "videos",
        audio=root / "audio",
        transcripts=root / "transcripts",
        metadata=root / "metadata",
        archive_file=root / "archive.txt",
        manifest_file=root / "manifest.json",
    )

    dirs = [
        DATA_DIR,
        UPLOAD_DIR,
        UPLOAD_AUDIO_DIR,
        UPLOAD_VIDEO_DIR,
        TRANSCRIPT_DIR,
        PLAYLISTS_DIR,
        METADATA_OUTPUT_DIR,
        METADATA_SINGLE_DIR,
        METADATA_BATCH_DIR,
        METADATA_PLAYLIST_DIR,
        METADATA_EXCEL_IMPORT_DIR,
        MODEL_DIR,
        WHISPER_CACHE_DIR,
        LOG_DIR,
        TEMP_DIR,
        TEMP_SINGLE_BATCH_DIR,
        SERVER_UPLOAD_TEMP_DIR,
        DATASET_DIR,
        paths.root,
        paths.videos,
        paths.audio,
        paths.transcripts,
        paths.metadata,
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    return paths


# ---------------------------------------------------
# Create global directories automatically
# ---------------------------------------------------

dirs = [
    DATA_DIR,
    UPLOAD_DIR,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
    TRANSCRIPT_DIR,
    PLAYLISTS_DIR,
    METADATA_OUTPUT_DIR,
    METADATA_SINGLE_DIR,
    METADATA_BATCH_DIR,
    METADATA_PLAYLIST_DIR,
    METADATA_EXCEL_IMPORT_DIR,
    MODEL_DIR,
    WHISPER_CACHE_DIR,
    LOG_DIR,
    TEMP_DIR,
    TEMP_SINGLE_BATCH_DIR,
    SERVER_UPLOAD_TEMP_DIR,
    DATASET_DIR,
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)
