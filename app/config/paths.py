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
# Model directories
# ---------------------------------------------------

MODEL_DIR = BASE_DIR / "models"
WHISPER_CACHE_DIR = MODEL_DIR / "whisper_cache"

# ---------------------------------------------------
# Logs
# ---------------------------------------------------

LOG_DIR = BASE_DIR / "logs"

# ---------------------------------------------------
# Temporary processing files
# ---------------------------------------------------

TEMP_DIR = DATA_DIR / "temp"

# ---------------------------------------------------
# Dataset export
# ---------------------------------------------------

DATASET_DIR = DATA_DIR / "dataset"

# ---------------------------------------------------
# External tools
# ---------------------------------------------------

FFMPEG_DIR = Path(r"D:\tools\ffmpeg\bin")
FFMPEG_EXE = FFMPEG_DIR / "ffmpeg.exe"
FFPROBE_EXE = FFMPEG_DIR / "ffprobe.exe"


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
        MODEL_DIR,
        WHISPER_CACHE_DIR,
        LOG_DIR,
        TEMP_DIR,
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
    MODEL_DIR,
    WHISPER_CACHE_DIR,
    LOG_DIR,
    TEMP_DIR,
    DATASET_DIR,
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)