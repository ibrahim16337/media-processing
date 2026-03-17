from pathlib import Path

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
# Create directories automatically
# ---------------------------------------------------

dirs = [
    DATA_DIR,
    UPLOAD_DIR,
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
    TRANSCRIPT_DIR,
    MODEL_DIR,
    WHISPER_CACHE_DIR,
    LOG_DIR,
    TEMP_DIR,
    DATASET_DIR
]

for d in dirs:
    d.mkdir(parents=True, exist_ok=True)