# Media Processing App

An end-to-end **Streamlit-based media processing system** for:

- transcribing Urdu / multilingual lectures from YouTube and local media
- generating structured metadata from transcripts
- exporting transcript and metadata files
- browsing and uploading videos to a remote server over SFTP

The app is designed for **high-volume media workflows** and uses a centralized path/config system so the project can be moved across machines by updating only the configuration.

---

## Overview

This project automates the full pipeline from media ingestion to publishing support:

1. **Input media**
   - Single YouTube video
   - Single local media file
   - Batch local media files
   - Full YouTube playlist

2. **Audio preparation**
   - Downloads or ingests source media
   - Standardizes audio with FFmpeg into Whisper-ready WAV files

3. **Transcription**
   - Uses **Faster-Whisper** with **`large-v3`**
   - GPU-first workflow with CUDA support
   - Supports Urdu with English / Arabic / Persian mixed terms
   - Progress reporting in Streamlit

4. **Metadata generation**
   - Generates title, description, tags, hashtags
   - Builds structured outputs from transcripts
   - Supports JSON and Excel export

5. **Server upload**
   - Browse remote folders over SFTP
   - View uploaded files already on server
   - Create folders, search files, delete files
   - Upload one or many videos with optional rename before upload

---

## Key Features

### Transcription
- Single YouTube transcription workflow
- Single local media transcription workflow
- Batch transcription for multiple files
- Playlist download + transcription workflow
- Live progress updates in the UI
- Uses **Whisper `large-v3`**
- Centralized shared Whisper cache path via `app/config/paths.py`

### Media Preparation
- FFmpeg-based audio standardization
- Converts source media into consistent WAV audio for transcription
- Video dimension inspection for video-type classification

### Transcript / Export Support
- Transcript preview in the UI
- Export-ready transcript files
- Playlist Excel export
- Transcript metadata schema support for downstream workflows

### Metadata Generation
- Generate:
  - `title`
  - `description`
  - `tags`
  - `hashtags`
- Supports transcript-driven metadata generation
- JSON and Excel export support
- Built for media publishing workflows

### Server Upload
- Remote folder browser
- Breadcrumb navigation
- Create remote folder
- Search remote files/videos
- Delete remote file
- Upload video files to selected server folder
- Rename files before upload
- Public URL generation for uploaded videos

### Configuration / Portability
- Centralized machine-specific paths in `app/config/paths.py`
- Shared external Whisper model cache supported
- Easy migration to another PC by updating config only

---

## Tech Stack

- **Python**
- **Streamlit**
- **Faster-Whisper**
- **CTranslate2**
- **FFmpeg / FFprobe**
- **yt-dlp**
- **openpyxl**
- **Paramiko**
- **Ollama / local LLM support** for metadata generation

---

## Project Structure

```text
media_processing/
├── app/
│   ├── config/
│   │   ├── paths.py
│   │   └── server_upload_config.py
│   │
│   ├── pipelines/
│   │   ├── media_pipeline/
│   │   │   ├── audio_standardizer.py
│   │   │   └── youtube_downloader.py
│   │   │
│   │   ├── transcription_pipeline/
│   │   │   ├── transcription_runner.py
│   │   │   └── transcriber_engine.py
│   │   │
│   │   ├── workflow_pipeline/
│   │   │   ├── transcription_workflows.py
│   │   │   ├── transcript_export_workflows.py
│   │   │   └── server_upload_workflows.py
│   │   │
│   │   ├── metadata_generation_pipeline/
│   │   │   ├── metadata_runner.py
│   │   │   ├── ollama_client.py
│   │   │   ├── prompt_builder.py
│   │   │   ├── response_parser.py
│   │   │   └── transcript_sources.py
│   │   │
│   │   ├── export_pipeline/
│   │   │   └── playlist_excel_exporter.py
│   │   │
│   │   ├── playlist_pipeline/
│   │   │   ├── playlist_downloader.py
│   │   │   └── playlist_runner.py
│   │   │
│   │   └── server_upload_pipeline/
│   │       ├── server_client.py
│   │       └── upload_runner.py
│   │
│   └── ui/
│       └── streamlit_app.py
│
├── data/
│   ├── uploads/
│   │   ├── audio/
│   │   └── video/
│   ├── transcripts/
│   ├── playlists/
│   ├── metadata_outputs/
│   └── temp/
│
├── models/
├── logs/
├── requirements.txt
└── README.md
```

---

## Configuration

All machine-specific paths should be configured in:

```python
app/config/paths.py
```

### Important path settings
Update these when moving to another PC:

- `AI_MODELS_DIR`
- `WHISPER_CACHE_DIR`
- `FFMPEG_DIR`
- `CUDA_BIN_DIR`

Example:

```python
AI_MODELS_DIR = Path(r"D:\AI_Models")
WHISPER_BASE_DIR = AI_MODELS_DIR / "whisper"
WHISPER_CACHE_DIR = WHISPER_BASE_DIR / "faster_whisper_cache"

FFMPEG_DIR = Path(r"D:\tools\ffmpeg\bin")
CUDA_BIN_DIR = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin")
```

---

## Server Upload Configuration

Remote upload settings are handled in:

```python
app/config/server_upload_config.py
```

Or through environment variables, depending on your current implementation.

Typical values include:

- host
- port
- username
- password
- remote root
- public web URL

---

## Installation

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate environment

#### Windows PowerShell
```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

Make sure the following are available on your machine:

- Python 3.11 / 3.12 recommended
- FFmpeg + FFprobe
- NVIDIA GPU driver
- CUDA toolkit (if using GPU transcription)
- Faster-Whisper compatible dependencies
- Whisper model cache downloaded or reachable

---

## Running the App

From project root:

```bash
streamlit run app/ui/streamlit_app.py
```

---

## Typical Workflow

### A. Transcription
1. Choose a source:
   - YouTube
   - local media
   - batch folder
   - playlist
2. Standardize audio
3. Transcribe using Faster-Whisper `large-v3`
4. Preview transcript
5. Export transcript / continue to metadata generation

### B. Metadata Generation
1. Load transcript(s)
2. Generate title, description, tags, hashtags
3. Export metadata to JSON / Excel

### C. Server Upload
1. Browse remote server
2. Select target folder
3. Optionally create a new folder
4. Choose local videos
5. Rename before upload if needed
6. Upload and capture public URLs

---

## Current Output Types

Depending on workflow, the app can generate:

- `.txt` transcript files
- `.xlsx` transcript / playlist export files
- `.json` metadata files
- `.xlsx` metadata files
- remote upload result summaries with public URLs

---

## Design Goals

- Centralized path management
- Cross-machine portability
- GPU-first transcription
- Clean Streamlit UX with progress reporting
- Reusable pipelines and workflow orchestration
- Production-friendly media processing for publishing teams

---

## Notes

- This project is optimized for **Urdu lecture-style transcription** with mixed-language religious and technical terms.
- Whisper models are intended to be shared across projects using a common external cache path.
- If you move the project to another PC, update only the configuration in `app/config/paths.py` and related config files.

---

## Future Improvements

- More advanced transcript post-processing
- Bulk metadata generation improvements
- Recursive remote search
- Safer server-side file operations
- Better queueing / concurrency controls
- Automated publish pipeline integration

---

## Author

Built as a practical media automation system for transcription, metadata generation, and server publishing workflows.
