import os
import sys
import subprocess
import time
import re
import shutil
from pathlib import Path
import streamlit as st
import zipfile
import io

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config.paths import (
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
    TRANSCRIPT_DIR,
    PLAYLISTS_DIR,
)
from app.pipelines.media_pipeline.ingestion_runner import run_ingestion
from app.pipelines.media_pipeline.youtube_downloader import download_youtube_audio
from app.pipelines.media_pipeline.audio_standardizer import standardize_audio
from app.pipelines.transcription_pipeline.transcription_runner import build_transcription_cmd
from app.pipelines.playlist_pipeline.playlist_runner import run_playlist_download

# --------------------------------------------------
# Ensure required directories exist
# --------------------------------------------------

UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------

if "batch_uploader_nonce" not in st.session_state:
    st.session_state["batch_uploader_nonce"] = 0

if "batch_notice" not in st.session_state:
    st.session_state["batch_notice"] = None

if "global_notice" not in st.session_state:
    st.session_state["global_notice"] = None

# --------------------------------------------------
# Constants
# --------------------------------------------------

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Media Transcription",
    page_icon="🎙",
    layout="wide"
)

st.title("AI Media Transcription")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def show_notice(notice):
    if not notice:
        return

    level = notice.get("level", "info")
    message = notice.get("message", "")

    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    elif level == "error":
        st.error(message)
    else:
        st.info(message)


def set_batch_notice(level: str, message: str):
    st.session_state["batch_notice"] = {"level": level, "message": message}


def set_global_notice(level: str, message: str):
    st.session_state["global_notice"] = {"level": level, "message": message}


def reset_batch_uploader():
    st.session_state["batch_uploader_nonce"] += 1


def save_uploaded_files(uploaded_files):
    audio_count = 0
    video_count = 0

    for file in uploaded_files:
        ext = Path(file.name).suffix.lower()

        if ext in AUDIO_EXTS:
            save_path = UPLOAD_AUDIO_DIR / file.name
            audio_count += 1
        elif ext in VIDEO_EXTS:
            save_path = UPLOAD_VIDEO_DIR / file.name
            video_count += 1
        else:
            continue

        with open(save_path, "wb") as f:
            f.write(file.read())

    return audio_count, video_count


def clear_files_in_folder(folder: Path, allowed_exts=None):
    removed = 0

    for file in folder.glob("*"):
        if not file.is_file():
            continue

        if allowed_exts is not None and file.suffix.lower() not in allowed_exts:
            continue

        try:
            file.unlink()
            removed += 1
        except Exception:
            pass

    return removed


def clear_folder_contents(folder: Path):
    removed_files = 0
    removed_dirs = 0

    for item in folder.glob("*"):
        try:
            if item.is_file():
                item.unlink()
                removed_files += 1
            elif item.is_dir():
                shutil.rmtree(item)
                removed_dirs += 1
        except Exception:
            pass

    return removed_files, removed_dirs


def clear_generated_project_data():
    removed_audio = clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS | {".webm"})
    removed_video = clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)

    removed_transcripts = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json"})
    playlist_files, playlist_dirs = clear_folder_contents(PLAYLISTS_DIR)

    return {
        "audio_files": removed_audio,
        "video_files": removed_video,
        "transcript_files": removed_transcripts,
        "playlist_files": playlist_files,
        "playlist_dirs": playlist_dirs,
    }


def delete_playlist_folder_by_slug(slug: str):
    slug = slug.strip()

    if not slug:
        return False, "Please enter a playlist slug."

    playlist_folder = PLAYLISTS_DIR / slug

    if not playlist_folder.exists():
        return False, f"Playlist folder not found: {slug}"

    try:
        shutil.rmtree(playlist_folder)
        return True, f"Deleted playlist folder: {slug}"
    except Exception as e:
        return False, f"Failed to delete playlist folder: {e}"


def build_zip_from_folder(folder: Path, pattern: str = "*.txt"):
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(folder.glob(pattern)):
            zip_file.write(file, arcname=file.name)

    zip_buffer.seek(0)
    return zip_buffer


def get_audio_files(folder: Path):
    return sorted(
        [
            f for f in folder.glob("*")
            if f.is_file() and f.suffix.lower() in AUDIO_EXTS
        ]
    )


def get_video_files(folder: Path):
    return sorted(
        [
            f for f in folder.glob("*")
            if f.is_file() and f.suffix.lower() in VIDEO_EXTS
        ]
    )


def read_text_file(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "Unable to read transcript file."


def run_transcription_with_progress(input_path: Path, output_dir: Path):
    progress_pattern = re.compile(r"(\d+\.\d+)%")

    progress_bar = st.progress(0)
    percent_container = st.empty()
    elapsed_container = st.empty()
    eta_container = st.empty()
    output_box = st.empty()

    start_transcribe = time.time()
    output_lines = []

    cmd = build_transcription_cmd(input_path, output_dir)

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env
    )

    last_update = 0

    while True:
        line = process.stdout.readline() if process.stdout else ""

        if not line and process.poll() is not None:
            break

        if line:
            clean_line = line.strip()

            if clean_line:
                output_lines.append(clean_line)
                output_box.code("\n".join(output_lines[-15:]))

            match = progress_pattern.search(line)

            if match:
                percent = float(match.group(1))
                progress_bar.progress(max(0, min(100, int(percent))))

                elapsed = time.time() - start_transcribe
                eta = ((elapsed / percent) * (100 - percent)) if percent > 0 else 0

                now = time.time()
                if now - last_update > 0.5:
                    percent_container.markdown(f"Progress: **{percent:.2f}%**")
                    elapsed_container.markdown(f"Elapsed: **{int(elapsed)} sec**")
                    eta_container.markdown(f"ETA: **{int(eta)} sec**")
                    last_update = now

    process.wait()

    transcription_time = time.time() - start_transcribe
    success = process.returncode == 0

    if success:
        progress_bar.progress(100)

    return success, transcription_time, output_lines


# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "YouTube",
    "Upload & Batch Process",
    "Transcripts",
    "Playlist Download & Transcription"
])

# ==================================================
# TAB 1 — YOUTUBE
# ==================================================

with tab1:
    st.header("YouTube Transcription")

    show_notice(st.session_state["global_notice"])
    st.session_state["global_notice"] = None

    col_a, col_b = st.columns(2)

    with col_a:
        if st.button("Clear All Generated Project Data", key="tab1_clear_all_data"):
            summary = clear_generated_project_data()
            set_global_notice(
                "success",
                (
                    f"Cleared data: {summary['audio_files']} audio file(s), "
                    f"{summary['video_files']} video file(s), "
                    f"{summary['transcript_files']} transcript file(s), "
                    f"{summary['playlist_dirs']} playlist folder(s), "
                    f"{summary['playlist_files']} extra playlist file(s)."
                ),
            )
            reset_batch_uploader()
            st.rerun()

    youtube_url = st.text_input("Paste YouTube Link", key="youtube_url")

    if st.button("Download & Transcribe", key="yt_button"):
        if not youtube_url.strip():
            st.warning("Please paste a YouTube link first.")
        else:
            try:
                progress_bar = st.progress(0)
                percent_box = st.empty()
                eta_box = st.empty()
                speed_box = st.empty()

                start_download = time.time()

                def progress_hook(d):
                    if d["status"] == "downloading":
                        total = d.get("total_bytes") or d.get("total_bytes_estimate")
                        downloaded = d.get("downloaded_bytes", 0)

                        if total:
                            percent = downloaded / total * 100
                            progress_bar.progress(max(0, min(100, int(percent))))
                            percent_box.markdown(f"Download: **{percent:.2f}%**")

                        eta = d.get("eta")
                        if eta is not None:
                            eta_box.markdown(f"ETA: **{eta} sec**")

                        speed = d.get("speed")
                        if speed:
                            speed_box.markdown(f"Speed: **{speed / 1024 / 1024:.2f} MB/s**")

                    elif d["status"] == "finished":
                        progress_bar.progress(100)

                st.write("Downloading audio...")
                audio_file = download_youtube_audio(
                    youtube_url,
                    progress_callback=progress_hook
                )
                download_time = time.time() - start_download

                st.success(f"Downloaded: {audio_file.name}")

                st.write("Standardizing audio...")
                convert_start = time.time()
                wav_file = standardize_audio(audio_file, UPLOAD_AUDIO_DIR)
                convert_time = time.time() - convert_start

                st.success(f"Audio standardized: {wav_file.name}")

                st.subheader("Transcription")
                success, transcription_time, _ = run_transcription_with_progress(
                    wav_file,
                    TRANSCRIPT_DIR
                )

                if success:
                    transcript_file = TRANSCRIPT_DIR / f"{wav_file.stem}.txt"

                    st.subheader("Statistics")
                    st.write(f"Download Time: **{download_time:.2f} sec**")
                    st.write(f"Conversion Time: **{convert_time:.2f} sec**")
                    st.write(f"Transcription Time: **{transcription_time:.2f} sec**")

                    if transcript_file.exists():
                        text = read_text_file(transcript_file)

                        st.success("Transcription completed successfully.")

                        st.download_button(
                            label="Download This Transcript",
                            data=text.encode("utf-8"),
                            file_name=transcript_file.name,
                            mime="text/plain",
                            key=f"download_single_{wav_file.stem}"
                        )

                        st.text_area(
                            "Transcript Preview",
                            value=text,
                            height=320,
                            key=f"yt_preview_{wav_file.stem}"
                        )
                    else:
                        st.warning("Transcription finished, but transcript file was not found.")
                else:
                    st.error("Transcription failed.")

            except Exception as e:
                st.error(f"Error: {e}")

# ==================================================
# TAB 2 — UPLOAD & BATCH PROCESS
# ==================================================

with tab2:
    st.header("Upload & Batch Transcription")

    show_notice(st.session_state["batch_notice"])
    st.session_state["batch_notice"] = None

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)

    with row1_col1:
        if st.button("Clear Audio Uploads", key="clear_audio_uploads"):
            removed = clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS)
            set_batch_notice("success", f"Removed {removed} audio file(s).")
            reset_batch_uploader()
            st.rerun()

    with row1_col2:
        if st.button("Clear Video Uploads", key="clear_video_uploads"):
            removed = clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)
            set_batch_notice("success", f"Removed {removed} video file(s).")
            reset_batch_uploader()
            st.rerun()

    with row1_col3:
        if st.button("Clear Transcripts", key="clear_batch_transcripts"):
            removed = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json"})
            set_batch_notice("success", f"Removed {removed} transcript file(s).")
            st.rerun()

    with row1_col4:
        if st.button("Clear All Generated Data", key="clear_batch_all_generated"):
            summary = clear_generated_project_data()
            set_batch_notice(
                "success",
                (
                    f"Cleared data: {summary['audio_files']} audio file(s), "
                    f"{summary['video_files']} video file(s), "
                    f"{summary['transcript_files']} transcript file(s), "
                    f"{summary['playlist_dirs']} playlist folder(s), "
                    f"{summary['playlist_files']} extra playlist file(s)."
                ),
            )
            reset_batch_uploader()
            st.rerun()

    uploaded_files = st.file_uploader(
        "Upload audio or video files",
        accept_multiple_files=True,
        type=[
            "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma",
            "mp4", "mkv", "mov", "avi", "webm"
        ],
        key=f"batch_uploader_{st.session_state['batch_uploader_nonce']}"
    )

    save_col1, save_col2 = st.columns(2)

    with save_col1:
        if st.button("Save Uploaded Files", key="save_batch_files"):
            if uploaded_files:
                audio_count, video_count = save_uploaded_files(uploaded_files)
                set_batch_notice(
                    "success",
                    f"Saved {len(uploaded_files)} file(s) ({audio_count} audio, {video_count} video)."
                )
                reset_batch_uploader()
                st.rerun()
            else:
                st.warning("Please choose files first.")

    current_audio_files = get_audio_files(UPLOAD_AUDIO_DIR)
    current_video_files = get_video_files(UPLOAD_VIDEO_DIR)

    st.write(f"Audio files available: **{len(current_audio_files)}**")
    st.write(f"Video files available: **{len(current_video_files)}**")

    if st.button("Start Batch Transcription", key="batch_transcribe_button"):
        try:
            batch_start = time.time()

            if current_video_files:
                st.subheader("Standardizing Video Files")
                standardized_files = run_ingestion(UPLOAD_VIDEO_DIR, UPLOAD_AUDIO_DIR)
                st.success(f"{len(standardized_files)} video file(s) standardized to audio.")
            else:
                st.info("No video files found to standardize.")

            audio_files_after_standardization = get_audio_files(UPLOAD_AUDIO_DIR)

            if not audio_files_after_standardization:
                st.warning("No audio files available for transcription.")
            else:
                st.subheader("Transcription")
                success, transcription_time, _ = run_transcription_with_progress(
                    UPLOAD_AUDIO_DIR,
                    TRANSCRIPT_DIR
                )

                batch_time = time.time() - batch_start

                if success:
                    st.success("Batch transcription completed successfully.")
                    st.subheader("Batch Statistics")
                    st.write(f"Audio files processed: **{len(audio_files_after_standardization)}**")
                    st.write(f"Total batch time: **{batch_time:.2f} sec**")
                    st.write(f"Transcription time: **{transcription_time:.2f} sec**")
                else:
                    st.error("Batch transcription failed.")

        except Exception as e:
            st.error(f"Error: {e}")

# ==================================================
# TAB 3 — TRANSCRIPTS
# ==================================================

with tab3:
    st.header("Transcripts")

    show_notice(st.session_state["global_notice"])
    st.session_state["global_notice"] = None

    tab3_col1, tab3_col2 = st.columns(2)

    with tab3_col1:
        if st.button("Clear Transcript Files", key="tab3_clear_transcripts"):
            removed = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json"})
            set_global_notice("success", f"Removed {removed} transcript file(s).")
            st.rerun()

    with tab3_col2:
        if st.button("Clear All Generated Project Data", key="tab3_clear_all_data"):
            summary = clear_generated_project_data()
            set_global_notice(
                "success",
                (
                    f"Cleared data: {summary['audio_files']} audio file(s), "
                    f"{summary['video_files']} video file(s), "
                    f"{summary['transcript_files']} transcript file(s), "
                    f"{summary['playlist_dirs']} playlist folder(s), "
                    f"{summary['playlist_files']} extra playlist file(s)."
                ),
            )
            reset_batch_uploader()
            st.rerun()

    transcripts = sorted(TRANSCRIPT_DIR.glob("*.txt"))

    if transcripts:
        zip_buffer = build_zip_from_folder(TRANSCRIPT_DIR, "*.txt")

        st.download_button(
            label="Download All Transcripts (ZIP)",
            data=zip_buffer,
            file_name="transcripts.zip",
            mime="application/zip",
            key="download_all_transcripts_zip"
        )
    else:
        st.info("No transcript files found yet.")

    for t in transcripts:
        with st.expander(t.name):
            text = read_text_file(t)

            st.download_button(
                label=f"Download {t.name}",
                data=text.encode("utf-8"),
                file_name=t.name,
                mime="text/plain",
                key=f"download_{t.stem}"
            )

            st.text_area(
                label="Transcript",
                value=text,
                height=300,
                key=f"global_transcript_{t.stem}"
            )

# ==================================================
# TAB 4 — PLAYLIST DOWNLOAD & TRANSCRIPTION
# ==================================================

with tab4:
    st.header("Playlist Download & Transcription")

    show_notice(st.session_state["global_notice"])
    st.session_state["global_notice"] = None

    manage_col1, manage_col2 = st.columns(2)

    with manage_col1:
        if st.button("Clear All Playlist Folders", key="clear_all_playlists"):
            removed_files, removed_dirs = clear_folder_contents(PLAYLISTS_DIR)
            set_global_notice(
                "success",
                f"Removed {removed_dirs} playlist folder(s) and {removed_files} extra file(s)."
            )
            st.rerun()

    with manage_col2:
        if st.button("Clear All Generated Project Data", key="tab4_clear_all_data"):
            summary = clear_generated_project_data()
            set_global_notice(
                "success",
                (
                    f"Cleared data: {summary['audio_files']} audio file(s), "
                    f"{summary['video_files']} video file(s), "
                    f"{summary['transcript_files']} transcript file(s), "
                    f"{summary['playlist_dirs']} playlist folder(s), "
                    f"{summary['playlist_files']} extra playlist file(s)."
                ),
            )
            reset_batch_uploader()
            st.rerun()

    playlist_slug_to_clear = st.text_input(
        "Enter playlist slug to delete",
        key="playlist_slug_to_clear",
        placeholder="e.g. drisrar-ahmad-short-clips"
    )

    if st.button("Delete This Playlist Folder", key="clear_playlist_folder_button"):
        ok, msg = delete_playlist_folder_by_slug(playlist_slug_to_clear)
        if ok:
            set_global_notice("success", msg)
        else:
            set_global_notice("warning", msg)
        st.rerun()

    playlist_url = st.text_input("Paste YouTube Playlist Link", key="playlist_url")
    quality = st.selectbox(
        "Select Download Quality",
        ["480p", "720p", "1080p"],
        index=1,
        key="playlist_quality"
    )

    if st.button("Download Playlist and Transcribe", key="playlist_button"):
        if not playlist_url.strip():
            st.warning("Please paste a playlist link first.")
        else:
            try:
                st.subheader("Downloading Playlist")

                playlist_progress = st.progress(0)
                download_status = st.empty()
                current_file_box = st.empty()

                def playlist_progress_hook(d):
                    status = d.get("status", "")

                    if status == "downloading":
                        filename = Path(d.get("filename", "")).name if d.get("filename") else "current file"
                        current_file_box.markdown(f"Downloading: **{filename}**")

                        total = d.get("total_bytes") or d.get("total_bytes_estimate")
                        downloaded = d.get("downloaded_bytes", 0)

                        if total:
                            percent = int((downloaded / total) * 100)
                            percent = max(0, min(100, percent))
                            playlist_progress.progress(percent)

                    elif status == "finished":
                        filename = Path(d.get("filename", "")).name if d.get("filename") else "file"
                        download_status.markdown(f"Finished: **{filename}**")
                        playlist_progress.progress(100)

                playlist_start = time.time()

                playlist_data = run_playlist_download(
                    url=playlist_url,
                    quality=quality,
                    progress_callback=playlist_progress_hook
                )

                playlist_elapsed = time.time() - playlist_start

                result = playlist_data["result"]
                standardized_files = playlist_data["standardized_files"]
                paths = result["paths"]
                manifest = result["manifest"]

                st.success("Playlist download complete.")
                st.write(f"Playlist: **{manifest.get('playlist_title', 'Unknown')}**")
                st.write(f"Videos found: **{manifest.get('entry_count', 0)}**")
                st.write(f"Quality selected: **{manifest.get('requested_quality', quality)}**")
                st.write(f"Download time: **{playlist_elapsed:.2f} sec**")
                st.write(f"Audio files prepared: **{len(standardized_files)}**")

                st.subheader("Transcribing Playlist")

                success, transcription_time, _ = run_transcription_with_progress(
                    paths.audio,
                    paths.transcripts
                )

                if success:
                    st.success("Playlist transcription complete.")
                    st.write(f"Transcription time: **{transcription_time:.2f} sec**")

                    playlist_transcripts = sorted(paths.transcripts.glob("*.txt"))

                    if playlist_transcripts:
                        transcript_zip = build_zip_from_folder(paths.transcripts, "*.txt")

                        st.download_button(
                            label="Download Playlist Transcripts (ZIP)",
                            data=transcript_zip,
                            file_name=f"{paths.root.name}_transcripts.zip",
                            mime="application/zip",
                            key=f"playlist_zip_{paths.root.name}"
                        )

                        for t in playlist_transcripts:
                            with st.expander(t.name):
                                text = read_text_file(t)

                                st.download_button(
                                    label=f"Download {t.name}",
                                    data=text.encode("utf-8"),
                                    file_name=t.name,
                                    mime="text/plain",
                                    key=f"playlist_download_{paths.root.name}_{t.stem}"
                                )

                                st.text_area(
                                    label="Transcript",
                                    value=text,
                                    height=300,
                                    key=f"playlist_transcript_{paths.root.name}_{t.stem}"
                                )
                    else:
                        st.warning("Playlist transcription finished, but no transcript files were found.")
                else:
                    st.error("Playlist transcription failed.")

            except Exception as e:
                st.error(f"Error: {e}")