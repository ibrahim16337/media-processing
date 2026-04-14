import sys
import shutil
import zipfile
import io
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.config.paths import (
    UPLOAD_AUDIO_DIR,
    UPLOAD_VIDEO_DIR,
    TRANSCRIPT_DIR,
    PLAYLISTS_DIR,
    METADATA_OUTPUT_DIR,
    METADATA_SINGLE_DIR,
    METADATA_BATCH_DIR,
    METADATA_PLAYLIST_DIR,
    METADATA_EXCEL_IMPORT_DIR,
)
from app.pipelines.workflow_pipeline.transcription_workflows import (
    transcribe_single_youtube,
    transcribe_batch_media,
    transcribe_playlist,
)
from app.pipelines.workflow_pipeline.metadata_workflows import (
    generate_metadata_from_single_youtube,
    generate_metadata_from_batch_media,
    generate_metadata_from_playlist,
    generate_metadata_from_single_transcript_file,
    generate_metadata_from_transcript_folder,
    generate_metadata_from_transcript_files,
    generate_metadata_from_excel,
)
from app.pipelines.workflow_pipeline.transcript_export_workflows import (
    export_global_transcripts_to_excel,
    export_global_transcripts_to_json,
    export_playlist_transcripts_to_excel,
    export_playlist_transcripts_to_json,
    build_global_transcripts_zip_bytes,
    build_transcript_folder_zip_bytes,
    build_transcript_zip_bytes,
)
from app.pipelines.metadata_generation_pipeline.ollama_client import (
    DEFAULT_MODEL,
    DEFAULT_BASE_URL,
    DEFAULT_TIMEOUT,
    DEFAULT_RETRIES,
    DEFAULT_SLEEP_MS,
    DEFAULT_TEMPERATURE,
    DEFAULT_NUM_CTX,
    DEFAULT_NUM_PREDICT,
)
from app.pipelines.metadata_generation_pipeline.transcript_sources import (
    get_excel_sheet_names,
    get_excel_columns,
)

# --------------------------------------------------
# Ensure directories exist
# --------------------------------------------------

UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
PLAYLISTS_DIR.mkdir(parents=True, exist_ok=True)
METADATA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
METADATA_SINGLE_DIR.mkdir(parents=True, exist_ok=True)
METADATA_BATCH_DIR.mkdir(parents=True, exist_ok=True)
METADATA_PLAYLIST_DIR.mkdir(parents=True, exist_ok=True)
METADATA_EXCEL_IMPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Session state
# --------------------------------------------------

st.session_state.setdefault("transcription_single_result", None)
st.session_state.setdefault("transcription_batch_result", None)
st.session_state.setdefault("transcription_playlist_result", None)

st.session_state.setdefault("metadata_single_result", None)
st.session_state.setdefault("metadata_batch_result", None)
st.session_state.setdefault("metadata_playlist_result", None)
st.session_state.setdefault("metadata_existing_result", None)

# --------------------------------------------------
# Constants
# --------------------------------------------------

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
TEXT_EXTS = {".txt"}
EXCEL_EXTS = {".xlsx", ".xlsm"}

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------

st.set_page_config(
    page_title="AI Media Transcription & SEO Metadata",
    page_icon="🎙",
    layout="wide",
)

st.title("AI Media Transcription & SEO Metadata")

# --------------------------------------------------
# Helpers
# --------------------------------------------------

def _safe_string(value):
    if value is None:
        return ""
    return str(value).strip()


def save_uploaded_file(uploaded_file, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    save_path = destination_dir / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


def save_uploaded_files(uploaded_files):
    audio_count = 0
    video_count = 0
    saved_paths = []

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
            f.write(file.getbuffer())

        saved_paths.append(save_path)

    return audio_count, video_count, saved_paths


def save_uploaded_transcript_files(uploaded_files, destination_dir: Path) -> list[Path]:
    saved_paths: list[Path] = []

    for file in uploaded_files:
        if Path(file.name).suffix.lower() not in TEXT_EXTS:
            continue
        saved_paths.append(save_uploaded_file(file, destination_dir))

    return saved_paths


def save_uploaded_excel(uploaded_file) -> Path:
    upload_dir = METADATA_EXCEL_IMPORT_DIR / "uploaded_excels"
    return save_uploaded_file(uploaded_file, upload_dir)


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
    removed_audio = clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS | VIDEO_EXTS)
    removed_video = clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)
    removed_transcripts = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json", ".xlsx"})
    playlist_files, playlist_dirs = clear_folder_contents(PLAYLISTS_DIR)
    metadata_files, metadata_dirs = clear_folder_contents(METADATA_OUTPUT_DIR)

    return {
        "audio_files": removed_audio,
        "video_files": removed_video,
        "transcript_files": removed_transcripts,
        "playlist_files": playlist_files,
        "playlist_dirs": playlist_dirs,
        "metadata_files": metadata_files,
        "metadata_dirs": metadata_dirs,
    }


def build_clear_summary_message(summary: dict):
    return (
        f"Cleared data: {summary['audio_files']} audio file(s), "
        f"{summary['video_files']} video file(s), "
        f"{summary['transcript_files']} transcript/export file(s), "
        f"{summary['playlist_dirs']} playlist folder(s), "
        f"{summary['playlist_files']} extra playlist file(s), "
        f"{summary['metadata_dirs']} metadata folder(s), "
        f"{summary['metadata_files']} metadata file(s)."
    )


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


def get_audio_files(folder: Path):
    return sorted([f for f in folder.glob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXTS])


def get_video_files(folder: Path):
    return sorted([f for f in folder.glob("*") if f.is_file() and f.suffix.lower() in VIDEO_EXTS])


def get_all_transcript_files():
    files = []
    files.extend(sorted(TRANSCRIPT_DIR.glob("*.txt")))
    files.extend(sorted(PLAYLISTS_DIR.glob("*/transcripts/*.txt")))

    unique = {}
    for file_path in files:
        unique[str(file_path.resolve())] = file_path

    return sorted(unique.values(), key=lambda p: str(p).lower())


def get_transcript_source_folders():
    folders = []

    if TRANSCRIPT_DIR.exists():
        folders.append(TRANSCRIPT_DIR)

    playlist_transcript_dirs = sorted(
        [p for p in PLAYLISTS_DIR.glob("*/transcripts") if p.is_dir()],
        key=lambda p: str(p).lower(),
    )
    folders.extend(playlist_transcript_dirs)

    unique = {}
    for folder in folders:
        unique[str(folder.resolve())] = folder

    return sorted(unique.values(), key=lambda p: str(p).lower())


def get_existing_excel_sources():
    files = []
    files.extend(sorted(PLAYLISTS_DIR.glob("*/metadata/*.xlsx")))
    files.extend(sorted(PLAYLISTS_DIR.glob("*/metadata/*.xlsm")))

    uploaded_excels_dir = METADATA_EXCEL_IMPORT_DIR / "uploaded_excels"
    if uploaded_excels_dir.exists():
        files.extend(sorted(uploaded_excels_dir.glob("*.xlsx")))
        files.extend(sorted(uploaded_excels_dir.glob("*.xlsm")))

    unique = {}
    for file_path in files:
        unique[str(file_path.resolve())] = file_path

    return sorted(unique.values(), key=lambda p: str(p).lower())


def get_playlist_slugs():
    return sorted([p.name for p in PLAYLISTS_DIR.glob("*") if p.is_dir()])


def format_path_for_display(path: Path):
    try:
        return str(path.relative_to(ROOT))
    except Exception:
        return str(path)


def read_text_file(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return "Unable to read file."

    
def create_progress_reporter(section_key: str):
    progress_bar = st.progress(0)
    stage_box = st.empty()
    detail_box = st.empty()

    def report(event: dict):
        percent = int(float(event.get("percent", 0)))
        stage = str(event.get("stage", "")).strip()
        message = str(event.get("message", "")).strip()
        current = event.get("current")
        total = event.get("total")

        progress_bar.progress(max(0, min(100, percent)))

        if stage:
            stage_box.markdown(f"**Stage:** {stage}")

        if current is not None and total is not None and total:
            detail_box.markdown(f"{message}  \n**Progress:** {current}/{total}")
        else:
            detail_box.markdown(message)

    return report


def render_text_download(path: Path, label: str, key_prefix: str):
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime="text/plain",
            key=f"{key_prefix}_{path.name}_txt_download",
        )


def render_binary_download(path: Path, label: str, mime: str, key_prefix: str):
    if path.exists():
        st.download_button(
            label=label,
            data=path.read_bytes(),
            file_name=path.name,
            mime=mime,
            key=f"{key_prefix}_{path.name}_download",
        )


def render_transcript_preview(transcript_file: Path, key_prefix: str):
    if not transcript_file.exists():
        return

    transcript_text = read_text_file(transcript_file)

    st.download_button(
        label="Download Transcript TXT",
        data=transcript_text.encode("utf-8"),
        file_name=transcript_file.name,
        mime="text/plain",
        key=f"{key_prefix}_{transcript_file.stem}_download_txt",
    )

    st.text_area(
        "Transcript Preview",
        value=transcript_text,
        height=300,
        key=f"{key_prefix}_{transcript_file.stem}_preview",
    )


def render_metadata_outputs(metadata_result: dict, key_prefix: str):
    if not metadata_result:
        return

    excel_output = Path(metadata_result.get("excel_output", ""))
    json_output = Path(metadata_result.get("json_output", ""))
    ok_log = Path(metadata_result.get("ok_log", ""))
    err_log = Path(metadata_result.get("err_log", ""))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_binary_download(
            excel_output,
            "Download Metadata Excel",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            f"{key_prefix}_metadata_excel",
        )

    with col2:
        render_binary_download(
            json_output,
            "Download Metadata JSON",
            "application/json",
            f"{key_prefix}_metadata_json",
        )

    with col3:
        render_binary_download(
            ok_log,
            "Download OK Log",
            "text/tab-separated-values",
            f"{key_prefix}_metadata_oklog",
        )

    with col4:
        render_binary_download(
            err_log,
            "Download Error Log",
            "text/tab-separated-values",
            f"{key_prefix}_metadata_errlog",
        )

    rows = metadata_result.get("rows", [])
    if rows:
        with st.expander("Metadata Preview", expanded=True):
            st.dataframe(rows, use_container_width=True)

    if metadata_result.get("error_count", 0) > 0:
        st.warning("Some items failed during the Ollama call or response parsing. Check the error log.")


def render_playlist_management_ui(key_prefix: str):
    manage_col1, manage_col2 = st.columns(2)

    with manage_col1:
        if st.button("Clear All Playlist Folders", key=f"{key_prefix}_clear_all_playlists"):
            removed_files, removed_dirs = clear_folder_contents(PLAYLISTS_DIR)
            st.success(f"Removed {removed_dirs} playlist folder(s) and {removed_files} extra file(s).")

    with manage_col2:
        if st.button("Clear All Generated Data", key=f"{key_prefix}_clear_all_data"):
            summary = clear_generated_project_data()
            st.success(build_clear_summary_message(summary))

    playlist_slug_to_clear = st.text_input(
        "Enter playlist slug to delete",
        key=f"{key_prefix}_playlist_slug_to_clear",
        placeholder="e.g. drisrar-ahmad-short-clips",
    )

    if st.button("Delete This Playlist Folder", key=f"{key_prefix}_delete_playlist_folder"):
        ok, msg = delete_playlist_folder_by_slug(playlist_slug_to_clear)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)


# --------------------------------------------------
# Tabs
# --------------------------------------------------

transcription_tab, metadata_tab = st.tabs([
    "Transcription",
    "Metadata / SEO",
])

# ==================================================
# TAB 1 — TRANSCRIPTION
# ==================================================

with transcription_tab:
    t_single, t_batch, t_playlist, t_exports = st.tabs([
        "Single YouTube",
        "Batch Media",
        "Playlist",
        "Transcript Exports",
    ])

    # ----------------------------------------------
    # Single YouTube
    # ----------------------------------------------

    with t_single:
        st.subheader("Single YouTube Transcription")

        yt_url = st.text_input("Paste YouTube Link", key="transcription_single_youtube_url")

        single_col1, single_col2 = st.columns(2)

        with single_col1:
            if st.button("Transcribe YouTube Video", key="transcription_single_youtube_button"):
                if not yt_url.strip():
                    st.warning("Please paste a YouTube link first.")
                else:
                    try:
                        progress_callback = create_progress_reporter("transcription_single_youtube")

                        result = transcribe_single_youtube(
                            yt_url.strip(),
                            progress_callback=progress_callback,
                        )

                        st.session_state["transcription_single_result"] = result

                        if result.get("ok", False):
                            st.success("Transcription completed successfully.")
                        else:
                            st.error("Transcription failed.")
                    except Exception as e:
                        st.error(f"Error: {e}")

        with single_col2:
            if st.button("Clear All Generated Data", key="transcription_single_clear_all"):
                summary = clear_generated_project_data()
                st.success(build_clear_summary_message(summary))

        single_result = st.session_state.get("transcription_single_result")

        if single_result:
            st.write(f"Download time: **{single_result.get('download_time_sec', 0):.2f} sec**")
            st.write(f"Standardize time: **{single_result.get('standardize_time_sec', 0):.2f} sec**")
            st.write(f"Transcription time: **{single_result.get('transcription_time_sec', 0):.2f} sec**")

            transcript_file = Path(single_result.get("transcript_file", ""))
            if transcript_file.exists():
                render_transcript_preview(transcript_file, "transcription_single")

    # ----------------------------------------------
    # Batch Media
    # ----------------------------------------------

    with t_batch:
        st.subheader("Batch Media Transcription")

        batch_manage_col1, batch_manage_col2, batch_manage_col3, batch_manage_col4 = st.columns(4)

        with batch_manage_col1:
            if st.button("Clear Audio Uploads", key="transcription_batch_clear_audio"):
                removed = clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS | VIDEO_EXTS)
                st.success(f"Removed {removed} audio file(s).")

        with batch_manage_col2:
            if st.button("Clear Video Uploads", key="transcription_batch_clear_video"):
                removed = clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)
                st.success(f"Removed {removed} video file(s).")

        with batch_manage_col3:
            if st.button("Clear Transcript Files", key="transcription_batch_clear_transcripts"):
                removed = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json", ".xlsx"})
                st.success(f"Removed {removed} transcript/export file(s).")

        with batch_manage_col4:
            if st.button("Clear All Generated Data", key="transcription_batch_clear_all"):
                summary = clear_generated_project_data()
                st.success(build_clear_summary_message(summary))

        uploaded_batch_files = st.file_uploader(
            "Upload audio or video files",
            accept_multiple_files=True,
            type=[
                "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma",
                "mp4", "mkv", "mov", "avi", "webm",
            ],
            key="transcription_batch_media_uploader",
        )

        batch_action_col1, batch_action_col2 = st.columns(2)

        with batch_action_col1:
            if st.button("Save Uploaded Media Files", key="transcription_batch_save_files"):
                if uploaded_batch_files:
                    audio_count, video_count, _ = save_uploaded_files(uploaded_batch_files)
                    st.success(f"Saved {len(uploaded_batch_files)} file(s) ({audio_count} audio, {video_count} video).")
                else:
                    st.warning("Please choose files first.")

        with batch_action_col2:
            if st.button("Start Batch Transcription", key="transcription_batch_start"):
                try:
                    progress_callback = create_progress_reporter("transcription_batch")

                    result = transcribe_batch_media(
                        progress_callback=progress_callback,
                    )

                    st.session_state["transcription_batch_result"] = result

                    if result.get("ok", False):
                        st.success("Batch transcription completed successfully.")
                    else:
                        st.error(result.get("error", "Batch transcription failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        current_audio_files = get_audio_files(UPLOAD_AUDIO_DIR)
        current_video_files = get_video_files(UPLOAD_VIDEO_DIR)

        st.write(f"Audio files available: **{len(current_audio_files)}**")
        st.write(f"Video files available: **{len(current_video_files)}**")

        batch_result = st.session_state.get("transcription_batch_result")

        if batch_result:
            st.write(f"Audio files processed: **{batch_result.get('audio_file_count', 0)}**")
            st.write(f"Video files processed: **{batch_result.get('video_file_count', 0)}**")
            st.write(f"Standardize time: **{batch_result.get('standardize_time_sec', 0):.2f} sec**")
            st.write(f"Transcription time: **{batch_result.get('transcription_time_sec', 0):.2f} sec**")

            transcript_files = [Path(p) for p in batch_result.get("transcript_files", []) if Path(p).exists()]
            if transcript_files:
                zip_bytes = build_transcript_zip_bytes(transcript_files)

                st.download_button(
                    label="Download Batch Transcripts ZIP",
                    data=zip_bytes,
                    file_name="batch_transcripts.zip",
                    mime="application/zip",
                    key="transcription_batch_zip_download",
                )

                for transcript_file in transcript_files[:10]:
                    with st.expander(transcript_file.name):
                        st.text_area(
                            "Transcript",
                            value=read_text_file(transcript_file),
                            height=250,
                            key=f"transcription_batch_preview_{transcript_file.stem}",
                        )

    # ----------------------------------------------
    # Playlist
    # ----------------------------------------------

    with t_playlist:
        st.subheader("Playlist Download & Transcription")

        render_playlist_management_ui("transcription_playlist_manage")

        playlist_url = st.text_input("Paste YouTube Playlist Link", key="transcription_playlist_url")
        playlist_quality = st.selectbox(
            "Select Download Quality",
            ["480p", "720p", "1080p"],
            index=1,
            key="transcription_playlist_quality",
        )

        if st.button("Download Playlist and Transcribe", key="transcription_playlist_button"):
            if not playlist_url.strip():
                st.warning("Please paste a playlist link first.")
            else:
                try:
                    progress_callback = create_progress_reporter("transcription_playlist")

                    result = transcribe_playlist(
                        playlist_url=playlist_url.strip(),
                        quality=playlist_quality,
                        progress_callback=progress_callback,
                    )

                    st.session_state["transcription_playlist_result"] = result

                    if result.get("ok", False):
                        st.success("Playlist transcription completed successfully.")
                    else:
                        st.error("Playlist transcription failed.")
                except Exception as e:
                    st.error(f"Error: {e}")

        playlist_result = st.session_state.get("transcription_playlist_result")

        if playlist_result:
            manifest = playlist_result.get("manifest", {}) or {}
            st.write(f"Playlist: **{manifest.get('playlist_title', 'Unknown')}**")
            st.write(f"Videos found: **{manifest.get('entry_count', 0)}**")
            st.write(f"Quality selected: **{playlist_result.get('quality', '')}**")
            st.write(f"Download time: **{playlist_result.get('download_time_sec', 0):.2f} sec**")
            st.write(f"Transcription time: **{playlist_result.get('transcription_time_sec', 0):.2f} sec**")

            transcripts_dir = Path(playlist_result.get("transcripts_dir", ""))
            playlist_slug = Path(playlist_result.get("playlist_root", "")).name

            if transcripts_dir.exists():
                zip_result = build_transcript_folder_zip_bytes(transcripts_dir)

                st.download_button(
                    label="Download Playlist Transcripts ZIP",
                    data=zip_result["zip_bytes"],
                    file_name=f"{playlist_slug}_transcripts.zip",
                    mime="application/zip",
                    key=f"playlist_txt_zip_{playlist_slug}",
                )

            playlist_detailed_excel = Path(playlist_result.get("playlist_excel_file", ""))
            if playlist_detailed_excel.exists():
                render_binary_download(
                    playlist_detailed_excel,
                    "Download Playlist Detailed Excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    f"playlist_detailed_excel_{playlist_slug}",
                )

            if playlist_slug:
                try:
                    playlist_struct_excel = export_playlist_transcripts_to_excel(playlist_slug)
                    playlist_struct_json = export_playlist_transcripts_to_json(playlist_slug)

                    excel_path = Path(playlist_struct_excel.get("output_file", ""))
                    json_path = Path(playlist_struct_json.get("output_file", ""))

                    export_col1, export_col2 = st.columns(2)

                    with export_col1:
                        render_binary_download(
                            excel_path,
                            "Download Transcript Excel",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            f"playlist_struct_excel_{playlist_slug}",
                        )

                    with export_col2:
                        render_binary_download(
                            json_path,
                            "Download Transcript JSON",
                            "application/json",
                            f"playlist_struct_json_{playlist_slug}",
                        )
                except Exception as e:
                    st.warning(f"Could not prepare structured playlist transcript exports: {e}")

            transcript_files = sorted(transcripts_dir.glob("*.txt")) if transcripts_dir.exists() else []
            for transcript_file in transcript_files[:10]:
                with st.expander(transcript_file.name):
                    st.text_area(
                        "Transcript",
                        value=read_text_file(transcript_file),
                        height=250,
                        key=f"playlist_transcript_preview_{transcript_file.stem}",
                    )

    # ----------------------------------------------
    # Transcript Exports
    # ----------------------------------------------

    with t_exports:
        st.subheader("Transcript Exports")

        export_col1, export_col2 = st.columns(2)

        with export_col1:
            if st.button("Clear Transcript Files", key="transcript_exports_clear_transcripts"):
                removed = clear_files_in_folder(TRANSCRIPT_DIR, {".txt", ".zip", ".json", ".xlsx"})
                st.success(f"Removed {removed} transcript/export file(s).")

        with export_col2:
            if st.button("Clear All Generated Data", key="transcript_exports_clear_all"):
                summary = clear_generated_project_data()
                st.success(build_clear_summary_message(summary))

        global_transcripts = sorted(TRANSCRIPT_DIR.glob("*.txt"))

        st.markdown("### Global Transcript Exports")

        if global_transcripts:
            try:
                global_zip = build_global_transcripts_zip_bytes()
                global_excel = export_global_transcripts_to_excel(TRANSCRIPT_DIR / "transcripts_export.xlsx")
                global_json = export_global_transcripts_to_json(TRANSCRIPT_DIR / "transcripts_export.json")

                gcol1, gcol2, gcol3 = st.columns(3)

                with gcol1:
                    st.download_button(
                        label="Download TXT ZIP",
                        data=global_zip["zip_bytes"],
                        file_name="transcripts.zip",
                        mime="application/zip",
                        key="global_transcript_zip_download",
                    )

                with gcol2:
                    render_binary_download(
                        Path(global_excel.get("output_file", "")),
                        "Download Excel",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "global_transcript_excel",
                    )

                with gcol3:
                    render_binary_download(
                        Path(global_json.get("output_file", "")),
                        "Download JSON",
                        "application/json",
                        "global_transcript_json",
                    )

                with st.expander("Preview Global Transcripts", expanded=False):
                    for transcript_file in global_transcripts[:10]:
                        st.text_area(
                            transcript_file.name,
                            value=read_text_file(transcript_file),
                            height=200,
                            key=f"global_export_preview_{transcript_file.stem}",
                        )
            except Exception as e:
                st.error(f"Global transcript export failed: {e}")
        else:
            st.info("No transcript files found in the global transcript directory.")

        st.markdown("### Playlist Transcript Exports")

        playlist_slugs = get_playlist_slugs()

        if playlist_slugs:
            selected_playlist_slug = st.selectbox(
                "Select Playlist",
                options=playlist_slugs,
                key="transcript_exports_playlist_slug",
            )

            if selected_playlist_slug:
                try:
                    playlist_zip_result = build_transcript_folder_zip_bytes(
                        PLAYLISTS_DIR / selected_playlist_slug / "transcripts"
                    )
                    playlist_excel_result = export_playlist_transcripts_to_excel(selected_playlist_slug)
                    playlist_json_result = export_playlist_transcripts_to_json(selected_playlist_slug)

                    pcol1, pcol2, pcol3 = st.columns(3)

                    with pcol1:
                        st.download_button(
                            label="Download TXT ZIP",
                            data=playlist_zip_result["zip_bytes"],
                            file_name=f"{selected_playlist_slug}_transcripts.zip",
                            mime="application/zip",
                            key=f"playlist_export_zip_{selected_playlist_slug}",
                        )

                    with pcol2:
                        render_binary_download(
                            Path(playlist_excel_result.get("output_file", "")),
                            "Download Excel",
                            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            f"playlist_export_excel_{selected_playlist_slug}",
                        )

                    with pcol3:
                        render_binary_download(
                            Path(playlist_json_result.get("output_file", "")),
                            "Download JSON",
                            "application/json",
                            f"playlist_export_json_{selected_playlist_slug}",
                        )
                except Exception as e:
                    st.error(f"Playlist transcript export failed: {e}")
        else:
            st.info("No playlist folders found yet.")

# ==================================================
# TAB 2 — METADATA / SEO
# ==================================================

with metadata_tab:
    m_single, m_batch, m_playlist, m_existing = st.tabs([
        "Single YouTube",
        "Batch Inputs",
        "Playlist",
        "Existing Transcripts / Excel",
    ])

    # ----------------------------------------------
    # Single YouTube
    # ----------------------------------------------

    with m_single:
        st.subheader("Single YouTube → Transcript + Metadata")

        single_meta_url = st.text_input("Paste YouTube Link", key="metadata_single_youtube_url")

        with st.expander("LLM Settings", expanded=False):
            llm_col1, llm_col2 = st.columns(2)

            with llm_col1:
                single_model = st.text_input("Ollama Model", value=DEFAULT_MODEL, key="metadata_single_model")
                single_base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, key="metadata_single_base_url")
                single_timeout = st.number_input("Timeout (sec)", min_value=1, value=DEFAULT_TIMEOUT, step=1, key="metadata_single_timeout")
                single_retries = st.number_input("Retries", min_value=0, value=DEFAULT_RETRIES, step=1, key="metadata_single_retries")

            with llm_col2:
                single_sleep_ms = st.number_input("Sleep Between Items (ms)", min_value=0, value=DEFAULT_SLEEP_MS, step=50, key="metadata_single_sleep_ms")
                single_temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(DEFAULT_TEMPERATURE), step=0.1, key="metadata_single_temperature")
                single_num_ctx = st.number_input("Context Window", min_value=256, value=DEFAULT_NUM_CTX, step=256, key="metadata_single_num_ctx")
                single_num_predict = st.number_input("Max Output Tokens", min_value=64, value=DEFAULT_NUM_PREDICT, step=64, key="metadata_single_num_predict")

            single_seed_text = st.text_input("Seed (optional)", value="", key="metadata_single_seed")

        if st.button("Generate Transcript + Metadata", key="metadata_single_youtube_button"):
            if not single_meta_url.strip():
                st.warning("Please paste a YouTube link first.")
            else:
                try:
                    seed_value = int(single_seed_text.strip()) if single_seed_text.strip() else None
                    progress_callback = create_progress_reporter("metadata_single_youtube")

                    result = generate_metadata_from_single_youtube(
                        youtube_url=single_meta_url.strip(),
                        model=single_model.strip() or DEFAULT_MODEL,
                        base_url=single_base_url.strip() or DEFAULT_BASE_URL,
                        timeout=int(single_timeout),
                        retries=int(single_retries),
                        sleep_ms=int(single_sleep_ms),
                        temperature=float(single_temperature),
                        num_ctx=int(single_num_ctx),
                        num_predict=int(single_num_predict),
                        seed=seed_value,
                        progress_callback=progress_callback,
                    )

                    st.session_state["metadata_single_result"] = result

                    if result.get("ok", False):
                        st.success("Transcript and metadata generation completed successfully.")
                    else:
                        st.error(result.get("error", "Metadata generation failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        single_meta_result = st.session_state.get("metadata_single_result")

        if single_meta_result:
            transcription_result = single_meta_result.get("transcription", {}) or {}
            transcript_file = Path(single_meta_result.get("transcript_file", ""))

            st.write(f"Download time: **{transcription_result.get('download_time_sec', 0):.2f} sec**")
            st.write(f"Standardize time: **{transcription_result.get('standardize_time_sec', 0):.2f} sec**")
            st.write(f"Transcription time: **{transcription_result.get('transcription_time_sec', 0):.2f} sec**")

            if transcript_file.exists():
                render_transcript_preview(transcript_file, "metadata_single")

            render_metadata_outputs(single_meta_result.get("metadata", {}) or {}, "metadata_single_outputs")

    # ----------------------------------------------
    # Batch Inputs
    # ----------------------------------------------

    with m_batch:
        st.subheader("Batch Inputs → Metadata")

        batch_mode = st.radio(
            "Batch Input Type",
            ["Media Files", "Transcript TXT Files"],
            horizontal=True,
            key="metadata_batch_mode",
        )

        with st.expander("LLM Settings", expanded=False):
            batch_llm_col1, batch_llm_col2 = st.columns(2)

            with batch_llm_col1:
                batch_model = st.text_input("Ollama Model", value=DEFAULT_MODEL, key="metadata_batch_model")
                batch_base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, key="metadata_batch_base_url")
                batch_timeout = st.number_input("Timeout (sec)", min_value=1, value=DEFAULT_TIMEOUT, step=1, key="metadata_batch_timeout")
                batch_retries = st.number_input("Retries", min_value=0, value=DEFAULT_RETRIES, step=1, key="metadata_batch_retries")

            with batch_llm_col2:
                batch_sleep_ms = st.number_input("Sleep Between Items (ms)", min_value=0, value=DEFAULT_SLEEP_MS, step=50, key="metadata_batch_sleep_ms")
                batch_temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(DEFAULT_TEMPERATURE), step=0.1, key="metadata_batch_temperature")
                batch_num_ctx = st.number_input("Context Window", min_value=256, value=DEFAULT_NUM_CTX, step=256, key="metadata_batch_num_ctx")
                batch_num_predict = st.number_input("Max Output Tokens", min_value=64, value=DEFAULT_NUM_PREDICT, step=64, key="metadata_batch_num_predict")

            batch_seed_text = st.text_input("Seed (optional)", value="", key="metadata_batch_seed")

        if batch_mode == "Media Files":
            media_manage_col1, media_manage_col2, media_manage_col3 = st.columns(3)

            with media_manage_col1:
                if st.button("Clear Upload Audio", key="metadata_batch_media_clear_audio"):
                    removed = clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS | VIDEO_EXTS)
                    st.success(f"Removed {removed} audio file(s).")

            with media_manage_col2:
                if st.button("Clear Upload Video", key="metadata_batch_media_clear_video"):
                    removed = clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)
                    st.success(f"Removed {removed} video file(s).")

            with media_manage_col3:
                if st.button("Clear Generated Metadata", key="metadata_batch_media_clear_metadata"):
                    removed_files, removed_dirs = clear_folder_contents(METADATA_BATCH_DIR)
                    st.success(f"Removed {removed_dirs} folder(s) and {removed_files} file(s).")

            uploaded_meta_media_files = st.file_uploader(
                "Upload audio or video files",
                accept_multiple_files=True,
                type=[
                    "mp3", "wav", "m4a", "flac", "ogg", "opus", "aac", "wma",
                    "mp4", "mkv", "mov", "avi", "webm",
                ],
                key="metadata_batch_media_uploader",
            )

            st.write(f"Current upload audio files: **{len(get_audio_files(UPLOAD_AUDIO_DIR))}**")
            st.write(f"Current upload video files: **{len(get_video_files(UPLOAD_VIDEO_DIR))}**")

            if st.button("Generate Metadata from Uploaded Media Files", key="metadata_batch_media_button"):
                if not uploaded_meta_media_files:
                    st.warning("Please upload media files first.")
                else:
                    try:
                        seed_value = int(batch_seed_text.strip()) if batch_seed_text.strip() else None

                        clear_files_in_folder(UPLOAD_AUDIO_DIR, AUDIO_EXTS | VIDEO_EXTS)
                        clear_files_in_folder(UPLOAD_VIDEO_DIR, VIDEO_EXTS)

                        save_uploaded_files(uploaded_meta_media_files)

                        progress_callback = create_progress_reporter("metadata_batch_media")

                        result = generate_metadata_from_batch_media(
                            model=batch_model.strip() or DEFAULT_MODEL,
                            base_url=batch_base_url.strip() or DEFAULT_BASE_URL,
                            timeout=int(batch_timeout),
                            retries=int(batch_retries),
                            sleep_ms=int(batch_sleep_ms),
                            temperature=float(batch_temperature),
                            num_ctx=int(batch_num_ctx),
                            num_predict=int(batch_num_predict),
                            seed=seed_value,
                            progress_callback=progress_callback,
                        )

                        st.session_state["metadata_batch_result"] = result

                        if result.get("ok", False):
                            st.success("Batch media metadata generation completed successfully.")
                        else:
                            st.error(result.get("error", "Batch media metadata generation failed."))
                    except Exception as e:
                        st.error(f"Error: {e}")

        else:
            transcript_upload_dir = METADATA_BATCH_DIR / "uploaded_txt_inputs"
            uploaded_meta_txt_files = st.file_uploader(
                "Upload transcript TXT files",
                accept_multiple_files=True,
                type=["txt"],
                key="metadata_batch_txt_uploader",
            )

            if st.button("Generate Metadata from Uploaded Transcript Files", key="metadata_batch_txt_button"):
                if not uploaded_meta_txt_files:
                    st.warning("Please upload transcript TXT files first.")
                else:
                    try:
                        seed_value = int(batch_seed_text.strip()) if batch_seed_text.strip() else None

                        saved_paths = save_uploaded_transcript_files(uploaded_meta_txt_files, transcript_upload_dir)

                        with st.spinner("Generating metadata from transcript files..."):
                            result = generate_metadata_from_transcript_files(
                                transcript_files=saved_paths,
                                output_name="uploaded_transcript_files",
                                metadata_output_dir=METADATA_BATCH_DIR,
                                model=batch_model.strip() or DEFAULT_MODEL,
                                base_url=batch_base_url.strip() or DEFAULT_BASE_URL,
                                timeout=int(batch_timeout),
                                retries=int(batch_retries),
                                sleep_ms=int(batch_sleep_ms),
                                temperature=float(batch_temperature),
                                num_ctx=int(batch_num_ctx),
                                num_predict=int(batch_num_predict),
                                seed=seed_value,
                            )

                        st.session_state["metadata_batch_result"] = result

                        if result.get("ok", False):
                            st.success("Metadata generated successfully from transcript files.")
                        else:
                            st.error(result.get("error", "Metadata generation failed."))
                    except Exception as e:
                        st.error(f"Error: {e}")

        batch_meta_result = st.session_state.get("metadata_batch_result")

        if batch_meta_result:
            transcript_files_used = [Path(p) for p in batch_meta_result.get("transcript_files_used", []) if Path(p).exists()]
            if transcript_files_used:
                zip_bytes = build_transcript_zip_bytes(transcript_files_used)

                st.download_button(
                    label="Download Used Transcript TXT ZIP",
                    data=zip_bytes,
                    file_name="batch_used_transcripts.zip",
                    mime="application/zip",
                    key="metadata_batch_used_transcripts_zip",
                )

            render_metadata_outputs(batch_meta_result.get("metadata", {}) or {}, "metadata_batch_outputs")

    # ----------------------------------------------
    # Playlist
    # ----------------------------------------------

    with m_playlist:
        st.subheader("Playlist → Transcript + Metadata")

        render_playlist_management_ui("metadata_playlist_manage")

        metadata_playlist_url = st.text_input("Paste YouTube Playlist Link", key="metadata_playlist_url")
        metadata_playlist_quality = st.selectbox(
            "Select Download Quality",
            ["480p", "720p", "1080p"],
            index=1,
            key="metadata_playlist_quality",
        )

        with st.expander("LLM Settings", expanded=False):
            playlist_llm_col1, playlist_llm_col2 = st.columns(2)

            with playlist_llm_col1:
                playlist_model = st.text_input("Ollama Model", value=DEFAULT_MODEL, key="metadata_playlist_model")
                playlist_base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, key="metadata_playlist_base_url")
                playlist_timeout = st.number_input("Timeout (sec)", min_value=1, value=DEFAULT_TIMEOUT, step=1, key="metadata_playlist_timeout")
                playlist_retries = st.number_input("Retries", min_value=0, value=DEFAULT_RETRIES, step=1, key="metadata_playlist_retries")

            with playlist_llm_col2:
                playlist_sleep_ms = st.number_input("Sleep Between Items (ms)", min_value=0, value=DEFAULT_SLEEP_MS, step=50, key="metadata_playlist_sleep_ms")
                playlist_temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(DEFAULT_TEMPERATURE), step=0.1, key="metadata_playlist_temperature")
                playlist_num_ctx = st.number_input("Context Window", min_value=256, value=DEFAULT_NUM_CTX, step=256, key="metadata_playlist_num_ctx")
                playlist_num_predict = st.number_input("Max Output Tokens", min_value=64, value=DEFAULT_NUM_PREDICT, step=64, key="metadata_playlist_num_predict")

            playlist_seed_text = st.text_input("Seed (optional)", value="", key="metadata_playlist_seed")

        if st.button("Generate Playlist Transcript + Metadata", key="metadata_playlist_button"):
            if not metadata_playlist_url.strip():
                st.warning("Please paste a playlist link first.")
            else:
                try:
                    seed_value = int(playlist_seed_text.strip()) if playlist_seed_text.strip() else None
                    progress_callback = create_progress_reporter("metadata_playlist")

                    result = generate_metadata_from_playlist(
                        playlist_url=metadata_playlist_url.strip(),
                        quality=metadata_playlist_quality,
                        model=playlist_model.strip() or DEFAULT_MODEL,
                        base_url=playlist_base_url.strip() or DEFAULT_BASE_URL,
                        timeout=int(playlist_timeout),
                        retries=int(playlist_retries),
                        sleep_ms=int(playlist_sleep_ms),
                        temperature=float(playlist_temperature),
                        num_ctx=int(playlist_num_ctx),
                        num_predict=int(playlist_num_predict),
                        seed=seed_value,
                        progress_callback=progress_callback,
                    )

                    st.session_state["metadata_playlist_result"] = result

                    if result.get("ok", False):
                        st.success("Playlist metadata generation completed successfully.")
                    else:
                        st.error(result.get("error", "Playlist metadata generation failed."))
                except Exception as e:
                    st.error(f"Error: {e}")

        metadata_playlist_result = st.session_state.get("metadata_playlist_result")

        if metadata_playlist_result:
            transcription_result = metadata_playlist_result.get("transcription", {}) or {}
            manifest = transcription_result.get("manifest", {}) or {}
            transcripts_dir = Path(metadata_playlist_result.get("transcripts_dir", ""))

            st.write(f"Playlist: **{manifest.get('playlist_title', 'Unknown')}**")
            st.write(f"Videos found: **{manifest.get('entry_count', 0)}**")
            st.write(f"Quality selected: **{transcription_result.get('quality', '')}**")
            st.write(f"Download time: **{transcription_result.get('download_time_sec', 0):.2f} sec**")
            st.write(f"Transcription time: **{transcription_result.get('transcription_time_sec', 0):.2f} sec**")

            if transcripts_dir.exists():
                zip_result = build_transcript_folder_zip_bytes(transcripts_dir)
                st.download_button(
                    label="Download Playlist Transcript TXT ZIP",
                    data=zip_result["zip_bytes"],
                    file_name=f"{Path(transcription_result.get('playlist_root', '')).name}_transcripts.zip",
                    mime="application/zip",
                    key="metadata_playlist_txt_zip",
                )

            playlist_detailed_excel = Path(metadata_playlist_result.get("playlist_excel", ""))
            if playlist_detailed_excel.exists():
                render_binary_download(
                    playlist_detailed_excel,
                    "Download Playlist Detailed Excel",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "metadata_playlist_detailed_excel",
                )

            render_metadata_outputs(metadata_playlist_result.get("metadata", {}) or {}, "metadata_playlist_outputs")

    # ----------------------------------------------
    # Existing Transcripts / Excel
    # ----------------------------------------------

    with m_existing:
        st.subheader("Existing Transcripts / Excel → Metadata Only")

        existing_mode = st.radio(
            "Input Type",
            ["Single TXT", "Transcript Folder", "Excel File"],
            horizontal=True,
            key="metadata_existing_mode",
        )

        selected_transcript_file = None
        selected_transcript_folder = None
        selected_excel_path = None
        selected_sheet_name = None
        selected_transcript_column = None
        selected_filename_column = None

        if existing_mode == "Single TXT":
            single_txt_mode = st.radio(
                "Transcript Source",
                ["Existing Generated Transcript", "Upload TXT File"],
                horizontal=True,
                key="metadata_existing_single_mode",
            )

            if single_txt_mode == "Existing Generated Transcript":
                transcript_files = get_all_transcript_files()

                if not transcript_files:
                    st.info("No transcript files found in the app output folders.")
                else:
                    file_options = [str(p) for p in transcript_files]

                    selected_file_str = st.selectbox(
                        "Select Transcript File",
                        options=file_options,
                        format_func=lambda p: format_path_for_display(Path(p)),
                        key="metadata_existing_single_select",
                    )

                    selected_transcript_file = Path(selected_file_str)

            else:
                uploaded_txt = st.file_uploader(
                    "Upload Transcript TXT File",
                    type=["txt"],
                    key="metadata_existing_single_upload",
                )

                if uploaded_txt is not None:
                    selected_transcript_file = save_uploaded_file(
                        uploaded_txt,
                        METADATA_SINGLE_DIR / "uploaded_txt_inputs",
                    )
                    st.success(f"Uploaded transcript saved: {selected_transcript_file.name}")

        elif existing_mode == "Transcript Folder":
            transcript_folders = get_transcript_source_folders()

            if not transcript_folders:
                st.info("No transcript folders found.")
            else:
                folder_options = [str(p) for p in transcript_folders]

                selected_folder_str = st.selectbox(
                    "Select Transcript Folder",
                    options=folder_options,
                    format_func=lambda p: format_path_for_display(Path(p)),
                    key="metadata_existing_folder_select",
                )

                selected_transcript_folder = Path(selected_folder_str)

                txt_count = len(list(selected_transcript_folder.glob("*.txt")))
                st.write(f"Transcript files found: **{txt_count}**")

        else:
            excel_input_mode = st.radio(
                "Excel Source",
                ["Existing Generated Excel File", "Upload Excel File"],
                horizontal=True,
                key="metadata_existing_excel_mode",
            )

            if excel_input_mode == "Existing Generated Excel File":
                excel_files = get_existing_excel_sources()

                if not excel_files:
                    st.info("No Excel files found.")
                else:
                    excel_options = [str(p) for p in excel_files]

                    selected_excel_str = st.selectbox(
                        "Select Excel File",
                        options=excel_options,
                        format_func=lambda p: format_path_for_display(Path(p)),
                        key="metadata_existing_excel_select",
                    )

                    selected_excel_path = Path(selected_excel_str)
            else:
                uploaded_excel = st.file_uploader(
                    "Upload Excel File",
                    type=["xlsx", "xlsm"],
                    key="metadata_existing_excel_upload",
                )

                if uploaded_excel is not None:
                    selected_excel_path = save_uploaded_excel(uploaded_excel)
                    st.success(f"Uploaded Excel saved: {selected_excel_path.name}")

            if selected_excel_path:
                try:
                    sheet_names = get_excel_sheet_names(selected_excel_path)

                    if sheet_names:
                        selected_sheet_name = st.selectbox(
                            "Select Sheet",
                            options=sheet_names,
                            key="metadata_existing_excel_sheet",
                        )

                        columns = get_excel_columns(selected_excel_path, selected_sheet_name)

                        if columns:
                            lower_columns = [c.strip().lower() for c in columns]
                            transcript_default_index = lower_columns.index("transcription") if "transcription" in lower_columns else 0

                            selected_transcript_column = st.selectbox(
                                "Transcript Column",
                                options=columns,
                                index=transcript_default_index,
                                key="metadata_existing_excel_transcript_column",
                            )

                            filename_options = ["(auto-generate row names)"] + columns
                            filename_default_index = lower_columns.index("title") + 1 if "title" in lower_columns else 0

                            filename_choice = st.selectbox(
                                "Filename Column (optional)",
                                options=filename_options,
                                index=filename_default_index,
                                key="metadata_existing_excel_filename_column",
                            )

                            if filename_choice != "(auto-generate row names)":
                                selected_filename_column = filename_choice

                except Exception as e:
                    st.error(f"Failed to inspect Excel file: {e}")

        with st.expander("LLM Settings", expanded=False):
            existing_llm_col1, existing_llm_col2 = st.columns(2)

            with existing_llm_col1:
                existing_model = st.text_input("Ollama Model", value=DEFAULT_MODEL, key="metadata_existing_model")
                existing_base_url = st.text_input("Base URL", value=DEFAULT_BASE_URL, key="metadata_existing_base_url")
                existing_timeout = st.number_input("Timeout (sec)", min_value=1, value=DEFAULT_TIMEOUT, step=1, key="metadata_existing_timeout")
                existing_retries = st.number_input("Retries", min_value=0, value=DEFAULT_RETRIES, step=1, key="metadata_existing_retries")

            with existing_llm_col2:
                existing_sleep_ms = st.number_input("Sleep Between Items (ms)", min_value=0, value=DEFAULT_SLEEP_MS, step=50, key="metadata_existing_sleep_ms")
                existing_temperature = st.number_input("Temperature", min_value=0.0, max_value=2.0, value=float(DEFAULT_TEMPERATURE), step=0.1, key="metadata_existing_temperature")
                existing_num_ctx = st.number_input("Context Window", min_value=256, value=DEFAULT_NUM_CTX, step=256, key="metadata_existing_num_ctx")
                existing_num_predict = st.number_input("Max Output Tokens", min_value=64, value=DEFAULT_NUM_PREDICT, step=64, key="metadata_existing_num_predict")

            existing_seed_text = st.text_input("Seed (optional)", value="", key="metadata_existing_seed")

        if st.button("Generate Metadata", key="metadata_existing_generate_button"):
            try:
                seed_value = int(existing_seed_text.strip()) if existing_seed_text.strip() else None

                with st.spinner("Generating metadata..."):
                    if existing_mode == "Single TXT":
                        if selected_transcript_file is None:
                            st.warning("Please choose or upload a transcript file first.")
                        else:
                            result = generate_metadata_from_single_transcript_file(
                                transcript_file=selected_transcript_file,
                                model=existing_model.strip() or DEFAULT_MODEL,
                                base_url=existing_base_url.strip() or DEFAULT_BASE_URL,
                                timeout=int(existing_timeout),
                                retries=int(existing_retries),
                                sleep_ms=int(existing_sleep_ms),
                                temperature=float(existing_temperature),
                                num_ctx=int(existing_num_ctx),
                                num_predict=int(existing_num_predict),
                                seed=seed_value,
                            )
                            st.session_state["metadata_existing_result"] = result

                    elif existing_mode == "Transcript Folder":
                        if selected_transcript_folder is None:
                            st.warning("Please select a transcript folder first.")
                        else:
                            result = generate_metadata_from_transcript_folder(
                                transcript_folder=selected_transcript_folder,
                                model=existing_model.strip() or DEFAULT_MODEL,
                                base_url=existing_base_url.strip() or DEFAULT_BASE_URL,
                                timeout=int(existing_timeout),
                                retries=int(existing_retries),
                                sleep_ms=int(existing_sleep_ms),
                                temperature=float(existing_temperature),
                                num_ctx=int(existing_num_ctx),
                                num_predict=int(existing_num_predict),
                                seed=seed_value,
                            )
                            st.session_state["metadata_existing_result"] = result

                    else:
                        if selected_excel_path is None or not selected_transcript_column:
                            st.warning("Please choose an Excel file and transcript column first.")
                        else:
                            result = generate_metadata_from_excel(
                                excel_file=selected_excel_path,
                                transcript_column=selected_transcript_column,
                                filename_column=selected_filename_column,
                                sheet_name=selected_sheet_name,
                                model=existing_model.strip() or DEFAULT_MODEL,
                                base_url=existing_base_url.strip() or DEFAULT_BASE_URL,
                                timeout=int(existing_timeout),
                                retries=int(existing_retries),
                                sleep_ms=int(existing_sleep_ms),
                                temperature=float(existing_temperature),
                                num_ctx=int(existing_num_ctx),
                                num_predict=int(existing_num_predict),
                                seed=seed_value,
                            )
                            st.session_state["metadata_existing_result"] = result

                existing_result = st.session_state.get("metadata_existing_result")
                if existing_result:
                    if existing_result.get("ok", False):
                        st.success("Metadata generation completed successfully.")
                    else:
                        st.error(existing_result.get("error", "Metadata generation failed."))
            except Exception as e:
                st.error(f"Error: {e}")

        existing_result = st.session_state.get("metadata_existing_result")
        if existing_result:
            render_metadata_outputs(existing_result.get("metadata", {}) or existing_result, "metadata_existing_outputs")
