import sys
import subprocess
import time
import re
from pathlib import Path
import streamlit as st
import zipfile
import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from app.config.paths import UPLOAD_AUDIO_DIR, TRANSCRIPT_DIR
from app.pipelines.media_pipeline.ingestion_runner import run_ingestion
from app.pipelines.media_pipeline.youtube_downloader import download_youtube_audio
from app.pipelines.media_pipeline.audio_standardizer import standardize_audio

UPLOAD_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AI Media Transcription", page_icon="🎙", layout="wide")

st.title("AI Media Transcription")

# --------------------------------------------------
# Tabs
# --------------------------------------------------

tab1, tab2, tab3 = st.tabs(["YouTube", "Upload & Batch Process", "Transcripts"])

# ==================================================
# TAB 1 — YOUTUBE
# ==================================================

with tab1:

    st.header("YouTube Transcription")

    youtube_url = st.text_input("Paste YouTube Link")

    if st.button("Download & Transcribe", key="yt_button"):

        if youtube_url:

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

                        progress_bar.progress(int(percent))

                        percent_box.markdown(f"Download: **{percent:.2f}%**")

                    eta = d.get("eta")

                    if eta:
                        eta_box.markdown(f"ETA: **{eta} sec**")

                    speed = d.get("speed")

                    if speed:
                        speed_box.markdown(
                            f"Speed: **{speed/1024/1024:.2f} MB/s**"
                        )

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

            st.success("Audio standardized")

            st.subheader("Transcription")

            progress_pattern = re.compile(r"(\d+\.\d+)%")

            progress_bar = st.progress(0)

            percent_container = st.empty()
            elapsed_container = st.empty()
            eta_container = st.empty()

            start_transcribe = time.time()

            cmd = [
                sys.executable,
                "app/pipelines/transcription_pipeline/transcriber_engine.py",
                str(wav_file),
                "--device", "cuda",
                "--model", "large-v3",
                "--compute_type", "float16",
                "--batch_size", "8",
                "--beam_size", "2",
                "--vad",
                "--local_only"
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )

            last_update = 0

            while True:

                line = process.stdout.readline() if process.stdout else ""

                if not line and process.poll() is not None:
                    break

                if line:

                    match = progress_pattern.search(line)

                    if match:

                        percent = float(match.group(1))

                        progress_bar.progress(int(percent))

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

            st.subheader("Statistics")

            st.write(f"Download Time: **{download_time:.2f} sec**")
            st.write(f"Conversion Time: **{convert_time:.2f} sec**")
            st.write(f"Transcription Time: **{transcription_time:.2f} sec**")


# ==================================================
# TAB 2 — UPLOAD & BATCH PROCESS
# ==================================================

with tab2:

    st.header("Upload & Batch Transcription")

    uploaded_files = st.file_uploader(
        "Upload audio or video files",
        accept_multiple_files=True,
        type=[
            "mp3","wav","m4a","flac","ogg","opus","aac","wma",
            "mp4","mkv","mov","avi"
        ]
    )

    if uploaded_files:

        for file in uploaded_files:

            save_path = UPLOAD_AUDIO_DIR / file.name

            with open(save_path, "wb") as f:
                f.write(file.read())

        st.success(f"{len(uploaded_files)} files uploaded")

    media_files = sorted(UPLOAD_AUDIO_DIR.glob("*"))

    if media_files:

        if st.button("Start Batch Transcription"):

            batch_start = time.time()

            st.subheader("Standardizing Audio")

            standardized_files = run_ingestion(UPLOAD_AUDIO_DIR)

            st.success(f"{len(standardized_files)} files standardized")

            st.subheader("Transcription")

            progress_pattern = re.compile(r"(\d+\.\d+)%")

            stats = []

            for audio in standardized_files:

                st.write(f"Processing **{audio.name}**")

                progress_bar = st.progress(0)

                percent_box = st.empty()
                elapsed_box = st.empty()
                eta_box = st.empty()

                start_file = time.time()

                cmd = [
                    sys.executable,
                    "app/pipelines/transcription_pipeline/transcriber_engine.py",
                    str(audio),
                    "--device", "cuda",
                    "--model", "large-v3",
                    "--compute_type", "float16",
                    "--batch_size", "8",
                    "--beam_size", "2",
                    "--vad",
                    "--local_only"
                ]

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                last_update = 0

                while True:

                    line = process.stdout.readline() if process.stdout else ""

                    if not line and process.poll() is not None:
                        break

                    if line:

                        match = progress_pattern.search(line)

                        if match:

                            percent = float(match.group(1))

                            progress_bar.progress(int(percent))

                            elapsed = time.time() - start_file

                            eta = ((elapsed / percent) * (100 - percent)) if percent > 0 else 0

                            now = time.time()

                            if now - last_update > 0.5:

                                percent_box.markdown(f"Progress: **{percent:.2f}%**")

                                elapsed_box.markdown(f"Elapsed: **{int(elapsed)} sec**")

                                eta_box.markdown(f"ETA: **{int(eta)} sec**")

                                last_update = now

                process.wait()

                transcription_time = time.time() - start_file

                stats.append({
                    "file": audio.name,
                    "time": transcription_time
                })

            batch_time = time.time() - batch_start

            st.subheader("Batch Statistics")

            st.write(f"Files processed: **{len(stats)}**")
            st.write(f"Total batch time: **{batch_time:.2f} sec**")


# ==================================================
# TAB 3 — TRANSCRIPTS
# ==================================================

with tab3:

    st.header("Transcripts")

    transcripts = sorted(TRANSCRIPT_DIR.glob("*.txt"))

    if transcripts:

        zip_buffer = io.BytesIO()

        with zipfile.ZipFile(zip_buffer, "w") as zip_file:

            for file in transcripts:
                zip_file.write(file, arcname=file.name)

        zip_buffer.seek(0)

        st.download_button(
            label="Download All Transcripts (ZIP)",
            data=zip_buffer,
            file_name="transcripts.zip",
            mime="application/zip"
        )

    for t in transcripts:

        with st.expander(f"{t.name}"):

            with open(t, "r", encoding="utf-8") as f:
                text = f.read()

            st.text_area(
                label="Transcript",
                value=text,
                height=300,
                label_visibility="collapsed"
            )