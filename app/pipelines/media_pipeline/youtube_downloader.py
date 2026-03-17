from pathlib import Path
import yt_dlp

from app.config.paths import UPLOAD_AUDIO_DIR


def download_youtube_audio(url, progress_callback=None):

    output_template = str(UPLOAD_AUDIO_DIR / "%(title)s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "progress_hooks": [progress_callback] if progress_callback else [],
        "quiet": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(url, download=True)

        filename = ydl.prepare_filename(info)

    return Path(filename)