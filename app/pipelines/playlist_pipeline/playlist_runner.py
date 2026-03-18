from app.pipelines.playlist_pipeline.playlist_downloader import download_playlist
from app.pipelines.media_pipeline.ingestion_runner import run_ingestion

def run_playlist_download(url: str, quality: str = "720p", progress_callback=None):
    result = download_playlist(
        url=url,
        quality=quality,
        progress_callback=progress_callback,
    )
    paths = result["paths"]
    standardized_files = run_ingestion(paths.videos, paths.audio)
    return {
        "result": result,
        "standardized_files": standardized_files,
    }