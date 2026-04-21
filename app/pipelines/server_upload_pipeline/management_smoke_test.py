from pprint import pprint

from app.pipelines.workflow_pipeline.server_upload_workflows import (
    browse_remote_root,
    search_remote_videos,
)

def main() -> None:
    root = browse_remote_root()
    pprint(
        {
            "current_path": root["current_path"],
            "folder_count": root["folder_count"],
            "file_count": root["file_count"],
            "video_count": root["video_count"],
        }
    )

    search_result = search_remote_videos(root["current_path"], "mp4")
    pprint(
        {
            "query": search_result["query"],
            "match_count": search_result["match_count"],
        }
    )

if __name__ == "__main__":
    main()