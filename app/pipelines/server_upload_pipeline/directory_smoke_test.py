from pprint import pprint

from app.config.server_upload_config import (
    get_server_upload_config,
    mask_server_upload_config,
)
from app.pipelines.server_upload_pipeline.server_client import SFTPServerClient


def main() -> None:
    config = get_server_upload_config()

    print("Loaded server config:")
    pprint(mask_server_upload_config(config))
    print("-" * 60)

    with SFTPServerClient() as client:
        snapshot = client.list_root()

    print("Connected successfully.")
    print(f"Remote path: {snapshot['path']}")
    print(f"Folders: {snapshot['folder_count']}")
    print(f"Files: {snapshot['file_count']}")
    print(f"Videos: {snapshot['video_count']}")
    print("-" * 60)

    print("Folders:")
    for folder in snapshot["folders"][:20]:
        print(f"  [DIR]  {folder['name']} -> {folder['path']}")

    print("-" * 60)
    print("Files:")
    for file in snapshot["files"][:20]:
        marker = "VIDEO" if file["is_video"] else "FILE "
        print(
            f"  [{marker}] {file['name']} | "
            f"size={file['size_bytes']} | modified={file['modified']}"
        )


if __name__ == "__main__":
    main()