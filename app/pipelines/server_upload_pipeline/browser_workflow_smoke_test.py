from pprint import pprint

from app.pipelines.workflow_pipeline.server_upload_workflows import (
    browse_remote_root,
    browse_remote_child,
    browse_remote_parent,
)


def main() -> None:
    root_snapshot = browse_remote_root()

    print("ROOT SNAPSHOT")
    print("-" * 60)
    print(f"Current Path: {root_snapshot['current_path']}")
    print(f"Parent Path: {root_snapshot['parent_path']}")
    print(f"Folders: {root_snapshot['folder_count']}")
    print(f"Files: {root_snapshot['file_count']}")
    print(f"Videos: {root_snapshot['video_count']}")
    print("Breadcrumbs:")
    pprint(root_snapshot["breadcrumbs"])

    if not root_snapshot["folders"]:
        print("\nNo folders found in remote root.")
        return

    first_folder = root_snapshot["folders"][0]
    first_folder_name = first_folder["name"]

    print("\n" + "=" * 60)
    print(f"OPENING FIRST CHILD FOLDER: {first_folder_name}")
    print("=" * 60)

    child_snapshot = browse_remote_child(
        current_path=root_snapshot["current_path"],
        child_folder_name=first_folder_name,
    )

    print(f"Current Path: {child_snapshot['current_path']}")
    print(f"Parent Path: {child_snapshot['parent_path']}")
    print(f"Folders: {child_snapshot['folder_count']}")
    print(f"Files: {child_snapshot['file_count']}")
    print(f"Videos: {child_snapshot['video_count']}")
    print("Breadcrumbs:")
    pprint(child_snapshot["breadcrumbs"])

    print("\n" + "=" * 60)
    print("GOING BACK TO PARENT")
    print("=" * 60)

    parent_snapshot = browse_remote_parent(child_snapshot["current_path"])

    print(f"Current Path: {parent_snapshot['current_path']}")
    print(f"Parent Path: {parent_snapshot['parent_path']}")
    print(f"Folders: {parent_snapshot['folder_count']}")
    print(f"Files: {parent_snapshot['file_count']}")
    print(f"Videos: {parent_snapshot['video_count']}")
    print("Breadcrumbs:")
    pprint(parent_snapshot["breadcrumbs"])


if __name__ == "__main__":
    main()