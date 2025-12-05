from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    repo_id = "vishal-1344/sci"  # adjust if needed
    api = HfApi()

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, repo_type="model")

    # Upload entire project folder
    upload_folder(
        folder_path=".",
        repo_id=repo_id,
        repo_type="model",
        commit_message="Initial SCI framework push",
    )


if __name__ == "__main__":
    main()
