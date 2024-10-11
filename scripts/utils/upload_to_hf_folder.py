import huggingface_hub
import os
import argparse
from tqdm import tqdm
import time
import hashlib


def get_repo_files(api, repo_id):
    repo_files = {}
    for file_info in api.list_repo_files(repo_id=repo_id, repo_type="dataset"):
        repo_files[file_info] = file_info
    print(f"Found {len(repo_files)} files in the repository")
    return repo_files

def upload_file_with_progress(api, item_path, item, repo_id):
    file_size = os.path.getsize(item_path)
    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {item}") as pbar:
        def update_progress(num_bytes):
            pbar.update(num_bytes)

        api.upload_file(
            path_or_fileobj=item_path,
            path_in_repo=item,
            repo_id=repo_id,
            repo_type="dataset",
        )

def count_files_in_folder(folder_path):
    total_files = 0
    for root, dirs, files in os.walk(folder_path):
        total_files += len(files)
    return total_files

def upload_folder_with_progress(api, item_path, item, repo_id, existing_files):
    total_files = count_files_in_folder(item_path)
    files_to_upload = []
    any_file_exists = False

    # First, walk through the folder and check if any items are in existing_files
    for root, _, files in os.walk(item_path):
        for file in files:
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, item_path)
            repo_path = os.path.join(item, relative_path).replace("\\", "/")
            
            if repo_path in existing_files:
                any_file_exists = True
                print(f"File already exists in repo: {repo_path}")
            else:
                files_to_upload.append((file_path, repo_path))

    # If no files exist in the repo, upload the entire folder at once
    if not any_file_exists:
        print(f"No existing files found. Uploading entire folder {item}")
        api.upload_folder(
            folder_path=item_path,
            path_in_repo=item,
            repo_id=repo_id,
            repo_type="dataset",
            multi_commits=True,
            multi_commits_verbose=True,
        )
        print(f"Uploaded {total_files} files in folder {item}")
    else:
        # If some files exist, upload individually
        print(f"Some files already exist. Uploading {len(files_to_upload)} new files individually.")
        with tqdm(total=len(files_to_upload), desc=f"Uploading new files in {item}") as pbar:
            for file_path, repo_path in files_to_upload:
                try:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                        repo_type="dataset",
                    )
                    pbar.update(1)
                except ValueError as e:
                    continue

        print(f"Uploaded {len(files_to_upload)} new files in folder {item}")


def upload_individual_items(args):
    print(f"Uploading items from {args.folder} to {args.repo}")
    api = huggingface_hub.HfApi()

    try:
        print(f"Creating a new repo {args.repo}")
        api.create_repo(
            args.repo,
            repo_type="dataset",
            exist_ok=True,
            private=True
        )
    except Exception as e:
        pass

    print("Fetching existing files in the repository...")
    existing_files = get_repo_files(api, args.repo)

    items = os.listdir(args.folder)
    for item in tqdm(items, desc="Processing items"):
        item_path = os.path.join(args.folder, item)
        
        if os.path.isfile(item_path):
            if item in existing_files:
                print(f"Skipping file: {item}")
                continue

            print(f"Uploading file: {item}")
            try:
                upload_file_with_progress(api, item_path, item, args.repo)
            except Exception as e:
                print(f"Error uploading {item}: {e}")
                print("Retrying after 30 seconds...")
                time.sleep(30)
                upload_file_with_progress(api, item_path, item, args.repo)
        
        elif os.path.isdir(item_path):
            print(f"Processing folder: {item}")
            try:
                upload_folder_with_progress(api, item_path, item, args.repo, existing_files)
            except Exception as e:
                print(f"Error processing folder {item}: {e}")
                print("Retrying after 30 seconds...")
                time.sleep(30)
                upload_folder_with_progress(api, item_path, item, args.repo, existing_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload individual items from a folder to Hugging Face Hub")
    parser.add_argument("-f", "--folder", type=str, help="The folder containing items to upload", required=True)
    parser.add_argument("-r", "--repo", type=str, help="The repo to upload to", required=True)
    parser.add_argument("--skip_create", action="store_true", help="Skip creating the repository")
    args = parser.parse_args()
    upload_individual_items(args)

# Example usage:
# python upload_to_hf_individual.py -f /path/to/your/folder -r your-username/your-repo-name