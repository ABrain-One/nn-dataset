import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, hf_hub_download

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    # Fallback for testing
    HF_TOKEN = ""
    print('⚠️ Warning: No HF_TOKEN found. Please set environment variable.')


def download(repo_id, filename, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=Path(filename).name,
        local_dir=local_dir)
    shutil.rmtree(local_dir / '.cache')
    return local_path


def upload_file(repo_id, local_file, path_in_repo, remove: bool = False):
    api = HfApi(token=HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)

    if local_file and os.path.exists(local_file):
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=str(Path(path_in_repo).name),
            repo_id=repo_id,
            repo_type='model')
        if remove: os.remove(local_file)
