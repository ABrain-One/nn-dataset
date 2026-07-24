"""
Minimal, robust HuggingFace model downloader.

We deliberately do NOT rely on `transformers.from_pretrained()` to *download*
models. Newer transformers versions can crash during the internal
download/resolution step with:

    AttributeError: 'NoneType' object has no attribute 'endswith'

(which happens inside modeling_utils when `checkpoint_files` ends up None).

Instead we use `huggingface_hub.snapshot_download`, which ONLY transfers files
(robust, no model-loading logic). After that, callers load with
`local_files_only=True`, which reads straight from the cache and never hits the
buggy download path.

This keeps the professor's `./train.sh` fully automatic: the first run fetches
missing models, subsequent runs are instant (cache hit).
"""
import os
import time
import tempfile

# Disable Xet before importing huggingface_hub because it caused
# xet-read-token 404 errors on clean-machine downloads.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

# Small, non-complex retry: just a few attempts for transient network errors.
# No FileLock, no exponential backoff, no rate-limit parsing.
_MAX_RETRIES = 3
_RETRY_SLEEP = 5  # seconds

def ensure_hf_model(repo_id: str) -> str:
    """
    Ensure an HF model/tokenizer repo is available locally.

    - If `repo_id` is an existing local path, it is returned as-is (no download).
    - If already ensured in this process, returns instantly without terminal spam.
    - Otherwise `snapshot_download` fetches it into the HF cache
      (returns instantly if already cached).
    - A tiny retry loop handles transient network errors.

    Returns the local directory/path that can be passed to
    `from_pretrained(..., local_files_only=True)`.
    """
    # Already a local path (e.g. a custom tokenizer dir) -> nothing to download.
    if os.path.exists(repo_id):
        return repo_id

    # Use a temporary file to share state across PyTorch worker processes!
    # Environment variables don't pass from child to parent, causing double prints.
    safe_name = repo_id.replace('/', '_').replace('-', '_').upper()
    flag_file = os.path.join(tempfile.gettempdir(), f"hf_ensured_{safe_name}.flag")
    if os.path.exists(flag_file):
        try:
            # We must return the actual absolute path to the snapshot cache, NOT the repo_id.
            # local_files_only=True instantly returns the cached path without hitting the network.
            return snapshot_download(repo_id=repo_id, local_files_only=True)
        except Exception:
            pass  # If it fails (cache deleted but flag remains), fall through to download

    print(f"[HF] Ensuring model available locally: {repo_id}")
    last_err: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            res = snapshot_download(
                repo_id=repo_id,
                ignore_patterns=[
                    # Skip non-weight binary formats that are never needed for inference
                    "*.msgpack", "*.h5", "*.ot", "*.tflite", "*.onnx", "*.pb",
                ]
            )
            # Mark as ensured so other processes/workers don't print
            with open(flag_file, 'w') as f:
                f.write('1')
            return res
        except Exception as err:  # transient network / rate-limit errors
            last_err = err
            print(
                f"[HF] Download attempt {attempt}/{_MAX_RETRIES} for '{repo_id}' "
                f"failed: {err}"
            )
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_SLEEP)

    raise RuntimeError(
        f"[HF] Failed to download '{repo_id}' after {_MAX_RETRIES} attempts: {last_err}"
    )
