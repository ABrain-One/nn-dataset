import hashlib
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from filelock import FileLock, Timeout
from huggingface_hub import HfApi, hf_hub_download

# =============================================================================
# Configuration
# =============================================================================

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    # Fallback for testing
    HF_TOKEN = None
    print('⚠️ Warning: No HF_TOKEN found. Please set environment variable.')

# Retry settings
_DOWNLOAD_MAX_RETRIES = int(os.environ.get("HF_DOWNLOAD_MAX_RETRIES", "10"))
_DOWNLOAD_RETRY_SLEEP = int(os.environ.get("HF_DOWNLOAD_RETRY_SLEEP", "30"))
_DOWNLOAD_MAX_RETRY_SLEEP = int(os.environ.get("HF_DOWNLOAD_MAX_RETRY_SLEEP", "600"))

_LOCK_TIMEOUT = int(os.environ.get("HF_DOWNLOAD_LOCK_TIMEOUT", "3600"))

_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
_LOCK_DIR = Path(os.environ.get("HF_LOCK_DIR", _HF_HOME / "download-locks"))
_LOCK_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Internal helpers
# =============================================================================

def _retry_sleep_time(attempt: int) -> int:
    return min(
        _DOWNLOAD_RETRY_SLEEP * (2 ** max(attempt - 1, 0)),
        _DOWNLOAD_MAX_RETRY_SLEEP,
    )


def _is_rate_limit(error: Exception) -> bool:
    status_code = getattr(getattr(error, "response", None), "status_code", None)
    if status_code == 429:
        return True
    message = str(error).lower()
    return "429" in message or "too many requests" in message or "rate limit" in message or "ratelimit" in message


def _is_local_cache_miss(error: Exception) -> bool:
    if isinstance(error, (FileNotFoundError, OSError)):
        message = str(error).lower()
        cache_miss_markers = (
            "not found", "cannot find", "couldn't find", "could not find",
            "local cache", "local_files_only", "offline mode",
            "does not appear to have a file", "configuration file", "no such file",
        )
        return any(marker in message for marker in cache_miss_markers)
    return False


def _safe_resource_name(value: str) -> str:
    readable = value.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"{readable[:80]}_{digest}"


def _get_lock_path(resource_name: str) -> Path:
    safe_name = _safe_resource_name(resource_name)
    return _LOCK_DIR / f"{safe_name}.lock"


def _add_token(kwargs: dict[str, Any]) -> dict[str, Any]:
    updated = dict(kwargs)
    if HF_TOKEN and "token" not in updated:
        updated["token"] = HF_TOKEN
    return updated


def _offline_mode_enabled() -> bool:
    values = (os.environ.get("HF_HUB_OFFLINE", ""), os.environ.get("TRANSFORMERS_OFFLINE", ""))
    return any(str(value).strip().lower() in {"1", "true", "yes", "on"} for value in values)


# =============================================================================
# Hub file download
# =============================================================================

def hf_hub_download_with_retry(repo_id: str, filename: str, local_dir: str, **kwargs: Any) -> str:
    os.makedirs(local_dir, exist_ok=True)
    filename = Path(filename).name
    online_kwargs = _add_token(kwargs)

    local_kwargs = dict(online_kwargs)
    local_kwargs["local_files_only"] = True

    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, **local_kwargs)
    except Exception as error:
        if not _is_local_cache_miss(error):
            raise

    if _offline_mode_enabled():
        raise FileNotFoundError(f"Offline mode is enabled: repo='{repo_id}', file='{filename}'.")

    resource_name = f"file_{repo_id}_{filename}"
    lock_path = _get_lock_path(resource_name)

    try:
        with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT):
            try:
                return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, **local_kwargs)
            except Exception as error:
                if not _is_local_cache_miss(error):
                    raise

            for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 2):
                try:
                    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir, **online_kwargs)
                except Exception as error:
                    retries_used = attempt - 1
                    if not _is_rate_limit(error) or retries_used >= _DOWNLOAD_MAX_RETRIES:
                        raise

                    sleep_time = _retry_sleep_time(attempt)
                    print(f"\n[WARN] Download rate limit (429) hit! "
                          f"Sleeping {sleep_time}s "
                          f"(retry {attempt}/{_DOWNLOAD_MAX_RETRIES})...")
                    time.sleep(sleep_time)

    except Timeout as error:
        raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from error

    raise RuntimeError(f"Unexpected failure for repo='{repo_id}', file='{filename}'.")


def download(repo_id: str, filename: str, local_dir: str) -> str:
    local_path = hf_hub_download_with_retry(repo_id=repo_id, filename=filename, local_dir=local_dir)
    metadata_cache = Path(local_dir) / ".cache"
    if metadata_cache.exists():
        shutil.rmtree(metadata_cache)
    return local_path


# =============================================================================
# Transformers from_pretrained
# =============================================================================

def from_pretrained_with_retry(loader: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    if not args:
        raise ValueError("Requires model ID or local path as first argument.")

    model_id_or_path = str(args[0])
    online_kwargs = _add_token(kwargs)

    local_kwargs = dict(online_kwargs)
    local_kwargs["local_files_only"] = True

    try:
        return loader(*args, **local_kwargs)
    except Exception as error:
        if not _is_local_cache_miss(error):
            raise

    if _offline_mode_enabled():
        raise FileNotFoundError(f"Model '{model_id_or_path}' not available offline.")

    loader_name = getattr(loader, "__qualname__", repr(loader))
    # Loader agnostic lock to prevent tokenizer and model fetching same repo concurrently online
    resource_name = f"model_{model_id_or_path}"
    lock_path = _get_lock_path(resource_name)

    try:
        with FileLock(str(lock_path), timeout=_LOCK_TIMEOUT):
            try:
                return loader(*args, **local_kwargs)
            except Exception as error:
                if not _is_local_cache_miss(error):
                    raise

            for attempt in range(1, _DOWNLOAD_MAX_RETRIES + 2):
                try:
                    return loader(*args, **online_kwargs)
                except Exception as error:
                    retries_used = attempt - 1
                    if not _is_rate_limit(error) or retries_used >= _DOWNLOAD_MAX_RETRIES:
                        raise

                    sleep_time = _retry_sleep_time(attempt)
                    print(f"\n[WARN] from_pretrained rate limit (429) hit! "
                          f"Loader: {loader_name}\n"
                          f"Sleeping {sleep_time}s "
                          f"(retry {attempt}/{_DOWNLOAD_MAX_RETRIES})...")
                    time.sleep(sleep_time)

    except Timeout as error:
        raise TimeoutError(f"Timed out waiting for lock: {lock_path}") from error

    raise RuntimeError(f"Unexpected from_pretrained failure for '{model_id_or_path}'.")


# =============================================================================
# Upload
# =============================================================================

def upload_file(repo_id, local_file, path_in_repo, remove: bool = False, hf_token=None):
    api = HfApi(token=hf_token or HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type='model', exist_ok=True)

    if local_file and os.path.exists(local_file):
        api.upload_file(
            path_or_fileobj=str(local_file),
            path_in_repo=str(Path(path_in_repo).name),
            repo_id=repo_id,
            repo_type='model')
        if remove: Path(local_file).unlink()
