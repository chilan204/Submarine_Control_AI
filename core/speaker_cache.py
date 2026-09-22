import json
import os
import threading
import time
from pathlib import Path

import requests


SPEAKER_CACHE = {}
cache_lock = threading.Lock()
_CACHE_RETRY_INTERVAL = 5


def _internal_headers():
    token = os.getenv("AI_INTERNAL_TOKEN", "").strip()
    return {"X-AI-Internal-Token": token} if token else {}


def resolve_path(relative_path: str):
    base_voice_dir = os.getenv("BASE_VOICE_DIR")
    if not base_voice_dir:
        raise RuntimeError("BASE_VOICE_DIR is not configured")

    base = Path(base_voice_dir).resolve()
    path = (base / relative_path).resolve()
    if path != base and base not in path.parents:
        raise ValueError("Voice sample path escapes BASE_VOICE_DIR")
    return str(path)


def load_speaker_db():
    java_api = os.getenv("JAVA_API_URL")
    if not java_api:
        raise RuntimeError("JAVA_API_URL is not configured")

    response = requests.get(java_api, headers=_internal_headers(), timeout=5)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not isinstance(data, list):
        raise ValueError("Speaker API returned invalid data")

    speaker_db = {}
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Speaker API returned an invalid speaker record")

        user_id = item.get("userId")
        file_path = item.get("filePath")
        embedding = item.get("embedding")
        if user_id is None or not file_path:
            continue

        embedding_list = None
        if embedding:
            try:
                embedding_list = json.loads(embedding)
            except (TypeError, json.JSONDecodeError) as e:
                raise ValueError(f"Invalid embedding for user {user_id}") from e

        speaker_db[str(user_id)] = {
            "filePath": resolve_path(str(file_path)),
            "embedding": embedding_list,
        }

    return speaker_db


def _replace_cache(data):
    with cache_lock:
        SPEAKER_CACHE.clear()
        SPEAKER_CACHE.update(data)


def initialize_speaker_cache():
    attempt = 1
    while True:
        print(f"[SPEAKER CACHE] Retry {attempt}...")
        try:
            data = load_speaker_db()
            _replace_cache(data)
            print(f"[SPEAKER CACHE] Ready ({len(data)})")
            return
        except Exception as e:
            print("[SPEAKER DB ERROR]", e)
            time.sleep(_CACHE_RETRY_INTERVAL)
            attempt += 1


def reload_speaker_cache():
    # Fetch and validate first. On failure the last-known-good cache survives.
    data = load_speaker_db()
    _replace_cache(data)
    print(f"[SPEAKER CACHE] Reloaded ({len(data)})")


def get_speaker_cache():
    with cache_lock:
        return {
            speaker_id: info.copy()
            for speaker_id, info in SPEAKER_CACHE.items()
        }
