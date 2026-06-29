import os
import time
import threading
import requests
import json

SPEAKER_CACHE = {}
cache_lock = threading.Lock()

JAVA_API = os.getenv("JAVA_API_URL")
BASE_VOICE_DIR = os.getenv("BASE_VOICE_DIR")
CACHE_RETRY_INTERVAL = 5


def resolve_path(relative_path: str):
    return os.path.join(BASE_VOICE_DIR, relative_path)


def load_speaker_db():
    try:
        response = requests.get(JAVA_API, timeout=5)

        if response.status_code != 200:
            return {}

        data = response.json().get("data", [])

        speaker_db = {}

        for item in data:
            user_id = str(item.get("userId"))
            file_path = item.get("filePath")
            emb_str = item.get("embedding")

            embedding_list = None
            if emb_str:
                try:
                    embedding_list = json.loads(emb_str)
                except Exception:
                    pass

            if user_id and file_path:
                speaker_db[user_id] = {
                    "filePath": resolve_path(file_path),
                    "embedding": embedding_list
                }

        return speaker_db

    except Exception as e:
        print("[SPEAKER DB ERROR]", e)
        return {}


def reload_speaker_cache():
    global SPEAKER_CACHE

    data = load_speaker_db()

    with cache_lock:
        SPEAKER_CACHE.clear()
        SPEAKER_CACHE.update(data)

    print(f"[SPEAKER CACHE] Reloaded ({len(data)})")


def initialize_speaker_cache():
    global SPEAKER_CACHE

    attempt = 1

    while True:
        print(f"[SPEAKER CACHE] Retry {attempt}...")

        data = load_speaker_db()

        if data:
            with cache_lock:
                SPEAKER_CACHE = data

            print(f"[SPEAKER CACHE] Ready ({len(data)})")
            break

        time.sleep(CACHE_RETRY_INTERVAL)
        attempt += 1


def get_speaker_cache():
    with cache_lock:
        return SPEAKER_CACHE.copy()
