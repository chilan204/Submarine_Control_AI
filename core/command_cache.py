import os
import threading
import time

import requests


COMMAND_CACHE = []
cache_lock = threading.Lock()
_CACHE_RETRY_INTERVAL = 5


def _internal_headers():
    token = os.getenv("AI_INTERNAL_TOKEN", "").strip()
    return {"X-AI-Internal-Token": token} if token else {}


def load_command_db():
    command_api = os.getenv("COMMAND_API_URL")
    if not command_api:
        raise RuntimeError("COMMAND_API_URL is not configured")

    response = requests.get(command_api, headers=_internal_headers(), timeout=5)
    response.raise_for_status()
    data = response.json().get("data", [])
    if not isinstance(data, list):
        raise ValueError("Command API returned invalid data")

    commands = []
    for item in data:
        if not isinstance(item, dict):
            raise ValueError("Command API returned an invalid command record")

        keyword = str(item.get("keyword") or "").strip().lower()
        action = str(item.get("action") or "").strip().upper()
        if not keyword or not action:
            raise ValueError("Every command must have a non-empty keyword and action")

        direction = item.get("direction")
        commands.append({
            "keyword": keyword,
            "action": action,
            "direction": str(direction).strip().upper() if direction else None,
            "hasValue": bool(item.get("hasValue", False)),
        })

    return commands


def _replace_cache(data):
    with cache_lock:
        COMMAND_CACHE.clear()
        COMMAND_CACHE.extend(data)


def initialize_command_cache():
    attempt = 1
    while True:
        print(f"[COMMAND CACHE] Retry {attempt}...")
        try:
            data = load_command_db()
            _replace_cache(data)
            print(f"[COMMAND CACHE] Ready ({len(data)})")
            return
        except Exception as e:
            print("[COMMAND DB ERROR]", e)
            time.sleep(_CACHE_RETRY_INTERVAL)
            attempt += 1


def reload_command_cache():
    # Fetch and validate first. On failure the last-known-good cache survives.
    data = load_command_db()
    _replace_cache(data)
    print(f"[COMMAND CACHE] Reloaded ({len(data)})")


def get_command_cache():
    with cache_lock:
        return [command.copy() for command in COMMAND_CACHE]
