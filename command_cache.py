import os
import time
import threading
import requests

COMMAND_CACHE = []
cache_lock = threading.Lock()

CACHE_RETRY_INTERVAL = 5


def load_command_db():
    try:
        command_api = os.getenv("COMMAND_API_URL")

        if not command_api:
            raise RuntimeError("COMMAND_API_URL is not configured")

        response = requests.get(command_api, timeout=5)
        response.raise_for_status()

        data = response.json().get("data", [])

        commands = []

        for item in data:
            commands.append({
                "keyword": item.get("keyword", "").lower(),
                "action": item.get("action"),
                "direction": item.get("direction"),
                "hasValue": item.get("hasValue", True)
            })

        return commands

    except Exception as e:
        print("[COMMAND DB ERROR]", e)
        return []


def initialize_command_cache():
    global COMMAND_CACHE

    attempt = 1

    while True:
        print(f"[COMMAND CACHE] Retry {attempt}...")

        data = load_command_db()

        if data:
            with cache_lock:
                COMMAND_CACHE.clear()
                COMMAND_CACHE.extend(data)

            print(f"[COMMAND CACHE] Ready ({len(data)})")
            break

        time.sleep(CACHE_RETRY_INTERVAL)
        attempt += 1


def reload_command_cache():
    global COMMAND_CACHE

    data = load_command_db()

    with cache_lock:
        COMMAND_CACHE.clear()
        COMMAND_CACHE.extend(data)

    print(f"[COMMAND CACHE] Reloaded ({len(COMMAND_CACHE)})")


def get_command_cache():
    with cache_lock:
        return COMMAND_CACHE.copy()