import os
import threading
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask

from api.routes import register_routes
from core.command_cache import initialize_command_cache
from core.speaker_cache import initialize_speaker_cache


_PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(_PROJECT_DIR / ".env")


def _start_cache_initializers():
    for name, initializer in (
        ("speaker-cache-init", initialize_speaker_cache),
        ("command-cache-init", initialize_command_cache),
    ):
        threading.Thread(
            target=initializer,
            name=name,
            daemon=True,
        ).start()


def create_app():
    application = Flask(__name__)
    application.config.update(
        MAX_CONTENT_LENGTH=int(os.getenv("MAX_AUDIO_BYTES", 5 * 1024 * 1024)),
        MAX_AUDIO_SECONDS=float(os.getenv("MAX_AUDIO_SECONDS", "15")),
        UPLOAD_FOLDER=str(_PROJECT_DIR / "data" / "uploads"),
        AI_INTERNAL_TOKEN=os.getenv("AI_INTERNAL_TOKEN", "").strip(),
    )

    register_routes(application)
    _start_cache_initializers()
    return application


# WSGI servers can import this object without skipping cache initialization.
app = create_app()


if __name__ == "__main__":
    print("VOICE AI SERVICE STARTING...")
    app.run(
        host=os.getenv("AI_HOST", "127.0.0.1"),
        port=int(os.getenv("AI_PORT", "5000")),
        debug=False,
    )
