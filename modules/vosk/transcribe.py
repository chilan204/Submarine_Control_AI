import json
import os
import threading
import wave
from pathlib import Path

from vosk import KaldiRecognizer, Model, SetLogLevel

from core.command_cache import get_command_cache


SetLogLevel(-1)

_DEFAULT_MODEL_DIR = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "pretrained_models"
    / "vosk-model-small-vn-0.4"
)
_MODELS = {}
_MODEL_LOCK = threading.Lock()


def _model_path(language):
    language_key = (language or "vi").upper()
    configured_path = (
        os.getenv(f"VOSK_MODEL_PATH_{language_key}")
        or os.getenv("VOSK_MODEL_PATH")
    )
    return Path(configured_path).expanduser() if configured_path else _DEFAULT_MODEL_DIR


def _get_model(language):
    path = _model_path(language).resolve()
    cache_key = str(path)

    with _MODEL_LOCK:
        model = _MODELS.get(cache_key)
        if model is not None:
            return model

        if not path.is_dir():
            raise RuntimeError(
                f"Vosk model not found at '{path}'. "
                "Set VOSK_MODEL_PATH (or VOSK_MODEL_PATH_VI/VOSK_MODEL_PATH_EN) "
                "to an extracted Vosk model directory."
            )

        print(f"[VOSK] Loading model: {path}")
        model = Model(model_path=str(path))
        _MODELS[cache_key] = model
        print("[VOSK] Model ready")
        return model


def _command_grammar():
    phrases = []
    seen = set()

    for command in get_command_cache():
        phrase = str(command.get("keyword") or "").strip().lower()
        if phrase and phrase not in seen:
            seen.add(phrase)
            phrases.append(phrase)

    if not phrases:
        raise RuntimeError("Cannot recognize speech because the command cache is empty")

    # Explicitly model out-of-grammar speech instead of forcing it to the nearest
    # AUV command. parse_command() will reject it as UNKNOWN.
    phrases.append("[unk]")
    return json.dumps(phrases, ensure_ascii=False)


def _result_text(result):
    try:
        return json.loads(result).get("text", "").strip()
    except (TypeError, json.JSONDecodeError):
        return ""


def transcribe_audio(file_path, language=None):
    model = _get_model(language)
    grammar = _command_grammar()

    with wave.open(file_path, "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError("Vosk requires mono WAV audio")
        if audio.getsampwidth() != 2:
            raise ValueError("Vosk requires 16-bit PCM WAV audio")

        recognizer = KaldiRecognizer(model, audio.getframerate(), grammar)
        segments = []

        while True:
            chunk = audio.readframes(4000)
            if not chunk:
                break
            if recognizer.AcceptWaveform(chunk):
                text = _result_text(recognizer.Result())
                if text:
                    segments.append(text)

        final_text = _result_text(recognizer.FinalResult())
        if final_text:
            segments.append(final_text)

    return " ".join(segments).strip()
