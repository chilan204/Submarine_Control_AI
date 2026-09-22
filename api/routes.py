import hmac
import ipaddress
import os
import uuid
import wave
from functools import wraps

from flask import current_app, jsonify, request

from core.command_cache import reload_command_cache
from core.parser import parse_command
from core.speaker_cache import get_speaker_cache, reload_speaker_cache
from modules.speechbrain.identify import extract_embedding, identify_speaker, load_audio
from modules.speechbrain.verify import verify_speaker
from modules.vosk.transcribe import transcribe_audio


def _require_internal_access(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        expected_token = current_app.config.get("AI_INTERNAL_TOKEN")
        supplied_token = request.headers.get("X-AI-Internal-Token", "")

        if expected_token:
            if not hmac.compare_digest(expected_token, supplied_token):
                return jsonify({"error": "Forbidden"}), 403
        else:
            try:
                if not ipaddress.ip_address(request.remote_addr).is_loopback:
                    return jsonify({"error": "Forbidden"}), 403
            except ValueError:
                return jsonify({"error": "Forbidden"}), 403

        return view(*args, **kwargs)

    return wrapped


def _save_wav_upload():
    if "file" not in request.files:
        return None, (jsonify({"error": "No file"}), 400)

    file_path = os.path.join(current_app.config["UPLOAD_FOLDER"], f"{uuid.uuid4()}.wav")
    request.files["file"].save(file_path)
    return file_path, None


def _validate_wav(file_path):
    max_seconds = current_app.config["MAX_AUDIO_SECONDS"]
    with wave.open(file_path, "rb") as audio:
        if audio.getnchannels() != 1:
            raise ValueError("Audio must be mono")
        if audio.getsampwidth() != 2:
            raise ValueError("Audio must use 16-bit PCM")
        if audio.getframerate() != 16000:
            raise ValueError("Audio must use a 16000 Hz sample rate")

        duration = audio.getnframes() / audio.getframerate()
        if duration <= 0 or duration > max_seconds:
            raise ValueError(f"Audio duration must be between 0 and {max_seconds} seconds")


def _remove_upload(file_path):
    if not file_path:
        return
    try:
        os.remove(file_path)
    except FileNotFoundError:
        pass
    except OSError:
        current_app.logger.warning("Could not remove temporary upload %s", file_path)


def register_routes(app):
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

    @app.errorhandler(413)
    def upload_too_large(_error):
        return jsonify({"error": "Audio file is too large"}), 413

    @app.route("/extract-embedding", methods=["POST"])
    @_require_internal_access
    def extract_embedding_route():
        file_path = None
        try:
            file_path, error = _save_wav_upload()
            if error:
                return error

            _validate_wav(file_path)
            signal = load_audio(file_path)
            embedding = extract_embedding(signal)
            return jsonify({"embedding": embedding.tolist()})
        except (ValueError, wave.Error) as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            current_app.logger.exception("Embedding extraction failed")
            return jsonify({"error": "Embedding extraction failed"}), 500
        finally:
            _remove_upload(file_path)

    @app.route("/reload-speaker-cache", methods=["POST"])
    @_require_internal_access
    def reload_speaker_cache_endpoint():
        try:
            reload_speaker_cache()
            return jsonify({"message": "Speaker cache reloaded"})
        except Exception:
            current_app.logger.exception("Speaker cache reload failed")
            return jsonify({"error": "Speaker cache reload failed"}), 503

    @app.route("/reload-command-cache", methods=["POST"])
    @_require_internal_access
    def reload_command_cache_endpoint():
        try:
            reload_command_cache()
            return jsonify({"message": "Command cache reloaded"})
        except Exception:
            current_app.logger.exception("Command cache reload failed")
            return jsonify({"error": "Command cache reload failed"}), 503

    @app.route("/voice-command", methods=["POST"])
    @_require_internal_access
    def voice_command():
        file_path = None
        try:
            speaker_db = get_speaker_cache()
            if not speaker_db:
                return jsonify({"error": "Speaker DB empty"}), 503

            file_path, error = _save_wav_upload()
            if error:
                return error
            _validate_wav(file_path)

            import time
            start_time = time.perf_counter()

            id_start = time.perf_counter()
            speaker_id, identification_score = identify_speaker(file_path, speaker_db)
            speaker_id = str(speaker_id)
            id_time = (time.perf_counter() - id_start) * 1000

            reference = speaker_db.get(speaker_id)
            if not reference or not reference.get("filePath"):
                return jsonify({"error": "Speaker not found"}), 401

            verify_start = time.perf_counter()
            verification_score, verification_prediction = verify_speaker(
                file_path,
                reference["filePath"],
            )
            verify_time = (time.perf_counter() - verify_start) * 1000

            tx_start = time.perf_counter()
            language = request.form.get("language")
            text = transcribe_audio(file_path, language)
            tx_time = (time.perf_counter() - tx_start) * 1000

            parse_start = time.perf_counter()
            command = parse_command(text)
            parse_time = (time.perf_counter() - parse_start) * 1000

            total_time = (time.perf_counter() - start_time) * 1000
            current_app.logger.info(
                "voice-command timings ms: identify=%.2f verify=%.2f "
                "transcribe=%.2f parse=%.2f total=%.2f",
                id_time,
                verify_time,
                tx_time,
                parse_time,
                total_time,
            )

            return jsonify({
                "text": text,
                "speaker_id": speaker_id,
                "command": command,
                "speaker_score": float(identification_score),
                "verification_score": float(verification_score),
                "verified": bool(verification_prediction),
            })
        except (ValueError, wave.Error) as e:
            return jsonify({"error": str(e)}), 400
        except Exception:
            current_app.logger.exception("Voice command processing failed")
            return jsonify({"error": "Voice command processing failed"}), 500
        finally:
            _remove_upload(file_path)
