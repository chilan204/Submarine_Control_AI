from flask import request, jsonify
import os
import uuid

from core.speaker_cache import (
    get_speaker_cache,
    reload_speaker_cache
)

from core.command_cache import reload_command_cache

from core.parser import parse_command
from modules.speechbrain.verify import verify_speaker
from modules.speechbrain.identify import identify_speaker
from modules.whisper.transcribe import transcribe_audio

UPLOAD_FOLDER = "data/uploads"


def register_routes(app):

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # -------------------------
    # reload speaker cache
    # -------------------------
    @app.route("/reload-speaker-cache", methods=["POST"])
    def reload_speaker_cache_endpoint():
        try:
            reload_speaker_cache()
            return jsonify({
                "message": "Speaker cache reloaded"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------
    # reload command cache
    # -------------------------
    @app.route("/reload-command-cache", methods=["POST"])
    def reload_command_cache_endpoint():
        try:
            reload_command_cache()
            return jsonify({
                "message": "Command cache reloaded"
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # -------------------------
    # voice command
    # -------------------------
    @app.route("/voice-command", methods=["POST"])
    def voice_command():

        file_path = None

        try:
            speaker_db = get_speaker_cache()

            if not speaker_db:
                return jsonify({"error": "Speaker DB empty"}), 500

            if "file" not in request.files:
                return jsonify({"error": "No file"}), 400

            file = request.files["file"]

            filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)

            file.save(file_path)

            speaker_id, score = identify_speaker(file_path, speaker_db)
            speaker_id = str(speaker_id)

            ref_path = speaker_db.get(speaker_id)

            if not ref_path:
                return jsonify({"error": "Speaker not found"}), 400

            verify_score, verify_result = verify_speaker(file_path, ref_path)

            if not verify_result:
                return jsonify({"status": "denied"}), 403

            text = transcribe_audio(file_path, None)
            command = parse_command(text)

            return jsonify({
                "text": text,
                "speaker_id": speaker_id,
                "command": command,
                "speaker_score": float(score),
                "verification_score": float(verify_score)
            })

        finally:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass