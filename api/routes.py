from flask import request, jsonify
import os
import uuid

from core.speaker_cache import (
    get_speaker_cache,
    reload_speaker_cache
)

from core.command_cache import reload_command_cache

from core.parser import parse_command
from modules.speechbrain.identify import identify_speaker, extract_embedding, load_audio
from modules.whisper.transcribe import transcribe_audio

UPLOAD_FOLDER = "data/uploads"


def register_routes(app):

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    # -------------------------
    # extract embedding
    # -------------------------
    @app.route("/extract-embedding", methods=["POST"])
    def extract_embedding_route():
        try:
            if "file" not in request.files:
                return jsonify({"error": "No file"}), 400
            file = request.files["file"]
            filename = f"temp_{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            signal = load_audio(file_path)
            emb = extract_embedding(signal)
            emb_list = emb.tolist()

            if os.path.exists(file_path):
                os.remove(file_path)

            return jsonify({
                "embedding": emb_list
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

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

            import time
            start_time = time.time()

            file = request.files["file"]
            filename = f"{uuid.uuid4()}_{file.filename}"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            
            save_time = (time.time() - start_time) * 1000

            # 1. Speaker Identification
            id_start = time.time()
            speaker_id, score = identify_speaker(file_path, speaker_db)
            speaker_id = str(speaker_id)
            id_time = (time.time() - id_start) * 1000

            ref_info = speaker_db.get(speaker_id)
            if not ref_info or not ref_info.get("filePath"):
                return jsonify({"error": "Speaker not found"}), 400

            # 2. Transcription
            tx_start = time.time()
            language = request.form.get("language")
            text = transcribe_audio(file_path, language)
            tx_time = (time.time() - tx_start) * 1000

            # 3. Parsing
            parse_start = time.time()
            command = parse_command(text)
            parse_time = (time.time() - parse_start) * 1000

            total_time = (time.time() - start_time) * 1000
            print(f"\n=== PROFILING /voice-command ===")
            print(f"File Save:        {save_time:.2f} ms")
            print(f"Identify Speaker: {id_time:.2f} ms")
            print(f"Transcribe:       {tx_time:.2f} ms")
            print(f"Parse Command:    {parse_time:.2f} ms")
            print(f"Total AI Process: {total_time:.2f} ms")
            print(f"================================\n")

            return jsonify({
                "text": text,
                "speaker_id": speaker_id,
                "command": command,
                "speaker_score": float(score),
                "verification_score": float(score)
            })

        finally:
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                pass