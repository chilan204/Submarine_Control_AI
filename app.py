from flask import Flask
from dotenv import load_dotenv

load_dotenv()

from speaker_cache import initialize_speaker_cache
from command_cache import initialize_command_cache
from routes import register_routes

app = Flask(__name__)

# register routes
register_routes(app)

if __name__ == "__main__":

    print("VOICE AI SERVICE STARTING...")

    # init caches
    initialize_speaker_cache()
    initialize_command_cache()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )