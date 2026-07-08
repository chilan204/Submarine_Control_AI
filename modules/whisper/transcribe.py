from faster_whisper import WhisperModel

# Use CPU for Whisper to avoid external cuBLAS/cuDNN DLL dependency issues on Windows.
# "int8" quantization makes it run extremely fast on CPU (under 0.5 seconds for short commands).
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print(f"[WHISPER] Using faster-whisper | device: {DEVICE} | compute: {COMPUTE_TYPE} | model: base")

model = WhisperModel("base", device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=4)


from core.command_cache import get_command_cache

def transcribe_audio(file_path, language=None):
    if language not in ["vi", "en"]:
        language = None

    # Dynamically generate prompt based on current cached command keywords
    initial_prompt = None
    try:
        commands = get_command_cache()
        keywords = [cmd["keyword"] for cmd in commands if cmd.get("keyword")]
        if keywords:
            initial_prompt = ", ".join(keywords)
    except Exception as e:
        print("[WHISPER PROMPT ERROR]", e)

    segments, info = model.transcribe(
        file_path,
        language=language,
        beam_size=1,
        vad_filter=True,
        initial_prompt=initial_prompt,
    )

    text = " ".join(segment.text for segment in segments).strip()

    return text
