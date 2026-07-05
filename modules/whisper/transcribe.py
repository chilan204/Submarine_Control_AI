from faster_whisper import WhisperModel

# Use CPU for Whisper to avoid external cuBLAS/cuDNN DLL dependency issues on Windows.
# "int8" quantization makes it run extremely fast on CPU (under 0.5 seconds for short commands).
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

print(f"[WHISPER] Using faster-whisper | device: {DEVICE} | compute: {COMPUTE_TYPE} | model: tiny")

model = WhisperModel("tiny", device=DEVICE, compute_type=COMPUTE_TYPE, cpu_threads=4)


def transcribe_audio(file_path, language=None):
    if language not in ["vi", "en"]:
        language = None

    segments, info = model.transcribe(
        file_path,
        language=language,
        beam_size=1,
        vad_filter=True,
    )

    text = " ".join(segment.text for segment in segments).strip()

    return text
