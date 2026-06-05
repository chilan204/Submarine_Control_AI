import whisper

model = whisper.load_model("base")

def transcribe_audio(file_path, language=None):
    audio = whisper.load_audio(file_path)
    audio = whisper.pad_or_trim(audio)
    
    if language in ["vi", "en"]:
        detected_lang = language
    else:
        try:
            mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)
        except TypeError:
            mel = whisper.log_mel_spectrogram(audio).to(model.device)
            
        _, probs = model.detect_language(mel)
        
        allowed_langs = {"vi": probs.get("vi", 0.0), "en": probs.get("en", 0.0)}
        detected_lang = max(allowed_langs, key=allowed_langs.get)
    
    result = model.transcribe(file_path, language=detected_lang)
    return result["text"]