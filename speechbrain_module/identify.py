import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    run_opts={"device": "cpu"}
)

def load_audio(path):
    signal, fs = torchaudio.load(path)

    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)

    return signal


def extract_embedding(signal):
    return classifier.encode_batch(signal).squeeze().flatten()


def identify_speaker(input_path, speaker_db):
    input_signal = load_audio(input_path)
    input_emb = extract_embedding(input_signal)

    best_score = -1
    best_speaker = "unknown"

    for name, path in speaker_db.items():
        ref_signal = load_audio(path)
        ref_emb = extract_embedding(ref_signal)

        score = torch.cosine_similarity(input_emb, ref_emb, dim=0).item()

        if score > best_score:
            best_score = score
            best_speaker = name

    return best_speaker, best_score