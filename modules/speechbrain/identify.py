import os
import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Get the data directory path relative to this file
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/pretrained_models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SPEECHBRAIN IDENTIFY] Using device: {DEVICE}")

classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(MODELS_DIR, "spkrec"),
    run_opts={"device": DEVICE}
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

    for name, info in speaker_db.items():
        ref_emb_list = info.get("embedding")
        if ref_emb_list:
            ref_emb = torch.tensor(ref_emb_list, device=DEVICE)
        else:
            ref_signal = load_audio(info["filePath"])
            ref_emb = extract_embedding(ref_signal)

        score = torch.cosine_similarity(input_emb, ref_emb, dim=0).item()

        if score > best_score:
            best_score = score
            best_speaker = name

    return best_speaker, best_score
