import os
import torch
import torchaudio
from speechbrain.pretrained import SpeakerRecognition

# Get the data directory path relative to this file
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../data/pretrained_models")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SPEECHBRAIN VERIFY] Using device: {DEVICE}")

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir=os.path.join(MODELS_DIR, "spkrec"),
    run_opts={"device": DEVICE}
)

def load_audio(path):
    signal, fs = torchaudio.load(path)

    if fs != 16000:
        signal = torchaudio.transforms.Resample(fs, 16000)(signal)

    return signal


def verify_speaker(input_path, ref_path):
    signal1 = load_audio(input_path)
    signal2 = load_audio(ref_path)

    prediction, score = verification.verify_batch(signal1, signal2)

    score_val = score.flatten()[0].item()
    pred_val = prediction.flatten()[0].item()

    return score_val, pred_val
