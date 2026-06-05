import torchaudio
from speechbrain.pretrained import SpeakerRecognition

verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="pretrained_models/spkrec",
    run_opts={"device": "cpu"}
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