import os
from tqdm import tqdm
import torch
import torchaudio
from torch.nn.functional import mse_loss

from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram


def create_voceded_dataset(data_dir, output_dir):
    """
    Create a dataset of vocoded audio files from the given audio dir. Copy the same dir and subdirs to the output dir.
    :param data_dir:
    :param output_dir:
    :return:
    """
    # make the directory structure + tqdm
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in tqdm(os.walk(data_dir)):
        for dir in dirs:
            os.makedirs(os.path.join(output_dir, root, dir), exist_ok=True)
        for file in files:
            if file.endswith(".flac"):
                file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_dir, root, file)
                if not os.path.exists(output_file_path):
                    audio, sr = torchaudio.load(file_path)
                    mel, _ = mel_spectogram(audio=audio.squeeze(), sample_rate=22500, hop_length=256,
                                              win_length=None,
                                              n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                              min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                              compression=True)
                    mel = mel.to("cuda")
                    vocoded_audio = hifi_gan.decode_batch(mel).squeeze(1)
                    vocoded_audio = vocoded_audio[:, :audio.shape[1]].to("cpu")
                    torchaudio.save(output_file_path, vocoded_audio, sr)


if __name__ == "__main__":
    hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                    savedir="vocoder_pretrained_model/tts-hifigan-ljspeech0",
                                    run_opts={"device": "cuda"})

    hifi_gan.eval()
    # hifi_gan = hifi_gan.to("cuda")
    print(hifi_gan.device)
    create_voceded_dataset("LibriSpeech", "VocodedLibriSpeech")
    print("Done")
