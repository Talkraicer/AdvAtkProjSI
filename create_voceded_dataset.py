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
            if file.endswith(".wav"):
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
    # aug_attacked_dirs = ["CWinf_eps_0.005", "CWinf_eps_0.0005", "FGSM_eps_0.005", "FGSM_eps_0.0005", "PGD_eps_0.005",
    #                      "PGD_eps_0.0005"]
    # aug_attacked_dirs = [os.path.join("attacks\\clean_cnn_eps_0.5", attack,) for attack in aug_attacked_dirs]
    aug_attacked_dirs = ["attacks/clean_cnn_eps_0.5/FGSM_eps_0.0005"]
    vocoded_dirs = ["vocoded_" + attack for attack in aug_attacked_dirs]
    for i in range(len(aug_attacked_dirs)):
        print(f"Creating vocoded dataset for {aug_attacked_dirs[i]}")
        create_voceded_dataset(aug_attacked_dirs[i], vocoded_dirs[i])
    print("Done")
