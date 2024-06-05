import os
from tqdm import tqdm
import torch
import torchaudio
from torch.nn.functional import mse_loss

from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

from matplotlib import pyplot as plt


def calculate_mse_and_plot_dist(regular_audio_dir, vocoded_audio_dir):
    """
    Calculate the MSE between the regular and vocoded audio files and plot the distribution of the MSE values.
    :param regular_audio_dir:
    :param vocoded_audio_dir:
    :return:
    """
    waveform_mse_values = []
    spectrogram_mse_values = []
    for root, dirs, files in tqdm(os.walk(regular_audio_dir)):
        for file in files:
            if file.endswith(".flac"):
                regular_file_path = os.path.join(root, file)
                vocoded_file_path = os.path.join(vocoded_audio_dir, root, file)
                if os.path.exists(vocoded_file_path):
                    regular_audio, sr = torchaudio.load(regular_file_path)
                    vocoded_audio, sr = torchaudio.load(vocoded_file_path)
                    mse = mse_loss(regular_audio, vocoded_audio)
                    waveform_mse_values.append(mse.item())

                    mel_regular, _ = mel_spectogram(audio=regular_audio.squeeze(), sample_rate=22500, hop_length=256,
                                                    win_length=None,
                                                    n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                                    normalized=False,
                                                    min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                                    compression=True)
                    mel_vocoded, _ = mel_spectogram(audio=vocoded_audio.squeeze(), sample_rate=22500, hop_length=256,
                                                    win_length=None,
                                                    n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                                    normalized=False,
                                                    min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                                    compression=True)
                    mse = mse_loss(mel_regular, mel_vocoded)
                    spectrogram_mse_values.append(mse.item())

    return mse_values


def plot_mse_distribution(mse_values, title):
    """
    Plot the distribution of the MSE values.
    :param mse_values:
    :param title:
    :return:
    """
    plt.hist(mse_values, bins=100)
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.show()
    # save the figure
    plt.savefig(f"{title}.png")


if __name__ == "__main__":
    mse_values = calculate_mse_and_plot_dist("LibriSpeech", "VocodedLibriSpeech")
    plot_mse_distribution(mse_values, "Waveform MSE Distribution")
    plot_mse_distribution(mse_values, "Spectrogram MSE Distribution")
    print("Done")
