import os
from tqdm import tqdm
import torch
import torchaudio
from torch.nn.functional import mse_loss
import numpy as np

from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

from matplotlib import pyplot as plt


def calculate_mse_and_plot_dist(regular_audio_dir, vocoded_audio_dir):
    """
    Calculate the MSE between the regular and vocoded audio files and plot the distribution of the MSE values.
    :param regular_audio_dir:
    :param vocoded_audio_dir:
    :return:
    """
    waveform_mse_values = {}
    spectrogram_mse_values = {}
    print("Calculating MSE values:")
    first_voc = True if "vocoded" in regular_audio_dir.lower() else False
    counter = 0
    for root, dirs, files in tqdm(os.walk(regular_audio_dir)):
        for file in files:
            if file.endswith(".flac"):
                counter += 1
                if first_voc:
                    regular_file_path = os.path.join(root, file)
                    vocoded_file_path = os.path.join(vocoded_audio_dir + root.split("/LibriSpeech\\train-clean-100")[-1], file)
                else:
                    regular_file_path = os.path.join(root, file)
                    vocoded_file_path = os.path.join(vocoded_audio_dir, root, file)
                if os.path.exists(vocoded_file_path):
                    regular_audio, sr = torchaudio.load(regular_file_path)
                    vocoded_audio, sr = torchaudio.load(vocoded_file_path)
                    mse = mse_loss(regular_audio, vocoded_audio)
                    waveform_mse_values[os.path.join(root, file)] = mse.item()

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
                    spectrogram_mse_values[os.path.join(root, file)] = mse.item()
    print(f"Total Files: {counter}")
    return waveform_mse_values, spectrogram_mse_values


def plot_waveform_comparison(percentile_5_files, percentile_95_files, title):
    # select random from each list and plot the waveform\spectrogram
    perc_5_file = np.random.choice(percentile_5_files)
    perc_95_file = np.random.choice(percentile_95_files)
    perc_5_audio_reg, sr = torchaudio.load(perc_5_file.split("/")[-1])
    perc_5_audio_voc, sr = torchaudio.load(perc_5_file)
    perc_95_audio_reg, sr = torchaudio.load(perc_95_file.split("/")[-1])
    perc_95_audio_voc, sr = torchaudio.load(perc_95_file)
    # plot in the same figure
    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].plot(perc_5_audio_reg.squeeze().numpy())
    axs[0, 0].set_title(f" Regular - 5% percentile")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].scatter([0, 0], [-1, 1], alpha=0)
    axs[0, 1].plot(perc_5_audio_voc.squeeze().numpy())
    axs[0, 1].set_title(f" Vocoded - 5% percentile")
    axs[0, 1].scatter([0, 0], [-1, 1], alpha=0)
    axs[1, 0].plot(perc_95_audio_reg.squeeze().numpy())
    axs[1, 0].set_title(f" Regular - 95% percentile")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 0].scatter([0, 0], [-1, 1], alpha=0)
    axs[1, 1].plot(perc_95_audio_voc.squeeze().numpy())
    axs[1, 1].set_title(f" Vocoded - 95% percentile")
    axs[1, 1].scatter([0, 0], [-1, 1], alpha=0)
    # transform all x-axis to time
    for i, ax in enumerate(axs.flat):
        ax.set(xlabel='Time', )
        if i >= 2:
            # set the ticks to represent seconds
            ax.set_xticks([i * 48000 for i in range(1 + len(perc_95_audio_reg.squeeze()) // 48000)])
            # set the tick labels to represent seconds
            ax.set_xticklabels([f"{i * 3}" for i in range(1 + len(perc_95_audio_reg.squeeze()) // (48000))])
        else:
            # set the ticks to represent seconds
            ax.set_xticks([i * 48000 for i in range(1 + len(perc_5_audio_reg.squeeze()) // 48000)])
            # set the tick labels to represent seconds
            ax.set_xticklabels([f"{i * 3}" for i in range(1 + len(perc_5_audio_reg.squeeze()) // (48000))])

    # add space between subplots
    fig.tight_layout(pad=1.0)
    plt.show()
    # save the figure
    title = title.replace("Distribution", "")
    plt.savefig(f"graphs_and_results/{title}_waveform_comparison.jpg")
    plt.cla()
    plt.clf()


def plot_spectrogram_comparison(percentile_5_files, percentile_95_files, title):
    perc_5_file = np.random.choice(percentile_5_files)
    perc_95_file = np.random.choice(percentile_95_files)
    perc_5_audio_reg, sr = torchaudio.load(perc_5_file.split("/")[-1])
    perc_5_audio_voc, sr = torchaudio.load(perc_5_file)
    perc_95_audio_reg, sr = torchaudio.load(perc_95_file.split("/")[-1])
    perc_95_audio_voc, sr = torchaudio.load(perc_95_file)
    mel_5_reg, _ = mel_spectogram(audio=perc_5_audio_reg.squeeze(), sample_rate=22500, hop_length=256,
                                  win_length=None,
                                  n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                  normalized=False,
                                  min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                  compression=True)
    mel_5_voc, _ = mel_spectogram(audio=perc_5_audio_voc.squeeze(), sample_rate=22500, hop_length=256,
                                  win_length=None,
                                  n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                  normalized=False,
                                  min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                  compression=True)
    mel_95_reg, _ = mel_spectogram(audio=perc_95_audio_reg.squeeze(), sample_rate=22500, hop_length=256,
                                   win_length=None,
                                   n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                   normalized=False,
                                   min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                   compression=True)
    mel_95_voc, _ = mel_spectogram(audio=perc_95_audio_voc.squeeze(), sample_rate=22500, hop_length=256,
                                   win_length=None,
                                   n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                   normalized=False,
                                   min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                   compression=True)
    # plot in the same figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(mel_5_reg.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[0, 0].set_title("Regular - 5% percentile")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(False)
    axs[0, 1].imshow(mel_5_voc.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[0, 1].set_title("Vocoded - 5% percentile")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].grid(False)
    axs[1, 0].imshow(mel_95_reg.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[1, 0].set_title("Regular - 95% percentile")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(False)
    axs[1, 1].imshow(mel_95_voc.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[1, 1].set_title("Vocoded - 95% percentile")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].grid(False)
    fig.tight_layout(pad=3.0)
    # save the figure
    title = title.replace("Distribution", "")
    plt.savefig(f"graphs_and_results/{title}_spectrogram_comparison.jpg")
    plt.cla()
    plt.clf()


def plot_mse_distribution(mse_dict, title, ):
    """
    Plot the distribution of the MSE values.
    :param mse_values:
    :param title:
    :return:
    """
    mse_values = list(mse_dict.values())
    # find the 5 and 95 percentile files names
    percentile_5 = np.percentile(mse_values, 5)
    percentile_95 = np.percentile(mse_values, 95)
    percentile_5_files = []
    percentile_95_files = []
    for file, mse in mse_dict.items():
        if mse <= percentile_5:
            percentile_5_files.append(file)
        if mse >= percentile_95:
            percentile_95_files.append(file)

    if "waveform" in title.lower():
        plot_waveform_comparison(percentile_5_files, percentile_95_files, title)
    elif "spectrogram" in title.lower():
        plot_spectrogram_comparison(percentile_5_files, percentile_95_files, title)

    plt.hist(mse_values, bins=100)
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.show()
    # save the figure
    plt.savefig(f"{title}.jpg")
    # plot the same figure with 5 and 95 percentile
    plt.clf()
    plt.cla()
    plt.hist(mse_values, bins=100)
    plt.title(title)
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    plt.axvline(x=np.percentile(mse_values, 5), color='r', linestyle='dashed', linewidth=2)
    plt.axvline(x=np.percentile(mse_values, 95), color='r', linestyle='dashed', linewidth=2)
    plt.show()
    # save the figure
    plt.savefig(f"graphs_and_results/{title}_with_percentile.png")
    # save for text file the following statistics: mean, median, 5 percentile, 95 percentile, std, max, min
    with open(f"graphs_and_results/{title}_statistics.txt", "w") as f:
        f.write(f"Mean: {np.mean(mse_values)}\n")
        f.write(f"Median: {np.median(mse_values)}\n")
        f.write(f"5 Percentile: {np.percentile(mse_values, 5)}\n")
        f.write(f"95 Percentile: {np.percentile(mse_values, 95)}\n")
        f.write(f"STD: {np.std(mse_values)}\n")
        f.write(f"Max: {np.max(mse_values)}\n")
        f.write(f"Min: {np.min(mse_values)}\n")


if __name__ == "__main__":
    waveform_mse_values, spectrogram_mse_values = calculate_mse_and_plot_dist(
        "LibriSpeech", "VocodedLibriSpeech")
    print(waveform_mse_values)
    plot_mse_distribution(waveform_mse_values, "Waveform MSE Distribution - LibriSpeech vs VocodedLibriSpeech")
    plot_mse_distribution(spectrogram_mse_values, "Spectrogram MSE Distribution - LibriSpeech vs VocodedLibriSpeech")
    print("Done")
