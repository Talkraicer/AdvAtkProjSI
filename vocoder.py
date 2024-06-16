import os
from tqdm import tqdm
import torch
import torchaudio
from torch.nn.functional import mse_loss

import librosa
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import seaborn as sns

sns.set()

hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech",
                                savedir="vocoder_pretrained_model/tts-hifigan-ljspeech")


def apply_vocoder(audio_path, output_path, saving_sr=16000):
    waveform, sample_rate = torchaudio.load(audio_path)
    spectrogram, _ = mel_spectogram(audio=waveform.squeeze(), sample_rate=22500, hop_length=256, win_length=None,
                                    n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                    min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)

    waveform_reconstructed = hifi_gan.decode_batch(spectrogram).squeeze(1)[:, :waveform.shape[1]]
    if waveform_reconstructed.shape[1] < waveform.shape[1]:
        print(f"Waveform reconstructed is shorter than original waveform - {audio_path}")
    torchaudio.save(output_path, waveform_reconstructed.squeeze(1), saving_sr)


# plot the original waveform and the vocoded waveform, the attacked waveform and the vocoded attacked waveform
def plot_waveforms(clean_audio, attacked_audio, saving_sr=16000):
    clean_waveform = torch.tensor(librosa.load(clean_audio, sr=saving_sr)[0])
    attacked_waveform = torch.tensor(librosa.load(attacked_audio, sr=saving_sr)[0])

    clean_mel_spectrogram, _ = mel_spectogram(audio=clean_waveform.squeeze(), sample_rate=22500, hop_length=256,
                                              win_length=None,
                                              n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                              min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                              compression=True)
    attacked_mel_spectrogram, _ = mel_spectogram(audio=attacked_waveform.squeeze(), sample_rate=22500, hop_length=256,
                                                 win_length=None,
                                                 n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                                 normalized=False,
                                                 min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                                 compression=True)

    clean_waveform_reconstructed = hifi_gan.decode_batch(clean_mel_spectrogram).squeeze(1)[:, :clean_waveform.shape[1]]
    attacked_waveform_reconstructed = hifi_gan.decode_batch(attacked_mel_spectrogram).squeeze(1)[:,
                                      :attacked_waveform.shape[1]]

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs[0, 0].plot(clean_waveform.squeeze().numpy())
    axs[0, 0].set_title("Original waveform")
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 1].plot(clean_waveform_reconstructed.squeeze().numpy())
    axs[0, 1].set_title("Vocoded waveform")
    axs[1, 0].plot(attacked_waveform.squeeze().numpy())
    axs[1, 0].set_title("Attacked waveform")
    axs[1, 0].set_ylabel("Amplitude")
    axs[1, 1].plot(attacked_waveform_reconstructed.squeeze().numpy())
    axs[1, 1].set_title("Vocoded attacked waveform")
    # transform all x-axis to time
    print(len(clean_waveform.squeeze()) // 48000, len(attacked_waveform.squeeze()))
    for ax in axs.flat:
        ax.set(xlabel='Time', )
        # set the ticks to represent seconds
        ax.set_xticks([i * 48000 for i in range(1 + len(clean_waveform.squeeze()) // 48000)])
        # set the tick labels to represent seconds
        ax.set_xticklabels([f"{i * 3}" for i in range(1 + len(clean_waveform.squeeze()) // (48000))])

    # add space between subplots
    fig.tight_layout(pad=1.0)
    plt.show()
    # set title
    fig.suptitle("Waveforms", fontsize=16)
    # save the figure
    fig.savefig("waveforms_clean_attack_recon.jpg")

    # add mse
    mse_mult_factor = 10000
    fig.text(0.5, 0.75, f"MSE \n{mse_mult_factor*mse_loss(clean_waveform, clean_waveform_reconstructed):.2f}", ha='center',
             va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the bottom subplots
    fig.text(0.5, 0.25, f"MSE \n{mse_mult_factor*mse_loss(attacked_waveform, attacked_waveform_reconstructed):.2f}", ha='center',
             va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the left subplots
    fig.text(0.25, 0.5, f"MSE \n{mse_mult_factor*mse_loss(clean_waveform, attacked_waveform):.2f}", ha='center',
             va='center', fontsize=12, rotation='vertical', bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the right subplots
    fig.text(0.70, 0.5,
             f"MSE \n{mse_mult_factor*mse_loss(attacked_waveform_reconstructed, clean_waveform_reconstructed):.2f}",
             ha='center',
             va='center', fontsize=12, rotation='vertical', bbox=dict(facecolor='gray', alpha=0.3))

    fig.text(0.5, 0.5, f"MSE \n{mse_mult_factor*mse_loss(attacked_waveform_reconstructed, clean_waveform):.2f}",
             ha='center',
             va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3), rotation=-45)
    plt.show()
    fig.tight_layout(pad=3.0)
    # set title
    fig.suptitle(r"Waveforms $MSE \cdot 10^5$", fontsize=16)
    # save the figure
    fig.savefig("waveforms_clean_attack_recon_mse.jpg")



    clean_mel_spec_recon, _ = mel_spectogram(audio=clean_waveform_reconstructed.squeeze(), sample_rate=22500,
                                             hop_length=256, win_length=None,
                                             n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                             normalized=False,
                                             min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                             compression=True)
    attacked_mel_spec_recon, _ = mel_spectogram(audio=attacked_waveform_reconstructed.squeeze(), sample_rate=22500,
                                                hop_length=256, win_length=None,
                                                n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1,
                                                normalized=False,
                                                min_max_energy_norm=True, norm="slaney", mel_scale="slaney",
                                                compression=True)

    # plot the mel spectrograms using plt.imshow without grid
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(clean_mel_spectrogram.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[0, 0].set_title("Original mel spectrogram")
    axs[0, 0].set_xlabel("Time")
    axs[0, 0].set_ylabel("Frequency")
    axs[0, 0].grid(False)
    axs[0, 1].imshow(clean_mel_spec_recon.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[0, 1].set_title("Vocoded mel spectrogram")
    axs[0, 1].set_xlabel("Time")
    axs[0, 1].grid(False)
    axs[1, 0].imshow(attacked_mel_spectrogram.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[1, 0].set_title("Attacked mel spectrogram")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Frequency")
    axs[1, 0].grid(False)
    axs[1, 1].imshow(attacked_mel_spec_recon.squeeze().numpy(), aspect="auto", cmap="viridis")
    axs[1, 1].set_title("Vocoded attacked mel spectrogram")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].grid(False)

    fig.text(0.5, 0.75, f"MSE \n{mse_loss(clean_mel_spectrogram, clean_mel_spec_recon):.2f}", ha='center',
             va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the bottom subplots
    fig.text(0.5, 0.25, f"MSE \n{mse_loss(attacked_mel_spectrogram, attacked_mel_spec_recon):.2f}", ha='center',
             va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the left subplots
    fig.text(0.25, 0.5, f"MSE \n{mse_loss(clean_mel_spectrogram, attacked_mel_spectrogram):.2f}", ha='center',
             va='center', fontsize=12, rotation='vertical', bbox=dict(facecolor='gray', alpha=0.3))

    # Add text between the right subplots
    fig.text(0.75, 0.5, f"MSE \n{mse_loss(attacked_mel_spec_recon, clean_mel_spec_recon):.2f}", ha='center',
             va='center', fontsize=12, rotation='vertical', bbox=dict(facecolor='gray', alpha=0.3))

    fig.text(0.5, 0.5, f"MSE \n{mse_loss(attacked_mel_spec_recon, clean_mel_spectrogram):.2f}", ha='center',
                va='center', fontsize=12, bbox=dict(facecolor='gray', alpha=0.3), rotation=-45)

    # set the x-axis to represent time
    # add space between subplots
    fig.tight_layout(pad=3.0)
    plt.show()
    # set title
    fig.suptitle("Mel spectrograms", fontsize=16)
    # save the figure
    fig.savefig("mel_spectrograms_clean_attack_recon_no_grid.jpg")

    # create table of mse between the original, the reconstructed, the attacked and the attacked reconstructed
    # mel spectrograms
    clean_clean = torch.nn.functional.mse_loss(clean_mel_spectrogram, clean_mel_spectrogram)
    clean_clean_recon = torch.nn.functional.mse_loss(clean_mel_spectrogram, clean_mel_spec_recon)
    clean_attacked = torch.nn.functional.mse_loss(clean_mel_spectrogram, attacked_mel_spectrogram)
    clean_attacked_recon = torch.nn.functional.mse_loss(clean_mel_spectrogram, attacked_mel_spec_recon)

    attacked_clean_recon = torch.nn.functional.mse_loss(attacked_mel_spectrogram, clean_mel_spec_recon)
    attacked_attacked = torch.nn.functional.mse_loss(attacked_mel_spectrogram, attacked_mel_spectrogram)
    attacked_attacked_recon = torch.nn.functional.mse_loss(attacked_mel_spectrogram, attacked_mel_spec_recon)

    clean_recon_clean_recon = torch.nn.functional.mse_loss(clean_mel_spec_recon, clean_mel_spec_recon)
    clena_recon_attacked_recon = torch.nn.functional.mse_loss(clean_mel_spec_recon, attacked_mel_spec_recon)

    attacked_recon_attacked_recon = torch.nn.functional.mse_loss(attacked_mel_spec_recon, attacked_mel_spec_recon)


    mse_table = [[clean_clean.item(), clean_clean_recon.item(), clean_attacked.item(), clean_attacked_recon.item()],
                 [clean_clean_recon.item(), clean_recon_clean_recon.item(), attacked_clean_recon.item(), clena_recon_attacked_recon.item()],
                [clean_attacked.item(), attacked_clean_recon.item(), attacked_attacked.item(), attacked_attacked_recon.item()],
                [clean_attacked_recon.item(), clena_recon_attacked_recon.item(), attacked_attacked_recon.item(), attacked_recon_attacked_recon.item()]]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    # round the values to 2 decimal places
    for i in range(4):
        for j in range(4):
            mse_table[i][j] = round(mse_table[i][j], 3)

    ax.table(cellText=mse_table, rowLabels=["Original", "Original Vocoded", "Attacked", "Attacked Vocoded"],
                colLabels=["Original", "Original Vocoded", "Attacked", "Attacked Vocoded"], loc="center")
    fig.tight_layout()
    # add the title
    fig.suptitle("MSE between mel spectrograms", fontsize=16)
    plt.show()




# mse - calculates the mean squared error between two spectrograms (the original and the reconstructed)
def mse_paths(audio1, audio2):
    spec1, _ = mel_spectogram(audio=audio1.squeeze(), sample_rate=22500, hop_length=256, win_length=None,
                              n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                              min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)
    spec2, _ = mel_spectogram(audio=audio2.squeeze(), sample_rate=22500, hop_length=256, win_length=None,
                              n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                              min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)
    return torch.nn.functional.mse_loss(spec1, spec2)


def mse_of_dir(audio_dir1, audio_dir2):
    mse = 0
    for audio1, audio2 in zip(os.listdir(audio_dir1), os.listdir(audio_dir2)):
        audio1_path = os.path.join(audio_dir1, audio1)
        audio2_path = os.path.join(audio_dir2, audio2)
        mse += mse_paths(audio1_path, audio2_path)
    return mse / len(os.listdir(audio_dir1))


if __name__ == "__main__":
    audio_source_dir = "C:/Adverserial/LibriSpeech/train-clean-100/103/1240"
    audio_attacked_dir = "C:/Adverserial/clean_4000_96.7/wavs/103/1240"
    audio_file ="103-1240-0047.flac"
    audio_output_dir = "103/1240_vocoded_clean"
    plot_waveforms(f"{audio_source_dir}/{audio_file}", f"{audio_attacked_dir}/{audio_file.replace('.flac', '.wav')}")

    plot_waveforms("LibriSpeech/train-clean-100/103/1240/103-1240-0047.flac",
                   "attacks/clean_cnn_eps_0.5/PGD_eps_0.0005/wavs/103/1240/103-1240-0047.wav")

    if not os.path.exists(audio_output_dir):
        os.makedirs(audio_output_dir)
    for audio_file in tqdm(os.listdir(audio_source_dir)):
        if audio_file.endswith(".wav") or audio_file.endswith(".flac"):
            index = audio_file.split("-")[-1][:-5]
            if int(index) >= 47:
                audio_path = os.path.join(audio_source_dir, audio_file)
                output_path = os.path.join(audio_output_dir, audio_file)
                apply_vocoder(audio_path, output_path)
                break
