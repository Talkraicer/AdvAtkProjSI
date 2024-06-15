import os
from tqdm import tqdm
import torch
import torchaudio
import librosa
from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram

import matplotlib.pyplot as plt
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

    fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(clean_waveform.squeeze().numpy())
    axs[0, 0].set_title("Original waveform")
    axs[0, 1].plot(clean_waveform_reconstructed.squeeze().numpy())
    axs[0, 1].set_title("Vocoded waveform")
    axs[1, 0].plot(attacked_waveform.squeeze().numpy())
    axs[1, 0].set_title("Attacked waveform")
    axs[1, 1].plot(attacked_waveform_reconstructed.squeeze().numpy())
    axs[1, 1].set_title("Vocoded attacked waveform")
    # transform all x-axis to time
    print(len(clean_waveform.squeeze()) // 48000, len(attacked_waveform.squeeze()))
    for ax in axs.flat:
        ax.set(xlabel='Time', ylabel='Amplitude')
        # set the ticks to represent seconds
        ax.set_xticks([i * 48000 for i in range(1 + len(clean_waveform.squeeze()) // 48000)])
        # set the tick labels to represent seconds
        ax.set_xticklabels([f"{i * 3}" for i in range(1 + len(clean_waveform.squeeze()) // (48000))])

    # add space between subplots
    fig.tight_layout(pad=1.0)
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

    # if not os.path.exists(audio_output_dir):
    #     os.makedirs(audio_output_dir)
    # #TODO: Check the saving sr
    # for audio_file in tqdm(os.listdir(audio_source_dir)):
    #     if audio_file.endswith(".wav") or audio_file.endswith(".flac"):
    #         index = audio_file.split("-")[-1][:-5]
    #         if int(index) >= 47:
    #             audio_path = os.path.join(audio_source_dir, audio_file)
    #             output_path = os.path.join(audio_output_dir, audio_file)
    #             apply_vocoder(audio_path, output_path)
    #             break
