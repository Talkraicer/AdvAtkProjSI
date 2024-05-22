from speechbrain.inference.vocoders import HIFIGAN
from speechbrain.lobes.models.FastSpeech2 import mel_spectogram
import os
import torchaudio
from tqdm import tqdm
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
    # plot mel spectrogram for src and tgt
    fig, axs = plt.subplots(2)
    axs[0].imshow(spectrogram.detach().cpu().numpy(), aspect='auto', origin='lower')
    axs[0].set_title("Mel spectrogram of original audio")
    tgt_spectrogram, _ = mel_spectogram(audio=waveform_reconstructed.squeeze(), sample_rate=16000, hop_length=256,
                                        win_length=None,
                                        n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                        min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)
    axs[1].imshow(tgt_spectrogram.detach().cpu().numpy(), aspect='auto', origin='lower')
    axs[1].set_title("Mel spectrogram of vocoded audio")
    # add space between subplots
    fig.tight_layout(pad=1.0)
    plt.show()


# plot the original waveform and the vocoded waveform, the attacked waveform and the vocoded attacked waveform
def plot_waveforms(clean_audio, attacked_audio, saving_sr=16000):
    clean_waveform, sample_rate = torchaudio.load(clean_audio,)
    attacked_waveform, sample_rate = torchaudio.load(attacked_audio)

    clean_mel_spectrogram, _ = mel_spectogram(audio=clean_waveform.squeeze(), sample_rate=22500, hop_length=256, win_length=None,
                                    n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                    min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)
    attacked_mel_spectrogram, _ = mel_spectogram(audio=attacked_waveform.squeeze(), sample_rate=22500, hop_length=256, win_length=None,
                                    n_mels=80, n_fft=1024, f_min=0.0, f_max=8000.0, power=1, normalized=False,
                                    min_max_energy_norm=True, norm="slaney", mel_scale="slaney", compression=True)

    clean_waveform_reconstructed = hifi_gan.decode_batch(clean_mel_spectrogram).squeeze(1)[:, :clean_waveform.shape[1]]
    attacked_waveform_reconstructed = hifi_gan.decode_batch(attacked_mel_spectrogram).squeeze(1)[:, :attacked_waveform.shape[1]]

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
        ax.set_xticklabels([f"{i * 3 }" for i in range(1 + len(clean_waveform.squeeze()) // (48000))])

    # add space between subplots
    fig.tight_layout(pad=1.0)
    plt.show()



if __name__ == "__main__":
    audio_source_dir = "LibriSpeech/train-clean-100/103/1240"
    audio_output_dir = "103/1240_vocoded_clean"

    plot_waveforms("LibriSpeech/train-clean-100/103/1240/103-1240-0047.flac", "103/1240/103-1240-0047.wav")

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