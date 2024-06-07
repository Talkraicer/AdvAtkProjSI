class Hparams():
    def __init__(self):
        self.sr = 16_000
        self.n_mels = 32
        self.n_fft = 1024
        self.hop_length = 160
        self.win_length = 800
        self.n_frames = 128
        self.max_db = 100
        self.ref_db = 20
        self.top_db = 15
        self.preemphasis = 0.97
        self.n_iter = 100
        self.fmin = 0
        self.fmax = self.sr // 2
        self.min_mel = 1e-5
        self.aux_context_window = 2

        #self.data_root = "/data/speech"
        self.data_root = "C:\Adverserial\VocodedLibriSpeech\VocodedLibriSpeech"
        self.data_subset = "train-clean-100"
        self.train_speaker_ratio = 1
        self.train_utterance_ratio = 0.8
        self.val_utter_ratio = 0.1

hp = Hparams()
