import torch
import torch.nn as nn
from dev.transforms import Preprocessor


class RawAudioCNN(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""

    def __init__(self, num_class):
        super().__init__()
        self.prep = Preprocessor()

        # =========== EXPERIMENTAL pre-filtering ======
        # 32 x 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[5, 5], stride=1, padding=[2, 2]),
            nn.BatchNorm2d(1),
        )
        # =========== ============= ======

        # 32 x 100
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 100
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 100
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 25
        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # maybe replace pool with dropout here
            # nn.MaxPool1d(2, stride=2)
        )

        # 32 x 30
        self.fc = nn.Linear(32, num_class)

    def forward(self, x):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        embedding = self.encode(x)
        logits = self.fc(embedding)
        return logits

    def encode(self, x):
        x = self.prep(x.squeeze(1))

        # ===== pre-filtering ========
        # [B, F, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        # ===== pre-filtering ========

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x, _ = x.max(2)
        return x

    def predict_from_embeddings(self, x):
        return self.fc(x)


class SpectrogramCNN(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""

    def __init__(self, num_class):
        super().__init__()
        self.prep = Preprocessor()

        # =========== EXPERIMENTAL pre-filtering ======
        # 32 x 100
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=[5, 5], stride=1, padding=[2, 2]),
            nn.BatchNorm2d(1),
        )
        # =========== ============= ======

        # 32 x 100
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 100
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 100
        self.conv4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv5 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 128 x 50
        self.conv6 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            # nn.MaxPool1d(2, stride=2)
        )
        # 128 x 25
        self.conv7 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )
        # 64 x 25
        self.conv8 = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # maybe replace pool with dropout here
            # nn.MaxPool1d(2, stride=2)
        )

        # 32 x 30
        self.fc = nn.Linear(32, num_class)

    def forward(self, x):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        embedding = self.encode(x)
        logits = self.fc(embedding)
        return logits

    def encode(self, x):
        # ===== pre-filtering ========
        # [B, F, T]
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = x.squeeze(1)
        # ===== pre-filtering ========

        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x, _ = x.max(2)
        return x

    def predict_from_embeddings(self, x):
        return self.fc(x)


class DoubleModelCNN(nn.Module):
    """Adaption of AudioNet (arXiv:1807.03418)."""

    def __init__(self, num_class, cnn_audio, cnn_spec):
        super().__init__()
        self.prep = Preprocessor()

        self.cnn_audio = cnn_audio
        self.cnn_spec = cnn_spec

        self.fc = nn.Linear(64, num_class)

    def forward(self, audio):
        """
        Inputs:
            x: [B, 1, T] waveform
        Outputs:
            x: [B, 1, T] waveform
        """
        spectrogram = self.prep(audio.squeeze(1))
        embedding1 = self.cnn_audio.encode(audio)
        embedding2 = self.cnn_spec.encode(spectrogram)
        embedding = torch.cat([embedding1, embedding2], dim=1)
        logits = self.fc(embedding)
        return logits

    def encode(self, audio):
        spectrogram = self.prep(audio.squeeze(1))
        embedding1 = self.cnn_audio.encode(audio)
        embedding2 = self.cnn_spec.encode(spectrogram)
        return torch.cat([embedding1, embedding2], dim=1)

    def predict_from_embeddings(self, x):
        return self.fc(x)

    def freeze_cnns(self):
        for param in self.cnn_audio.parameters():
            param.requires_grad = False
        for param in self.cnn_spec.parameters():
            param.requires_grad = False

