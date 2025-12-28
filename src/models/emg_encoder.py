import torch
import torch.nn as nn


class EMGEncoder(nn.Module):
    """CNN backbone for extracting features from raw EMG"""

    def __init__(self, input_channels: int = 16, output_dim: int = 256):
        super(EMGEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, output_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x.squeeze(-1)


class GestureBranch(nn.Module):
    """Extracts position-invariant gesture features"""

    def __init__(self, input_dim: int = 256, output_dim: int = 128, dropout: float = 0.3):
        super(GestureBranch, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, x):
        return self.network(x)


class PositionBranch(nn.Module):
    """Extracts gesture-invariant position features"""

    def __init__(self, input_dim: int = 256, output_dim: int = 64, dropout: float = 0.3):
        super(PositionBranch, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.network(x)


class GestureClassifier(nn.Module):
    def __init__(self, input_dim: int = 128, num_classes: int = 6):
        super(GestureClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


class PositionClassifier(nn.Module):
    def __init__(self, input_dim: int = 64, num_positions: int = 5):
        super(PositionClassifier, self).__init__()
        self.classifier = nn.Linear(input_dim, num_positions)

    def forward(self, x):
        return self.classifier(x)


class PositionDiscriminator(nn.Module):
    """Predicts position from gesture features for adversarial training"""

    def __init__(self, input_dim: int = 128, num_positions: int = 5):
        super(PositionDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_positions)
        )

    def forward(self, x):
        return self.discriminator(x)


class DisentangledEMGNet(nn.Module):
    """Complete disentangled EMG network"""

    def __init__(
        self,
        num_gestures: int = 6,
        num_positions: int = 5,
        encoder_dim: int = 256,
        gesture_dim: int = 128,
        position_dim: int = 64
    ):
        super(DisentangledEMGNet, self).__init__()

        self.encoder = EMGEncoder(input_channels=16, output_dim=encoder_dim)
        self.gesture_branch = GestureBranch(encoder_dim, gesture_dim)
        self.position_branch = PositionBranch(encoder_dim, position_dim)
        self.gesture_classifier = GestureClassifier(gesture_dim, num_gestures)
        self.position_classifier = PositionClassifier(position_dim, num_positions)
        self.position_discriminator = PositionDiscriminator(gesture_dim, num_positions)

    def forward(self, x, return_features=False):
        shared_features = self.encoder(x)
        gesture_features = self.gesture_branch(shared_features)
        position_features = self.position_branch(shared_features)
        gesture_logits = self.gesture_classifier(gesture_features)
        position_logits = self.position_classifier(position_features)
        discriminator_logits = self.position_discriminator(gesture_features)

        if return_features:
            return {
                'shared_features': shared_features,
                'gesture_features': gesture_features,
                'position_features': position_features,
                'gesture_logits': gesture_logits,
                'position_logits': position_logits,
                'discriminator_logits': discriminator_logits
            }
        else:
            return gesture_logits, position_logits, discriminator_logits
