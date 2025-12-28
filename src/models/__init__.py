"""Neural network models for EMG classification"""

from .emg_encoder import (
    EMGEncoder,
    GestureBranch,
    PositionBranch,
    GestureClassifier,
    PositionClassifier,
    PositionDiscriminator,
    DisentangledEMGNet
)

from .losses import (
    GestureClassificationLoss,
    PositionClassificationLoss,
    NTXentLoss,
    AdversarialLoss,
    DisentanglementLoss,
    GradientReversalLayer
)

__all__ = [
    'EMGEncoder',
    'GestureBranch',
    'PositionBranch',
    'GestureClassifier',
    'PositionClassifier',
    'PositionDiscriminator',
    'DisentangledEMGNet',
    'GestureClassificationLoss',
    'PositionClassificationLoss',
    'NTXentLoss',
    'AdversarialLoss',
    'DisentanglementLoss',
    'GradientReversalLayer'
]
