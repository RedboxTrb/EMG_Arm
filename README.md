# EMG Prosthetic Control

Position-invariant gesture recognition for myoelectric prosthetics using deep learning.

## Overview

Neural network approach for classifying hand gestures from EMG signals across different arm positions. Traditional methods suffer from significant accuracy drops when arm position changes - this addresses that through representation learning.

## Dataset

"EMG Dataset for Gesture Recognition with Arm Translation" (Kyranou et al., 2024)
- 8 participants, 6 gestures, 5 positions
- 16-channel EMG @ 2kHz
- DOI: 10.1038/s41597-024-04296-8

## Structure

```
src/
├── models/          # Neural architectures and losses
├── utils/           # Data loading and preprocessing
└── training/        # Training scripts
```

## Setup

```bash
conda activate torch
pip install -r requirements.txt
```

## Training

Baseline:
```bash
python src/training/train_baseline_lda.py --within-position
python src/training/train_baseline_lda.py --cross-position
```

Disentangled model:
```bash
python src/training/train_disentangled.py --epochs 100
```

## Approach

**Baseline**: LDA classifier (~78% cross-position)

**Phase 2**: Disentangled CNN with multi-task learning (target: 85-90%)

**Phase 3**: Prototypical networks for one-shot learning (planned)
