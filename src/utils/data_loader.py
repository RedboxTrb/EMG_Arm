"""
Data loading utilities for EMG dataset

Dataset structure:
- 8 participants
- 2 days × 2 blocks = 4 blocks per participant
- 150 trials per block (6 gestures × 5 positions × 5 repetitions)
- Each trial: 5 seconds @ 2000 Hz = 10,000 samples
- 16 EMG channels
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
from torch.utils.data import Dataset


class EMGDataset(Dataset):
    """
    PyTorch Dataset for loading EMG data

    Args:
        data_root: Path to dataset root (e.g., 'EMG_Arm_Dataset/data')
        participants: List of participant IDs (1-8)
        days: List of days (1-2)
        blocks: List of blocks (1-2)
        positions: List of positions to include (None = all)
        grasps: List of grasps to include (None = all)
        transform: Optional data transformation
    """

    def __init__(
        self,
        data_root: str,
        participants: List[int] = None,
        days: List[int] = None,
        blocks: List[int] = None,
        positions: Optional[List[int]] = None,
        grasps: Optional[List[int]] = None,
        transform=None
    ):
        self.data_root = Path(data_root)
        self.transform = transform

        # Default: all participants, days, blocks
        self.participants = participants if participants is not None else list(range(1, 9))
        self.days = days if days is not None else [1, 2]
        self.blocks = blocks if blocks is not None else [1, 2]
        self.positions_filter = positions
        self.grasps_filter = grasps

        # Load all trial metadata
        self.samples = self._load_metadata()

        print(f"Loaded {len(self.samples)} trials")
        print(f"  Participants: {self.participants}")
        print(f"  Days: {self.days}, Blocks: {self.blocks}")
        if self.positions_filter:
            print(f"  Positions: {self.positions_filter}")
        if self.grasps_filter:
            print(f"  Grasps: {self.grasps_filter}")

    def _load_metadata(self) -> List[Dict]:
        """Load metadata for all trials matching the filters"""
        samples = []

        for participant in self.participants:
            participant_dir = self.data_root / f"participant_{participant}"

            if not participant_dir.exists():
                print(f"Warning: {participant_dir} not found")
                continue

            for day in self.days:
                for block in self.blocks:
                    block_dir = participant_dir / f"participant{participant}_day{day}_block{block}"

                    if not block_dir.exists():
                        continue

                    # Load trials.csv
                    trials_file = block_dir / "trials.csv"
                    if not trials_file.exists():
                        print(f"Warning: {trials_file} not found")
                        continue

                    df = pd.read_csv(trials_file)

                    # Filter by position and grasp
                    if self.positions_filter is not None:
                        df = df[df['target_position'].isin(self.positions_filter)]
                    if self.grasps_filter is not None:
                        df = df[df['grasp'].isin(self.grasps_filter)]

                    # Create sample entries
                    for _, row in df.iterrows():
                        samples.append({
                            'participant': participant,
                            'day': day,
                            'block': block,
                            'position': int(row['target_position']),
                            'grasp': int(row['grasp']),
                            'trial_no': int(row['trial_no']),
                            'row_number': int(row['row_number']),
                            'block_dir': block_dir
                        })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """
        Get a single trial

        Returns:
            emg_data: Tensor of shape (16, 10000) - 16 channels × 10000 samples
            grasp: Grasp label (0-5)
            position: Position label (0-8)
        """
        sample = self.samples[idx]
        block_dir = sample['block_dir']

        # Load EMG data from HDF5
        emg_file = block_dir / "emg_data.hdf5"
        with h5py.File(emg_file, 'r') as f:
            # Get the trial data
            # The HDF5 structure is: {row_number: emg_array}
            row_key = str(sample['row_number'])
            emg_data = f[row_key][:]  # Shape: (16, 10000)

        # Convert to float32 tensor
        emg_tensor = torch.from_numpy(emg_data).float()

        # Apply transform if specified
        if self.transform:
            emg_tensor = self.transform(emg_tensor)

        return emg_tensor, sample['grasp'], sample['position']

    def get_sample_info(self, idx: int) -> Dict:
        """Get metadata for a sample"""
        return self.samples[idx]


def create_data_splits(
    data_root: str,
    test_participant: int,
    val_participant: Optional[int] = None
) -> Tuple[EMGDataset, Optional[EMGDataset], EMGDataset]:
    """
    Create train/val/test splits for leave-one-subject-out cross-validation

    Args:
        data_root: Path to dataset root
        test_participant: Participant ID for testing (1-8)
        val_participant: Optional participant ID for validation (1-8)

    Returns:
        train_dataset, val_dataset (or None), test_dataset
    """
    all_participants = list(range(1, 9))

    # Test set: one participant
    test_participants = [test_participant]

    # Validation set: optional
    if val_participant is not None:
        val_participants = [val_participant]
        train_participants = [p for p in all_participants
                             if p not in [test_participant, val_participant]]
    else:
        val_participants = None
        train_participants = [p for p in all_participants
                             if p != test_participant]

    # Create datasets
    train_dataset = EMGDataset(data_root, participants=train_participants)

    val_dataset = EMGDataset(data_root, participants=val_participants) \
        if val_participants else None

    test_dataset = EMGDataset(data_root, participants=test_participants)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # Test the data loader
    print("Testing EMG Data Loader")
    print("=" * 60)

    data_root = "EMG_Arm_Dataset/data"

    # Load data for participant 1 only
    dataset = EMGDataset(
        data_root=data_root,
        participants=[1],
        days=[1],
        blocks=[1]
    )

    print(f"\nDataset size: {len(dataset)}")

    # Get first sample
    if len(dataset) > 0:
        emg, grasp, position = dataset[0]
        print(f"\nFirst sample:")
        print(f"  EMG shape: {emg.shape}")
        print(f"  EMG dtype: {emg.dtype}")
        print(f"  Grasp: {grasp}")
        print(f"  Position: {position}")
        print(f"  EMG stats: min={emg.min():.3f}, max={emg.max():.3f}, mean={emg.mean():.3f}")

        # Show sample info
        info = dataset.get_sample_info(0)
        print(f"\nSample metadata: {info}")

    # Test data splits
    print("\n" + "=" * 60)
    print("Testing train/val/test split (LOSO)")
    print("=" * 60)

    train_ds, val_ds, test_ds = create_data_splits(
        data_root=data_root,
        test_participant=8,
        val_participant=7
    )

    print(f"\nTrain size: {len(train_ds)}")
    print(f"Val size: {len(val_ds) if val_ds else 0}")
    print(f"Test size: {len(test_ds)}")
