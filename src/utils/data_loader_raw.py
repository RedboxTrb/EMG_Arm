import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class RawEMGDataset(Dataset):
    """Dataset for loading raw EMG signals (16, 10000)"""

    def __init__(
        self,
        data_root: str,
        participants: List[int] = None,
        days: List[int] = None,
        blocks: List[int] = None,
        positions: Optional[List[int]] = None,
        grasps: Optional[List[int]] = None,
        augment: bool = False
    ):
        self.data_root = Path(data_root)
        self.augment = augment

        # Defaults
        self.participants = participants if participants is not None else list(range(1, 9))
        self.days = days if days is not None else [1, 2]
        self.blocks = blocks if blocks is not None else [1, 2]
        self.positions_filter = positions
        self.grasps_filter = grasps

        # Load metadata
        self.samples = self._load_metadata()

        # Create position mapping (for position labels)
        all_positions = sorted(set(s['position'] for s in self.samples))
        self.position_to_idx = {pos: idx for idx, pos in enumerate(all_positions)}
        self.idx_to_position = {idx: pos for pos, idx in self.position_to_idx.items()}

        print(f"Loaded {len(self.samples)} trials")
        print(f"  Participants: {self.participants}")
        print(f"  Days: {self.days}, Blocks: {self.blocks}")
        if self.positions_filter:
            print(f"  Positions: {self.positions_filter}")
        if self.grasps_filter:
            print(f"  Grasps: {self.grasps_filter}")
        print(f"  Position mapping: {self.position_to_idx}")

    def _load_metadata(self) -> List[dict]:
        samples = []

        for participant in self.participants:
            participant_dir = self.data_root / f"participant_{participant}"

            if not participant_dir.exists():
                continue

            for day in self.days:
                for block in self.blocks:
                    block_dir = participant_dir / f"participant{participant}_day{day}_block{block}"

                    if not block_dir.exists():
                        continue

                    trials_file = block_dir / "trials.csv"
                    if not trials_file.exists():
                        continue

                    df = pd.read_csv(trials_file)

                    # Filter
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
        sample = self.samples[idx]
        block_dir = sample['block_dir']

        # Load raw EMG
        emg_file = block_dir / "emg_data.hdf5"
        with h5py.File(emg_file, 'r') as f:
            row_key = str(sample['row_number'])
            emg_data = f[row_key][:]  # (16, N)

        # Ensure correct length (10000 samples)
        if emg_data.shape[1] < 10000:
            # Pad if too short
            pad_width = 10000 - emg_data.shape[1]
            emg_data = np.pad(emg_data, ((0, 0), (0, pad_width)), mode='edge')
        elif emg_data.shape[1] > 10000:
            # Trim if too long
            emg_data = emg_data[:, :10000]

        # Convert to tensor
        emg_tensor = torch.from_numpy(emg_data).float()

        # Apply augmentation if requested
        if self.augment:
            emg_tensor = self._augment(emg_tensor)

        # Get labels
        grasp_label = sample['grasp']  # Keep 1-6 indexing from dataset
        position_label = self.position_to_idx[sample['position']]  # Map to 0-indexed

        return emg_tensor, grasp_label, position_label

    def _augment(self, emg: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.9, 1.1)
            emg = emg * scale

        if np.random.rand() < 0.5:
            noise = torch.randn_like(emg) * 0.01 * emg.std()
            emg = emg + noise

        if np.random.rand() < 0.5:
            shift = np.random.randint(-200, 200)
            emg = torch.roll(emg, shift, dims=1)

        return emg

    def get_sample_info(self, idx: int) -> dict:
        return self.samples[idx]


def create_balanced_sampler(dataset: RawEMGDataset) -> WeightedRandomSampler:
    gesture_position_counts = {}
    for sample in dataset.samples:
        key = (sample['grasp'], sample['position'])
        gesture_position_counts[key] = gesture_position_counts.get(key, 0) + 1

    weights = []
    for sample in dataset.samples:
        key = (sample['grasp'], sample['position'])
        weight = 1.0 / gesture_position_counts[key]
        weights.append(weight)

    weights = torch.DoubleTensor(weights)
    return WeightedRandomSampler(weights, len(weights), replacement=True)


def create_disentanglement_dataloaders(
    data_root: str,
    train_participants: List[int],
    val_participants: List[int] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    use_augmentation: bool = True,
    use_balanced_sampling: bool = True
):
    # Train dataset
    train_dataset = RawEMGDataset(
        data_root=data_root,
        participants=train_participants,
        days=[1, 2],
        blocks=[1, 2],
        augment=use_augmentation
    )

    # Create sampler if requested
    if use_balanced_sampling:
        train_sampler = create_balanced_sampler(train_dataset)
        shuffle = False  # Sampler handles shuffling
    else:
        train_sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=shuffle if train_sampler is None else False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Val dataset
    val_loader = None
    if val_participants is not None:
        val_dataset = RawEMGDataset(
            data_root=data_root,
            participants=val_participants,
            days=[1, 2],
            blocks=[1, 2],
            augment=False
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

    return train_loader, val_loader
