"""
EMG Signal Preprocessing

Pipeline:
1. Bandpass filter: 20-450 Hz (4th-order Butterworth)
2. Notch filter: 50/60 Hz (powerline noise)
3. Windowing: 128ms window, 50ms stride (60% overlap)
4. Feature extraction: Hudgins features (MAV, ZC, SSC, WL)
5. Normalization: Z-score per channel
"""

import numpy as np
import torch
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch
from typing import Tuple, Optional


class EMGPreprocessor:
    """
    EMG signal preprocessing and feature extraction

    Args:
        fs: Sampling frequency (Hz)
        window_size_ms: Window size in milliseconds
        stride_ms: Window stride in milliseconds
        bandpass_low: Low cutoff for bandpass filter (Hz)
        bandpass_high: High cutoff for bandpass filter (Hz)
        notch_freq: Notch filter frequency for powerline noise (Hz)
        normalize: Whether to apply z-score normalization
    """

    def __init__(
        self,
        fs: int = 2000,
        window_size_ms: int = 128,
        stride_ms: int = 50,
        bandpass_low: float = 20.0,
        bandpass_high: float = 450.0,
        notch_freq: float = 60.0,  # 60 Hz for US, 50 Hz for Europe
        normalize: bool = True
    ):
        self.fs = fs
        self.window_size = int(window_size_ms * fs / 1000)  # samples
        self.stride = int(stride_ms * fs / 1000)  # samples
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.normalize = normalize

        # Design filters
        self.bp_b, self.bp_a = self._design_bandpass_filter()
        self.notch_b, self.notch_a = self._design_notch_filter()

        print(f"EMG Preprocessor initialized:")
        print(f"  Sampling rate: {fs} Hz")
        print(f"  Window: {window_size_ms} ms ({self.window_size} samples)")
        print(f"  Stride: {stride_ms} ms ({self.stride} samples)")
        print(f"  Overlap: {(1 - stride_ms/window_size_ms)*100:.0f}%")
        print(f"  Bandpass: {bandpass_low}-{bandpass_high} Hz")
        print(f"  Notch: {notch_freq} Hz")

    def _design_bandpass_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design 4th-order Butterworth bandpass filter"""
        nyquist = self.fs / 2
        low = self.bandpass_low / nyquist
        high = self.bandpass_high / nyquist
        b, a = butter(4, [low, high], btype='band')
        return b, a

    def _design_notch_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Design notch filter for powerline noise"""
        nyquist = self.fs / 2
        freq = self.notch_freq / nyquist
        Q = 30  # Quality factor
        b, a = iirnotch(freq, Q)
        return b, a

    def filter_signal(self, emg: np.ndarray) -> np.ndarray:
        """
        Apply bandpass and notch filters to EMG signal

        Args:
            emg: EMG data, shape (n_channels, n_samples)

        Returns:
            Filtered EMG data, same shape
        """
        n_channels = emg.shape[0]
        filtered = np.zeros_like(emg)

        for ch in range(n_channels):
            # Bandpass filter
            temp = filtfilt(self.bp_b, self.bp_a, emg[ch])
            # Notch filter
            filtered[ch] = filtfilt(self.notch_b, self.notch_a, temp)

        return filtered

    def extract_hudgins_features(self, window: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Extract Hudgins time-domain features from a window

        Features per channel:
        - MAV: Mean Absolute Value
        - ZC: Zero Crossings
        - SSC: Slope Sign Changes
        - WL: Waveform Length

        Args:
            window: EMG window, shape (n_channels, window_size)
            threshold: Threshold for ZC and SSC

        Returns:
            Features, shape (n_channels * 4,)
        """
        n_channels = window.shape[0]
        features = []

        for ch in range(n_channels):
            x = window[ch]

            # 1. Mean Absolute Value (MAV)
            mav = np.mean(np.abs(x))

            # 2. Zero Crossings (ZC)
            zc = 0
            for i in range(len(x) - 1):
                if (x[i] * x[i+1] < 0) and (np.abs(x[i] - x[i+1]) >= threshold):
                    zc += 1

            # 3. Slope Sign Changes (SSC)
            ssc = 0
            for i in range(1, len(x) - 1):
                if ((x[i] - x[i-1]) * (x[i] - x[i+1]) >= threshold):
                    ssc += 1

            # 4. Waveform Length (WL)
            wl = np.sum(np.abs(np.diff(x)))

            features.extend([mav, zc, ssc, wl])

        return np.array(features)

    def window_signal(self, emg: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Segment signal into overlapping windows

        Args:
            emg: Filtered EMG, shape (n_channels, n_samples)

        Returns:
            windows: Array of windows, shape (n_windows, n_channels, window_size)
            n_windows: Number of windows
        """
        n_channels, n_samples = emg.shape
        windows = []

        start = 0
        while start + self.window_size <= n_samples:
            window = emg[:, start:start + self.window_size]
            windows.append(window)
            start += self.stride

        windows = np.array(windows)
        return windows, len(windows)

    def extract_features_from_windows(self, windows: np.ndarray) -> np.ndarray:
        """
        Extract Hudgins features from all windows

        Args:
            windows: Array of windows, shape (n_windows, n_channels, window_size)

        Returns:
            features: Array of features, shape (n_windows, n_features)
                     where n_features = n_channels * 4
        """
        n_windows = windows.shape[0]
        features = []

        for i in range(n_windows):
            feat = self.extract_hudgins_features(windows[i])
            features.append(feat)

        features = np.array(features)

        # Normalize if requested
        if self.normalize:
            # Z-score normalization per feature
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)

        return features

    def process_trial(self, emg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline for one trial

        Args:
            emg: Raw EMG data, shape (n_channels, n_samples)

        Returns:
            features: Extracted features, shape (n_windows, n_features)
            windows: Windowed raw data, shape (n_windows, n_channels, window_size)
        """
        # 1. Filter
        filtered = self.filter_signal(emg)

        # 2. Window
        windows, n_windows = self.window_signal(filtered)

        # 3. Extract features
        features = self.extract_features_from_windows(windows)

        return features, windows


class EMGTransform:
    """
    PyTorch transform for EMG data preprocessing

    Usage:
        transform = EMGTransform(return_features=True)
        dataset = EMGDataset(data_root, transform=transform)
    """

    def __init__(
        self,
        fs: int = 2000,
        window_size_ms: int = 128,
        stride_ms: int = 50,
        return_features: bool = True,
        return_windows: bool = False
    ):
        self.preprocessor = EMGPreprocessor(
            fs=fs,
            window_size_ms=window_size_ms,
            stride_ms=stride_ms
        )
        self.return_features = return_features
        self.return_windows = return_windows

    def __call__(self, emg_tensor: torch.Tensor) -> torch.Tensor:
        """
        Apply preprocessing to EMG tensor

        Args:
            emg_tensor: Raw EMG, shape (n_channels, n_samples)

        Returns:
            Preprocessed tensor (features or windows)
        """
        # Convert to numpy
        emg_np = emg_tensor.numpy()

        # Process
        features, windows = self.preprocessor.process_trial(emg_np)

        # Return requested output
        if self.return_features:
            return torch.from_numpy(features).float()
        elif self.return_windows:
            return torch.from_numpy(windows).float()
        else:
            # Return filtered signal
            filtered = self.preprocessor.filter_signal(emg_np)
            return torch.from_numpy(filtered).float()


if __name__ == "__main__":
    # Test preprocessing
    print("Testing EMG Preprocessing")
    print("=" * 60)

    # Create dummy EMG data (16 channels, 5 seconds @ 2000 Hz = 10,000 samples)
    fs = 2000
    duration = 5  # seconds
    n_samples = fs * duration
    n_channels = 16

    # Generate dummy data with some signal
    t = np.linspace(0, duration, n_samples)
    emg_dummy = np.zeros((n_channels, n_samples))
    for ch in range(n_channels):
        # Mix of low and high frequencies
        emg_dummy[ch] = (
            0.5 * np.sin(2 * np.pi * 50 * t) +  # 50 Hz (powerline noise)
            0.3 * np.sin(2 * np.pi * 100 * t) +  # 100 Hz
            0.2 * np.random.randn(n_samples)  # Noise
        )

    print(f"\nDummy EMG shape: {emg_dummy.shape}")
    print(f"Dummy EMG range: [{emg_dummy.min():.3f}, {emg_dummy.max():.3f}]")

    # Initialize preprocessor
    preprocessor = EMGPreprocessor(
        fs=fs,
        window_size_ms=128,
        stride_ms=50
    )

    # Process
    features, windows = preprocessor.process_trial(emg_dummy)

    print(f"\nAfter preprocessing:")
    print(f"  Features shape: {features.shape}")  # (n_windows, 64)
    print(f"  Windows shape: {windows.shape}")  # (n_windows, 16, 256)
    print(f"  Expected windows: {(10000 - 256) // 100 + 1} (approx)")
    print(f"  Features per window: {features.shape[1]} (16 channels × 4 features)")
    print(f"  Feature stats: mean={features.mean():.3f}, std={features.std():.3f}")

    # Test PyTorch transform
    print("\n" + "=" * 60)
    print("Testing PyTorch Transform")
    print("=" * 60)

    transform = EMGTransform(return_features=True)
    emg_tensor = torch.from_numpy(emg_dummy).float()

    transformed = transform(emg_tensor)
    print(f"\nTransformed shape: {transformed.shape}")
    print(f"Transformed dtype: {transformed.dtype}")
