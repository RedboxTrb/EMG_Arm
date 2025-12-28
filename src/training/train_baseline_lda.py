"""
Baseline LDA Classifier

Goal: Reproduce paper's baseline results
- Within-position: ~96% accuracy (Table 2)
- Cross-position: ~86% accuracy (Tables 3-4)

Usage:
    conda activate torch
    python src/training/train_baseline_lda.py --within-position
    python src/training/train_baseline_lda.py --cross-position
"""

import sys
sys.path.append('.')

import numpy as np
from pathlib import Path
import argparse
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.data_loader import EMGDataset
from src.utils.preprocessing import EMGPreprocessor


def load_and_preprocess_data(
    dataset: EMGDataset,
    preprocessor: EMGPreprocessor,
    max_samples: int = None
):
    """
    Load dataset and preprocess all trials

    Args:
        dataset: EMGDataset instance
        preprocessor: EMGPreprocessor instance
        max_samples: Maximum number of samples to load (for testing)

    Returns:
        X: Features array (n_samples, n_features)
        y_grasp: Grasp labels (n_samples,)
        y_position: Position labels (n_samples,)
    """
    X_list = []
    y_grasp_list = []
    y_position_list = []

    n_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    print(f"Processing {n_samples} trials...")

    for i in range(n_samples):
        if i % 100 == 0:
            print(f"  Progress: {i}/{n_samples}")

        # Get raw EMG
        emg, grasp, position = dataset[i]
        emg_np = emg.numpy()

        # Preprocess and extract features
        features, _ = preprocessor.process_trial(emg_np)

        # For LDA, we'll use the mean features across all windows in a trial
        # This gives us one feature vector per trial
        mean_features = features.mean(axis=0)

        X_list.append(mean_features)
        y_grasp_list.append(grasp)
        y_position_list.append(position)

    X = np.array(X_list)
    y_grasp = np.array(y_grasp_list)
    y_position = np.array(y_position_list)

    print(f"Preprocessing complete!")
    print(f"  X shape (before normalization): {X.shape}")
    print(f"  X range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"  X mean: {X.mean():.3f}, std: {X.std():.3f}")

    # Z-score normalization across the full dataset
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    print(f"  After normalization - X mean: {X.mean():.3f}, std: {X.std():.3f}")

    print(f"  Unique grasps: {np.unique(y_grasp)}")
    print(f"  Unique positions: {np.unique(y_position)}")
    print(f"  Samples per grasp: {np.bincount(y_grasp)}")

    return X, y_grasp, y_position


def plot_confusion_matrix(cm, class_names, title, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()


def train_within_position(data_root: str, participant: int = 1, test_size: float = 0.2):
    """
    Train and evaluate LDA within a single position

    Target: ~96% accuracy (Table 2 from paper)

    Strategy:
    - Use data from participant 1, ALL days and blocks
    - Filter to position 5 (neutral position)
    - 80/20 train/test split
    """
    print("=" * 60)
    print("WITHIN-POSITION BASELINE (Target: ~96%)")
    print("=" * 60)

    # Load data for one participant, position 5 (neutral)
    # Use ALL days and blocks for more data
    dataset = EMGDataset(
        data_root=data_root,
        participants=[participant],
        days=[1, 2],  # Both days
        blocks=[1, 2],  # Both blocks
        positions=[5]  # Neutral position only
    )

    # Preprocessor (disable normalization - we'll normalize the full dataset later)
    preprocessor = EMGPreprocessor(normalize=False)

    # Load and preprocess
    X, y_grasp, y_position = load_and_preprocess_data(dataset, preprocessor)

    # Split train/test
    n_total = len(X)
    n_train = int(n_total * (1 - test_size))

    # Shuffle
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_grasp[train_idx], y_grasp[test_idx]

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Train LDA
    print("\nTraining LDA...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Evaluate
    y_pred_train = lda.predict(X_train)
    y_pred_test = lda.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc * 100:.2f}%")
    print(f"  Test accuracy: {test_acc * 100:.2f}%")
    print(f"  Target accuracy: ~96%")

    # Classification report (only for classes present in test set)
    print("\nClassification Report:")
    unique_classes = sorted(np.unique(y_test))
    target_names = [f"Grasp {i}" for i in unique_classes]
    print(classification_report(y_test, y_pred_test,
                                labels=unique_classes,
                                target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    class_names = [f"G{i}" for i in unique_classes]
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"Within-Position LDA (Acc: {test_acc*100:.1f}%)",
        save_path="results/baseline_lda_within_position.png"
    )

    return test_acc


def train_cross_position(data_root: str, participant: int = 1):
    """
    Train and evaluate LDA across positions

    Target: ~86% accuracy (Tables 3-4 from paper)

    Strategy:
    - Train on position 5 (neutral)
    - Test on all other positions
    """
    print("=" * 60)
    print("CROSS-POSITION BASELINE (Target: ~86%)")
    print("=" * 60)

    # Preprocessor (disable normalization - we'll normalize the full dataset later)
    preprocessor = EMGPreprocessor(normalize=False)

    # Train data: Position 5 only, ALL days and blocks
    train_dataset = EMGDataset(
        data_root=data_root,
        participants=[participant],
        days=[1, 2],  # Both days
        blocks=[1, 2],  # Both blocks
        positions=[5]
    )

    print("Loading training data (Position 5)...")
    X_train, y_train, _ = load_and_preprocess_data(train_dataset, preprocessor)

    # Test data: All other positions, ALL days and blocks
    # Note: Based on dataset examination, positions are [2, 4, 5, 6, 8] not [1-9]
    # So we'll use all positions except 5
    test_positions = [2, 4, 6, 8]  # Exclude position 5
    test_dataset = EMGDataset(
        data_root=data_root,
        participants=[participant],
        days=[1, 2],  # Both days
        blocks=[1, 2],  # Both blocks
        positions=test_positions
    )

    print(f"\nLoading test data (Positions {test_positions})...")
    X_test, y_test, y_test_pos = load_and_preprocess_data(test_dataset, preprocessor)

    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size: {len(X_test)}")

    # Train LDA
    print("\nTraining LDA on Position 5...")
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    # Evaluate
    y_pred_train = lda.predict(X_train)
    y_pred_test = lda.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"\nResults:")
    print(f"  Train accuracy (Pos 5): {train_acc * 100:.2f}%")
    print(f"  Test accuracy (Other Pos): {test_acc * 100:.2f}%")
    print(f"  Target accuracy: ~86%")
    print(f"  Performance drop: {(train_acc - test_acc) * 100:.2f}%")

    # Per-position accuracy
    print("\nPer-Position Accuracy:")
    for pos in test_positions:
        mask = (y_test_pos == pos)
        if mask.sum() > 0:
            pos_acc = accuracy_score(y_test[mask], y_pred_test[mask])
            print(f"  Position {pos}: {pos_acc * 100:.2f}% ({mask.sum()} samples)")

    # Classification report (only for classes present in test set)
    print("\nClassification Report:")
    unique_classes = sorted(np.unique(y_test))
    target_names = [f"Grasp {i}" for i in unique_classes]
    print(classification_report(y_test, y_pred_test,
                                labels=unique_classes,
                                target_names=target_names))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred_test)
    class_names = [f"G{i}" for i in unique_classes]
    plot_confusion_matrix(
        cm,
        class_names=class_names,
        title=f"Cross-Position LDA (Acc: {test_acc*100:.1f}%)",
        save_path="results/baseline_lda_cross_position.png"
    )

    return test_acc


def main():
    parser = argparse.ArgumentParser(description="Train baseline LDA classifier")
    parser.add_argument('--within-position', action='store_true',
                       help='Train within-position baseline')
    parser.add_argument('--cross-position', action='store_true',
                       help='Train cross-position baseline')
    parser.add_argument('--participant', type=int, default=1,
                       help='Participant ID (1-8)')
    parser.add_argument('--data-root', type=str, default='EMG_Arm_Dataset/data',
                       help='Path to dataset')

    args = parser.parse_args()

    # Create results directory
    Path("results").mkdir(exist_ok=True)

    # Set random seed
    np.random.seed(42)

    if args.within_position:
        test_acc = train_within_position(args.data_root, args.participant)
        print(f"\n{'='*60}")
        print(f"WITHIN-POSITION FINAL ACCURACY: {test_acc*100:.2f}%")
        print(f"{'='*60}")

    if args.cross_position:
        test_acc = train_cross_position(args.data_root, args.participant)
        print(f"\n{'='*60}")
        print(f"CROSS-POSITION FINAL ACCURACY: {test_acc*100:.2f}%")
        print(f"{'='*60}")

    if not args.within_position and not args.cross_position:
        print("Please specify --within-position or --cross-position")
        print("Example: python src/training/train_baseline_lda.py --within-position")


if __name__ == "__main__":
    main()
