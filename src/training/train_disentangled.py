import argparse
import json
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from src.models import DisentangledEMGNet, DisentanglementLoss
from src.utils.data_loader_raw import create_disentanglement_dataloaders


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0
    loss_components = {
        'total': 0,
        'gesture': 0,
        'position': 0,
        'contrastive': 0,
        'adversarial': 0
    }

    correct_gesture = 0
    correct_position = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (emg, gesture_labels, position_labels) in enumerate(pbar):
        # Move to device
        emg = emg.to(device)
        gesture_labels = gesture_labels.to(device)
        position_labels = position_labels.to(device)

        # Note: gesture_labels are 1-6, need to convert to 0-5 for loss
        gesture_labels = gesture_labels - 1

        # Forward pass
        optimizer.zero_grad()

        outputs = model(emg, return_features=True)
        gesture_logits = outputs['gesture_logits']
        position_logits = outputs['position_logits']
        discriminator_logits = outputs['discriminator_logits']
        gesture_features = outputs['gesture_features']

        # Compute loss
        loss, loss_dict = criterion(
            gesture_logits,
            position_logits,
            discriminator_logits,
            gesture_features,
            gesture_labels,
            position_labels
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        for key in loss_components.keys():
            loss_components[key] += loss_dict[key]

        # Accuracy
        _, predicted_gesture = gesture_logits.max(1)
        _, predicted_position = position_logits.max(1)
        total += gesture_labels.size(0)
        correct_gesture += predicted_gesture.eq(gesture_labels).sum().item()
        correct_position += predicted_position.eq(position_labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'gest_acc': 100. * correct_gesture / total,
            'pos_acc': 100. * correct_position / total
        })

    # Average losses
    n_batches = len(train_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components.keys():
        loss_components[key] /= n_batches

    gesture_acc = 100. * correct_gesture / total
    position_acc = 100. * correct_position / total

    return avg_loss, loss_components, gesture_acc, position_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()

    total_loss = 0
    loss_components = {
        'total': 0,
        'gesture': 0,
        'position': 0,
        'contrastive': 0,
        'adversarial': 0
    }

    correct_gesture = 0
    correct_position = 0
    total = 0

    with torch.no_grad():
        for emg, gesture_labels, position_labels in tqdm(val_loader, desc='Validation'):
            emg = emg.to(device)
            gesture_labels = gesture_labels.to(device)
            position_labels = position_labels.to(device)

            # Convert labels
            gesture_labels = gesture_labels - 1

            # Forward pass
            outputs = model(emg, return_features=True)
            gesture_logits = outputs['gesture_logits']
            position_logits = outputs['position_logits']
            discriminator_logits = outputs['discriminator_logits']
            gesture_features = outputs['gesture_features']

            # Compute loss
            loss, loss_dict = criterion(
                gesture_logits,
                position_logits,
                discriminator_logits,
                gesture_features,
                gesture_labels,
                position_labels
            )

            # Track metrics
            total_loss += loss.item()
            for key in loss_components.keys():
                loss_components[key] += loss_dict[key]

            # Accuracy
            _, predicted_gesture = gesture_logits.max(1)
            _, predicted_position = position_logits.max(1)
            total += gesture_labels.size(0)
            correct_gesture += predicted_gesture.eq(gesture_labels).sum().item()
            correct_position += predicted_position.eq(position_labels).sum().item()

    # Average
    n_batches = len(val_loader)
    avg_loss = total_loss / n_batches
    for key in loss_components.keys():
        loss_components[key] /= n_batches

    gesture_acc = 100. * correct_gesture / total
    position_acc = 100. * correct_position / total

    return avg_loss, loss_components, gesture_acc, position_acc


def main():
    parser = argparse.ArgumentParser(description='Train disentangled EMG model')
    parser.add_argument('--data-root', type=str, default='EMG_Arm_Dataset/data',
                       help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--test-participant', type=int, default=8,
                       help='Participant to hold out for testing (1-8)')
    parser.add_argument('--val-participant', type=int, default=7,
                       help='Participant to use for validation (1-8)')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of dataloader workers')
    parser.add_argument('--save-dir', type=str, default='results/disentangled',
                       help='Directory to save results')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    parser.add_argument('--no-balanced-sampling', action='store_true',
                       help='Disable balanced sampling')

    args = parser.parse_args()

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(save_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loaders
    print("\nCreating data loaders...")
    all_participants = list(range(1, 9))
    train_participants = [p for p in all_participants
                         if p not in [args.test_participant, args.val_participant]]

    train_loader, val_loader = create_disentanglement_dataloaders(
        data_root=args.data_root,
        train_participants=train_participants,
        val_participants=[args.val_participant],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_augmentation=not args.no_augmentation,
        use_balanced_sampling=not args.no_balanced_sampling
    )

    print(f"\nTrain participants: {train_participants}")
    print(f"Val participant: {args.val_participant}")
    print(f"Test participant: {args.test_participant} (held out)")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Model
    print("\nCreating model...")
    model = DisentangledEMGNet(
        num_gestures=6,
        num_positions=5,  # Will be mapped to available positions
        encoder_dim=256,
        gesture_dim=128,
        position_dim=64
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss function
    criterion = DisentanglementLoss(
        loss_weights={
            'gesture': 1.0,
            'position': 0.5,
            'contrastive': 0.5,
            'adversarial': 0.3
        },
        temperature=0.5
    )

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Training loop
    print("\nStarting training")

    best_val_gesture_acc = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_gesture_acc': [],
        'val_gesture_acc': [],
        'train_position_acc': [],
        'val_position_acc': [],
        'lr': []
    }

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_loss_components, train_gesture_acc, train_position_acc = \
            train_epoch(model, train_loader, criterion, optimizer, device, epoch)

        # Validate
        val_loss, val_loss_components, val_gesture_acc, val_position_acc = \
            validate(model, val_loader, criterion, device)

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # Print results
        print(f"\nEpoch {epoch} Results:")
        print(f"  Train - Loss: {train_loss:.4f}, "
              f"Gesture Acc: {train_gesture_acc:.2f}%, "
              f"Position Acc: {train_position_acc:.2f}%")
        print(f"  Val   - Loss: {val_loss:.4f}, "
              f"Gesture Acc: {val_gesture_acc:.2f}%, "
              f"Position Acc: {val_position_acc:.2f}%")
        print(f"  Loss components:")
        print(f"    Gesture: {train_loss_components['gesture']:.4f} / {val_loss_components['gesture']:.4f}")
        print(f"    Position: {train_loss_components['position']:.4f} / {val_loss_components['position']:.4f}")
        print(f"    Contrastive: {train_loss_components['contrastive']:.4f} / {val_loss_components['contrastive']:.4f}")
        print(f"    Adversarial: {train_loss_components['adversarial']:.4f} / {val_loss_components['adversarial']:.4f}")
        print(f"  Learning rate: {current_lr:.6f}")

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_gesture_acc'].append(train_gesture_acc)
        history['val_gesture_acc'].append(val_gesture_acc)
        history['train_position_acc'].append(train_position_acc)
        history['val_position_acc'].append(val_position_acc)
        history['lr'].append(current_lr)

        # Save best model
        if val_gesture_acc > best_val_gesture_acc:
            best_val_gesture_acc = val_gesture_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_gesture_acc': val_gesture_acc,
                'val_position_acc': val_position_acc,
            }, save_dir / 'best_model.pth')
            print(f"  Saved best model (val acc: {val_gesture_acc:.2f}%)")

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history,
            }, save_dir / f'checkpoint_epoch{epoch}.pth')

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
    }, save_dir / 'final_model.pth')

    # Save history
    with open(save_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print("\nTraining complete")
    print(f"Best val accuracy: {best_val_gesture_acc:.2f}%")
    print(f"Models saved to: {save_dir}")


if __name__ == "__main__":
    main()
