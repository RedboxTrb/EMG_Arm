import torch
import torch.nn as nn
import torch.nn.functional as F


class GestureClassificationLoss(nn.Module):
    def __init__(self):
        super(GestureClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)


class PositionClassificationLoss(nn.Module):
    def __init__(self):
        super(PositionClassificationLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        return self.criterion(logits, labels)


class NTXentLoss(nn.Module):
    """NT-Xent contrastive loss - pulls same gesture together, pushes different gestures apart"""

    def __init__(self, temperature: float = 0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, gesture_labels, position_labels):
        batch_size = features.shape[0]
        device = features.device

        features = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        gesture_mask = gesture_labels.unsqueeze(0) == gesture_labels.unsqueeze(1)
        position_mask = position_labels.unsqueeze(0) != position_labels.unsqueeze(1)
        positive_mask = gesture_mask & position_mask
        negative_mask = ~gesture_mask

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        positive_mask = positive_mask & ~self_mask
        negative_mask = negative_mask & ~self_mask

        losses = []
        for i in range(batch_size):
            pos_sims = similarity_matrix[i][positive_mask[i]]
            if len(pos_sims) == 0:
                continue

            neg_sims = similarity_matrix[i][negative_mask[i]]
            if len(neg_sims) == 0:
                continue

            pos_exp = torch.exp(pos_sims)
            neg_exp = torch.exp(neg_sims).sum()

            for pos in pos_exp:
                loss = -torch.log(pos / (pos + neg_exp + 1e-8))
                losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=device)

        return torch.stack(losses).mean()


class AdversarialLoss(nn.Module):
    """Adversarial loss with gradient reversal"""

    def __init__(self, lambda_adversarial: float = 1.0):
        super(AdversarialLoss, self).__init__()
        self.lambda_adversarial = lambda_adversarial
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, discriminator_logits, position_labels, reverse_gradient=True):
        loss = self.criterion(discriminator_logits, position_labels)
        if reverse_gradient:
            loss = -self.lambda_adversarial * loss
        return loss


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_: float = 1.0):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class DisentanglementLoss(nn.Module):
    """Combined loss for disentangled representation learning"""

    def __init__(self, loss_weights: dict = None, temperature: float = 0.5):
        super(DisentanglementLoss, self).__init__()

        if loss_weights is None:
            loss_weights = {
                'gesture': 1.0,
                'position': 0.5,
                'contrastive': 0.5,
                'adversarial': 0.3
            }

        self.loss_weights = loss_weights
        self.gesture_loss_fn = GestureClassificationLoss()
        self.position_loss_fn = PositionClassificationLoss()
        self.contrastive_loss_fn = NTXentLoss(temperature)
        self.adversarial_loss_fn = AdversarialLoss()

    def forward(
        self,
        gesture_logits,
        position_logits,
        discriminator_logits,
        gesture_features,
        gesture_labels,
        position_labels
    ):
        loss_gesture = self.gesture_loss_fn(gesture_logits, gesture_labels)
        loss_position = self.position_loss_fn(position_logits, position_labels)
        loss_contrastive = self.contrastive_loss_fn(
            gesture_features, gesture_labels, position_labels
        )
        loss_adversarial = self.adversarial_loss_fn(
            discriminator_logits, position_labels, reverse_gradient=True
        )

        total_loss = (
            self.loss_weights['gesture'] * loss_gesture +
            self.loss_weights['position'] * loss_position +
            self.loss_weights['contrastive'] * loss_contrastive +
            self.loss_weights['adversarial'] * loss_adversarial
        )

        loss_dict = {
            'total': total_loss.item(),
            'gesture': loss_gesture.item(),
            'position': loss_position.item(),
            'contrastive': loss_contrastive.item(),
            'adversarial': loss_adversarial.item()
        }

        return total_loss, loss_dict
