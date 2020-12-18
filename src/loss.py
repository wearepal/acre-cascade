from typing import Callable, Dict

import torch
import torch.nn as nn
from torch.tensor import Tensor


__all__ = ["Loss", "dice_loss", "DiceLoss", "jaccard_loss", "JaccardLoss", "MultiLoss"]

Loss = Callable[[Tensor, Tensor], Tensor]


def dice_loss(logits: Tensor, mask: Tensor):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        mask: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[mask.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = logits.sigmoid()
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[mask.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits.softmax(dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, mask.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2.0 * intersection / (cardinality + torch.finfo(logits.dtype).eps)).mean()
    return 1.0 - dice_loss


class DiceLoss(nn.Module):
    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        return dice_loss(logits=logits, mask=mask)


def jaccard_loss(logits: Tensor, mask: Tensor) -> Tensor:
    """Computes the Jaccard loss, a.k.a the IoU loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the jaccard loss so we
    return the negated jaccard loss.
    Args:
        mask: a tensor of shape [B, H, W] or [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
    Returns:
        jacc_loss: the Jaccard loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[mask.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = logits.sigmoid()
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[mask.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = logits.softmax(dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, mask.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    union = cardinality - intersection
    jacc_loss = (intersection / (union + torch.finfo(logits.dtype))).mean()
    return 1.0 - jacc_loss


class JaccardLoss(nn.Module):
    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        return jaccard_loss(logits=logits, mask=mask)


class MultiLoss(nn.Module):
    """Combines multiple losses with weighting."""

    def __init__(self, loss_fns: Dict[Loss, float]):
        """loss_fns should be a dictionary of loss functions and their prefactors."""
        self.loss_fns = loss_fns

    def forward(self, logits: Tensor, mask: Tensor) -> Tensor:
        """Adds the loss functions, weighted by the prefactor."""
        loss = logits.new_zeros(())
        for loss_fn, prefact in self.loss_fns.items():
            if prefact != 0:
                loss += prefact * loss_fn(logits, mask)
        return loss
