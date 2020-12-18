"""Utility functions."""
from abc import ABC
import time
from typing import Any, Callable, Dict, Literal, Type, TypeVar

from pytorch_lightning.metrics.functional.classification import dice_score
from torch import Tensor

__all__ = ["implements", "generate_timestamp", "MultiLoss", "DiceLoss"]

_F = TypeVar("_F", bound=Callable[..., Any])


class implements:  # pylint: disable=invalid-name
    """Mark a function as implementing an interface."""

    def __init__(self, interface: Type):
        """Instantiate the decorator.

        Args:
            interface: the interface that is implemented
        """
        self.interface = interface

    def __call__(self, func: _F) -> _F:
        """Take a function and return it unchanged."""
        super_method = getattr(self.interface, func.__name__, None)
        assert super_method is not None, f"'{func.__name__}' does not exist in {self.interface}"
        return func


def generate_timestamp() -> str:
    """Make a TimeStamp."""
    return str(int(time.time() - 1601376237))


Loss = Callable[[Tensor, Tensor], Tensor]


class DiceLoss:
    def __init__(
        self,
        bg: bool = False,
        nan_score: float = 0.0,
        no_fg_score: float = 0.0,
        reduction: Literal["elementwise_mean", "sum", "none"] = "elementwise_mean",
    ) -> None:
        self.bg = bg
        self.nan_score = nan_score
        self.no_fg_score = no_fg_score
        self.reduction = reduction

    super().__init__()

    def __call__(self, logits: Tensor, mask: Tensor) -> Tensor:
        probs = logits.softmax(dim=1)
        return dice_score(
            pred=probs,
            target=mask,
            bg=self.bg,
            nan_score=self.nan_score,
            no_fg_score=self.no_fg_score,
            reduction=self.reduction,
        )


class MultiLoss:
    """Combines multiple losses with weighting."""

    def __init__(self, loss_fns: Dict[Loss, float]):
        """loss_fns should be a dictionary of loss functions and their prefactors."""
        self.loss_fns = loss_fns

    def __call__(self, logits: Tensor, mask: Tensor) -> Tensor:
        """Adds the loss functions, weighted by the prefactor."""
        loss = logits.new_zeros(())
        for loss_fn, prefact in self.loss_fns.items():
            if prefact != 0:
                loss += prefact * loss_fn(logits, mask)
        return loss
