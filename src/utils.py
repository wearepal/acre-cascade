"""Utility functions."""
import time
from typing import Any, Callable, Dict, Type, TypeVar

from torch import Tensor

__all__ = ["implements", "generate_timestamp", "MultiLoss"]

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
    return str(int(time.time() - 1601376237))


Loss = Callable[[Tensor, Tensor], Tensor]


class MultiLoss:
    """Combines multiple losses with weighting."""

    def __init__(self, loss_fns: Dict[Loss, float]):
        """loss_fns should be a dictionary of loss functions and their prefactors."""

        self.loss_fns = loss_fns

    def __call__(self, logits: Tensor, mask: Tensor) -> Tensor:
        loss = logits.new_zeros(())
        for loss_fn, prefact in self.loss_fns.items():
            loss += prefact * loss_fn(logits, mask)
        return loss
