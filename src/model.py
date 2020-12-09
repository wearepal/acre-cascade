"""Model to train."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

from pl_examples.domain_templates.unet import UNet
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.tensor import Tensor

from src.data import TestBatch, TrainBatch
from src.utils import implements

__all__ = ["SegModel", "UNetSegModel"]


class SegModel(pl.LightningModule, ABC):
    """Semantic Segmentation Module.

    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    """

    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 1.0e-3,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.loss_fn = nn.CrossEntropyLoss()
        self.net = self.build()

    @abstractmethod
    def build() -> nn.Module:
        """Builds the underlying segmentation network"""
        ...

    @implements(nn.Module)
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)

    @implements(pl.LightningModule)
    def configure_optimizers(self) -> Tuple[List[Optimizer], List[_LRScheduler]]:
        opt = Adam(self.net.parameters(), lr=self.learning_rate)
        sch = CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @implements(pl.LightningModule)
    def training_step(self, batch: TrainBatch, batch_index: int) -> Dict[str, Any]:
        img = batch.image.float()
        mask = batch.mask.long()
        out = self(img)
        loss_val = self.loss_fn(out, mask)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    @implements(pl.LightningModule)
    def validation_step(self, batch: TrainBatch, batch_idx: int) -> Dict[str, Any]:
        img = batch.image.float()
        mask = batch.mask.long()
        out = self(img)
        loss_val = self.loss_fn(out, mask)
        return {"val_loss": loss_val}

    @implements(pl.LightningDataModule)
    def test_step(self, batch: TestBatch, batch_idx: int) -> Tensor:
        return self(batch.image)

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs) -> Dict[str, Any]:
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {
            "log": log_dict,
            "val_loss": log_dict["val_loss"],
            "progress_bar": log_dict,
        }


class UNetSegModel(SegModel):
    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        features_start: int,
        bilinear: bool,
        learning_rate: float,
    ):
        super().__init__(
            num_classes=num_classes,
            learning_rate=learning_rate,
        )
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

    @implements(SegModel)
    def build(self) -> None:
        self.net = UNet(
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
