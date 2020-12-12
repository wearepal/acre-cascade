"""Model to train."""

from abc import ABC, abstractmethod
from collections import ChainMap
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from pl_examples.domain_templates.unet import UNet
import pytorch_lightning as pl
from pytorch_lightning.metrics.functional.classification import dice_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.tensor import Tensor

from src.data import TestBatch, TrainBatch
from src.submission_generation import Submission, sample_to_submission
from src.utils import Loss, MultiLoss, implements

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
        lr: float = 1.0e-3,
        loss_fn: Loss = MultiLoss({nn.CrossEntropyLoss(): 1, dice_score: 0.1}),
    ):
        super().__init__()
        self.learning_rate = lr
        self.num_classes = num_classes

        self.loss_fn = loss_fn
        self.net = self.build()
        self.submission: Optional[Dict[str, Any]] = None

    @abstractmethod
    def build(self) -> nn.Module:
        """Builds the underlying segmentation network."""
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
        self.log("train_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return {"loss": loss_val}

    @implements(pl.LightningModule)
    def validation_step(self, batch: TrainBatch, batch_idx: int) -> Dict[str, Any]:
        img = batch.image.float()
        mask = batch.mask.long()
        out = self(img)
        loss_val = self.loss_fn(out, mask)
        self.log("val_loss", loss_val, on_step=True, on_epoch=True, prog_bar=True)
        return {"val_loss": loss_val}

    @implements(pl.LightningModule)
    def test_step(self, batch: TestBatch, batch_idx: int) -> Dict[str, Dict[str, Any]]:
        predicted_mask = self(batch.image).argmax(dim=1)
        submission_i = sample_to_submission(
            filename=batch.filename,
            team_name=batch.team,
            crop_name=batch.crop,
            mask=predicted_mask.cpu().detach().numpy(),
        )
        submission_dict_i = asdict(submission_i)
        return {submission_dict_i.pop("filename"): submission_dict_i}

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: List[Dict[str, Submission]]) -> None:
        self.submission = dict(ChainMap(*outputs))


class UNetSegModel(SegModel):
    """UNet based Segmentation model."""

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        features_start: int,
        bilinear: bool,
        lr: float,
    ):
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        super().__init__(
            num_classes=num_classes,
            lr=lr,
        )

    @implements(SegModel)
    def build(self) -> nn.Module:
        return UNet(
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
