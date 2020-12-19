"""Model to train."""

from abc import ABC, abstractmethod
from collections import ChainMap
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

from pl_examples.domain_templates.unet import UNet
import pytorch_lightning as pl
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.tensor import Tensor
from torchvision.transforms.functional import to_pil_image
import wandb

from src.data import CLASS_LABELS, TestBatch, TestTuple, TrainTuple
from src.loss import Loss
from src.submission_generation import Submission, sample_to_submission
from src.utils import implements

__all__ = ["SegModel", "UNetSegModel"]


class SegModel(pl.LightningModule, ABC):
    """The base class for Lightning-based Semantic Segmentation Modules."""

    def __init__(
        self,
        num_classes: int,
        lr: float = 1.0e-3,
        loss_fn: Loss = nn.CrossEntropyLoss(),
        T_max: int = 10,
    ):
        super().__init__()
        self.learning_rate = lr
        self.num_classes = num_classes
        self.loss_fn = loss_fn
        self.T_max = T_max

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
        sch = CosineAnnealingLR(opt, T_max=self.T_max)
        return [opt], [sch]

    @implements(pl.LightningModule)
    def training_step(self, batch: TrainTuple, batch_index: int) -> Tensor:
        img = batch[0].float()
        mask = batch[1].long()
        out = self(img)
        loss = self.loss_fn(out, mask)
        self.log("train_loss", loss, prog_bar=True, logger=False)
        logging_dict: Dict[str, Any] = {"training/loss": loss}

        if batch_index % 50 == 0:
            mask_list = []
            for _img, _mask, _out in zip(img, mask, out):
                mask_img = wandb.Image(
                    to_pil_image(_img),
                    masks={
                        "predictions": {
                            "mask_data": _out.argmax(dim=0).cpu().numpy(),
                            "class_labels": CLASS_LABELS,
                        },
                        "groud_truth": {
                            "mask_data": _mask.t().cpu().numpy(),
                            "class_labels": CLASS_LABELS,
                        },
                    },
                )
                mask_list.append(mask_img)
                logging_dict["training/predictions"] = mask_list
        self.logger.experiment.log(logging_dict)

        return loss

    @implements(pl.LightningModule)
    def validation_step(self, batch: TrainTuple, batch_idx: int) -> Tensor:
        img = batch[0].float()
        mask = batch[1].long()
        out = self(img)
        loss = self.loss_fn(out, mask)
        self.log("val_loss", loss, prog_bar=True, logger=False)
        logging_dict: Dict[str, Any] = {"validation/loss": loss}

        if batch_idx == 0:
            mask_list = []
            for _img, _mask, _out in zip(img, mask, out):
                mask_img = wandb.Image(
                    to_pil_image(_img),
                    masks={
                        "predictions": {
                            "mask_data": _out.argmax(dim=0).cpu().numpy(),
                            "class_labels": CLASS_LABELS,
                        },
                        "groud_truth": {
                            "mask_data": _mask.t().cpu().numpy(),
                            "class_labels": CLASS_LABELS,
                        },
                    },
                )
                mask_list.append(mask_img)
            logging_dict["validation/predictions"] = mask_list
        self.logger.experiment.log(logging_dict)

        return loss

    @implements(pl.LightningModule)
    def test_step(self, batch: TestBatch, batch_idx: int) -> Dict[str, Dict[str, Any]]:
        """"Predict the mask of a single test image and prepare it for submission."""
        predicted_mask = self(batch[0]).argmax(dim=1)
        submission_i = sample_to_submission(
            filename=batch[3][0],
            team_name=batch[1][0],
            crop_name=batch[2][0],
            mask=predicted_mask.squeeze().cpu().detach().numpy(),
        )
        submission_dict_i = asdict(submission_i)
        return {submission_dict_i.pop("filename"): submission_dict_i}

    @implements(pl.LightningModule)
    def test_epoch_end(self, outputs: List[Dict[str, Submission]]) -> None:
        """Collate the list of dictionaries produced by test_step into a single dictionary."""
        self.submission = dict(ChainMap(*outputs))


class UNetSegModel(SegModel):
    """Segmentation Module using U-Net as the underlying model."""

    def __init__(
        self,
        num_classes: int,
        num_layers: int,
        features_start: int,
        bilinear: bool,
        lr: float,
        loss_fn: Loss = nn.CrossEntropyLoss(),
        T_max: int = 10,
    ):
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        super().__init__(num_classes=num_classes, lr=lr, loss_fn=loss_fn, T_max=T_max)

    @implements(SegModel)
    def build(self) -> nn.Module:
        return UNet(
            num_classes=self.num_classes,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
