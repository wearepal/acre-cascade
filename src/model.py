"""Model to train."""

import pytorch_lightning as pl
import torch
from pl_examples.domain_templates.unet import UNet
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from src.utils import implements


class SegModel(pl.LightningModule):
    """Semantic Segmentation Module.

    This is a basic semantic segmentation module implemented with Lightning.
    It uses CrossEntropyLoss as the default loss function. May be replaced with
    other loss functions as required.
    It is specific to KITTI dataset i.e. dataloaders are for KITTI
    and Normalize transform uses the mean and standard deviation of this dataset.
    It uses the FCN ResNet50 model as an example.
    Adam optimizer is used along with Cosine Annealing learning rate scheduler.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        lr: float,
        num_layers: int,
        features_start: int,
        bilinear: bool,
        **kwargs
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.lr = lr
        self.num_layers = num_layers
        self.features_start = features_start
        self.bilinear = bilinear

        self.net = UNet(
            num_classes=19,
            num_layers=self.num_layers,
            features_start=self.features_start,
            bilinear=self.bilinear,
        )
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.35675976, 0.37380189, 0.3764753],
                    std=[0.32064945, 0.32098866, 0.32325324],
                ),
            ]
        )
        self.trainset = DATA_PLACEHOLDER(
            self.data_path, split="train", transform=self.transform
        )
        self.validset = DATA_PLACEHOLDER(
            self.data_path, split="valid", transform=self.transform
        )

    @implements(nn.Module)
    def forward(self, x):
        return self.net(x)

    @implements(pl.LightningModule)
    def training_step(self, batch, batch_nb):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        log_dict = {"train_loss": loss_val}
        return {"loss": loss_val, "log": log_dict, "progress_bar": log_dict}

    @implements(pl.LightningModule)
    def validation_step(self, batch, batch_idx):
        img, mask = batch
        img = img.float()
        mask = mask.long()
        out = self(img)
        loss_val = F.cross_entropy(out, mask, ignore_index=250)
        return {"val_loss": loss_val}

    @implements(pl.LightningModule)
    def validation_epoch_end(self, outputs):
        loss_val = torch.stack([x["val_loss"] for x in outputs]).mean()
        log_dict = {"val_loss": loss_val}
        return {
            "log": log_dict,
            "val_loss": log_dict["val_loss"],
            "progress_bar": log_dict,
        }

    @implements(pl.LightningModule)
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
        return [opt], [sch]

    @implements(pl.LightningModule)
    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True)

    @implements(pl.LightningModule)
    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, shuffle=False)
