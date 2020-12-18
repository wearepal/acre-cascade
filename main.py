"""Main script to run."""
from dataclasses import dataclass
from enum import Enum
import json
import logging
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import to_absolute_path
from omegaconf import MISSING, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from torch.nn.modules.loss import CrossEntropyLoss

from src.data import AcreCascadeDataModule
from src.loss import DiceLoss, MultiLoss
from src.model import UNetSegModel
from src.utils import generate_timestamp

Team = Enum("Team", "Bipbip Pead Roseau Weedelec")
Crop = Enum("Crop", "Haricot Mais")
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    data_dir: str = MISSING
    output_dir: str = MISSING
    train_batch_size: int = 16
    val_batch_size: int = 32
    val_pcnt: float = 0.2
    num_workers: int = 4
    lr: float = 1.0e-3
    num_layers: int = 4
    features_start: int = 32
    bilinear: bool = False
    log_offline: bool = False
    gpus: int = 0
    epochs: int = 100
    use_amp: bool = False
    seed: Optional[int] = 47
    download: bool = False
    teams: Optional[List[Team]] = None
    crop: Optional[Crop] = None
    xent_weight: float = 1.0
    dice_weight: float = 1.0


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config")
def main(cfg: Config) -> None:
    """Main script."""
    # Create a submdir within the output dir named with a timestamp
    output_dir = Path(to_absolute_path(cfg.output_dir))
    run_dir = output_dir / generate_timestamp()
    run_dir.mkdir(parents=True)

    # Set all seeds for reproducibility
    if cfg.seed is not None:
        pl.seed_everything(seed=cfg.seed)

    # ------------------------
    # 1 INIT DATAMODULE
    # ------------------------
    dm = AcreCascadeDataModule(
        data_dir=Path(to_absolute_path(cfg.data_dir)),
        train_batch_size=cfg.train_batch_size,
        val_batch_size=cfg.val_batch_size,
        val_pcnt=cfg.val_pcnt,
        num_workers=cfg.num_workers,
        download=cfg.download,
        teams=None if cfg.teams is None else [team.name for team in cfg.teams],  # type: ignore
        crop=None if cfg.crop is None else cfg.crop.name,  # type: ignore
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL
    # ------------------------
    loss_fn = MultiLoss({CrossEntropyLoss(): cfg.xent_weight, DiceLoss(): cfg.dice_weight})
    model = UNetSegModel(
        num_classes=dm.num_classes,
        num_layers=cfg.num_layers,
        features_start=cfg.features_start,
        lr=cfg.lr,
        bilinear=cfg.bilinear,
        loss_fn=loss_fn,
    )

    # ------------------------
    # 3 SET LOGGER
    # ------------------------
    logger = WandbLogger(
        config=OmegaConf.to_container(cfg, resolve=True, enum_to_str=True), offline=cfg.log_offline
    )

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=cfg.gpus,
        logger=logger,
        max_epochs=cfg.epochs,
        precision=16 if cfg.use_amp else 32,
        log_every_n_steps=1,
    )

    # ------------------------
    # 6 START TRAINING
    # ------------------------
    trainer.fit(model=model, datamodule=dm)

    # ------------------------
    # 7 START TESTING
    # ------------------------
    trainer.test(model=model, datamodule=dm)

    # ------------------------
    # 8 SAVE THE SUBMISSION
    # ------------------------
    submission_fp = run_dir / "submission.json"
    with open(submission_fp, "w") as f:
        json.dump(model.submission, f)
    LOGGER.info(f"Submission saved to {submission_fp.resolve()}")


if __name__ == "__main__":
    main()
