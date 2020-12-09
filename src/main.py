"""Mian script to run."""

from dataclasses import dataclass
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.data import AcreCascadeDataModule
from src.model import UNetSegModel
import typer


def experiment(
    data_dir: Path = typer.Option("data", "--data-dir", "-d"),
    train_batch_size: int = typer.Option(16, "--train-batch-size"),
    val_batch_size: int = typer.Option(32, "--val-batch-size"),
    val_pcnt: float = typer.Option(0.2, "--val-pcnt"),
    num_workers: int = typer.Option(4, "--num-workers"),
    learning_rate: float = typer.Option(1.0e-3, "--learning_rate", "-lr"),
    num_layers: int = typer.Option("--num-layers"),
    features_start: int = typer.Option("--features_start"),
    bilinear: bool = typer.Option(False, "--bilinear"),
    log_to_wandb: bool = typer.Option(False, "--log-to-wandb"),
    gpus: int = typer.Option(0, "--gpus"),
    epochs: int = typer.Option(100, "--epochs"),
    use_amp: bool = typer.Option(False, "--use_amp"),
) -> None:
    """Main script."""

    dm = AcreCascadeDataModule(
        data_dir=data_dir,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        val_pcnt=val_pcnt,
        num_workers=num_workers,
    )
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = UNetSegModel(
        num_classes=dm.num_classes,
        num_layers=num_layers,
        features_start=features_start,
        learning_rate=learning_rate,
        bilinear=bilinear,
    )

    # ------------------------
    # 2 SET LOGGER
    # ------------------------
    logger = False
    if log_to_wandb:
        logger = WandbLogger()
        # optional: log model topology
        logger.watch(model.net)

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=gpus,
        logger=logger,
        max_epochs=epochs,
        precision=16 if use_amp else 32,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    experiment()
