"""Mian script to run."""

from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import typer

from src.data import AcreCascadeDataModule
from src.model import UNetSegModel


app = typer.Typer()


@app.command(context_settings={"ignore_unknown_options": True})
def experiment(
    data_dir: Path = typer.Option("data", "--data-dir", "-d"),
    train_batch_size: int = typer.Option(16, "--train-batch-size"),
    val_batch_size: int = typer.Option(32, "--val-batch-size"),
    val_pcnt: float = typer.Option(0.2, "--val-pcnt"),
    num_workers: int = typer.Option(4, "--num-workers"),
    lr: float = typer.Option(1.0e-3, "--learning-rate", "-lr"),
    num_layers: int = typer.Option("--num-layers"),
    features_start: int = typer.Option("--features-start"),
    bilinear: bool = typer.Option(False, "--bilinear"),
    log_to_wandb: bool = typer.Option(False, "--log-to-wandb"),
    gpus: int = typer.Option(0, "--gpus"),
    epochs: int = typer.Option(100, "--epochs"),
    use_amp: bool = typer.Option(False, "--use-amp"),
    seed: int = typer.Option(42, "--seed"),
) -> None:
    """Main script."""
    # Set all seeds for reproducibility
    pl.seed_everything(seed=seed)
    # ------------------------
    # 1 INIT DATAMODULE
    # ------------------------
    dm = AcreCascadeDataModule(
        data_dir=data_dir,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        val_pcnt=val_pcnt,
        num_workers=num_workers,
    )
    # ------------------------
    # 2 INIT LIGHTNING MODEL
    # ------------------------
    model = UNetSegModel(
        num_classes=dm.num_classes,
        num_layers=num_layers,
        features_start=features_start,
        lr=lr,
        bilinear=bilinear,
    )

    # ------------------------
    # 3 SET LOGGER
    # ------------------------
    logger = False
    if log_to_wandb:
        logger = WandbLogger()
        # optional: log model topology
        logger.watch(model.net)

    # ------------------------
    # 4 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=gpus,
        logger=logger,
        max_epochs=epochs,
        precision=16 if use_amp else 32,
    )

    # ------------------------
    # 6 START TRAINING
    # ------------------------
    trainer.fit(model=model, datamodule=dm)

    # ------------------------
    # 7 START TESTING
    # ------------------------
    trainer.test(model=model, datamodule=dm)


if __name__ == "__main__":
    app()
