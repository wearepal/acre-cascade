"""Main script to run."""
import json
from pathlib import Path
from typing import List, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from src.data import AcreCascadeDataModule
from src.model import UNetSegModel
from src.utils import generate_timestamp
import typer

app = typer.Typer()


@app.command(context_settings={"ignore_unknown_options": True})
def experiment(
    data_dir: Path = typer.Option("data", "--data-dir", "-d"),
    output_dir: Path = typer.Option("output", "--output", "-o"),
    train_batch_size: int = typer.Option(16, "--train-batch-size"),
    val_batch_size: int = typer.Option(32, "--val-batch-size"),
    val_pcnt: float = typer.Option(0.2, "--val-pcnt"),
    num_workers: int = typer.Option(4, "--num-workers"),
    lr: float = typer.Option(1.0e-3, "--learning-rate", "-lr"),
    num_layers: int = typer.Option(4, "--num-layers"),
    features_start: int = typer.Option(32, "--features-start"),
    bilinear: bool = typer.Option(False, "--bilinear"),
    log_to_wandb: bool = typer.Option(False, "--log-to-wandb"),
    gpus: int = typer.Option(0, "--gpus"),
    epochs: int = typer.Option(100, "--epochs"),
    use_amp: bool = typer.Option(False, "--use-amp"),
    seed: Optional[int] = typer.Option(47, "--seed"),
    download: bool = typer.Option(False, "--download", "-dl"),
    team: Optional[List[str]] = typer.Option(None, "--team"),
    crop: Optional[str] = typer.Option(None, "--crop"),
) -> None:
    """Main script."""
    if not team:  # Needed because typer converts None to an empty tuple
        team = None
    # Create a submdir within the output dir named with a timestamp
    run_dir = output_dir / generate_timestamp()
    run_dir.mkdir(parents=True)

    # Set all seeds for reproducibility
    if seed is not None:
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
        download=download,
        teams=team,  # type: ignore
        crop=crop,  # type: ignore
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
    logger: Union[bool, WandbLogger] = False
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
    typer.echo(f"Submission saved to {submission_fp.resolve()}")


if __name__ == "__main__":
    app()
