"""Mian script to run."""

from dataclasses import dataclass

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from src.model import SegModel


class HParams(dataclass):
    """Haparams. to replace with typer."""

    log_wandb = False


def main():
    """Main script."""
    hparams = HParams()

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = SegModel(**vars(hparams))

    # ------------------------
    # 2 SET LOGGER
    # ------------------------
    logger = False
    if hparams.log_wandb:
        logger = WandbLogger()

        # optional: log model topology
        logger.watch(model.net)

    # ------------------------
    # 3 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        gpus=hparams.gpus,
        logger=logger,
        max_epochs=hparams.epochs,
        accumulate_grad_batches=hparams.grad_batches,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_amp else 32,
    )

    # ------------------------
    # 5 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == "__main__":
    main()
