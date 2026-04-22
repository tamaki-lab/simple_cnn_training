import torch
import lightning.pytorch as pl
from lightning.pytorch.plugins import TorchSyncBatchNorm


# from cfg import ArgParse
from logger import configure_logger_pl
from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel

import hydra
from omegaconf import DictConfig

CONFIG_PATH = "config"
CONFIG_NAME = "config"


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    assert torch.cuda.is_available()

    # cfg = ArgParse.get()

    loggers, exp_name = configure_logger_pl(
        command_line_cfg=cfg,
        model_name=cfg.model.model_name,
        disable_logging=cfg.disable_comet,
        save_dir=cfg.log_dirs.comet_log_dir,
    )
    data_module = TrainValDataModule(
        command_line_cfg=cfg,
        dataset_name=cfg.dataset.dataset_name,
    )
    model_lightning = SimpleLightningModel(
        command_line_args=cfg,
        n_classes=data_module.n_classes,
        exp_name=exp_name
    )

    callbacks = configure_callbacks()

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        devices=cfg.GPU.devices,
        accelerator="gpu",
        strategy="auto",
        max_epochs=cfg.training.num_epochs,
        logger=loggers,
        log_every_n_steps=cfg.training.log_interval_steps,
        accumulate_grad_batches=cfg.optimizer.grad_accum,
        num_sanity_val_steps=0,
        # precision="16-true",  # for FP16 training, use with caution for nan/inf
        # fast_dev_run=True, # only for debug
        # fast_dev_run=5,  # only for debug
        # limit_train_batches=15,  # only for debug
        # limit_val_batches=15,  # only for debug
        callbacks=callbacks,
        plugins=[TorchSyncBatchNorm()],
        # profiler="simple",
    )

    trainer.fit(
        model=model_lightning,
        datamodule=data_module,
        ckpt_path=cfg.checkpoint_file.checkpoint_to_resume,
    )


if __name__ == "__main__":
    main()
