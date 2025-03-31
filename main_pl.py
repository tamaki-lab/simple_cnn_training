
from copy import deepcopy
from argparse import Namespace
import torch
import lightning.pytorch as pl
import yaml
import os
from lightning.pytorch.plugins import TorchSyncBatchNorm


import argparse
from logger import configure_logger_pl
from callback import configure_callbacks
from dataset import TrainValDataModule
from model import SimpleLightningModel


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def merge_configs(default_cfg, override_cfg):
    merged = deepcopy(default_cfg)
    for k, v in override_cfg.items():
        if isinstance(v, dict) and k in merged:
            merged[k] = merge_configs(merged[k], v)
        else:
            merged[k] = v
    return merged


def dict_to_namespace(d):
    return Namespace(**d)


def get_args():
    # Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    return parser.parse_args()


def main(args):
    assert torch.cuda.is_available()
    default_config = load_yaml("config/default_config.yaml")
    config_path = args.model.split('_')[0] + '/' + args.model + '.yaml'
    config = load_yaml(os.path.join('config', config_path))
    merged_config = merge_configs(default_config, config)
    args = dict_to_namespace(merged_config)

    loggers, exp_name = configure_logger_pl(
        model_name=args.model_name,
        disable_logging=args.disable_comet,
        save_dir=args.comet_log_dir,
    )
    data_module = TrainValDataModule(
        command_line_args=args,
        dataset_name=args.dataset_name,
    )
    model_lightning = SimpleLightningModel(
        command_line_args=args,
        n_classes=data_module.n_classes,
        exp_name=exp_name
    )

    callbacks = configure_callbacks()

    # https://lightning.ai/docs/pytorch/stable/common/trainer.html
    # https://lightning.ai/docs/pytorch/stable/common/trainer.html#trainer-flags
    trainer = pl.Trainer(
        devices=args.devices,
        accelerator="gpu",
        strategy="auto",
        max_epochs=args.num_epochs,
        logger=loggers,
        log_every_n_steps=args.log_interval_steps,
        accumulate_grad_batches=args.grad_accum,
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
        ckpt_path=args.checkpoint_to_resume,
    )


if __name__ == "__main__":
    main(get_args())
