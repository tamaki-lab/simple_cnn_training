import argparse
import os


import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint


from utils import compute_topk_accuracy
from model import configure_model, ModelConfig
from setup import configure_optimizer, configure_scheduler
from omegaconf import DictConfig


class SimpleLightningModel(pl.LightningModule):
    """A simple lightning module

        see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#methods
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#properties
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#hooks

        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#method-order
    """

    def __init__(
            self,
            command_line_args: DictConfig,
            n_classes: int,
            exp_name: str,
    ):
        """constructor

        see
        https://lightning.ai/docs/pytorch/stable/starter/style_guide.html#init

        Args:
            command_line_args (argparse): args
            n_classes (int): number of categories
            exp_name (str): experiment name of comet.ml
        """
        super().__init__()
        self.args = command_line_args
        self.exp_name = exp_name

        self.model = configure_model(ModelConfig(
            model_name=self.args.model.model_name,
            use_pretrained=self.args.model.use_pretrained,
            torch_home=self.args.model.torch_home,
            n_classes=n_classes,
        ))

        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#save-hyperparameters
        self.save_hyperparameters()

    def configure_optimizers(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """

        optimizer = configure_optimizer(
            optimizer_name=self.args.optimizer.optimizer_name,
            lr=self.args.optimizer.lr,
            weight_decay=self.args.optimizer.weight_decay,
            momentum=self.args.optimizer.momentum,
            model_params=self.model.parameters()
        )
        scheduler = configure_scheduler(
            optimizer=optimizer,
            use_scheduler=self.args.optimizer.use_scheduler
        )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def configure_callbacks(self):
        """see
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-callbacks
        """

        save_checkpoint_dir = os.path.join(self.args.checkpoint_file.save_checkpoint_dir, self.exp_name)
        if self.global_rank == 0:
            os.makedirs(save_checkpoint_dir, exist_ok=True)

        checkpoint_callbacks = [
            ModelCheckpoint(
                dirpath=save_checkpoint_dir,
                monitor="val_top1",
                mode="max",  # larger is better
                save_top_k=2,
                filename="epoch{epoch}_step{step}_acc={val_top1:.2f}",
                auto_insert_metric_name=False,
            ),
        ]

        return checkpoint_callbacks

    def log_train_loss_top15(self, loss, top1, top5, batch_size):
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-epoch-level-metrics
        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log_dict
        self.log_dict(
            {
                "train_loss": loss.item(),
                "train_top1": top1,
            },
            prog_bar=True,  # show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

        # https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.log
        self.log(
            "train_top5",
            top5,
            prog_bar=False,  # do not show on the progress bar
            on_step=True,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,
            batch_size=batch_size,
        )

    def training_step(self, batch, batch_idx):
        """a single training step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Returns:
            tensor: loss (used for backward by lightning)

        Note:
            DO NOT USE .to() or model.train() here
                (automatically send to multi-GPUs)
            DO NOT USE loss.backward() here
                (automatically performed by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-loop
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.training_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)
        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_train_loss_top15(loss, top1, top5, batch_size)

        return loss

    def log_val_loss_top15(self, loss, top1, top5, batch_size):
        self.log_dict(
            {
                "val_loss": loss.item(),
                "val_top1": top1,
                "val_top5": top5,
            },
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            rank_zero_only=False,
            sync_dist=True,  # sync log metrics for validation
            batch_size=batch_size,
        )

    def validation_step(self, batch, batch_idx):
        """a single validation step for a batch

        Args:
            batch (Tuple[tensor]): a batch of data samples and labels
                (actual type depends on the dataloader)
            batch_idx (int): index of the batch in the epoch

        Note:
            DO NOT USE .to() or model.eval() here
                (automatically send to multi-GPUs)
            DO NOT USE with torch.no_grad() here
                (automatically handled by lightning)
            see
                https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation
                https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningModule.html#lightning.pytorch.core.LightningModule.validation_step
        """

        data, labels = batch  # (BCHW, B) or {'video': BCTHW, 'label': B}
        batch_size = data.size(0)

        outputs = self.model(data, labels=labels)
        loss = outputs.loss

        top1, top5, *_ = compute_topk_accuracy(outputs.logits, labels, topk=(1, 5))
        self.log_val_loss_top15(loss, top1, top5, batch_size)
