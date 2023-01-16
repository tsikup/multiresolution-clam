import os
import torch
import argparse
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from pytorch.loggers import get_loggers
from pytorch.callbacks import get_callbacks
from utils.config import process_config
from pytorch.data_helpers.data_augmentors import get_augmentor
from pytorch.data_helpers.data_loaders import (
    FeatureDatasetHDF5,
)
from pytorch.models.classification.clam import CLAM_Features_PL


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "-o",
        "--output_dir",
        dest="output_dir",
        help="Output directory to save experiment.",
        required=True,
    )

    argparser.add_argument(
        "--num-nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Number of training nodes.",
        required=False,
    )

    argparser.add_argument(
        "--name",
        dest="name",
        help="Experiment name.",
        required=True,
    )

    argparser.add_argument(
        "--fold",
        dest="fold",
        type=int,
        help="KFold number.",
        required=True,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    args = argparser.parse_args()
    return args


def get_default_transform(config):
    return {
        "train": get_augmentor(
            split="train",
            patch_size=config.model.input_shape[1],
            enable_augmentation=config.augment.enable,
            enable_stain_augmentation=config.augment.stain_augmentation.enable,
            replace_background=config.preprocess.replace_black_with_white_background_online,
            constant_pad_value=config.preprocess.filters2apply.constant_pad_value,
            additional_targets={"context": "image"}
            if "x_context" in config.dataset.data_cols
            else None,
        ),
        "val": get_augmentor(
            split="val",
            patch_size=config.model.input_shape[1],
            enable_augmentation=False,
            enable_stain_augmentation=False,
            replace_background=config.preprocess.replace_black_with_white_background_online,
            constant_pad_value=config.preprocess.filters2apply.constant_pad_value,
            additional_targets={"context": "image"}
            if "x_context" in config.dataset.data_cols
            else None,
        ),
        "test": get_augmentor(
            split="val",
            patch_size=config.model.input_shape[1],
            enable_augmentation=False,
            enable_stain_augmentation=False,
            replace_background=config.preprocess.replace_black_with_white_background_online,
            constant_pad_value=config.preprocess.filters2apply.constant_pad_value,
            additional_targets={"context": "image"}
            if "x_context" in config.dataset.data_cols
            else None,
        ),
    }


def main(args, config):
    data = dict()

    train_dataset = FeatureDatasetHDF5(
        data_dir=config.dataset.train_folder,
        data_cols=config.dataset.data_cols,
        base_label=config.dataset.base_label,
    )

    val_dataset = FeatureDatasetHDF5(
        data_dir=config.dataset.val_folder,
        data_cols=config.dataset.data_cols,
        base_label=config.dataset.base_label,
    )

    config.trainer.num_batches_per_epoch = len(train_dataset) / (
        config.trainer.batch_size * config.trainer.accumulate_grad_batches
    )

    data["train"] = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=config.trainer.shuffle,
        pin_memory=True,
        prefetch_factor=config.trainer.prefetch_factor,
        persistent_workers=config.trainer.persistent_workers,
    )

    data["val"] = DataLoader(
        val_dataset,
        batch_size=config.trainer.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=config.trainer.prefetch_factor,
    )

    model = CLAM_Features_PL(
        config=config,
        n_classes=config.dataset.num_classes,
        size_arg=config.model.clam.size_arg,
        gate=config.model.clam.gated,
        dropout=config.model.clam.dropout,
        k_sample=8,
        instance_eval=config.model.clam.instance_eval,
        instance_loss=config.model.clam.instance_loss,
        instance_loss_weight=0.3,
        subtyping=config.model.clam.subtype,
        multibranch=False,
        multires_aggregation=dict(config.multires_aggregation)
        if config.multires_aggregation is not None
        else None,
        autoscale_network=config.model.clam.autoscale_network,
        attention_depth=config.model.clam.attention_depth,
        classifier_depth=config.model.clam.classifier_depth,
    )

    print(model.model)

    # loggers
    loggers = get_loggers(config)

    # callbacks
    callbacks = get_callbacks(config)

    # initialize trainer
    num_gpus = torch.cuda.device_count()
    num_nodes = args.num_nodes
    strategy = "ddp" if num_gpus > 1 or num_nodes > 1 else None
    trainer = Trainer(
        # num_sanity_val_steps=0,
        accelerator="gpu",
        precision=16,
        accumulate_grad_batches=config.trainer.accumulate_grad_batches,
        devices=num_gpus,
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=config.trainer.epochs,
        logger=loggers,
        callbacks=callbacks,
        check_val_every_n_epoch=config.trainer.check_val_every_n_epoch,
        reload_dataloaders_every_n_epochs=config.trainer.reload_dataloaders_every_n_epochs,
    )

    checkpoint = model.model.state_dict()
    torch.save(
        checkpoint, os.path.join(config.callbacks.checkpoint_dir, "initial_clam.ckpt")
    )

    print("Fitting model on datamodule.")
    trainer.fit(
        model,
        train_dataloaders=data["train"],
        val_dataloaders=data["val"],
        ckpt_path=None,
    )

    trainer.validate(dataloaders=data["val"], ckpt_path="best", verbose=True)

    trainer.save_checkpoint(os.path.join(config.callbacks.checkpoint_dir, "final.ckpt"))


if __name__ == "__main__":
    args = get_args()

    config = process_config(
        args.config,
        name=args.name,
        output_dir=args.output_dir,
        dirs=True,
        config_copy=True,
    )
    config.filename = args.config

    config.dataset.train_folder = os.path.join(
        config.dataset.train_folder, f"{args.fold}_fold/train"
    )
    config.dataset.val_folder = os.path.join(
        config.dataset.val_folder, f"{args.fold}_fold/val"
    )

    main(args, config)
