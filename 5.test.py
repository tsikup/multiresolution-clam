import os
import argparse
import glob
from natsort import os_sorted
from pathlib import Path
from pytorch_lightning import Trainer, LightningDataModule
from torch.utils.data import DataLoader

from pytorch.loggers import get_loggers
from utils.config import get_config
from pytorch.data_helpers.data_loaders import FeatureDatasetHDF5
from pytorch.models.classification.clam import CLAM_Features_PL
from pytorch.utils.metrics.metrics import get_metrics


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    args = argparser.parse_args()
    return args


def main(args, config, model_folder):
    
    feature_dataset = FeatureDatasetHDF5(
            data_dir=config.dataset.test_folder,
            data_cols=config.dataset.data_cols,
            base_label=config.dataset.base_label
        )
    
    data = DataLoader(
        feature_dataset,
        batch_size=config.trainer.batch_size,
    )
    
    folder = os.path.join(model_folder, '*fold*')
    folds = os_sorted(glob.glob(folder))
    
    for idx, fold in enumerate(folds):
        print(fold)
        config.callbacks.tensorboard_log_dir = os.path.join(config.callbacks.tensorboard_log_dir, f'fold_{idx}')
        model_ckpt = os.path.join(fold, 'checkpoints/version_0/final.ckpt')
        model = CLAM_Features_PL.load_from_checkpoint(model_ckpt, strict=False)
        model.test_metrics = get_metrics(
            config,
            n_classes=2,
            dist_sync_on_step=False,
            mode="test",
            segmentation=False,
        ).clone(prefix="test_")
        
        # loggers
        loggers = get_loggers(config)

        # initialize trainer
        trainer = Trainer(
            accelerator="gpu",
            precision=16,
            devices=1,
            strategy=None,
            logger=loggers,
        )
            
        print("Testing model.")
        if isinstance(data, LightningDataModule):
            trainer.test(model, datamodule=data, ckpt_path=None)
        else:
            trainer.test(model, dataloaders=data, ckpt_path=None, verbose=True)


if __name__ == "__main__":
    args = get_args()
    
    model_folder = 'path/to/trained/model/folds'
    
    config, _ = get_config(args.config)
    config.callbacks.tensorboard_log_dir = os.path.join(model_folder, 'test_metrics')
    Path(config.callbacks.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    config.filename = args.config

    main(args, config, model_folder)
