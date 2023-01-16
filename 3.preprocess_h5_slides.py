import os
import glob
import h5py
import argparse
import numpy as np
import albumentations as A
import pandas as pd
import torch
import traceback
from torchvision import transforms
from tqdm.contrib.telegram import tqdm
from typing import Union
from pathlib import Path
from natsort import os_sorted
from albumentations.pytorch import ToTensorV2

from utils.chat import send_noti_to_telegram
from utils.config import get_config
from he_preprocessing.utils.timer import Timer
from he_preprocessing.utils.image import is_blurry, pad_image
from pytorch.data_helpers.data_loaders import MultiResDatasetHDF5
from pytorch.data_helpers.utils import get_channels_sums_from_ndarray
from he_preprocessing.filter.filter import apply_filters_to_image, keep_tile
from pytorch.models.ssl_features.vit import ViT
from pytorch.models.ssl_features.resnets import ResNet50_SimCLR


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input-folders",
        dest="input_folders",
        nargs="+",
        help="The input folder containing the hdf5 datasets.",
        required=True,
    )

    argparser.add_argument(
        "--checkpoint-dir",
        dest="ckpt_dir",
        default=None,
        help="The directory containing the checkpoints of the pretrained models.",
        required=False,
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to save dataframe.",
        required=True,
    )

    argparser.add_argument(
        "--labels-csv",
        dest="labels_csv",
        default=None,
        help="Labels for each slide.",
        required=False,
    )

    argparser.add_argument(
        "--label",
        dest="label",
        default=None,
        help="Which label to use.",
        required=False,
    )

    argparser.add_argument(
        "--segmentation",
        dest="segmentation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable segmentation mode.",
        required=False,
    )

    argparser.add_argument(
        "--size", dest="size", type=int, default=512, help="Image size.", required=False
    )

    argparser.add_argument(
        "--color-mode",
        dest="color_mode",
        default="rgb",
        help="Color mode to load images with. 'rgb' or 'grayscale'",
        required=False,
    )

    argparser.add_argument(
        "--node-id",
        dest="node_id",
        type=int,
        default=0,
        help="Node id.",
        required=False,
    )

    argparser.add_argument(
        "--num-nodes",
        dest="num_nodes",
        type=int,
        default=1,
        help="Total number of nodes.",
        required=False,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    args = argparser.parse_args()
    return args


def get_ssl_models(ckpt_dir):
    vit = ViT(arch="small", ckpt=os.path.join(ckpt_dir, "vits_tcga_brca_dino.pt"))
    vit.eval()
    resnet50 = ResNet50_SimCLR(
        ckpt=os.path.join(ckpt_dir, "resnet50_tcga_brca_simclr.pt")
    )
    resnet50.eval()
    return {
        "vit": vit,
        "resnet50": resnet50,
    }


def eval_transforms(pretrained=False):
    if pretrained:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    else:
        mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    trnsfrms_val = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    )
    return trnsfrms_val


def compute_embeddings(model, x, transform):
    return np.squeeze(model.forward(transform(x).unsqueeze(0)).detach().cpu().numpy())


def create_hdf5(
    slides,
    output_dir,
    tile_size,
    blurriness_threshold,
    keep_tile_percentage,
    filters2apply,
    color_mode,
    normalize_bluriness: bool = True,
    segmentation: bool = False,
    labels_csv: Union[str, Path] = None,
    label_key: str = None,
    telegram_token=None,
    telegram_id=None,
):
    # ************************** #
    # Create output hdf5 dataset #
    # ************************** #
    if labels_csv is not None:
        assert segmentation is False

    if labels_csv is not None:
        assert label_key is not None

    labels_df = pd.read_csv(labels_csv)
    labels_df["image_name"] = labels_df["image_name"].str.rstrip()
    labels_df["image_name"] = labels_df["image_name"].apply(
        lambda x: Path(x).stem if x.endswith(".ndpi") or x.endswith(".svs") else x
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if args.ckpt_dir is not None:
        ssl_models = get_ssl_models(args.ckpt_dir)
        transform = eval_transforms(pretrained=False)

    total_slides = len(slides)
    for slide_idx, slide in enumerate(slides):
        # ****************** #
        # Process each slide #
        # ****************** #
        idx = 0

        slide_h5_path = glob.glob(os.path.join(slide, "*.h5"))[0]
        slide_name = Path(slide).name

        if slide_name.endswith(".h5") or slide_name.endswith(".hdf5"):
            slide_name = Path(slide_name).stem

        hdf5_path = os.path.join(output_dir, f"{slide_name}.h5")

        if os.path.exists(hdf5_path):
            print(f"HDF5 dataset exists at {hdf5_path}. Aborting...")
            exit(0)

        f = h5py.File(hdf5_path, mode="w")

        dataset_len = 1000
        target_size = (int(tile_size), int(tile_size), 3 if color_mode == "rgb" else 1)
        label_size = (int(tile_size), int(tile_size), 1)
        images_shape = (dataset_len,) + target_size
        labels_shape = (dataset_len,) + label_size if segmentation else (dataset_len,)

        # Images
        d_x_target = f.create_dataset(
            "x_target",
            images_shape,
            dtype=np.uint8,
            maxshape=(None, *images_shape[1:]),
        )

        d_x_context = f.create_dataset(
            "x_context",
            images_shape,
            dtype=np.uint8,
            maxshape=(None, *images_shape[1:]),
        )

        if args.ckpt_dir is not None:
            d_embeddings_resnet_target = f.create_dataset(
                "embeddings_resnet_target",
                (dataset_len, 2048),
                dtype=np.float32,
                maxshape=(None, 2048),
            )

            d_embeddings_resnet_context = f.create_dataset(
                "embeddings_resnet_context",
                (dataset_len, 2048),
                dtype=np.float32,
                maxshape=(None, 2048),
            )

            d_embeddings_vit_target = f.create_dataset(
                "embeddings_vit_target",
                (dataset_len, 384),
                dtype=np.float32,
                maxshape=(None, 384),
            )

            d_embeddings_vit_context = f.create_dataset(
                "embeddings_vit_context",
                (dataset_len, 384),
                dtype=np.float32,
                maxshape=(None, 384),
            )

        # labels
        if segmentation:
            d_y_target = f.create_dataset(
                "y_target",
                labels_shape,
                dtype=np.uint8,
                maxshape=(None, *labels_shape[1:]),
            )

            d_y_context = f.create_dataset(
                "y_context",
                labels_shape,
                dtype=np.uint8,
                maxshape=(None, *labels_shape[1:]),
            )
        else:
            d_y = f.create_dataset(
                "y",
                labels_shape,
                dtype=np.uint8,
                maxshape=(None,),
            )

        if segmentation:
            dataset = MultiResDatasetHDF5(
                data_dir=os.path.dirname(slide_h5_path),
                data_name=os.path.basename(slide_h5_path),
                data_cols={
                    "x_target": "x_target",
                    "x_context": "x_context",
                    "y_target": "y_target",
                    "y_context": "y_context",
                },
                transform=A.Compose([ToTensorV2()]),
                segmentation=True,
                channels_last=True,
            )
        else:
            dataset = MultiResDatasetHDF5(
                data_dir=os.path.dirname(slide_h5_path),
                data_name=os.path.basename(slide_h5_path),
                data_cols={
                    "x_target": "x_target",
                    "x_context": "x_context",
                    "labels": None,
                },
                transform=A.Compose([ToTensorV2()]),
                segmentation=False,
                channels_last=True,
            )

        with torch.no_grad():
            for i in tqdm(
                range(len(dataset)),
                desc=f"{slide}: preprocessing slide {slide_idx} / {total_slides - 1}",
                token=telegram_token,
                chat_id=telegram_id,
            ):
                current_size = d_x_target.shape[0]
                if idx >= current_size:
                    d_x_target.resize(current_size + 1000, axis=0)
                    d_x_context.resize(current_size + 1000, axis=0)

                    if args.ckpt_dir is not None:
                        d_embeddings_resnet_target.resize(current_size + 1000, axis=0)
                        d_embeddings_resnet_context.resize(current_size + 1000, axis=0)
                        d_embeddings_vit_target.resize(current_size + 1000, axis=0)
                        d_embeddings_vit_context.resize(current_size + 1000, axis=0)

                    if segmentation:
                        d_y_target.resize(current_size + 1000, axis=0)
                        d_y_context.resize(current_size + 1000, axis=0)
                    else:
                        d_y.resize(current_size + 1000, axis=0)

                data = dataset.get_item(i)
                x_target = data["x_target"].numpy()
                x_context = data["x_context"].numpy()
                if segmentation:
                    y_target = data["y_target"].numpy()
                    y_context = data["y_context"].numpy()
                else:
                    label = labels_df.loc[labels_df["image_name"] == slide_name][
                        label_key
                    ].tolist()[0]

                if blurriness_threshold["target"] is not None and is_blurry(
                    x_target,
                    threshold=blurriness_threshold["target"],
                    normalize=normalize_bluriness,
                    masked=False,
                ):
                    continue

                if blurriness_threshold["context"] is not None and is_blurry(
                    x_context,
                    threshold=blurriness_threshold["context"],
                    normalize=normalize_bluriness,
                    masked=False,
                ):
                    continue

                x_target = pad_image(
                    x_target,
                    tile_size,
                    value=config.preprocess.filters2apply.constant_pad_value,
                )
                x_context = pad_image(
                    x_context,
                    tile_size,
                    value=config.preprocess.filters2apply.constant_pad_value,
                )

                if keep_tile(
                    x_target,
                    tile_size=tile_size,
                    tissue_threshold=keep_tile_percentage,
                    pad=False,
                ):
                    # Preprocess image
                    if segmentation:
                        y_target = pad_image(y_target, tile_size, value=-1)
                        y_context = pad_image(y_context, tile_size, value=-1)

                    x_target, _ = apply_filters_to_image(
                        x_target,
                        roi_f=None,
                        slide=slide_name,
                        filters2apply=filters2apply,
                        save=False,
                    )

                    x_context, _ = apply_filters_to_image(
                        x_context,
                        roi_f=None,
                        slide=slide_name,
                        filters2apply=filters2apply,
                        save=False,
                    )

                    _sums, _squared_sums = get_channels_sums_from_ndarray(
                        x_target, channels_last=True, max_value=255.0
                    )
                    target_sums = target_sums + _sums
                    target_squared_sums = target_squared_sums + _squared_sums

                    _sums, _squared_sums = get_channels_sums_from_ndarray(
                        x_context, channels_last=True, max_value=255.0
                    )
                    context_sums += _sums
                    context_squared_sums += _squared_sums

                    d_x_target[idx, ...] = x_target.astype(np.uint8)
                    d_x_context[idx, ...] = x_context.astype(np.uint8)

                    if segmentation:
                        d_y_target[idx, ...] = y_target.astype(np.uint8)
                        d_y_context[idx, ...] = y_context.astype(np.uint8)
                    else:
                        d_y[idx] = label

                    if args.ckpt_dir is not None:
                        d_embeddings_resnet_target[idx, ...] = compute_embeddings(
                            model=ssl_models["resnet50"], x=x_target, transform=transform
                        )

                        d_embeddings_resnet_context[idx, ...] = compute_embeddings(
                            model=ssl_models["resnet50"], x=x_context, transform=transform
                        )

                        d_embeddings_vit_target[idx, ...] = compute_embeddings(
                            model=ssl_models["vit"], x=x_target, transform=transform
                        )

                        d_embeddings_vit_context[idx, ...] = compute_embeddings(
                            model=ssl_models["vit"], x=x_context, transform=transform
                        )

                    idx = idx + 1

        d_sums_target[...] = target_sums
        d_squared_sums_target[...] = target_squared_sums
        d_sums_context[...] = context_sums
        d_squared_sums_context[...] = context_squared_sums
        d_num_batches[0] = idx

        # Trick to fix bug where hdf5 is resized more than needed
        d_x_target.resize(idx, axis=0)
        d_x_context.resize(idx, axis=0)

        if args.ckpt_dir is not None:
            d_embeddings_resnet_target.resize(idx, axis=0)
            d_embeddings_resnet_context.resize(idx, axis=0)
            d_embeddings_vit_target.resize(idx, axis=0)
            d_embeddings_vit_context.resize(idx, axis=0)

        if segmentation:
            d_y_target.resize(idx, axis=0)
            d_y_context.resize(idx, axis=0)
        else:
            d_y.resize(idx, axis=0)

        f.close()


if __name__ == "__main__":
    timer = Timer()

    args = get_args()

    config, _ = get_config(args.config)

    slides = []
    for in_folder in args.input_folders:
        slides.extend(os_sorted(glob.glob(os.path.join(in_folder, "*/"))))

    slides = os_sorted(slides)

    # SELECT SLIDES BASED ON NODE ID
    process_idx = args.node_id
    num_processes = args.num_nodes
    num_images = len(slides)
    images_per_process = num_images / num_processes
    start_index = process_idx * images_per_process + 1
    end_index = (process_idx + 1) * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    slides = slides[start_index - 1 : end_index]

    slides_df = pd.DataFrame(slides, columns=["slide_path"])
    slides_df["slide_name"] = slides_df["slide_path"].apply(lambda x: Path(x).name)

    labels_df = pd.read_csv(args.labels_csv)
    labels_df["image_name"] = labels_df["image_name"].str.rstrip()
    labels_df["image_name"] = labels_df["image_name"].apply(
        lambda x: Path(x).stem if x.endswith(".ndpi") or x.endswith(".svs") else x
    )

    slides_df = slides_df.join(
        labels_df.set_index("image_name")[args.label], on="slide_name"
    )
    slides, labels = slides_df["slide_path"].to_list(), slides_df[args.label].to_list()

    try:
        create_hdf5(
            slides,
            output_dir=args.output_dir,
            tile_size=args.size,
            blurriness_threshold={
                "target": config.preprocess.filters2apply.blurriness_threshold,
                "context": config.preprocess.filters2apply.blurriness_threshold_context,
            },
            keep_tile_percentage=config.preprocess.filters2apply.keep_tile_percentage,
            filters2apply=config.preprocess.filters2apply,
            normalize_bluriness=config.preprocess.filters2apply.normalize_bluriness,
            color_mode=args.color_mode,
            segmentation=args.segmentation,
            labels_csv=args.labels_csv,
            label_key=args.label,
            telegram_token=config.telegram.token,
            telegram_id=config.telegram.chat_id,
        )
        send_noti_to_telegram(
            f"Dataset HDF5 creation finished for node {args.node_id} in {timer.elapsed()}.",
            TELEGRAM_TOKEN=config.telegram.token,
            TELEGRAM_CHAT_ID=config.telegram.chat_id,
        )
    except Exception as e:
        send_noti_to_telegram(
            f"EXCEPTION: Dataset HDF5 creation finished for node {args.node_id} in {timer.elapsed()} with an ERROR.",
            TELEGRAM_TOKEN=config.telegram.token,
            TELEGRAM_CHAT_ID=config.telegram.chat_id,
        )
        print(e)
        traceback.print_exc()
