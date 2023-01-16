import os
import h5py
import shutil
import natsort
import argparse
import numpy as np
from pathlib import Path
from typing import Dict
from wholeslidedata.annotation.parser import AnnotationParser

from utils.config import get_config
from utils.chat import send_noti_to_telegram
from he_preprocessing.utils.image import is_blurry, keep_tile
from pytorch.data_helpers.wsi.utils import (
    create_batch_sampler,
    whole_slide_files_from_folder_factory,
)
from pytorch.data_helpers.wsi import QuPathAnnotationParser, MaskedTiledAnnotationHook


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)

    argparser.add_argument(
        "--slides-dir",
        dest="slides_dir",
        help="Directory containing the pathology slides.",
        required=True,
    )

    argparser.add_argument(
        "--annotations-dir",
        dest="annotations_dir",
        help="Directory containing the annotation files.",
        required=True,
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to save hdf5 files for each slide.",
        required=True,
    )

    argparser.add_argument(
        "--image-extension",
        dest="image_extension",
        default=".ndpi",
        help="Extension of digital pathology slides.",
        required=False,
    )

    argparser.add_argument(
        "--ann-extension",
        dest="ann_extension",
        default=".geojson",
        help="Extension of digital pathology annotations.",
        required=False,
    )

    argparser.add_argument(
        "--quality-control",
        dest="quality_control",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable quality control.",
        required=False,
    )

    argparser.add_argument(
        "--multiresolution",
        dest="multiresolution",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multiresolution data.",
        required=False,
    )

    argparser.add_argument(
        "--dont-include-labels",
        dest="no_labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include label data.",
        required=False,
    )

    argparser.add_argument(
        "--tile-size",
        dest="tile_size",
        type=int,
        default=512,
        help="Tile size.",
        required=False,
    )

    argparser.add_argument(
        "--blurriness-threshold",
        dest="blurriness_threshold",
        type=int,
        default=None,
        help="Blurriness threshold for target.",
        required=False,
    )

    argparser.add_argument(
        "--intersection-percentage",
        dest="intersection_percentage",
        type=float,
        default=1.0,
        help="Intersection threshold between tiled patch and annotation to include patch.",
        required=False,
    )

    argparser.add_argument(
        "--stride-overlap-percentage",
        dest="stride_overlap_percentage",
        type=float,
        default=0.0,
        help="Stride overlap percentage between sampled tile windows.",
        required=False,
    )

    argparser.add_argument(
        "--blurriness-threshold-context",
        dest="blurriness_threshold_context",
        type=int,
        default=None,
        help="Blurriness threshold for context.",
        required=False,
    )

    argparser.add_argument(
        "--gpu",
        dest="gpu",
        type=int,
        default=0,
        help="Which GPU to use.",
        required=False,
    )

    argparser.add_argument(
        "--num-gpus",
        dest="num_gpus",
        type=int,
        default=1,
        help="Number of available GPU to use.",
        required=False,
    )

    argparser.add_argument(
        "--tissue-percentage",
        dest="tissue_percentage",
        type=float,
        default=0.5,
        help="Tissue percentage to keep tile.",
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


INITIAL_SIZE = 1000
TISSUE_LABEL = 1
TUMOR_LABEL = 2
TARGET_SPACING = 0.5
CONTEXT_SPACING = 2.0


def get_files(
    slides_dir,
    annotations_dir,
    tile_size,
    labels,
    stride_overlap_percentage,
    file_type="mrwsi",
):

    if ann_extension == ".geojson":
        parser = QuPathAnnotationParser
    else:
        parser = AnnotationParser
    parser = parser(
        labels=labels,
        hooks=(
            MaskedTiledAnnotationHook(
                tile_size=args.tile_size,
                ratio=1,
                overlap=int(tile_size * stride_overlap_percentage),
                label_names=list(labels.keys()),
                full_coverage=True,
            ),
        ),
    )

    image_files = whole_slide_files_from_folder_factory(
        slides_dir,
        file_type,
        excludes=[
            "mask",
        ],
        filters=[
            slide_extension,
        ],
        image_backend="openslide",
    )

    annotation_files = whole_slide_files_from_folder_factory(
        annotations_dir,
        "wsa",
        excludes=["tif"],
        filters=[ann_extension],
        annotation_parser=parser,
    )
    return image_files, annotation_files


def extract_data(
    output_root_dir: Path,
    image_files,
    annotation_files,
    slide_extension,
    ann_extension,
    file_type,
    tile_size,
    batch_size,
    tissue_percentage,
    intersection_percentage,
    stride_overlap_percentage,
    labels,
    spacing,
    blurriness_threshold: Dict[str, int],
    tissue_threshold: float = 0.5,
    no_labels: bool = False,
    seed=123,
):
    for image_idx, image_file in enumerate(image_files):
        print("##############")
        print(image_file)
        print("##############")

        try:
            batch_sampler, batch_ref_sampler, batch_shape = create_batch_sampler(
                image_files=[image_file],
                annotation_files=annotation_files,
                slide_extension=slide_extension,
                ann_extension=ann_extension,
                file_type=file_type,
                tile_size=tile_size,
                batch_size=batch_size,
                tissue_percentage=tissue_percentage,
                stride_overlap_percentage=stride_overlap_percentage,
                intersection_percentage=intersection_percentage,
                blurriness_threshold=blurriness_threshold,
                labels=labels,
                spacing=spacing,
                seed=seed,
            )
        except ValueError:
            print(f"No annotations found in {image_file}")
            continue

        image_name = image_file.path.stem
        output_dir = Path(os.path.join(output_root_dir, image_name))
        output_dir.mkdir(parents=True, exist_ok=True)
        h5_path = os.path.join(output_dir, image_name + ".h5")
        assert not Path(
            h5_path
        ).exists(), f"HDF5 dataset {h5_path}, exists. Aborting..."

        h5_f = h5py.File(name=h5_path, mode="w")

        images_shape = batch_shape.shape[0]

        d_x_target = h5_f.create_dataset(
            "x_target",
            (INITIAL_SIZE, *images_shape),
            dtype=np.uint8,
            chunks=None,
            compression=None,
            maxshape=(None, *images_shape),
        )

        d_x_context = h5_f.create_dataset(
            "x_context",
            (INITIAL_SIZE, *images_shape),
            dtype=np.uint8,
            chunks=None,
            compression=None,
            maxshape=(None, *images_shape),
        )

        if not no_labels:
            d_y_target = h5_f.create_dataset(
                "y_target",
                (INITIAL_SIZE, *images_shape[:-1]),
                dtype=np.uint8,
                chunks=None,
                compression=None,
                maxshape=(None, *images_shape[:-1]),
            )

            d_y_context = h5_f.create_dataset(
                "y_context",
                (INITIAL_SIZE, *images_shape[:-1]),
                dtype=np.uint8,
                chunks=None,
                compression=None,
                maxshape=(None, *images_shape[:-1]),
            )

        target_key = ("target", spacing["target"])
        context_key = ("context", spacing["context"])
        shape = (args.tile_size, args.tile_size, 3)

        idx = 0
        while True:
            ref_batch = batch_ref_sampler.batch()
            if ref_batch == []:
                break
            try:
                total_labels = len(batch_ref_sampler)
            except TypeError:
                total_labels = -1
            ann_idx = f'{ref_batch[0]["reference"].wsa_index}_{ref_batch[0]["reference"].annotation_index}'
            print(
                f"************ Processing {ann_idx} out of {total_labels} total labels. {image_file}, {image_idx} out of {len(image_files)} ************"
            )

            x_batch, y_batch = batch_sampler.batch(ref_batch)
            if x_batch is None:
                break
            for x, y in zip(x_batch, y_batch):
                x_target = x[target_key][shape]
                x_context = x[context_key][shape]

                if not no_labels:
                    y_target = y[target_key][shape]
                    y_context = y[context_key][shape]

                if args.quality_control:
                    if not keep_tile(
                        x_target,
                        tile_size,
                        tissue_threshold=tissue_threshold,
                        roi_mask=None,
                        pad=True,
                    ):
                        print(
                            f"Tile {ref_batch} doesn't have enough tissue... Discarding..."
                        )
                        continue

                    if (
                        blurriness_threshold["target"] is not None
                        and is_blurry(
                            x_target,
                            threshold=blurriness_threshold["target"],
                            verbose=0,
                        )
                    ) or (
                        blurriness_threshold["context"] is not None
                        and is_blurry(
                            x_context,
                            threshold=blurriness_threshold["context"],
                            verbose=0,
                        )
                    ):
                        print(f"Tile {ref_batch} is blurry... Discarding...")
                        continue

                d_x_target[idx, ...] = x_target
                d_x_context[idx, ...] = x_context

                if not no_labels:
                    d_y_target[idx, ...] = y_target
                    d_y_context[idx, ...] = y_context

                idx += 1

                current_d_len = d_x_target.shape[0]
                if idx >= current_d_len:
                    d_x_target.resize(current_d_len + INITIAL_SIZE, axis=0)
                    d_x_context.resize(current_d_len + INITIAL_SIZE, axis=0)
                    if not no_labels:
                        d_y_target.resize(current_d_len + INITIAL_SIZE, axis=0)
                        d_y_context.resize(current_d_len + INITIAL_SIZE, axis=0)

        if idx == 0 and not np.any(d_x_target[0, ...]):
            h5_f.close()
            shutil.rmtree(output_dir)
            print(f"No multires tiles extracted for {image_file}.")
            continue

        idx -= 1
        # Trick to fix bug where hdf5 is resized more than needed
        d_x_target.resize(idx, axis=0)
        d_x_context.resize(idx, axis=0)

        Path(os.path.join(output_dir, f"{idx}.txt")).touch()

        if not no_labels:
            d_y_target.resize(idx, axis=0)
            d_y_context.resize(idx, axis=0)

        h5_f.close()

if __name__ == "__main__":
    args = get_args()

    no_labels = args.no_labels
    if no_labels is None:
        no_labels = False

    config, _ = get_config(args.config)

    if args.gpu == 0:
        send_noti_to_telegram(
            f"Multires tiles extraction started with {args.num_gpus} processes",
            TELEGRAM_TOKEN=config.telegram.token,
            TELEGRAM_CHAT_ID=config.telegram.chat_id,
        )

    slide_extension = args.image_extension
    if not slide_extension.startswith("."):
        slide_extension = "." + slide_extension

    ann_extension = args.ann_extension
    if not ann_extension.startswith("."):
        ann_extension = "." + ann_extension

    if args.multiresolution:
        file_type = "mrwsi"
    else:
        file_type = "wsi"

    labels = {"tumor": TUMOR_LABEL}

    spacing = {
        "target": TARGET_SPACING,
        "context": CONTEXT_SPACING,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_files, annotation_files = get_files(
        slides_dir=args.slides_dir,
        annotations_dir=args.annotations_dir,
        file_type=file_type,
        tile_size=args.tile_size,
        labels=labels,
        stride_overlap_percentage=args.stride_overlap_percentage,
    )

    image_files = natsort.natsorted(image_files, key=str)

    blurriness_threshold = dict(
        target=args.blurriness_threshold, context=args.blurriness_threshold_context
    )

    # how many processes to use
    process_idx = args.gpu
    num_processes = args.num_gpus
    num_images = len(image_files)
    if num_processes > num_images:
        num_processes = num_images
    images_per_process = num_images / num_processes

    start_index = process_idx * images_per_process + 1
    end_index = (process_idx + 1) * images_per_process
    start_index = int(start_index)
    end_index = int(end_index)
    image_files_sublist = image_files[start_index - 1 : end_index]

    extract_data(
        output_dir,
        image_files_sublist,
        annotation_files,
        slide_extension,
        ann_extension,
        file_type,
        args.tile_size,
        1,
        0.5,
        args.intersection_percentage,
        args.stride_overlap_percentage,
        labels,
        spacing,
        blurriness_threshold,
        args.tissue_percentage,
        no_labels,
        np.random.randint(2**32 - 1),
    )

    send_noti_to_telegram(
        f"Multires tiles extracted for all images for proccess {process_idx}",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )

    print(
        f"Multires tiles extracted for all images for proccess {process_idx}",
    )
