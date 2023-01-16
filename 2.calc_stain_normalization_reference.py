import os
import glob
import argparse
import multiprocessing
import h5py
import numpy as np
import pandas as pd
import albumentations as A
from typing import Union
from pathlib import Path
from natsort import os_sorted
from tqdm.contrib.telegram import tqdm

from utils.chat import send_noti_to_telegram
from utils.config import get_config
from he_preprocessing.utils.image import create_mosaic
from he_preprocessing.utils.timer import Timer
from he_preprocessing.normalization import stain_utils
from he_preprocessing.normalization.stain_norm import StainNormalizer
from pytorch.data_helpers.data_loaders import ImageOnlyDatasetHDF5


def get_args():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--input-folders",
        dest="input_folders",
        nargs='+',
        help="The input folder containing the hdf5 datasets.",
        required=True,
    )

    argparser.add_argument(
        "--output-dir",
        dest="output_dir",
        help="Output directory to save dataframe.",
        required=True,
    )

    argparser.add_argument(
        "--config",
        dest="config",
        help="Config file to use.",
        required=True,
    )

    argparser.add_argument(
        "--method",
        dest="method",
        default="macenko",
        help="Stain normalization method to use. One of [macenko]",
        required=False,
    )

    argparser.add_argument(
        "--size",
        dest="size",
        type=int,
        default=512,
        help="Size of tiles.",
        required=False,
    )

    argparser.add_argument(
        "--slide-image-num",
        dest="slide_image_number",
        type=int,
        default=100,
        help="How many images to sample (slide-level).",
        required=False,
    )

    argparser.add_argument(
        "--dataset-image-num",
        dest="dataset_image_number",
        type=int,
        default=3025,
        help="How many images to sample (dataset-level).",
        required=False,
    )

    argparser.add_argument(
        "--slide-downsample",
        dest="slide_downsample",
        type=int,
        default=2,
        help="Downsample at which the images are patched together (slide-level).",
        required=False,
    )

    argparser.add_argument(
        "--dataset-downsample",
        dest="dataset_downsample",
        type=int,
        default=5,
        help="Downsample at which the images are patched together (dataset-level).",
        required=False,
    )

    args = argparser.parse_args()
    return args


def calculate_stain_norm(patched_image, method):
    patched_image, luminosity_95_percentile = stain_utils.standardize_brightness(
        patched_image, percentile=None
    )

    # Geet saturation vector and h&e matrix
    normalizer = StainNormalizer(target=patched_image, method=method, luminosity=False)
    saturation_vector = normalizer.get_99_percentile_saturation_vector()
    he_matrix = normalizer.get_he_vector()

    return luminosity_95_percentile, saturation_vector, he_matrix


def sample_dataset(
    slide_h5_path: Union[str, Path],
    patched_image: np.ndarray,
    index: int,
    images_per_slide: int,
    remaining_images: int,
    luminosity: int,
    remove_from_remaining=False,
):
    h5_dataset = h5py.File(slide_h5_path, "r")
    patch_ids = np.random.randint(
        0,
        h5_dataset["x_target"].shape[0],
        size=min(images_per_slide, h5_dataset["x_target"].shape[0]),
    )

    # if h5 dataset has less images than images_per_slide, then add to the remaining_images
    if not remove_from_remaining:
        remaining_images += max(0, images_per_slide - h5_dataset["x_target"].shape[0])

    dataset = ImageOnlyDatasetHDF5(
        data_dir=os.path.dirname(slide_h5_path),
        data_name=os.path.basename(slide_h5_path),
        data_cols={"images": "x_target"},
        transform=A.Compose(
                [
                    A.Resize(width=patched_image.shape[1], height=patched_image.shape[2]),
                    A.pytorch.ToTensorV2()
                ]
        ),
        pytorch_transform=None,
        channels_last=True,
    )

    for i in patch_ids:
        x = dataset.get_item(i)
        x = x.numpy().astype(np.uint8)
        x, _ = stain_utils.standardize_brightness(x, percentile=luminosity)
        patched_image[index, ...] = x
        index += 1
        if remove_from_remaining:
            remaining_images -= 1

    return patched_image, remaining_images, index


def calculate_dataset_level_stain_norm(
    slides, tile_size, image_number, downsample, method, luminosity_df, telegram_id, telegram_key
):
    send_noti_to_telegram(
        f"Dataset-level stain normalization reference calculation started.",
        TELEGRAM_TOKEN=telegram_key,
        TELEGRAM_CHAT_ID=telegram_id
    )

    image_number = int(np.floor(np.sqrt(image_number)) ** 2)

    images_per_slide, remaining_images = int(image_number // len(slides)), int(
        image_number % len(slides)
    )

    original_images_per_slide = images_per_slide

    cols = int(np.sqrt(image_number))

    patched_image = np.zeros(
        shape=(
            image_number,
            int(np.ceil(tile_size / downsample)),
            int(np.ceil(tile_size / downsample)),
            3,
        ),
        dtype=np.uint8,
    )

    patched_idx = 0

    for slide in tqdm(slides, desc='dataset-level', token=telegram_key, chat_id=telegram_id):
        slide_h5_path = glob.glob(os.path.join(slide, "*.h5"))[0]

        luminosity = luminosity_df.loc[luminosity_df.slide == Path(slide).name].luminosity_95_percentile.to_numpy()[0]

        patched_image, remaining_images, patched_idx = sample_dataset(
            slide_h5_path,
            patched_image,
            patched_idx,
            images_per_slide,
            remaining_images,
            luminosity,
            remove_from_remaining=False,
        )

    pbar = tqdm(desc='dataset-level-remaining', total=remaining_images, token=telegram_key, chat_id=telegram_id)
    while remaining_images > 0:
        slide = np.random.choice(slides)
        slide_h5_path = glob.glob(os.path.join(slide, "*.h5"))[0]

        luminosity = luminosity_df.loc[luminosity_df.slide == Path(slide).name].luminosity_95_percentile.to_numpy()[0]

        images_per_slide = np.random.randint(1, min(0.2 * original_images_per_slide, remaining_images) + 1)

        patched_image, remaining_images, patched_idx = sample_dataset(
            slide_h5_path,
            patched_image,
            patched_idx,
            images_per_slide,
            remaining_images,
            luminosity,
            remove_from_remaining=True,
        )
        pbar.update(images_per_slide)

    
    # Create
    patched_image = create_mosaic(patched_image, ncols=cols)
    
    send_noti_to_telegram(
        f"Calculating stain matrix for dataset...",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )

    luminosity_95_percentile, saturation_vector, he_matrix = calculate_stain_norm(
        patched_image, method
    )

    reference_df = pd.DataFrame(
        [np.concatenate((saturation_vector.flatten(), he_matrix.flatten())).tolist()],
        columns=[
            "saturation_vector_0",
            "saturation_vector_1",
            "he_matrix_0",
            "he_matrix_1",
            "he_matrix_2",
            "he_matrix_3",
            "he_matrix_4",
            "he_matrix_5",
        ],
    )

    return reference_df


def calculate_slide_level_stain_norm(
    slides, tile_size, image_number, downsample, method, telegram_id, telegram_key
):
    slide_reference_df = pd.DataFrame()

    if multiprocessing.current_process()._identity[0] == 1:
        send_noti_to_telegram(f"Slide-level stain normalization reference calculation started.", TELEGRAM_TOKEN=telegram_key, TELEGRAM_CHAT_ID=telegram_id)
        slide_iterator = tqdm(slides, desc='slide-level', token=telegram_key, chat_id=telegram_id)
    else:
        slide_iterator = slides

    for slide in slide_iterator:
        slide_h5_path = glob.glob(os.path.join(slide, "*.h5"))[0]

        h5_dataset = h5py.File(slide_h5_path, "r")

        image_number = np.floor(np.sqrt(image_number)) ** 2
        image_number = int(
            min(image_number, np.floor(np.sqrt(h5_dataset["x_target"].shape[0])) ** 2)
        )

        patch_ids = np.random.randint(
            0, h5_dataset["x_target"].shape[0], size=image_number
        )

        cols = int(np.sqrt(image_number))

        patched_image = np.zeros(
            shape=(
                image_number,
                int(np.ceil(tile_size / downsample)),
                int(np.ceil(tile_size / downsample)),
                3,
            ),
            dtype=np.uint8,
        )

        dataset = ImageOnlyDatasetHDF5(
            data_dir=os.path.dirname(slide_h5_path),
            data_name=os.path.basename(slide_h5_path),
            data_cols={"images": "x_target"},
            transform=A.Compose(
                [
                    A.Resize(
                        width=patched_image.shape[1], height=patched_image.shape[2]
                    ),
                    A.pytorch.ToTensorV2()
                ]
            ),
            pytorch_transform=None,
            channels_last=True,
        )

        for i, index in enumerate(patch_ids):
            x = dataset.get_item(index)
            patched_image[i, ...] = x.numpy().astype(np.uint8)

        # Create
        patched_image = create_mosaic(patched_image, ncols=cols)

        luminosity_95_percentile, saturation_vector, he_matrix = calculate_stain_norm(
            patched_image, method
        )

        slide_reference_df = pd.concat(
            (
                slide_reference_df,
                pd.DataFrame(
                    [
                        [Path(slide).name, luminosity_95_percentile]
                        + np.concatenate(
                            (saturation_vector.flatten(), he_matrix.flatten())
                        ).tolist()
                    ],
                    columns=[
                        "slide",
                        "luminosity_95_percentile",
                        "saturation_vector_0",
                        "saturation_vector_1",
                        "he_matrix_0",
                        "he_matrix_1",
                        "he_matrix_2",
                        "he_matrix_3",
                        "he_matrix_4",
                        "he_matrix_5",
                    ],
                ),
            ),
            ignore_index=True,
        )

    return slide_reference_df


if __name__ == "__main__":
    """
    Calculate the slide-level stain normalization luminosity reference.
    """
    timer = Timer()

    args = get_args()

    config, _ = get_config(args.config)
    
    send_noti_to_telegram(
        f"Stain normalization reference calculation started for {args.input_folders}.",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )

    output_dir = args.output_dir

    slides = []
    for input_folder in args.input_folders:
        slides.append(glob.glob(os.path.join(input_folder, "*/")))
    slides = [s for slides_x in slides for s in slides_x]
    slides = os_sorted(slides)

    # *********** #
    # SLIDE-LEVEL #
    # *********** #

    # slides = [slides[0]]
    # slide_reference_df = calculate_slide_level_stain_norm(
    #     slides, args.size, args.slide_image_number, args.slide_downsample, 'macenko', config.telegram.chat_id, config.telegram.token
    # )

    # how many processes to use
    num_processes = 20 # multiprocessing.cpu_count()

    num_images = len(slides)
    if num_processes > num_images:
        num_processes = num_images
    images_per_process = num_images / num_processes

    pool = multiprocessing.Pool(num_processes)

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        input_sublist = slides[start_index - 1 : end_index]
        tasks.append(
            (input_sublist, args.size, args.slide_image_number, args.slide_downsample, args.method, config.telegram.chat_id, config.telegram.token)
        )
        print(
            "Task #"
            + str(num_process)
            + ": Process  "
            + str(len(input_sublist))
            + " slide"
        )

    # start tasks
    results = []
    for t in tasks:
        results.append(pool.apply_async(calculate_slide_level_stain_norm, t))

    for result in results:
        result.wait()

    slide_reference_df = pd.DataFrame()
    for result in results:
        slide_reference_df = pd.concat((slide_reference_df, result.get()), ignore_index=True)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    slide_reference_df.to_csv(
        os.path.join(args.output_dir, "stain_vectors_slide_level_reference.csv"),
        index=False,
    )

    # ************* #
    # DATASET-LEVEL #
    # ************* #
    # slides = [*slides, ] * 50
    dataset_reference_df = calculate_dataset_level_stain_norm(
        slides, args.size, args.dataset_image_number, args.dataset_downsample, "macenko", slide_reference_df, config.telegram.chat_id, config.telegram.token
    )
    
    dataset_reference_df.to_csv(
        os.path.join(args.output_dir, "stain_vectors_dataset_level_reference.csv"),
        index=False,
    )

    send_noti_to_telegram(
        f"Stain normalization reference calculation finished for {args.input_folders} in {timer.elapsed()}.",
        TELEGRAM_TOKEN=config.telegram.token,
        TELEGRAM_CHAT_ID=config.telegram.chat_id,
    )
