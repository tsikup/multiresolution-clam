# ------------------------------------------------------------------------
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------------------------
"""
This module contains functions for wsi image manipulation from
https://github.com/CODAIT/deep-histopath/blob/master/deephistopath/preprocessing.py
"""
import math
from typing import List, Union
import PIL
import openslide
import numpy as np
import tissueloc as tl
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.morphology as sk_morphology
import skimage.color as sk_color
from PIL import Image, ImageDraw, ImageFont
from openslide import OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
from scipy.ndimage import binary_fill_holes
from definitions import *
from he_preprocessing.utils.timer import Timer

# References for used modules and libraries
"""
@article{chen2019tissueloc,
  author    = {Pingjun Chen and Lin Yang},
  title     = {tissueloc: Whole slide digital pathology image tissue localization},
  journal   = {J. Open Source Software},
  volume    = {4},
  number    = {33},
  pages     = {1148},
  year      = {2019},
  url       = {https://doi.org/10.21105/joss.01148},
  doi       = {10.21105/joss.01148}
}
"""
"""
@article{goode2013openslide,
  title     = {OpenSlide: A vendor-neutral software foundation for digital pathology},
  author    = {Goode, Adam and Gilbert, Benjamin and Harkes, Jan and Jukic, Drazen and Satyanarayanan, Mahadev},
  journal   = {Journal of pathology informatics},
  volume    = {4},
  year      = {2013},
  publisher = {Wolters Kluwer--Medknow Publications}
}
"""


# Open Whole-Slide Image
def open_slide(slide_f, folder=None):
    """
    Open a whole-slide image, given an image filepath.
    Args:
      slide_f: Slide filename.
      folder: Directory in which the slides folder is stored, as a string.
    Returns:
      An OpenSlide object representing a whole-slide image.
    """
    if folder is None:
        filename = slide_f
    else:
        filename = os.path.join(folder, slide_f)
    try:
        slide = openslide.open_slide(filename)
    except OpenSlideError:
        slide = None
    except FileNotFoundError:
        slide = None
    return slide
    # return None


def select_slide_level(slide_path, downsample=32):
    """Find the slide level to perform tissue localization
    Parameters
    ----------
    slide_path : valid slide path
        The slide to process.
    downsample : int
        Desired downsample. MAY BE DIFFERENT
    Returns
    -------
    level : int
        Selected level.
    d_factor: int
        Downsampling factor of selected level compared to level 0
    Notes
    -----
    The slide should have hierarchical storage structure.
    -----
    Calls tissueloc.select_slide_level
    """
    slide = open_slide(slide_path)
    assert downsample > 0
    max_img_size = max(np.array(slide.dimensions) // downsample) + 1
    return tl.select_slide_level(slide_path, max_img_size)


# Open Whole-Slide Image
def open_slide_np(slide_f, folder=None, downsample=32):
    """
    Open a whole-slide image, given an image filepath as numpy array.
    Args:
      slide_f: Slide filename.
      folder: Directory in which the slides are stored, as a string.
      downsample: Downsample level to work on. MAY BE DIFFERENT at the end.
    Returns:
      A NumPy object representing a whole-slide image.
    """
    level, d_factor = select_slide_level(slide_f, downsample)
    slide = open_slide(slide_f, folder)
    if level > 1:
        slide_img = slide.read_region((0, 0), level, slide.level_dimensions[level])
    if isinstance(slide_img, Image.Image):
        slide_img = np.asarray(slide_img)
    if slide_img.shape[2] == 4:
        slide_img = slide_img[:, :, :-1]
    return slide_img, d_factor


# Open Whole-Slide Image
def open_image(image_f, folder=None):
    """
    Open an image, given an image filepath.
    Args:
      image_f: Image filename.
      folder: Directory in which the images are stored, as a string.
    Returns:
      An Image object representing an image.
    """
    if folder is None:
        filename = image_f
    else:
        filename = os.path.join(folder, image_f)
    try:
        img = Image.open(filename)
    except FileNotFoundError:
        img = None
    except Exception as e:
        img = None
    return img


# Open Image as NumPy Array
def open_image_np(image_f, folder=None):
    """
    Open an image, given an image filepath as numpy array.
    Args:
      image_f: Image filename.
      folder: Directory in which the images are stored, as a string.
    Returns:
      A NumPy object representing an image.
    """
    return pil_to_np(open_image(image_f, folder))


def resize_img(img, shape):
    """
    Resize an image according to its type
    Args:
        img: PIL Image object
        shape: target shape
    Returns:
        resized image
    """
    if type(img) is PIL.Image:
        return resize_pil(img, shape)
    else:
        return resize_np(img, shape)


def resize_pil(img, shape):
    """
    Resize a PIL image to shape.
    Args:
        img: PIL Image object
        shape: target shape
    Returns:
        resized PIL image object
    """
    return img.resize(shape, PIL.Image.BILINEAR)


def resize_np(img_np, shape):
    """
    Resize a NumPy array to shape
    Args:
        img_np: NumPy array
        shape: target shape
    Returns:
        resized image array
    """
    pil_img = Image.fromarray(img_np)
    img = pil_img.resize(shape, PIL.Image.BILINEAR)
    return np.array(img)


def replace_pixels(np_img, old_color: List[int], new_color: List[int]):
    """
    Replace old_color pixels with new color
    Args:
        np_img: Image as NumPy array.
        old_color: old color list with 2 or 3 elements
        new_color: new color list with 2 or 3 elements
    Returns:
        np_img: Image as NumPy array
    """
    assert len(old_color) == len(new_color) and len(old_color) in [2, 3]
    indices = np.where(np.all(np_img == old_color, axis=-1))
    np_img[indices] = new_color
    return np_img


# Apply a binary mask to an rgb image
def mask_rgb(rgb, mask, background=None):
    """
    Apply a binary (T/F, 1/0) mask to a 3-channel RGB image and output the result.

    Args:
        rgb: RGB image as a NumPy array.
        mask: An image mask to determine which pixels in the original image should be displayed.
        background: background color to apply; if none, then it applies the mask's default
    Returns:
        NumPy array representing an RGB image with mask applied.
    """
    if len(mask.shape) < 3 or mask.shape[-1] == 1:
        if background:
            if type(background) not in [list, tuple]:
                background = [background for _ in range(rgb.shape[-1])]
            result = rgb
            for i in range(rgb.shape[-1]):
                result[~mask, i] = background[i]
        else:
            result = rgb * np.dstack([mask, mask, mask])
    else:
        if background:
            if type(background) not in [list, tuple]:
                background = [background, background, background]
            result = rgb
            result[~mask] = background
        else:
            result = rgb * mask
    return result


# Apply a binary mask to an rgb image
def mask_gray(gray, mask):
    """
    Apply a binary (T/F, 1/0) mask to a 1-channel grayscale image and output the result.

    Args:
        gray: Grayscale image as a NumPy array.
        mask: An image mask to determine which pixels in the original image should be displayed.
    Returns:
        NumPy array representing an RGB image with mask applied.
    """
    result = gray * mask
    return result


def rgb2hed(np_img):
    """
    Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.
    Args:
      np_img: RGB image as a NumPy array.
    Returns:
      NumPy array (float or uint8) with HED channels.
    """
    return sk_color.rgb2hed(np_img)


def hed2rgb(np_img):
    """
    Filter HED channels to RGB.
    Args:
      np_img: HED image as a NumPy array.
    Returns:
      NumPy array (float or uint8) with RGB channels.
    """
    return sk_color.hed2rgb(np_img)


# TODO: Fix bug
def rgb2hsd(X):
    """
    Filter RGB channels to HSD (Hue, Saturation, Density).
    Ref:
        der Laak et al. Hue-Saturation-Density (HSD) Model for StainRecognition in Digital Images FromTransmitted Light Microscopy
    """
    eps = np.finfo(float).eps
    X[np.where(X == 0.0)] = eps

    OD = -np.log(X / 1.0)
    D = np.mean(OD, 3)
    D[np.where(D == 0.0)] = eps

    cx = OD[:, :, :, 0] / (D) - 1.0
    cy = (OD[:, :, :, 1] - OD[:, :, :, 2]) / (np.sqrt(3.0) * D)

    D = np.expand_dims(D, 3)
    cx = np.expand_dims(cx, 3)
    cy = np.expand_dims(cy, 3)

    X_HSD = np.concatenate((D, cx, cy), 3)
    return X_HSD


def hsd2rgb(X_HSD):
    X_HSD_0 = X_HSD[..., 0]
    X_HSD_1 = X_HSD[..., 1]
    X_HSD_2 = X_HSD[..., 2]
    D_R = np.expand_dims(np.multiply(X_HSD_1 + 1, X_HSD_0), 2)
    D_G = np.expand_dims(
        np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 + np.sqrt(3.0) * X_HSD_2), 2
    )
    D_B = np.expand_dims(
        np.multiply(0.5 * X_HSD_0, 2 - X_HSD_1 - np.sqrt(3.0) * X_HSD_2), 2
    )

    X_OD = np.concatenate((D_R, D_G, D_B), axis=2)
    X_RGB = 1.0 * np.exp(-X_OD)
    return X_RGB


def rgb2hsv(np_img):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
      np_img: RGB image as a NumPy array.
    Returns:
      Image as NumPy array in HSV representation.
    """
    return sk_color.rgb2hsv(np_img)


def hsv2rgb(np_img):
    """
    Filter HSV channels to RGB.
    Args:
      np_img: HSV image as a NumPy array.
    Returns:
      Image as NumPy array in RGB representation.
    """
    return sk_color.hsv2rgb(np_img)


def pad_image(np_img, size, value=230):
    """
    Pad image to (size,size,-1)
    Args:
      np_img: Image as NumPy array.
      size: Size to pad image to.
      value: constant value to pad with.
      debug: Show debug info if True.
    """
    width = np_img.shape[0]
    height = np_img.shape[1]
    padding_x = size - width
    assert padding_x >= 0
    padding_y = size - height
    assert padding_y >= 0

    if padding_y == 0 and padding_x == 0:
        return np_img

    padding_x1 = np.floor(padding_x / 2).astype(int)
    padding_x2 = np.ceil(padding_x / 2).astype(int)

    padding_y1 = np.floor(padding_y / 2).astype(int)
    padding_y2 = np.ceil(padding_y / 2).astype(int)

    if len(np_img.shape) == 3 and np_img.shape[-1] == 3:
        padded_img = np.pad(
            np_img,
            ((padding_x1, padding_x2), (padding_y1, padding_y2), (0, 0)),
            "constant",
            constant_values=value,
        )
    else:
        if len(np_img.shape) == 3 and np_img.shape[-1] == 1:
            np_img = np.squeeze(np_img, axis=-1)
        padded_img = np.pad(
            np_img,
            ((padding_x1, padding_x2), (padding_y1, padding_y2)),
            "constant",
            constant_values=value,
        )
        padded_img = np.expand_dims(padded_img, axis=-1)

    return padded_img


def invert(np_img):
    """
    Invert image array (black->white, white->black)
    Args:
      np_img: RGB Image as a NumPy array.
    Returns:
      Inverted image.
    """
    if np_img.dtype == int or np_img.dtype == np.uint8:
        inverted = 255 - np_img
    else:
        inverted = 255 - (np_img * 255).astype("uint8")
    return inverted


def uint8_to_bool(np_img):
    """
    Convert NumPy array of uint8 (255,0) values to bool (True,False) values
    Args:
      np_img: Binary image as NumPy rray of uint8 (255,0) values.
    Returns:
      NumPy array of bool (True,False) values.
    """
    result = (np_img / 255).astype(bool)
    return result


def contrast_stretch(np_img, low=40, high=60):
    """
    Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in a specified range.
    Args:
      np_img: Image as a NumPy array (gray or RGB).
      low: Range low value (0 to 255).
      high: Range high value (0 to 255).
    Returns:
      Image as NumPy array with contrast enhanced.
    """
    low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
    contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))
    return contrast_stretch


def rgb2gray(np_img):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.
    Shape (h, w, c) to (h, w).
    Args:
      np_img: RGB Image as a NumPy array.
    Returns:
      Grayscale image as NumPy array with shape (h, w).
    """
    gray = sk_color.rgb2gray(np_img)
    gray = (gray * 255).astype("uint8")
    return gray


def rgb2gray_1(tile):
    """
    Convert a rgb image to grayscale and applied several morphological operations.
    Args:
      tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
      A 2D NumPy array of shape (tile_size, tile_size)
      representing the grayscale version of the RGB image.
    """
    # Convert 3D RGB image to 2D grayscale image, from
    # 0 (dense tissue) to 1 (plain background).
    tile = rgb2gray(tile)
    # 8-bit depth complement, from 1 (dense tissue)
    # to 0 (plain background).
    tile = 1 - tile
    # Canny edge detection with hysteresis thresholding.
    # This returns a binary map of edges, with 1 equal to
    # an edge. The idea is that tissue would be full of
    # edges, while background would not.
    tile = sk_feature.canny(tile)
    # Binary closing, which is a dilation followed by
    # an erosion. This removes small dark spots, which
    # helps remove noise in the background.
    tile = sk_morphology.binary_closing(tile, sk_morphology.disk(10))
    # Binary dilation, which enlarges bright areas,
    # and shrinks dark areas. This helps fill in holes
    # within regions of tissue.
    tile = sk_morphology.binary_dilation(tile, sk_morphology.disk(10))
    # Fill remaining holes within regions of tissue.
    tile = binary_fill_holes(tile)
    return tile


def rgb2gray_2(tile):
    """
    Convert a rgb image to grayscale based on optical density and applied several morphological operations.
    Args:
      tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
      A 2D NumPy array of shape (tile_size, tile_size)
      representing the grayscale version of the RGB image.
    """
    # Convert to optical density values
    tile = rgb2od(tile)
    # Threshold at beta
    beta = 0.15
    tile = np.min(tile, axis=2) >= beta
    # Apply morphology for same reasons as above.
    tile = sk_morphology.binary_closing(tile, sk_morphology.disk(2))
    tile = sk_morphology.binary_dilation(tile, sk_morphology.disk(2))
    tile = binary_fill_holes(tile)
    return tile


# def optical_density(tile):
#     """
#     Convert a tile to optical density values.
#     Args:
#       tile: A 3D NumPy array of shape (tile_size, tile_size, channels).
#     Returns:
#       A 3D NumPy array of shape (tile_size, tile_size, channels)
#       representing optical density values.
#     """
#     tile = tile.astype(np.float64)
#     # od = -np.log10(tile/255 + 1e-8)
#     od = -np.log((tile + 1) / 240)
#     return od


def rgb2od(np_img):
    """
    Convert a rgb image from rgb to optical density
    Args:
      np_img: A 3D NumPy array of shape (tile_size, tile_size, channels).
    Returns:
      A 3D NumPy array of shape (tile_size, tile_size, channels) representing optical density values.
    """
    mask = np_img == 0
    np_img[mask] = 1
    return np.maximum(-1 * np.log(np_img / 255), 1e-6)


def od2rgb(OD):
    """
    Convert from optical density (OD_RGB) to RGB.
    RGB = 255 * exp(-1*OD_RGB)
    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    assert OD.min() >= 0, "Negative optical density."
    OD = np.maximum(OD, 1e-6)
    return (255 * np.exp(-1 * OD)).astype(np.uint8)


def bw_histogram(np_img, max_value="auto"):
    """
    Get the histogram of the grayscale version of an image
    Args:
      np_img: Image as a NumPy array.
      max_value: Maximum value that can be observed in the NumPy image.
                 Auto means that it will be either 255 if maximum value
                 is more than 1, otherwise it will be 1. It will be either
                 255 for 8-bit images or 1 if the image is normalized.
    Returns:
      histogram
    """
    if len(np_img.shape) == 3 and np_img.shape[-1] == 3:
        bw = rgb2gray(np_img)
    else:
        bw = np_img
    if max_value == "auto":
        if np.amax(bw) > 1:
            max_value = 255
        else:
            max_value = 1
    np_hist, _ = np.histogram(bw, bins=256, range=(0, max_value))
    np_hist = np_hist.astype(float)
    return np_hist


def tissue_percent_1(np_img):
    """
    Determine the percentage of a NumPy array that is tissue based on the rgb2gray1 function.
    Args:
      np_img: Image as a NumPy array.
    Returns:
      The percentage of the NumPy array that is tissue.
    """
    # Binarize image
    np_img_2 = rgb2gray_1(np_img)
    # Calculate percentage of tissue coverage.
    tissue_percentage = np_img_2.mean()
    return tissue_percentage


def tissue_percent_2(np_img):
    """
    Determine the percentage of a NumPy array that is tissue based on the rgb2gray2 function.

    Args:
      np_img: Image as a NumPy array.
    Returns:
      The percentage of the NumPy array that is tissue.
    """
    np_img_2 = rgb2gray_2(np_img)
    percentage = np_img_2.mean()
    return percentage


def keep_tile(
    tile, tile_size, tissue_threshold, pens_threshold=0.2, roi_mask=None, pad=False
):
    """
    Determine if a tile should be kept.
    This filters out tiles based on size and a tissue percentage
    threshold, using a custom algorithm. If a tile has height &
    width equal to (tile_size, tile_size), and contains greater
    than or equal to the given percentage, then it will be kept;
    otherwise it will be filtered out.
    Args:
      tile: Tile is a 3D NumPy array of shape (tile_size, tile_size, channels).
      tile_size: The width and height of a square tile to be generated.
      tissue_threshold: Tissue percentage threshold.
      pens_threshold: Pens percentage threshold.
      roi_mask: Roi mask to apply to tile before calculating the percentage.
    Returns:
      A Boolean indicating whether a tile should be kept for future usage.
    """
    if pad:
        if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
            tile = pad_image(tile, tile_size, 255)

    if (roi_mask is not None or roi_mask != np.array(None)) and np.any(
        uint8_to_bool(roi_mask) == False
    ):
        tile[np.where(uint8_to_bool(roi_mask) == False)] = [255, 255, 255]

    if tile.shape[0:2] == (tile_size, tile_size):

        # Check 1
        percentage1 = tissue_percent_1(tile)
        check1 = percentage1 >= tissue_threshold

        # Check 2
        percentage2 = tissue_percent_2(tile)
        check2 = percentage2 >= tissue_threshold

        check = check1 and check2
    else:
        check = False

    return check


def is_blurry(np_img, threshold: int, normalize: bool = True, masked=False, verbose=0):
    """
    Detects if an image is blurry using the laplacian method of cv2.
    Args:
        np_img: The NumPy array representing the image.
        threshold: The threshold under which the image is considered blurry.
        normalize: Normalize the variance per tissue area.
        masked: Calculate variance based only on tissue area.
        verbose: Return blurriness variance if verbose > 0.
    """
    import cv2 as cv

    gray = rgb2gray(np_img)

    # ddepth: Depth of the destination image. Since our input is CV_8U we define ddepth = CV_16S to avoid overflow
    # kernel_size: The kernel size of the Sobel operator to be applied internally.
    _blurry_var = None
    laplacian = cv.Laplacian(gray, ddepth=cv.CV_64F, ksize=1)

    if masked:
        mask = ~get_tissue_mask(np_img, method="od")
        laplacian = np.ma.array(laplacian, mask=mask)

    blurry_var = laplacian.var()
    blurry_max = laplacian.max()
    # blurry_var = skimage.filters.edges.laplace(np_img).var()
    _blurry_var = blurry_var

    if normalize:
        _blurry_var = normalize_per_tissue_area(np_img, _blurry_var)
        threshold = threshold / (np_img.shape[1] ** 2)

    blurry = _blurry_var < threshold

    if verbose > 0:
        return blurry, blurry_var, blurry_max, _blurry_var, threshold
    else:
        return blurry


def normalize_per_tissue_area(np_img, value):
    grayscale = get_tissue_mask(np_img, method="od")
    return value / np.count_nonzero(grayscale)


# Create Tile Generator
def create_tile_generator(slide, tile_size, overlap):
    """
    Create a tile generator for the given slide.
    This generator is able to extract tiles from the overall
    whole-slide image.
    Args:
      slide: An OpenSlide object representing a whole-slide image.
      tile_size: The width and height of a square tile to be generated.
      overlap: Number of pixels by which to overlap the tiles.
    Returns:
      A DeepZoomGenerator object representing the tile generator. Each
      extracted tile is a PIL Image with shape
      (tile_size, tile_size, channels).
      Note: This generator is not a true "Python generator function", but
      rather is an object that is capable of extracting individual tiles.
    """
    generator = DeepZoomGenerator(
        slide, tile_size=tile_size, overlap=overlap, limit_bounds=True
    )
    return generator
    # return None


# Determine 20x,40x,...x Magnification Zoom Level
def get_x_zoom_level(slide, generator, zoom):
    """
    Return the zoom level that corresponds to a 20x magnification.
    The generator can extract tiles from multiple zoom levels,
    downsampling by a factor of 2 per level from highest to lowest
    resolution.
    Args:
      slide: An OpenSlide object representing a whole-slide image.
      generator: A DeepZoomGenerator object representing a tile generator.
        Note: This generator is not a true "Python generator function",
        but rather is an object that is capable of extracting individual
        tiles.
      zoom: Magnification level (e.g. zoom=20 for 20x)
    Returns:
      Zoom level corresponding to a ZOOMx magnification, or as close as
      possible.
    """
    highest_zoom_level = generator.level_count - 1  # 0-based indexing
    try:
        mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])
        # `mag / zoom` gives the downsampling factor between the slide's
        # magnification and the desired 20x magnification.
        # `(mag / zoom) / 2` gives the zoom level offset from the highest
        # resolution level, based on a 2x downsampling factor in the
        # generator.
        offset = math.floor((mag / zoom) / 2)
        level = highest_zoom_level - offset
    except (ValueError, KeyError) as e:
        # In case the slide magnification level is unknown, just
        # use the highest resolution.
        level = highest_zoom_level
    return level
    # return 0


def is_image(I):
    """
    Is I an image.
    """
    if not isinstance(I, np.ndarray):
        return False
    if not I.ndim == 3:
        return False
    return True


def is_uint8_image(I):
    """
    Is I a uint8 image.
    """
    if not is_image(I):
        return False
    if I.dtype != np.uint8:
        return False
    return True


def tile_image(image, mask=None, tile_size=512, offset=512):
    """
    Create tiles from np_array
    Args:
        image: the image array with shape (height, width, channels)
        mask: the mask array with shape (height, width, channels)
    Returns:
        cropped_images
    """
    img_shape = image.shape
    tile_size = (tile_size, tile_size)
    offset = (offset, offset)
    cropped_images = []
    cropped_masks = []

    for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            cropped_images.append(
                image[
                    offset[1] * i : min(offset[1] * i + tile_size[1], img_shape[0]),
                    offset[0] * j : min(offset[0] * j + tile_size[0], img_shape[1]),
                ]
            )

            if mask is not None:
                cropped_masks.append(
                    mask[
                        offset[1] * i : min(offset[1] * i + tile_size[1], img_shape[0]),
                        offset[0] * j : min(offset[0] * j + tile_size[0], img_shape[1]),
                    ]
                )

    if mask is not None:
        return cropped_images, cropped_masks
    else:
        return cropped_images


def create_mosaic(array, ncols=10):
    """
    Create mosaic image from array of images
    Args:
        array: the image array with shape (num_of_images, height, width, channels)
        ncols: number of columns for the mosaic
    Returns:
        mosaic image
    """
    nindex, height, width, channels = array.shape
    nrows = nindex // ncols
    assert nindex == nrows * ncols
    # want result.shape = (height*nrows, width*ncols, channels)
    result = (
        array.reshape(nrows, ncols, height, width, channels)
        .swapaxes(1, 2)
        .reshape(height * nrows, width * ncols, channels)
    )

    return result


def save_img(img, filepath, folder=None):
    """
    Save image
    Args:
        img: the image value with the size (img_size_x, img_size_y, channels)
        filepath: the file path at which to save JPEGs
    """
    try:
        if folder is not None:
            if os.path.dirname(filepath) == "":
                filepath = os.path.join(folder, filepath)
            else:
                filepath = os.path.join(folder, os.path.basename(filepath))
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        if type(img) == np.ndarray:
            img = np_to_pil(img)
        img.save(filepath)
    except Exception as e:
        print("Error saving image.\n{}".format(e))


def save_thumbnail(pil_img, size, path, display_path=False):
    """
    Save a thumbnail of a PIL image, specifying the maximum width or height of the thumbnail.
    Args:
      pil_img: The PIL image to save as a thumbnail.
      size:  The maximum width or height of the thumbnail.
      path: The path to the thumbnail.
      display_path: If True, display thumbnail path in console.
    """
    try:
        if type(pil_img) == np.ndarray:
            pil_img = np_to_pil(pil_img)
        max_size = tuple(round(size * d / max(pil_img.size)) for d in pil_img.size)
        img = pil_img.resize(max_size, PIL.Image.BILINEAR)
        if display_path:
            print("Saving thumbnail to: " + path)
        directory = os.path.dirname(path)
        if directory != "" and not os.path.exists(directory):
            os.makedirs(directory)
        img.save(path)
    except Exception as e:
        print("Error saving thumbnail.\n{}".format(e))


def slide_to_scaled_pil_image(slide_f, scale_factor=SCALE_FACTOR, folder=None):
    """
    Convert a WSI training slide to a scaled-down PIL image.
    Args:
      slide_f: The slide filename.
      folder: The folder containing slide.
    Returns:
      Tuple consisting of scaled-down PIL image, original width, original height, new width, and new height.
    """
    print("Opening Slide %s" % slide_f)
    slide = open_slide(slide_f, folder)

    large_w, large_h = slide.dimensions
    new_w = math.floor(large_w / scale_factor)
    new_h = math.floor(large_h / scale_factor)
    level = slide.get_best_level_for_downsample(scale_factor)
    whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
    whole_slide_image = whole_slide_image.convert("RGB")
    img = whole_slide_image.resize((new_w, new_h), PIL.Image.BILINEAR)
    return img, large_w, large_h, new_w, new_h


def slide_to_scaled_np_image(slide_f, folder=None):
    """
    Convert a WSI training slide to a scaled-down NumPy image.
    Args:
      slide_f: The slide filename.
      folder: The folder containing slide.
    Returns:
      Tuple consisting of scaled-down NumPy image, original width, original height, new width, and new height.
    """
    pil_img, large_w, large_h, new_w, new_h = slide_to_scaled_pil_image(slide_f, folder)
    np_img = pil_to_np(pil_img)
    return np_img, large_w, large_h, new_w, new_h


def small_to_large_mapping(small_pixel, large_dimensions, scale_factor=SCALE_FACTOR):
    """
    Map a scaled-down pixel width and height to the corresponding pixel of the original whole-slide image.
    Args:
      small_pixel: The scaled-down width and height.
      large_dimensions: The width and height of the original whole-slide image.
    Returns:
      Tuple consisting of the scaled-up width and height.
    """
    small_x, small_y = small_pixel
    large_w, large_h = large_dimensions
    large_x = round(
        (large_w / scale_factor)
        / math.floor(large_w / scale_factor)
        * (scale_factor * small_x)
    )
    large_y = round(
        (large_h / scale_factor)
        / math.floor(large_h / scale_factor)
        * (scale_factor * small_y)
    )
    return large_x, large_y


def show_slide(slide_f, scale_factor=SCALE_FACTOR):
    """
    Display a WSI slide on the screen, where the slide has been scaled down and converted to a PIL image.
    Args:
      slide_f: The slide file path.
      scale_factor: downsample factor
    """
    pil_img = slide_to_scaled_pil_image(slide_f, scale_factor=scale_factor)[0]
    pil_img.show()


def pil_to_np(pil_img):
    """
    Convert a PIL Image to a NumPy array.

    Note that RGB PIL (w, h) -> NumPy (h, w, 1) or (h, w, 3).

    Args:
        pil_img: The PIL Image.

    Returns:
        The PIL image converted to a NumPy array.
    """
    t = Timer()
    rgb = np.asarray(pil_img)
    np_info(rgb, "PIL to NumPy Array", t.elapsed())
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.

    Args:
        np_img: The image represented as a NumPy array.

    Returns:
    The NumPy array converted to a PIL Image.
    """
    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")
    return Image.fromarray(np_img)


def np_info(np_arr, name=None, elapsed=None, additional_np_stats=False, debug=False):
    """
    Display information (shape, type, max, min, etc) about a NumPy array.

    Args:
        np_arr: The NumPy array.
        name: The (optional) name of the array.
        elapsed: The (optional) time elapsed to perform a filtering operation.
        additional_np_stats: Print additional np stats.
        debug: show np_stats
    """

    if debug:
        if name is None:
            name = "NumPy Array"
        if elapsed is None:
            elapsed = "---"

        try:
            if np_arr is None:
                print("%-20s | Time: %-14s" % (name, str(elapsed)))
            elif additional_np_stats is False:
                if type(np_arr) is not np.ndarray:
                    np_arr = np.asarray(np_arr)
                print(
                    "%-20s | Time: %-14s  Type: %-7s Shape: %s"
                    % (name, str(elapsed), np_arr.dtype, np_arr.shape)
                )
            else:
                if type(np_arr) is not np.ndarray:
                    np_arr = np.asarray(np_arr)
                max = np_arr.max()
                min = np_arr.min()
                mean = np_arr.mean()
                is_binary = "T" if (np.unique(np_arr).size == 2) else "F"
                print(
                    "%-20s | Time: %-14s Min: %6.2f  Max: %6.2f  Mean: %6.2f  Binary: %s  Type: %-7s Shape: %s"
                    % (
                        name,
                        str(elapsed),
                        min,
                        max,
                        mean,
                        is_binary,
                        np_arr.dtype,
                        np_arr.shape,
                    )
                )
        except Exception as e:
            print("An error occurred at np_info:\n{}".format(e))


def display_img(
    np_img,
    text=None,
    font_path="../../assets/fonts/Arial-Bold.ttf",
    size=48,
    color=(255, 0, 0),
    background=(255, 255, 255),
    border=(0, 0, 0),
    bg=False,
):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.

    Args:
        np_img: Image as a NumPy array.
        text: The text to add to the image.
        font_path: The path to the font to use.
        size: The font size
        color: The font color
        background: The background color
        border: The border color
        bg: If True, add rectangle background behind text
    """
    if type(np_img == np.ndarray):
        result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == "L":
        result = result.convert("RGB")
    draw = ImageDraw.Draw(result)
    if text is not None:
        font = ImageFont.truetype(font_path, size)
        if bg:
            (x, y) = draw.textsize(text, font)
            draw.rectangle([(0, 0), (x + 5, y + 4)], fill=background, outline=border)
        draw.text((2, 0), text, color, font=font)
    result.show()


def get_thumbnail(
    slide_path,
    downsample=10,
    segmentation=False,
    return_contours=False,
    level=None,
    min_tissue_size=10000,
):
    """
    Get image thumbnail with segmented tissue contours
    Args:
        slide_path:
        downsample:
        segmentation:
        return_contours:
        level:
        min_tissue_size:
    Returns:
        thumb(, contours, d_factor)
    """
    import cv2 as cv

    slide = open_slide(os.path.basename(slide_path), os.path.dirname(slide_path))
    assert slide is not None

    # max_img_size = max(slide.level_dimensions[slide.get_best_level_for_downsample(downsample=downsample)])
    max_img_size = max(np.array(slide.dimensions) // downsample) + 1

    contours, d_factor = tl.locate_tissue_cnts(
        slide_path,
        max_img_size=max_img_size,
        smooth_sigma=13,
        thresh_val=0.80,
        min_tissue_size=min_tissue_size,
    )
    thumb = pil_to_np(
        slide.get_thumbnail(size=tuple(map(lambda xx: xx / d_factor, slide.dimensions)))
    )
    if segmentation:
        for contour in contours:
            cv.drawContours(thumb, contour, -1, (0, 255, 0), 3)

    if return_contours:
        return thumb, contours, d_factor

    return thumb


# TODO check if contours have tissue, else discard
def get_slide_tissue_mask(
    slide_path,
    downsample=10,
    segmentation=False,
    return_contours=False,
    return_thumbnail=False,
    min_tissue_size=10000,
):
    """
    Get image thumbnail with tissue masks
    Args:
        slide_path:
        downsample:
        segmentation:
        return_contours:
        min_tissue_size:
        return_thumbnail:
    Returns:
        mask(, contours, thumb, d_factor)
    """
    import cv2 as cv

    thumb, contours, d_factor = get_thumbnail(
        slide_path,
        downsample=downsample,
        segmentation=segmentation,
        return_contours=True,
        min_tissue_size=min_tissue_size,
    )
    mask = np.zeros(shape=thumb.shape[:2])

    cv.drawContours(mask, contours, -1, 1, thickness=-1)

    if return_contours and return_thumbnail:
        return mask, contours, thumb, d_factor
    elif return_contours:
        return mask, contours, d_factor
    elif return_thumbnail:
        return mask, thumb, d_factor
    else:
        return mask, d_factor


def get_tissue_centered(slide_path, downsample=10, offset=50):
    """
    Get each detected tissue with mask
    Args:
        slide_path:
        downsample:
        offset:
    Returns:
        List[(tissue, mask)]
    """
    mask, contours, thumb, d_factor = get_slide_tissue_mask(
        slide_path, downsample, return_contours=True, return_thumbnail=True
    )

    tissues = []

    for contour in contours:
        y, x = contour.min(axis=0)[0]
        h, w = contour.max(axis=0)[0]
        w = w - x
        h = h - y

        y1 = max(0, y - offset)
        y2 = min(thumb.shape[1], y + h + offset)

        x1 = max(0, x - offset)
        x2 = min(thumb.shape[0], x + w + offset)

        tissue = thumb[x1:x2, y1:y2, :]
        tissue_mask = mask[x1:x2, y1:y2]
        tissues.append((tissue, tissue_mask))

    return tissues


def get_tissue_mask(np_img, method="od"):
    if method == "morph":
        return rgb2gray_1(np_img)
    elif method == "od":
        return rgb2gray_2(np_img)
    else:
        raise ValueError("method should be either 'morph' or 'od'.")


def resize_max_size(img, max_size):
    scale = max_size / max(img.shape[:2])
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    return resize_np(img, dim)


def overlay_mask(img: Union[np.ndarray, Image.Image], mask: Union[np.ndarray, Image.Image], alpha=0.3, max_size=None):
    import cv2
    mask = np.array(mask)
    if mask.dtype == np.bool:
        mask = mask.astype(np.uint8) * 255
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = img * 255
        img = img.astype(np.uint8)
    if len(mask.shape) != len(img.shape):
        mask = np.dstack((mask,) * img.shape[-1])
    overlay = cv2.addWeighted(np.array(img), 1-alpha, mask, alpha, 0)
    if max_size is not None:
        overlay = resize_max_size(overlay, max_size)
    return overlay
