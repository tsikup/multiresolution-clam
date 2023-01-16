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
import functools
import multiprocessing
import os
from pathlib import Path
import math
import cv2 as cv
import numpy as np
import pandas as pd
import skimage
import skimage.color as sk_color
import skimage.exposure as sk_exposure
import skimage.feature as sk_feature
import skimage.filters as sk_filters
import skimage.future as sk_future
import skimage.morphology as sk_morphology
import skimage.segmentation as sk_segmentation
from PIL import Image, ImageDraw
from shapely import geometry
from definitions import (
    ROOT_DIR,
    FILTER_DIR,
    DEST_TRAIN_DIR,
    ALL_FILTERS_DIR,
    THUMBNAIL_DIR,
    IMAGE_EXT,
    THUMBNAIL_EXT,
    THUMBNAIL_SIZE,
    OUTPUT_IMG_DIR,
    bcolors,
)
from he_preprocessing.utils import image as ut_image
from he_preprocessing.utils.image import (
    DEBUG,
    np_info,
    pil_to_np,
    save_img,
    save_thumbnail,
    display_img,
)
from he_preprocessing.normalization.stain_norm import StainNormalizer
from he_preprocessing.utils.timer import Timer
from utils.chat import send_noti_to_telegram


def filter_rgb_to_grayscale(np_img, output_type="uint8", debug=False):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.

    Shape (h, w, c) to (h, w).

    Args:
      np_img: RGB Image as a NumPy array.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.

    Returns:
      Grayscale image as NumPy array with shape (h, w).
    """
    if debug or DEBUG:
        t = Timer()
    # # Another common RGB ratio possibility: [0.299, 0.587, 0.114]
    # grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    grayscale = ut_image.rgb2gray(np_img)
    if output_type == "float":
        grayscale = grayscale / 255.0
    else:
        pass
    if debug or DEBUG:
        np_info(grayscale, "Gray", t.elapsed(), debug=(debug or DEBUG))
    return grayscale


def filter_invert(np_img, output_type="uint8", debug=False):
    """
    Invert image array (black->white, white->black)

    Args:
      np_img: RGB Image as a NumPy array.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.

    Returns:
      Inverted image.
    """
    if debug or DEBUG:
        t = Timer()
    inverted = ut_image.invert(np_img, output_type)
    if output_type == "float":
        inverted = (inverted / 255).astype("float")
    else:
        pass
    if debug or DEBUG:
        np_info(inverted, "Inverted", t.elapsed(), debug=(debug or DEBUG))
    return inverted


def filter_contrast_stretch(np_img, low=40, high=60, debug=False):
    """
    Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in a specified range.
    Args:
      np_img: Image as a NumPy array (gray or RGB).
      low: Range low value (0 to 255).
      high: Range high value (0 to 255).
      debug: Show debug info if True.
    Returns:
      Image as NumPy array with contrast enhanced.
    """
    if debug or DEBUG:
        t = Timer()
    contrast_stretch = ut_image.contrast_stretch(np_img, low, high)
    if debug or DEBUG:
        np_info(
            contrast_stretch, "Contrast Stretch", t.elapsed(), debug=(debug or DEBUG)
        )
    return contrast_stretch


def filter_histogram_equalization(np_img, nbins=256, output_type="uint8", debug=False):
    """
    Filter image (gray or RGB) using histogram equalization to increase contrast in image.
    Args:
      np_img: Image as a NumPy array (gray or RGB).
      nbins: Number of histogram bins.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.
    Returns:
       NumPy array (float or uint8) with contrast enhanced by histogram equalization.
    """
    if debug or DEBUG:
        t = Timer()
    # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
    if np_img.dtype == "uint8" and nbins != 256:
        np_img = np_img / 255
    hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
    if np_img.dtype == "uint8":
        hist_equ = (hist_equ * 255).astype("uint8")
    if output_type == "float":
        pass
    else:
        hist_equ = (hist_equ * 255).astype("uint8")
    if debug or DEBUG:
        np_info(hist_equ, "Hist Equalization", t.elapsed(), debug=(debug or DEBUG))
    return hist_equ


def filter_adaptive_equalization(
    np_img, nbins=256, clip_limit=0.01, output_type="uint8", debug=False
):
    """
    Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions
    is enhanced.
    Args:
      np_img: Image as a NumPy array (gray or RGB).
      nbins: Number of histogram bins.
      clip_limit: Clipping limit where higher value increases contrast.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.
    Returns:
       NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
    """
    if debug or DEBUG:
        t = Timer()
    adapt_equ = sk_exposure.equalize_adapthist(
        np_img, nbins=nbins, clip_limit=clip_limit
    )
    if output_type == "float":
        pass
    else:
        adapt_equ = (adapt_equ * 255).astype("uint8")
    if debug or DEBUG:
        np_info(adapt_equ, "Adapt Equalization", t.elapsed(), debug=(debug or DEBUG))
    return adapt_equ


def filter_local_equalization(np_img, disk_size=50, debug=False):
    """
    Filter image (gray) using local equalization, which uses local histograms based on the disk structuring element.
    Args:
      np_img: Image as a NumPy array.
      disk_size: Radius of the disk structuring element used for the local histograms
      debug: Show debug info if True.
    Returns:
      NumPy array with contrast enhanced using local equalization.
    """
    if debug or DEBUG:
        t = Timer()
    if np_img.shape[-1] == 3:
        np_img = ut_image.rgb2gray(np_img)
    local_equ = sk_filters.rank.equalize(
        np_img, footprint=sk_morphology.disk(disk_size)
    )
    if debug or DEBUG:
        np_info(local_equ, "Local Equalization", t.elapsed(), debug=(debug or DEBUG))
    return local_equ


def filter_rgb_to_hed(np_img, output_type="uint8", debug=False):
    """
    Filter RGB channels to HED (Hematoxylin - Eosin - Diaminobenzidine) channels.
    Args:
      np_img: RGB image as a NumPy array.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (float or uint8) with HED channels.
    """
    if debug or DEBUG:
        t = Timer()
    hed = ut_image.rgb2hed(np_img)
    if output_type == "float":
        hed = sk_exposure.rescale_intensity(hed, out_range=(0.0, 1.0))
    else:
        hed = (sk_exposure.rescale_intensity(hed, out_range=(0, 255))).astype("uint8")
    if debug or DEBUG:
        np_info(hed, "RGB to HED", t.elapsed(), debug=(debug or DEBUG))
    return hed


def filter_rgb_to_hsd(np_img, debug=False):
    """
    Filter RGB channels to HSD (Hue, Saturation, Density).
    Ref:
        der Laak et al. Hue-Saturation-Density (HSD) Model for StainRecognition in Digital Images FromTransmitted Light Microscopy
    Args:
      np_img: RGB image as a NumPy array.
      debug: Show debug info if True.
    Returns:
      Image as NumPy array in HSD representation.
    """
    if debug or DEBUG:
        t = Timer()
    hsd = ut_image.rgb2hsd(np_img)
    if debug or DEBUG:
        np_info(hsd, "RGB to HSD", t.elapsed(), debug=(debug or DEBUG))
    return hsd


def filter_hsd_to_rgb(np_img, debug=False):
    """
    Filter HSD channels to RGB.
    Ref:
        der Laak et al. Hue-Saturation-Density (HSD) Model for StainRecognition in Digital Images FromTransmitted Light Microscopy
    Args:
      np_img: HSD image as a NumPy array.
      debug: Show debug info if True.
    Returns:
      Image as NumPy array in RGB representation.
    """
    if debug or DEBUG:
        t = Timer()
    rgb = ut_image.hsd2rgb(np_img)
    if debug or DEBUG:
        np_info(rgb, "HSD to RGB", t.elapsed(), debug=(debug or DEBUG))
    return rgb


def filter_rgb_to_hsv(np_img, debug=False):
    """
    Filter RGB channels to HSV (Hue, Saturation, Value).
    Args:
      np_img: RGB image as a NumPy array.
      debug: Show debug info if True.
    Returns:
      Image as NumPy array in HSV representation.
    """
    if debug or DEBUG:
        t = Timer()
    hsv = ut_image.rgb2hsv(np_img)
    if debug or DEBUG:
        np_info(hsv, "RGB to HSV", t.elapsed(), debug=(debug or DEBUG))
    return hsv


def filter_hsv_to_h(hsv, flatten=False, output_type="int", debug=False):
    """
    Obtain hue values from HSV NumPy array as a 1-dimensional array. If output as an int array, the original float values are multiplied by 360 for their degree equivalents for simplicity. For more information, see https://en.wikipedia.org/wiki/HSL_and_HSV
    Args:
      hsv: HSV image as a NumPy array.
      flatten: Control whether hue channel is flattened.
      output_type: Type of array to return (float or int).
      display_np_info: If True, display NumPy array info and filter time.
      debug: Show debug info if True.
    Returns:
      Hue values (float or int) as a 1-dimensional NumPy array.
    """
    if debug or DEBUG:
        t = Timer()
    h = hsv[:, :, 0]
    if flatten:
        h = h.flatten()
    if output_type == "int":
        h *= 360
        h = h.astype("int")
    if debug or DEBUG:
        np_info(hsv, "HSV to H", t.elapsed(), debug=(debug or DEBUG))
    return h


def filter_hsv_to_s(hsv, flatten=False, debug=False):
    """
    Experimental HSV to S (saturation).
    Args:
      hsv:  HSV image as a NumPy array.
      flatten: Control whether hue channel is flattened.
      debug: Show debug info if True.
    Returns:
      Saturation values as a 1-dimensional NumPy array.
    """
    if debug or DEBUG:
        t = Timer()
    s = hsv[:, :, 1]
    if flatten:
        s = s.flatten()
    if debug or DEBUG:
        np_info(hsv, "HSV to S", t.elapsed(), debug=(debug or DEBUG))
    return s


def filter_hsv_to_v(hsv, flatten=False, debug=False):
    """
    Experimental HSV to V (value).
    Args:
      hsv:  HSV image as a NumPy array.
      flatten: Control whether hue channel is flattened.
      debug: Show debug info if True.
    Returns:
      Value values as a 1-dimensional NumPy array.
    """
    if debug or DEBUG:
        t = Timer()
    v = hsv[:, :, 2]
    if flatten:
        v = v.flatten()
    if debug or DEBUG:
        np_info(hsv, "HSV to S", t.elapsed(), debug=(debug or DEBUG))
    return v


def filter_hed_to_hematoxylin(np_img, output_type="uint8", debug=False):
    """
    Obtain Hematoxylin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.
    Args:
      np_img: HED image as a NumPy array.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array for Hematoxylin channel.
    """
    if debug or DEBUG:
        t = Timer()
    hema = np_img[:, :, 0]
    if output_type == "float":
        hema = sk_exposure.rescale_intensity(hema, out_range=(0.0, 1.0))
    else:
        hema = (sk_exposure.rescale_intensity(hema, out_range=(0, 255))).astype("uint8")
    if debug or DEBUG:
        np_info(hema, "HED to Hematoxylin", t.elapsed(), debug=(debug or DEBUG))
    return hema


def filter_hed_to_eosin(np_img, output_type="uint8", debug=False):
    """
    Obtain Eosin channel from HED NumPy array and rescale it (for example, to 0 to 255 for uint8) for increased
    contrast.
    Args:
      np_img: HED image as a NumPy array.
      output_type: Type of array to return (float or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array for Eosin channel.
    """
    if debug or DEBUG:
        t = Timer()
    eosin = np_img[:, :, 1]
    if output_type == "float":
        eosin = sk_exposure.rescale_intensity(eosin, out_range=(0.0, 1.0))
    else:
        eosin = (sk_exposure.rescale_intensity(eosin, out_range=(0, 255))).astype(
            "uint8"
        )
    if debug or DEBUG:
        np_info(eosin, "HED to Eosin", t.elapsed(), debug=(debug or DEBUG))
    return eosin


def filter_kmeans_segmentation(np_img, compactness=10, n_segments=800, debug=False):
    """
    Use K-means segmentation (color/space proximity) to segment RGB image where each segment is colored based on the average color for that segment.
    Args:
      np_img: Binary image as a NumPy array.
      compactness: Color proximity versus space proximity factor.
      n_segments: The number of segments.
      debug: Show debug info if True.
    Returns:
      NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
      color for that segment.
    """
    if debug or DEBUG:
        t = Timer()
    labels = sk_segmentation.slic(
        np_img, compactness=compactness, n_segments=n_segments
    )
    result = sk_color.label2rgb(labels, np_img, kind="avg")
    if debug or DEBUG:
        np_info(result, "K-Means Segmentation", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_rag_threshold(
    np_img, compactness=10, n_segments=800, threshold=9, debug=False
):
    """
    Use K-means segmentation to segment RGB image, build region adjacency graph based on the segments, combine similar regions based on threshold value, and then output these resulting region segments.
    Args:
      np_img: Binary image as a NumPy array.
      compactness: Color proximity versus space proximity factor.
      n_segments: The number of segments.
      threshold: Threshold value for combining regions.
      debug: Show debug info if True.
    Returns:
      NumPy array (uint8) representing 3-channel RGB image where each segment has been colored based on the average
      color for that segment (and similar segments have been combined).
    """
    if debug or DEBUG:
        t = Timer()
    labels = sk_segmentation.slic(
        np_img, compactness=compactness, n_segments=n_segments
    )
    g = sk_future.graph.rag_mean_color(np_img, labels)
    labels2 = sk_future.graph.cut_threshold(labels, g, threshold)
    result = sk_color.label2rgb(labels2, np_img, kind="avg")
    if debug or DEBUG:
        np_info(result, "RAG Threshold", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_threshold(np_img, threshold, output_type="bool", debug=False):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
      np_img: Binary image as a NumPy array.
      threshold: The threshold value to exceed.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array
      pixel exceeds the threshold value.
    """
    if debug or DEBUG:
        t = Timer()
    result = np_img > threshold
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Threshold", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_green_channel(
    np_img,
    green_thresh=200,
    avoid_overmask=True,
    overmask_thresh=90,
    output_type="bool",
    debug=False,
):
    """
    Create a mask to filter out pixels with a green channel value greater than a particular threshold, since hematoxylin and eosin are purplish and pinkish, which do not have much green to them.
    Args:
      np_img: RGB image as a NumPy array.
      green_thresh: Green channel threshold value (0 to 255). If value is greater than green_thresh, mask out pixel.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing a mask where pixels above a particular green channel threshold have been masked out.
    """
    if debug or DEBUG:
        t = Timer()

    g = np_img[:, :, 1]
    gr_ch_mask = (g < green_thresh) & (g > 0)
    mask_percentage = mask_percent(gr_ch_mask)
    if (
        (mask_percentage >= overmask_thresh)
        and (green_thresh < 255)
        and (avoid_overmask is True)
    ):
        new_green_thresh = math.ceil((255 - green_thresh) / 2 + green_thresh)
        if debug or DEBUG:
            print(
                "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Green Channel green_thresh=%d, so try %d"
                % (mask_percentage, overmask_thresh, green_thresh, new_green_thresh)
            )
        gr_ch_mask = filter_green_channel(
            np_img, new_green_thresh, avoid_overmask, overmask_thresh, output_type
        )
    np_img = gr_ch_mask

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    if debug or DEBUG:
        np_info(np_img, "Filter Green Channel", t.elapsed(), debug=(debug or DEBUG))
    return np_img


def filter_red(
    rgb,
    red_lower_thresh,
    green_upper_thresh,
    blue_upper_thresh,
    output_type="bool",
    debug=False,
):
    """
    Create a mask to filter out reddish colors, where the mask is based on a pixel being above a
    red channel threshold value, below a green channel threshold value, and below a blue channel threshold value.
    Args:
      rgb: RGB image as a NumPy array.
      red_lower_thresh: Red channel lower threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_upper_thresh: Blue channel upper threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    r = rgb[:, :, 0] > red_lower_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] < blue_upper_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Red", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_red_pen(rgb, output_type="bool", debug=False):
    """
    Create a mask to filter out red pen marks from a slide.
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    result = (
        filter_red(
            rgb,
            red_lower_thresh=150,
            green_upper_thresh=80,
            blue_upper_thresh=90,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=110,
            green_upper_thresh=20,
            blue_upper_thresh=30,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=185,
            green_upper_thresh=65,
            blue_upper_thresh=105,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=195,
            green_upper_thresh=85,
            blue_upper_thresh=125,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=220,
            green_upper_thresh=115,
            blue_upper_thresh=145,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=125,
            green_upper_thresh=40,
            blue_upper_thresh=70,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=200,
            green_upper_thresh=120,
            blue_upper_thresh=150,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=100,
            green_upper_thresh=50,
            blue_upper_thresh=65,
            debug=debug,
        )
        & filter_red(
            rgb,
            red_lower_thresh=85,
            green_upper_thresh=25,
            blue_upper_thresh=45,
            debug=debug,
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Red Pen", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_green(
    rgb,
    red_upper_thresh,
    green_lower_thresh,
    blue_lower_thresh,
    output_type="bool",
    debug=False,
):
    """
    Create a mask to filter out greenish colors, where the mask is based on a pixel being below a
    red channel threshold value, above a green channel threshold value, and above a blue channel threshold value.
    Note that for the green ink, the green and blue channels tend to track together, so we use a blue channel
    lower threshold value rather than a blue channel upper threshold value.
    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_lower_thresh: Green channel lower threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: If True, display NumPy array info and filter time.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] > green_lower_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Green", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_green_pen(rgb, output_type="bool", debug=False):
    """
    Create a mask to filter out green pen marks from a slide.
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to eturn (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    result = (
        filter_green(
            rgb,
            red_upper_thresh=150,
            green_lower_thresh=160,
            blue_lower_thresh=140,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=70,
            green_lower_thresh=110,
            blue_lower_thresh=110,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=45,
            green_lower_thresh=115,
            blue_lower_thresh=100,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=30,
            green_lower_thresh=75,
            blue_lower_thresh=60,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=195,
            green_lower_thresh=220,
            blue_lower_thresh=210,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=225,
            green_lower_thresh=230,
            blue_lower_thresh=225,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=170,
            green_lower_thresh=210,
            blue_lower_thresh=200,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=10,
            green_lower_thresh=10,
            blue_lower_thresh=0,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=20,
            green_lower_thresh=30,
            blue_lower_thresh=20,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=50,
            green_lower_thresh=60,
            blue_lower_thresh=40,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=30,
            green_lower_thresh=50,
            blue_lower_thresh=35,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=65,
            green_lower_thresh=70,
            blue_lower_thresh=60,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=100,
            green_lower_thresh=110,
            blue_lower_thresh=105,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=165,
            green_lower_thresh=180,
            blue_lower_thresh=180,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=140,
            green_lower_thresh=140,
            blue_lower_thresh=150,
            debug=debug,
        )
        & filter_green(
            rgb,
            red_upper_thresh=185,
            green_lower_thresh=195,
            blue_lower_thresh=195,
            debug=debug,
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Green Pen", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_blue(
    rgb,
    red_upper_thresh,
    green_upper_thresh,
    blue_lower_thresh,
    output_type="bool",
    debug=False,
):
    """
    Create a mask to filter out blueish colors, where the mask is based on a pixel being below a
    red channel threshold value, below a green channel threshold value, and above a blue channel threshold value.
    Args:
      rgb: RGB image as a NumPy array.
      red_upper_thresh: Red channel upper threshold value.
      green_upper_thresh: Green channel upper threshold value.
      blue_lower_thresh: Blue channel lower threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: If True, display NumPy array info and filter time.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    r = rgb[:, :, 0] < red_upper_thresh
    g = rgb[:, :, 1] < green_upper_thresh
    b = rgb[:, :, 2] > blue_lower_thresh
    result = ~(r & g & b)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Blue", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_blue_pen(rgb, output_type="bool", debug=False):
    """
    Create a mask to filter out blue pen marks from a slide.
    Args:
      rgb: RGB image as a NumPy array.
      output_type: Type of array to eturn (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing the mask.
    """
    if debug or DEBUG:
        t = Timer()
    result = (
        filter_blue(
            rgb,
            red_upper_thresh=60,
            green_upper_thresh=120,
            blue_lower_thresh=190,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=120,
            green_upper_thresh=170,
            blue_lower_thresh=200,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=175,
            green_upper_thresh=210,
            blue_lower_thresh=230,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=145,
            green_upper_thresh=180,
            blue_lower_thresh=210,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=37,
            green_upper_thresh=95,
            blue_lower_thresh=160,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=30,
            green_upper_thresh=65,
            blue_lower_thresh=130,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=130,
            green_upper_thresh=155,
            blue_lower_thresh=180,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=40,
            green_upper_thresh=35,
            blue_lower_thresh=85,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=30,
            green_upper_thresh=20,
            blue_lower_thresh=65,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=90,
            green_upper_thresh=90,
            blue_lower_thresh=140,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=60,
            green_upper_thresh=60,
            blue_lower_thresh=120,
            debug=debug,
        )
        & filter_blue(
            rgb,
            red_upper_thresh=110,
            green_upper_thresh=110,
            blue_lower_thresh=175,
            debug=debug,
        )
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Blue Pen", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_grays(rgb, tolerance=15, output_type="bool", debug=False):
    """
    Create a mask to filter out pixels where the red, green, and blue channel values are similar.
    Args:
      rgb: RGB image as a NumPy array.
      tolerance: Tolerance value to determine how similar the values must be in order to be filtered out
      output_type: Type of array to eturn (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array representing a mask where pixels with similar red, green, and blue values have been masked out.
    """
    if debug or DEBUG:
        t = Timer()
    (h, w, c) = rgb.shape

    rgb = rgb.astype(np.int)
    rg_diff = abs(rgb[:, :, 0] - rgb[:, :, 1]) <= tolerance
    rb_diff = abs(rgb[:, :, 0] - rgb[:, :, 2]) <= tolerance
    gb_diff = abs(rgb[:, :, 1] - rgb[:, :, 2]) <= tolerance
    result = ~(rg_diff & rb_diff & gb_diff)

    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255
    if debug or DEBUG:
        np_info(result, "Filter Grays", t.elapsed(), debug=(debug or DEBUG))
    return result


def filter_remove_small_objects(
    np_img,
    min_size=3000,
    avoid_overmask=True,
    overmask_thresh=95,
    output_type="uint8",
    debug=False,
):
    """
    Filter image to remove small objects (connected components) less than a particular minimum size. If avoid_overmask
    is True, this function can recursively call itself with progressively smaller minimum size objects to remove to
    reduce the amount of masking that this filter performs.
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Minimum size of small object to remove.
      avoid_overmask: If True, avoid masking above the overmask_thresh percentage.
      overmask_thresh: If avoid_overmask is True, avoid masking above this threshold percentage value.
      output_type: Type of array to eturn (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8).
    """
    if debug or DEBUG:
        t = Timer()

    rem_sm = np_img.astype(bool)  # make sure mask is boolean
    rem_sm = sk_morphology.remove_small_objects(rem_sm, min_size=min_size)
    mask_percentage = mask_percent(rem_sm)
    if (
        (mask_percentage >= overmask_thresh)
        and (min_size >= 1)
        and (avoid_overmask is True)
    ):
        new_min_size = min_size / 2
        print(
            "Mask percentage %3.2f%% >= overmask threshold %3.2f%% for Remove Small Objs size %d, so try %d"
            % (mask_percentage, overmask_thresh, min_size, new_min_size)
        )
        rem_sm = filter_remove_small_objects(
            np_img, new_min_size, avoid_overmask, overmask_thresh, output_type
        )
    np_img = rem_sm

    if output_type == "bool":
        pass
    elif output_type == "float":
        np_img = np_img.astype(float)
    else:
        np_img = np_img.astype("uint8") * 255

    if debug or DEBUG:
        np_info(np_img, "Remove Small Objs", t.elapsed(), debug=(debug or DEBUG))
    return np_img


def filter_remove_small_holes(np_img, min_size=3000, output_type="uint8", debug=False):
    """
    Filter image to remove small holes less than a particular size.
    Args:
      np_img: Image as a NumPy array of type bool.
      min_size: Remove small holes below this size.
      output_type: Type of array to eturn (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8).
    """
    if debug or DEBUG:
        t = Timer()

    rem_sm = sk_morphology.remove_small_holes(np_img, area_threshold=min_size)

    if output_type == "bool":
        pass
    elif output_type == "float":
        rem_sm = rem_sm.astype(float)
    else:
        rem_sm = rem_sm.astype("uint8") * 255

    if debug or DEBUG:
        np_info(rem_sm, "Remove Small Holes", t.elapsed(), debug=(debug or DEBUG))
    return rem_sm


def filter_hysteresis_threshold(
    np_img, low=50, high=100, output_type="uint8", debug=False
):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.

    Args:
      np_img: Image as a NumPy array.
      low: Low threshold.
      high: High threshold.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """
    if debug or DEBUG:
        t = Timer()
    hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
    if output_type == "bool":
        pass
    elif output_type == "float":
        hyst = hyst.astype(float)
    else:
        hyst = (255 * hyst).astype("uint8")
    if debug or DEBUG:
        np_info(hyst, "Hysteresis Threshold", t.elapsed(), debug=(debug or DEBUG))
    return hyst


def filter_otsu_threshold(np_img, output_type="uint8", debug=False):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.

    Args:
      np_img: Image as a NumPy array.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """
    if debug or DEBUG:
        t = Timer()
    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = np_img > otsu_thresh_value
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255
    if debug or DEBUG:
        np_info(otsu, "Otsu Threshold", t.elapsed(), debug=(debug or DEBUG))
    return otsu


def filter_local_otsu_threshold(np_img, disk_size=3, output_type="uint8", debug=False):
    """
    Compute local Otsu threshold for each pixel and return binary image based on pixels being less than the local Otsu threshold.

    Args:
      np_img: Image as a NumPy array.
      disk_size: Radius of the disk structuring element used to compute the Otsu threshold for each pixel.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8) where local Otsu threshold values have been applied to original image.
    """
    if debug or DEBUG:
        t = Timer()
    if np_img.shape[-1] == 3:
        np_img = filter_rgb_to_grayscale(np_img)
    local_otsu = sk_filters.rank.otsu(np_img, sk_morphology.disk(disk_size))
    if output_type == "bool":
        pass
    elif output_type == "float":
        local_otsu = local_otsu.astype(float)
    else:
        local_otsu = local_otsu.astype("uint8") * 255
    if debug or DEBUG:
        np_info(local_otsu, "Otsu Local Threshold", t.elapsed(), debug=(debug or DEBUG))
    return local_otsu


def filter_entropy(
    np_img, neighborhood=9, threshold=5, output_type="uint8", debug=False
):
    """
    Filter image based on entropy (complexity).

    Args:
      np_img: Image as a NumPy array.
      neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
      threshold: Threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
    """
    if debug or DEBUG:
        t = Timer()
    if np_img.shape[-1] == 3:
        np_img = filter_rgb_to_grayscale(np_img)
    entr = (
        sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood)))
        > threshold
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        entr = entr.astype(float)
    else:
        entr = entr.astype("uint8") * 255
    if debug or DEBUG:
        np_info(entr, "Entropy", t.elapsed(), debug=(debug or DEBUG))
    return entr


def filter_canny(
    np_img,
    sigma=1,
    low_threshold=None,
    high_threshold=None,
    output_type="uint8",
    debug=False,
):
    """
    Filter image based on Canny algorithm edges.

    Args:
      np_img: Image as a NumPy array.
      sigma: Width (std dev) of Gaussian.
      low_threshold: Low hysteresis threshold value.
      high_threshold: High hysteresis threshold value.
      output_type: Type of array to return (bool, float, or uint8).
      debug: Show debug info if True.
    Returns:
      NumPy array (bool, float, or uint8) representing Canny edge map (binary image).
    """
    if debug or DEBUG:
        t = Timer()
    if np_img.shape[-1] == 3:
        np_img = filter_rgb_to_grayscale(np_img)
    can = sk_feature.canny(
        np_img, sigma=sigma, low_threshold=low_threshold, high_threshold=high_threshold
    )
    if output_type == "bool":
        pass
    elif output_type == "float":
        can = can.astype(float)
    else:
        can = can.astype("uint8") * 255
    if debug or DEBUG:
        np_info(can, "Canny Edges", t.elapsed(), debug=(debug or DEBUG))
    return can


def mask_percent(np_img, debug=False):
    """
    Determine the percentage of a NumPy array that is masked (how many of the values are 0 values).

    Args:
      np_img: Image as a NumPy array.
      debug: Show debug info if True.
    Returns:
      The percentage of the NumPy array that is masked.
    """
    if debug or DEBUG:
        t = Timer()
    if (len(np_img.shape) == 3) and (np_img.shape[2] == 3):
        np_sum = np_img[:, :, 0] + np_img[:, :, 1] + np_img[:, :, 2]
        mask_percentage = 100 - np.count_nonzero(np_sum) / np_sum.size * 100
    else:
        mask_percentage = 100 - np.count_nonzero(np_img) / np_img.size * 100
    if debug or DEBUG:
        np_info(mask_percentage, "Mask Percentage", t.elapsed(), debug=(debug or DEBUG))
    return mask_percentage


def draw_tiles(
    image, tile_size=128, tissue_threshold=0.5, thickness=3, color=(255, 0, 0)
):
    """
    Function to draw tiles in an image
    Args:
        image: Image numpy array
        tile_size: Size of a tile
        tissue_threshold: Threshold under which a tile is discarded as it doen't contain enough tissue
        thickness: thickness of the tile contour
    Return:
        Image with tile contours
    """
    size = image.shape[:2]
    _img = image.copy()

    # get current x, y
    x = size[0]
    y = size[1]

    left = 0
    for m in range(math.ceil(x / tile_size)):
        upper = 0
        for n in range(math.ceil(y / tile_size)):
            right = left + tile_size
            lower = upper + tile_size

            left_chk = min(max(left, 0), x)
            upper_chk = min(max(upper, 0), y)
            right_chk = min(right, x)
            lower_chk = min(lower, y)

            box = (left_chk, upper_chk, right_chk, lower_chk)

            subImg = _img[left_chk:right_chk, upper_chk:lower_chk, :]

            if subImg.shape[0] <= 0 or subImg.shape[1] <= 0:
                continue

            if keep_tile(np.array(subImg), tile_size, tissue_threshold, pad=False):
                image = cv.rectangle(
                    image,
                    (upper_chk, left_chk),
                    (lower_chk, right_chk),
                    color,
                    thickness,
                )

            upper += tile_size
        left += tile_size

    return image


def keep_tile(
    tile,
    tile_size,
    tissue_threshold,
    pens_threshold=0.2,
    roi_mask=None,
    pad=False,
    debug=False,
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
      debug: Show debug info if True.
    Returns:
      A Boolean indicating whether a tile should be kept for future usage.
    """
    if debug or DEBUG:
        t = Timer()
    check = ut_image.keep_tile(
        tile, tile_size, tissue_threshold, pens_threshold, roi_mask, pad
    )
    if debug or DEBUG:
        np_info(None, "Keep tile", t.elapsed(), debug=(debug or DEBUG))
    return check


def tissue_percent_otsu(np_img, threshold=None, debug=False):
    """
    Determine the percentage of a NumPy array that is tissue.

    Args:
      np_img: Image as a NumPy array.
      threshold: Threshold to binarize image.
      debug: Show debug info if True.
    Returns:
      The percentage of the NumPy array that is tissue.
    """
    if debug or DEBUG:
        t = Timer()
    if threshold is None:
        threshold = otsu_threshold(np_img, debug=debug)

    np_tile_hist = bw_histogram(np_img, debug=debug)
    tissue_perc = np.sum(np_tile_hist[:threshold]) / np.sum(np_tile_hist)
    # tissue_perc = np.round(tissue_perc, decimals=5) if tissue_perc > 0 else 0
    if debug or DEBUG:
        np_info(None, "Tissue Percentage Otsu", t.elapsed(), debug=(debug or DEBUG))
    return tissue_perc


def background_percent_otsu(np_img, threshold=None, debug=False):
    """
    Determine the percentage of a NumPy array that is not tissue.

    Args:
      np_img: Image as a NumPy array.
      threshold: Threshold to binarize image.
      debug: Show debug info if True.
    Returns:
      The percentage of the NumPy array that is not tissue.
    """
    if debug or DEBUG:
        t = Timer()
    if debug or DEBUG:
        np_info(None, "Background Percentage Otsu", t.elapsed(), debug=(debug or DEBUG))
    return 1.0 - tissue_percent_otsu(np_img, threshold, debug=debug)


def bw_histogram(np_img, max_value="auto", debug=False):
    """
    Get the histogram of the grayscale version of an image

    Args:
      np_img: Image as a NumPy array.
      max_value: Maximum value that can be observed in the NumPy image.
                 Auto means that it will be either 255 if maximum value
                 is more than 1, otherwise it will be 1. It will be either
                 255 for 8-bit images or 1 if the image is normalized.
      debug: Show debug info if True.
    Returns:
      histogram
    """
    if debug or DEBUG:
        t = Timer()
    np_hist = ut_image.bw_histogram(np_img, max_value)
    if debug or DEBUG:
        np_info(None, "BW Histogram", t.elapsed(), debug=(debug or DEBUG))
    return np_hist


def otsu_threshold(np_img, debug=False):
    """
    Determine the otsu's threshold to binarize the image.

    Args:
      np_img: Image as a NumPy array.
      debug: Show debug info if True.
    Returns:
      Threshold value.
    """
    # heavily inspired by scikit-image otsu implementation
    # https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py#L237
    if debug or DEBUG:
        t = Timer()
    np_hist = bw_histogram(np_img, debug=debug)

    bins = np.arange(256)

    # class probabilities for all possible thresholds
    w1 = np.cumsum(np_hist)
    w2 = np.cumsum(np_hist[::-1])[::-1]

    # class means for all possible thresholds
    # https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero
    a1 = np.cumsum(np_hist * bins)
    mean1 = np.divide(a1, w1, out=np.zeros_like(a1), where=w1 != 0)

    a2 = np.cumsum((np_hist * bins)[::-1])
    mean2 = np.divide(a2, w2[::-1], out=np.zeros_like(a2), where=w2 != 0)[::-1]

    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = w1[:-1] * w2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    # print(variance12)

    idx = np.argmax(variance12)
    otsu_th = bins[:-1][idx]
    if debug or DEBUG:
        np_info(None, "Otsu Threshold", t.elapsed(), debug=(debug or DEBUG))
    return otsu_th


# TODO: Optimize it because there are too many false positives
def get_microtome_artifact(img, debug=False):
    """
    Get microtome artifact mask.
    Args:
        img: input image array
        debug: debug boolean
    Returns:
        mask for microtome artifact
    """
    if debug or DEBUG:
        t = Timer()
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    ret2, gray = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((30, 30), np.uint8)
    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, kernel)
    closing = 255 - closing
    kernel = np.ones((10, 10), np.uint8)
    microtome_mask = cv.morphologyEx(closing, cv.MORPH_CLOSE, kernel)
    if debug or DEBUG:
        np_info(microtome_mask, "Mask RGB", t.elapsed(), debug=(debug or DEBUG))
    return np.array(255 - microtome_mask)


def generate_tile_cell_mask(img_path, out_path, debug=False):
    tile_idx = os.path.basename(img_path).split("tile_")[-1].split(" (")[0]
    out_path = os.path.join(out_path, f"tile_{tile_idx}")

    # if os.path.exists(os.path.join(out_path, 'cellsMask.png')) and os.path.exists(os.path.join(out_path, 'nucleiMask.png')):
    #     print(f'Masks exist for {img_path} at {out_path}. Skipping...')
    #     return None, None

    print(f"Processing {img_path}")
    Path(out_path).mkdir(parents=True, exist_ok=True)

    downsample_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
    downsample = float(os.path.basename(downsample_dir).split("ds_")[1])
    tile_cells_csv = pd.read_csv(
        os.path.join(
            os.path.dirname(os.path.dirname(img_path)),
            "tile_cells_csv",
            f"cells_info_tile_{tile_idx}.csv",
        )
    )
    if not tile_cells_csv.empty:
        tile_cells_csv.sort_values(by=["cellID"], ignore_index=True, inplace=True)
    cell_colors_df = pd.read_csv(
        os.path.join(ROOT_DIR, "assets", "cell_color_coding.csv")
    )

    tile_x = int(img_path.split("x=")[1].split(",")[0])
    tile_y = int(img_path.split("y=")[1].split(",")[0])
    tile_w = int(img_path.split("w=")[1].split(",")[0])
    tile_h = int(img_path.split("h=")[1].split(",")[0])

    slide_width, slide_height = np.array([tile_x, tile_y]) + np.array([tile_w, tile_h])
    slide_width, slide_height = np.round(slide_width / downsample), np.round(
        slide_height / downsample
    )

    nucleusOverlay = Image.new(
        "RGB", (int(tile_w // downsample), int(tile_h // downsample)), "#000000"
    )
    cellOverlay = Image.new(
        "RGB", (int(tile_w // downsample), int(tile_h // downsample)), "#000000"
    )

    for _, cell in tile_cells_csv.iterrows():
        try:
            cell_type = cell["cellClass"]

            coreID = cell["coreID"]
            cellID = cell["cellID"]

            # Nucleus ROI
            nucleusX = np.array(
                [
                    float(coord)
                    for coord in cell["nucleus_roi_x"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            nucleusY = np.array(
                [
                    float(coord)
                    for coord in cell["nucleus_roi_y"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            nucleusPoints = np.array([nucleusX, nucleusY]).transpose()
            nucleusPolygon = geometry.Polygon(nucleusPoints)

            x, y = nucleusPolygon.exterior.xy
            nucleusXY = [
                (
                    np.round((slide_width - x1 / downsample)).astype(int),
                    np.round((slide_height - y1 / downsample)).astype(int),
                )
                for x1, y1 in zip(x, y)
            ]

            overlay1 = ImageDraw.Draw(nucleusOverlay)
            overlay1.polygon(
                nucleusXY,
                fill=cell_colors_df[cell_type].get(0),
                outline=cell_colors_df[cell_type].get(0),
            )

            # Cell ROI
            cellX = np.array(
                [
                    float(coord)
                    for coord in cell["cell_roi_x"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            cellY = np.array(
                [
                    float(coord)
                    for coord in cell["cell_roi_y"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )

            cellX = cellX if cellX is not np.NAN else nucleusX
            cellY = cellY if cellY is not np.NAN else nucleusY

            cellPoints = np.array([cellX, cellY]).transpose()
            cellPolygon = geometry.Polygon(cellPoints)

            x, y = cellPolygon.exterior.xy
            cellXY = [
                (
                    np.round((slide_width - x1 / downsample)).astype(int),
                    np.round((slide_height - y1 / downsample)).astype(int),
                )
                for x1, y1 in zip(x, y)
            ]

            overlay2 = ImageDraw.Draw(cellOverlay)
            overlay2.polygon(
                cellXY,
                fill=cell_colors_df[cell_type].get(0),
                outline=cell_colors_df[cell_type].get(0),
            )
        except Exception as e:
            print(
                f"exception occured for img: {img_path} and cell: {coreID}:{cellID}, skipping this cell only"
            )

    nucleusOverlay = nucleusOverlay.transpose(Image.FLIP_TOP_BOTTOM)
    nucleusOverlay = nucleusOverlay.transpose(Image.FLIP_LEFT_RIGHT)
    nucleusOverlay.save(os.path.join(out_path, "nucleiMask.png"))

    cellOverlay = cellOverlay.transpose(Image.FLIP_TOP_BOTTOM)
    cellOverlay = cellOverlay.transpose(Image.FLIP_LEFT_RIGHT)
    cellOverlay.save(os.path.join(out_path, "cellsMask.png"))

    return np.array(nucleusOverlay), np.array(cellOverlay)


def generate_cell_mask(img_path, out_path, save_colors=False, debug=False):
    if os.path.exists(os.path.join(out_path, "cellsMask.png")) and os.path.exists(
        os.path.join(out_path, "nucleiMask.png")
    ):
        print(f"Masks exist for {img_path} at {out_path}. Skipping...")
        return None, None

    Path(out_path).mkdir(parents=True, exist_ok=True)

    downsample_dir = os.path.dirname(os.path.dirname(os.path.dirname(img_path)))
    corepath = os.path.dirname(downsample_dir)
    downsample = float(os.path.basename(downsample_dir).split("ds_")[1])

    cells_csv = pd.read_csv(os.path.join(corepath, "cells_info.csv"))
    cells_csv.sort_values(by=["cellID"], ignore_index=True, inplace=True)

    cell_colors_df = pd.read_csv(
        os.path.join(ROOT_DIR, "assets", "cell_color_coding.csv")
    )

    tissue_coordinates_df = pd.read_csv(
        os.path.join(corepath, "tissue_coordinates.csv")
    )
    slide_width, slide_height = (
        tissue_coordinates_df[["x", "y"]].values[0]
        + tissue_coordinates_df[["w", "h"]].values[0]
    )
    slide_width, slide_height = np.round(slide_width / downsample), np.round(
        slide_height / downsample
    )

    img_w, img_h = tuple(tissue_coordinates_df[["w", "h"]].values[0])
    nucleusOverlay = Image.new(
        "RGB", (int(img_w // downsample), int(img_h // downsample)), "#000000"
    )
    cellOverlay = Image.new(
        "RGB", (int(img_w // downsample), int(img_h // downsample)), "#000000"
    )

    for _, cell in cells_csv.iterrows():
        try:
            cell_type = cell["cellClass"]
            coreID = cell["coreID"]
            cellID = cell["cellID"]

            # Nucleus ROI
            nucleusX = np.array(
                [
                    float(coord)
                    for coord in cell["nucleus_roi_x"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            nucleusY = np.array(
                [
                    float(coord)
                    for coord in cell["nucleus_roi_y"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            nucleusPoints = np.array([nucleusX, nucleusY]).transpose()
            nucleusPolygon = geometry.Polygon(nucleusPoints)

            x, y = nucleusPolygon.exterior.xy
            nucleusXY = [
                (
                    np.round((slide_width - x1 / downsample)).astype(int),
                    np.round((slide_height - y1 / downsample)).astype(int),
                )
                for x1, y1 in zip(x, y)
            ]

            overlay1 = ImageDraw.Draw(nucleusOverlay)
            overlay1.polygon(
                nucleusXY,
                fill=cell_colors_df[cell_type].get(0),
                outline=cell_colors_df[cell_type].get(0),
            )

            # Cell ROI
            cellX = np.array(
                [
                    float(coord)
                    for coord in cell["cell_roi_x"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )
            cellY = np.array(
                [
                    float(coord)
                    for coord in cell["cell_roi_y"]
                    .split("[")[1]
                    .split("]")[0]
                    .split(",")
                ]
            )

            cellPoints = np.array([cellX, cellY]).transpose()
            cellPolygon = geometry.Polygon(cellPoints)

            x, y = cellPolygon.exterior.xy
            cellXY = [
                (
                    np.round((slide_width - x1 / downsample)).astype(int),
                    np.round((slide_height - y1 / downsample)).astype(int),
                )
                for x1, y1 in zip(x, y)
            ]

            overlay2 = ImageDraw.Draw(cellOverlay)
            overlay2.polygon(
                cellXY,
                fill=cell_colors_df[cell_type].get(0),
                outline=cell_colors_df[cell_type].get(0),
            )
        except Exception as e:
            print(
                f"exception occured for img: {img_path} and cell: {coreID}:{cellID}, skipping this cell only"
            )

    nucleusOverlay = nucleusOverlay.transpose(Image.FLIP_TOP_BOTTOM)
    nucleusOverlay = nucleusOverlay.transpose(Image.FLIP_LEFT_RIGHT)
    nucleusOverlay.save(os.path.join(out_path, "nucleiMask.png"))

    cellOverlay = cellOverlay.transpose(Image.FLIP_TOP_BOTTOM)
    cellOverlay = cellOverlay.transpose(Image.FLIP_LEFT_RIGHT)
    cellOverlay.save(os.path.join(out_path, "cellsMask.png"))

    return np.array(nucleusOverlay), np.array(cellOverlay)


def apply_image_filters(
    np_img,
    roi_mask=None,
    filters2apply=None,
    slide=None,
    image_name=None,
    info=None,
    save=False,
    display=False,
    output_dir=FILTER_DIR,
    debug=False,
):
    """
    Apply filters to image as NumPy array and optionally save and/or display filtered images.
    Args:
      np_img: Image as NumPy array.
      roi_mask: ROI masking annotation in the image.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      image_name: The image file name (used for saving/displaying).
      save: If True, save image.
      display: If True, display image.
      output_dir: Output directory to save image.
      debug: Show debug info if True.
    Returns:
      Resulting filtered image as a NumPy array.
    """
    if debug or DEBUG:
        t = Timer()
    if filters2apply is None:
        filters2apply = {
            "tileSize": 512,
            "blurriness_threshold": 100,
            "apply_mask": False,
            "mask_background": [255, 255, 255],
            "green": False,
            "grays": False,
            "redPen": False,
            "greenPen": False,
            "bluePen": False,
            "remove_microtome_artifacts": False,
            "remove_small_objects": False,
            "stain_norm": True,
            "stain_norm_luminosity": True,
            "stain_norm_method": "macenko",
            "keep_tile_percentage": None,
        }
    rgb = np_img
    save_display(
        save,
        display,
        info,
        rgb,
        image_name,
        0,
        "Original",
        "rgb",
        output_dir=output_dir,
    )

    masks = []

    if filters2apply["green"]:
        mask_not_green = filter_green_channel(rgb, debug=debug)
        masks.append(mask_not_green)
        rgb_not_green = ut_image.mask_rgb(rgb, mask_not_green)
        save_display(
            save,
            display,
            info,
            rgb_not_green,
            image_name,
            1,
            "Not Green",
            "rgb-not-green",
            output_dir=output_dir,
        )

    if filters2apply["grays"]:
        mask_not_gray = filter_grays(rgb, debug=debug)
        masks.append(mask_not_gray)
        rgb_not_gray = ut_image.mask_rgb(rgb, mask_not_gray)
        save_display(
            save,
            display,
            info,
            rgb_not_gray,
            image_name,
            2,
            "Not Gray",
            "rgb-not-gray",
            output_dir=output_dir,
        )

    if filters2apply["redPen"]:
        mask_no_red_pen = filter_red_pen(rgb, debug=debug)
        mask_no_red_pen = skimage.morphology.binary_closing(mask_no_red_pen)
        masks.append(mask_no_red_pen)
        rgb_no_red_pen = ut_image.mask_rgb(rgb, mask_no_red_pen)
        save_display(
            save,
            display,
            info,
            rgb_no_red_pen,
            image_name,
            3,
            "No Red Pen",
            "rgb-no-red-pen",
            output_dir=output_dir,
        )

    if filters2apply["greenPen"]:
        mask_no_green_pen = filter_green_pen(rgb, debug=debug)
        mask_no_green_pen = skimage.morphology.binary_closing(mask_no_green_pen)
        masks.append(mask_no_green_pen)
        rgb_no_green_pen = ut_image.mask_rgb(rgb, mask_no_green_pen)
        save_display(
            save,
            display,
            info,
            rgb_no_green_pen,
            image_name,
            4,
            "No Green Pen",
            "rgb-no-green-pen",
            output_dir=output_dir,
        )

    if filters2apply["bluePen"]:
        mask_no_blue_pen = filter_blue_pen(rgb, debug=debug)
        mask_no_blue_pen = skimage.morphology.binary_closing(mask_no_blue_pen)
        masks.append(mask_no_blue_pen)
        rgb_no_blue_pen = ut_image.mask_rgb(rgb, mask_no_blue_pen)
        save_display(
            save,
            display,
            info,
            rgb_no_blue_pen,
            image_name,
            5,
            "No Blue Pen",
            "rgb-no-blue-pen",
            output_dir=output_dir,
        )

    if filters2apply["remove_microtome_artifacts"]:
        mask_microtome_artifact = get_microtome_artifact(rgb, debug=debug)
        masks.append(mask_microtome_artifact)
        rgb_no_microtome_artifact = ut_image.mask_rgb(rgb, mask_microtome_artifact)
        save_display(
            save,
            display,
            info,
            rgb_no_microtome_artifact,
            image_name,
            6,
            "No Microtome Artifact",
            "rgb-no-microtome-artifact",
            output_dir=output_dir,
        )

    if filters2apply["stain_norm"]:
        stain_normalizer = StainNormalizer(
            luminosity=filters2apply["stain_norm_luminosity"],
            method=filters2apply["stain_norm_method"],
            reference_dir=filters2apply["stain_norm_reference_dir"],
        )
        rgb = stain_normalizer.transform(rgb.astype(np.uint8), slide=slide)
        if filters2apply["stain_norm_mask"]:
            rgb = ut_image.mask_rgb(rgb, ut_image.get_tissue_mask(rgb), background=filters2apply["constant_pad_value"])
        save_display(
            save,
            display,
            info,
            rgb,
            image_name,
            7,
            "Stain Normalization",
            "rgb-stain-norm",
            output_dir=output_dir,
        )

    if filters2apply["remove_small_objects"] and not masks:
        gray_mask = filter_threshold(
            filter_invert(filter_rgb_to_grayscale(rgb, debug=debug)), 30, debug=debug
        )
        masks.append(gray_mask)
        rgb_gray = ut_image.mask_rgb(rgb, gray_mask)
        save_display(
            save,
            display,
            info,
            rgb_gray,
            image_name,
            8,
            "Gray Binarized",
            "gray-binarized",
            output_dir=output_dir,
        )

    if masks:
        all_masks = functools.reduce(lambda x, y: x & y, masks)
        rgb = ut_image.mask_rgb(
            rgb, all_masks, background=filters2apply["mask_background"]
        )
        save_display(
            save,
            display,
            info,
            rgb,
            image_name,
            9,
            "All Masks",
            "rgb-all-masks",
            output_dir=output_dir,
        )

    if filters2apply["remove_small_objects"]:
        mask_remove_small = filter_remove_small_objects(
            all_masks, min_size=500, output_type="bool", debug=debug
        )
        rgb_remove_small = ut_image.mask_rgb(rgb, mask_remove_small)
        save_display(
            save,
            display,
            info,
            rgb_remove_small,
            image_name,
            10,
            "All Masks,\nRemove Small Objects",
            "rgb-all-masks-remove-small",
            output_dir=output_dir,
        )
        img = rgb_remove_small
    else:
        img = rgb

    if roi_mask is not None and filters2apply["apply_mask"]:
        roi_mask = ut_image.uint8_to_bool(roi_mask)
        img = ut_image.mask_rgb(
            img, roi_mask, background=filters2apply["mask_background"]
        )
        save_display(
            save,
            display,
            info,
            img,
            image_name,
            11,
            "ROI Mask",
            "rgb-roi-mask",
            output_dir=output_dir,
        )

    if filters2apply["tileSize"] is not None and (
        img.shape[0] < filters2apply["tileSize"]
        or img.shape[1] < filters2apply["tileSize"]
    ):
        img = ut_image.pad_image(
            img, filters2apply["tileSize"], filters2apply["constant_pad_value"], debug
        )
        save_display(
            save,
            display,
            info,
            img,
            image_name,
            12,
            "Padded filtered image",
            "rgb-filtered-padded",
            output_dir=output_dir,
        )

    if debug or DEBUG:
        np_info(
            img, "Apply predefined image filters", t.elapsed(), debug=(debug or DEBUG)
        )
    return img


def apply_filters_to_slide(
    slide_f, filters2apply=None, downsample=32, save=False, display=False, debug=False
):
    """
    Apply a set of filters to a slide and optionally save and/or display filtered images.
    Args:
      slide_f: The slide filepath.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      save: If True, save filtered images.
      display: If True, display filtered images to screen.
    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    t = Timer()
    imageName = os.path.basename(slide_f).split(".")[0]
    print(f"Processing slide {imageName}")

    info = dict()

    if save and not os.path.exists(DEST_TRAIN_DIR):
        os.makedirs(DEST_TRAIN_DIR)
    np_orig, d_factor = ut_image.open_slide_np(slide_f, downsample=downsample)
    filtered_np_img = apply_image_filters(
        np_orig,
        roi_mask=None,
        filters2apply=filters2apply,
        image_name=imageName,
        info=info,
        save=save,
        display=display,
    )

    # if save:
    #     if debug or DEBUG:
    #         t1 = Timer()
    #     result_path = get_filter_image_result(imageName)
    #     pil_img = ut_image.np_to_pil(filtered_np_img)
    #     pil_img.save(result_path)
    #     print("%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t1.elapsed()), result_path))
    #
    #     if debug or DEBUG:
    #         t1 = Timer()
    #     thumbnail_path = get_filter_thumbnail_result(slide_num)
    #     save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_path)
    #     if debug or DEBUG:
    #         print("%-20s | Time: %-14s  Name: %s" % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path))

    if debug or DEBUG:
        print("Slide #%03d processing time: %s\n" % (imageName, str(t.elapsed())))

    return filtered_np_img


def apply_filters_to_image(
    np_img,
    roi_f=None,
    slide=None,
    filters2apply=None,
    save=True,
    output_dir=FILTER_DIR,
    output_dir_filters=ALL_FILTERS_DIR,
    output_roi_dir=None,
    thumbnail_dir=THUMBNAIL_DIR,
    image_name=None,
    image_ext=IMAGE_EXT,
    save_each_filter=False,
    display=False,
    debug=False,
):
    """
    Apply a set of filters to an image and optionally save and/or display filtered images.
    Args:
      np_img: Image as NumPy array.
      roi_f: ROI mask filepath.
      slide: corresponding slide name.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      save: If True, save filtered images.
      output_dir: Output directory to save image.
      output_dir_filters: Output directory to save each filter step.
      output_roi_dir: Output directory to save roi mask.
      thumbnail_dir: Output directory to save thumbnail.
      image_name: Image file name.
      image_ext: Extension to save image.
      save_each_filter: If True, save each filtered image.
      display: If True, display filtered images to screen.
      debug: Show debug info if True.
    Returns:
      Tuple consisting of 1) the resulting filtered image as a NumPy array, and 2) dictionary of image information
      (used for HTML page generation).
    """
    if debug or DEBUG:
        t = Timer()
    if image_name is not None:
        print("Processing image %s" % image_name)
    else:
        image_name = "no_name"

    info = dict()

    if save and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if type(np_img) is not np.ndarray:
        np_img = pil_to_np(np_img)

    np_roi = ut_image.open_image_np(roi_f) if roi_f is not None else None
    if np_roi is not None and type(np_roi) is not np.ndarray:
        np_roi = pil_to_np(np_roi)

    np_orig = np_img

    if save:
        result_path = os.path.join(output_dir, image_name + "." + image_ext)
        #
        # if os.path.exists(result_path):
        #     return None, None

    # if filters2apply['keep_tile_percentage'] is not None:
    #     if not keep_tile(np_orig, tile_size=filters2apply["tileSize"], tissue_threshold=filters2apply['keep_tile_percentage'], roi_mask=np_roi, pad=True, debug=debug):
    #         print("%s Name: %s. Discarding...\n" % ("Tile tissue percentage below threshold.", image_name))
    #         return None, None

    filtered_np_img = apply_image_filters(
        np_orig,
        roi_mask=np_roi,
        filters2apply=filters2apply,
        image_name=image_name,
        info=info,
        slide=slide,
        save=save_each_filter,
        display=display,
        output_dir=output_dir_filters,
        debug=debug,
    )

    if save:
        # # If image is blurry, discard it.
        # if filters2apply["blurriness_threshold"] is not None:
        #     if ut_image.is_blurry(np_img, threshold=filters2apply["blurriness_threshold"], verbose=0, debug=debug):
        #         print("Image %s is blurry. Discarding...\n" % image_name)
        #         return None, None

        if debug or DEBUG:
            t1 = Timer()
        save_img(filtered_np_img, result_path)
        if output_roi_dir is not None:
            Path(output_roi_dir).mkdir(parents=True, exist_ok=True)
            roi_result_path = os.path.join(
                output_roi_dir, image_name + "_mask." + image_ext
            )
            os.symlink(roi_f, roi_result_path)
        if debug or DEBUG:
            print(
                "%-20s | Time: %-14s  Name: %s"
                % ("Save Image", str(t1.elapsed()), result_path)
            )

        if thumbnail_dir is not None:
            if debug or DEBUG:
                t1 = Timer()
            thumbnail_path = os.path.join(
                thumbnail_dir, image_name + "." + THUMBNAIL_EXT
            )
            save_thumbnail(filtered_np_img, THUMBNAIL_SIZE, thumbnail_path)
            if debug or DEBUG:
                print(
                    "%-20s | Time: %-14s  Name: %s"
                    % ("Save Thumbnail", str(t1.elapsed()), thumbnail_path)
                )

    if debug or DEBUG:
        print("Image %s processing time: %s\n" % (image_name, str(t.elapsed())))
        np_info(
            filtered_np_img,
            "Apply filters to image",
            t.elapsed(),
            debug=(debug or DEBUG),
        )
    return filtered_np_img, info


def save_display(
    save,
    display,
    info,
    np_img,
    image_name,
    filter_num,
    display_text,
    file_text,
    output_dir=OUTPUT_IMG_DIR,
    thumbnail_dir=THUMBNAIL_DIR,
    image_ext=IMAGE_EXT,
    display_mask_percentage=True,
    debug=False,
):
    """
    Optionally save an image and/or display the image.
    Args:
      save: If True, save filtered images.
      display: If True, display filtered images to screen.
      info: Dictionary to store filter information.
      np_img: Image as a NumPy array.
      image_name: Image file name.
      filter_num: The filter number.
      display_text: Filter display name.
      file_text: Filter name for file.
      output_dir: Output directory to save image.
      thumbnail_dir: Output directory to save thumbnail.
      image_ext: Extension to save image.
      display_mask_percentage: If True, display mask percentage on displayed slide.
      debug: Show debug info if True.
    """
    mask_percentage = None
    if display_mask_percentage:
        mask_percentage = mask_percent(np_img)
        display_text = (
            display_text + "\n(" + mask_percentage_text(mask_percentage) + " masked)"
        )
    if image_name is None and filter_num is None:
        pass
    elif filter_num is None:
        display_text = "I_%s " % image_name + display_text
    elif image_name is None:
        display_text = "F%03d " % filter_num + display_text
    else:
        display_text = "I_%s-F%03d " % (image_name, filter_num) + display_text
    if display:
        display_img(np_img, display_text)
    if save:
        save_filtered_image(
            np_img,
            image_name,
            filter_num,
            file_text,
            output_dir,
            thumbnail_dir,
            image_ext,
            debug=debug,
        )
    if info is not None:
        info[image_name + "-f_" + str(filter_num)] = (
            image_name,
            filter_num,
            display_text,
            file_text,
            mask_percentage,
            output_dir,
            thumbnail_dir,
        )


def mask_percentage_text(mask_percentage):
    """
    Generate a formatted string representing the percentage that an image is masked.
    Args:
      mask_percentage: The mask percentage.
    Returns:
      The mask percentage formatted as a string.
    """
    return "%3.2f%%" % mask_percentage


def save_filtered_image(
    np_img,
    slide_name,
    filter_num,
    filter_text,
    output_dir=OUTPUT_IMG_DIR,
    thumbnail_dir=THUMBNAIL_DIR,
    image_ext=IMAGE_EXT,
    debug=False,
):
    """
    Save a filtered image to the file system.
    Args:
      np_img: Image as a NumPy array.
      slide_name:  Image file name.
      filter_num: The filter number.
      filter_text: Descriptive text to add to the image filename.
      output_dir: Output directory to save image.
      thumbnail_dir: Output directory to save thumbnail.
      image_ext: Extension to save image.
      debug: Show debug info if True.
    """
    if debug or DEBUG:
        t = Timer()
    filepath = os.path.join(
        output_dir,
        slide_name + "-" + str(filter_num) + "-" + filter_text + "." + image_ext,
    )
    if type(np_img) == np.ndarray:
        pil_img = ut_image.np_to_pil(np_img)
    else:
        pil_img = np_img
    save_img(pil_img, filepath)
    if debug or DEBUG:
        print(
            "%-20s | Time: %-14s  Name: %s" % ("Save Image", str(t.elapsed()), filepath)
        )

    if thumbnail_dir is not None:
        if debug or DEBUG:
            t1 = Timer()
        thumbnail_filepath = os.path.join(
            thumbnail_dir,
            slide_name
            + "-"
            + str(filter_num)
            + "-"
            + filter_text
            + "."
            + THUMBNAIL_EXT,
        )
        save_thumbnail(pil_img, THUMBNAIL_SIZE, thumbnail_filepath)
        if debug or DEBUG:
            print(
                "%-20s | Time: %-14s  Name: %s"
                % ("Save Thumbnail", str(t1.elapsed()), thumbnail_filepath)
            )


def apply_filters_to_image_list(
    image_list,
    filters2apply,
    save,
    display,
    slide_list=None,
    roi_list=None,
    output_dir=FILTER_DIR,
    output_roi_path_list=None,
    save_each_filter=False,
    config=None,
    debug=False,
):
    """
    Apply filters to a list of images.
    Args:
      image_list: Tuple containing image filenames.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      save: If True, save filtered images.
      display: If True, display filtered images to screen.
      roi_list: Tuple containing roi mask filenames.
      output_dir: Output directory to save images.
      output_roi_path_list: Output directory to save roi mask.
      save_each_filter: If True, save each filtered image.
      debug: Show debug info if True.
    Returns:
      Tuple consisting of 1) a list of image numbers, and 2) a dictionary of image filter information.
    """
    html_page_info = dict()
    if roi_list is None:
        roi_list = [None for i in range(len(image_list))]

    if slide_list is None:
        slide_list = [None for i in range(len(image_list))]

    if output_roi_path_list is None:
        output_roi_path_list = [None for i in range(len(image_list))]

    if type(output_dir) is not list:
        output_dir = [output_dir for i in range(len(image_list))]
    elif len(output_dir) != len(image_list):
        print(
            "Output dir list is not the same length as input image list. Using FILTER_DIR as default."
        )
        output_dir = [FILTER_DIR for i in range(len(image_list))]

    for idx, (image_f, output_f, roi_f, slide, output_roi_f) in enumerate(
        zip(image_list, output_dir, roi_list, slide_list, output_roi_path_list)
    ):
        # try:
        np_img = ut_image.open_image_np(image_f)
        image_name = os.path.splitext(os.path.basename(image_f))[0]
        _, info = apply_filters_to_image(
            np_img,
            roi_f=roi_f,
            slide=slide,
            filters2apply=filters2apply,
            image_name=image_name,
            output_dir=output_f,
            output_roi_dir=output_roi_f,
            save=save,
            save_each_filter=save_each_filter,
            display=display,
            debug=debug,
        )
        if info:
            html_page_info.update(info)
        if (
            multiprocessing.current_process() == 0
            and idx % 500 == 0
            and config is not None
        ):
            send_noti_to_telegram(
                f"Preprocessed 500 images finished",
                TELEGRAM_TOKEN=config.telegram.token,
                TELEGRAM_CHAT_ID=config.telegram.chat_id,
            )
        # except Exception as e:
        #     print(f"{bcolors.WARNING}Error: Exception occurred for image `{image_f}`: {e}{bcolors.ENDC}")

    return image_list, html_page_info


def apply_function_to_image_list(
    func,
    image_list,
    output_path_list,
    slide_list=None,
    roi_list=None,
    output_roi_path_list=None,
    debug=False,
):
    """
    Apply filters to a list of images.
    Args:
      func: Function to apply to each image.
      image_list: Tuple containing input image filepath.
      output_path_list: Specify a list of output files. One-one correspondence with image_list.
      roi_list: Tuple containing roi mask filenames.
      output_roi_path_list: Output directory for roi masks.
      debug: Show debug info if True.
    Returns:
      Dictionary containing the return object of the function for each input path.
    """
    results = dict()

    if slide_list is None:
        slide_list = [None for i in range(len(image_list))]

    if roi_list is None or roi_list is False:
        roi_list = [None]

    if len(roi_list) != len(image_list):
        roi_list = [None for i in range(len(image_list))]

    if output_roi_path_list is None or output_roi_path_list is False:
        output_roi_path_list = [None]

    if output_roi_path_list is None or len(output_roi_path_list) != len(image_list):
        output_roi_path_list = [None for i in range(len(image_list))]

    for image_f, output_f, slide, roi_f, output_roi_f in zip(
        image_list, output_path_list, slide_list, roi_list, output_roi_path_list
    ):
        try:
            if func == generate_cell_mask or func == generate_tile_cell_mask:
                results[image_f] = func(image_f, output_f, debug=debug)
            else:
                results[image_f] = func(
                    image_f, output_f, slide, roi_f, output_roi_f, debug=debug
                )
        except Exception as e:
            print(
                f"{bcolors.WARNING}Error: Exception occurred for image `{image_f}`: {e}{bcolors.ENDC}"
            )
    return results


def singleprocess_apply_function(
    func,
    image_list,
    output_path_list,
    roi_list=None,
    output_roi_path_list=None,
    filters2apply=None,
    save=True,
    save_each_filter=False,
    display=False,
    debug=False,
):
    """
    Apply a given function to all images using multiple processes (one process per core).
    Args:
      func: Function to apply to each image.
      image_list: Specify a list of image files.
      output_path_list: Specify a list of output files. One-one correspondence with image_list.
      roi_list: Tuple containing roi mask filenames.
      output_roi_path_list: Specify a list of output roi files. One-one correspondence with roi_list.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      save: If True, save filtered images.
      save_each_filter: If True, save each filtered image.
      display: If True, display filtered images to screen (multiprocessed display not recommended).
      debug: Show debug info if True.
    """
    t = Timer()
    print("Applying filters to images\n")

    if func == "filters":
        apply_filters_to_image_list(
            image_list,
            filters2apply,
            save,
            display,
            roi_list,
            output_path_list,
            output_roi_path_list,
            save_each_filter,
            debug,
        )
    else:
        apply_function_to_image_list(
            func, image_list, output_path_list, roi_list, output_roi_path_list, debug
        )

    time_elapsed = t.elapsed()
    print("Time to apply function `%s` to all images: %s\n" % func, str(time_elapsed))
    return time_elapsed


def multiprocess_apply_function(
    func,
    image_list,
    output_path_list,
    slide_list=None,
    roi_list=None,
    output_roi_path_list=None,
    filters2apply=None,
    thread_multiplication=1,
    save=True,
    save_each_filter=False,
    display=False,
    config=None,
    debug=False,
):
    """
    Apply a given function to all images using multiple processes (one process per core).
    Args:
      func: Function to apply to each image.
      image_list: Specify a list of image files.
      output_path_list: Specify a list of output files. One-one correspondence with image_list.
      roi_list: Tuple containing roi mask filenames.
      output_roi_path_list: Specify a list of output roi files. One-one correspondence with roi_list.
      filters2apply: Dictionary of filters to apply. If None, the default filters will be applied.
      save: If True, save filtered images.
      save_each_filter: If True, save each filtered image.
      display: If True, display filtered images to screen (multiprocessed display not recommended).
      debug: Show debug info if True.
    """
    timer = Timer()
    print("Applying function {} to images (multiprocess)\n".format(func))

    # how many processes to use
    num_processes = multiprocessing.cpu_count() * thread_multiplication
    pool = multiprocessing.Pool(num_processes)

    num_images = len(image_list)
    if num_processes > num_images:
        num_processes = num_images
    images_per_process = num_images / num_processes

    include_rois = True
    if roi_list is None or len(roi_list) != num_images:
        include_rois = False
        roi_list = [None for i in range(num_images)]

    if output_roi_path_list is None or len(output_roi_path_list) != num_images:
        output_roi_path_list = [None for i in range(num_images)]

    if slide_list is None or len(slide_list) != num_images:
        slide_list = [None for i in range(num_images)]

    print("Number of processes: " + str(num_processes))
    print("Number of images: " + str(num_images))

    tasks = []
    for num_process in range(1, num_processes + 1):
        start_index = (num_process - 1) * images_per_process + 1
        end_index = num_process * images_per_process
        start_index = int(start_index)
        end_index = int(end_index)
        input_sublist = image_list[start_index - 1 : end_index]
        output_sublist = output_path_list[start_index - 1 : end_index]
        slide_sublist = slide_list[start_index - 1 : end_index]
        roi_sublist = roi_list[start_index - 1 : end_index]
        output_roi_sublist = output_roi_path_list[start_index - 1 : end_index]
        if func == "filters":
            tasks.append(
                (
                    input_sublist,
                    filters2apply,
                    save,
                    display,
                    slide_sublist,
                    roi_sublist,
                    output_sublist,
                    output_roi_sublist,
                    save_each_filter,
                    config,
                    debug,
                )
            )
        else:
            if include_rois:
                tasks.append(
                    (
                        func,
                        input_sublist,
                        output_sublist,
                        slide_sublist,
                        roi_sublist,
                        output_roi_sublist,
                        config,
                        debug,
                    )
                )
            else:
                tasks.append((func, input_sublist, output_sublist, debug))
        print(
            "Task #"
            + str(num_process)
            + ": Process  "
            + str(len(input_sublist))
            + " images"
        )

    # start tasks
    results = []
    for t in tasks:
        if func == "filters":
            results.append(pool.apply_async(apply_filters_to_image_list, t))
        else:
            results.append(pool.apply_async(apply_function_to_image_list, t))

    for result in results:
        result.wait()

    time_elapsed = timer.elapsed()
    print(
        "Time to apply function to all images (multiprocess): %s\n" % str(time_elapsed)
    )
    return time_elapsed
