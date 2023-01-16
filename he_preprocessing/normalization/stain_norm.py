# Normalize staining
import numpy as np
import pandas as pd
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

from he_preprocessing.normalization import (
    stainNorm_Vahadane,
    stainNorm_Macenko,
    stainNorm_Reinhard,
)


def custom_macenko(
    sample, beta=0.15, alpha=1, light_intensity=240, stain_method="default"
):
    """
    Normalize the staining of H&E histology slides.

    This function normalizes the staining of H&E histology slides.

    References:
    - Macenko, Marc, et al. "A method for normalizing histology slides
    for quantitative analysis." Biomedical Imaging: From Nano to Macro,
    2009.  ISBI'09. IEEE International Symposium on. IEEE, 2009.
        - http://wwwx.cs.unc.edu/~mn/sites/default/files/macenko2009.pdf
    - https://github.com/mitkovetta/staining-normalization

    Args:
    sample: Sample is a 3D NumPy array of shape (H,W,C).
    stain_method: Used only when method is custom_macenko and should be either 'default' or 'matlab_default'

    Returns:
    Sample is a 3D NumPy array of shape (H,W,C) that has been stain normalized.
    """
    # Setup.
    HERef = {
        "matlab_default": np.array(
            [0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]
        ).reshape(3, 2),
        "default": np.array(
            [0.54598845, 0.322116, 0.72385198, 0.76419107, 0.42182333, 0.55879629]
        ).reshape(3, 2),
    }

    maxCRef = {
        "matlab_default": np.array([1.9705, 1.0308]).reshape(2, 1),
        "default": np.array([0.82791151, 0.61137274]).reshape(2, 1),
    }

    x = np.asarray(sample)
    h, w, c = x.shape
    x = x.reshape(-1, c).astype(np.float64)  # shape (H*W, C)

    # Reference stain vectors and stain saturations.  We will normalize all slides
    # to these references.  To create these, grab the stain vectors and stain
    # saturations from a desirable slide.

    # Values in reference implementation for use with eigendecomposition approach, natural log,
    # and `light_intensity=240`.
    # stain_ref = np.array([0.5626, 0.2159, 0.7201, 0.8012, 0.4062, 0.5581]).reshape(3,2)
    # max_sat_ref = np.array([1.9705, 1.0308]).reshape(2,1)

    # SVD w/ log10, and `light_intensity=255`.
    if stain_method not in ["default", "matlab_default"]:
        raise Exception('stain_method must be "default" or "matlab_default"')
    stain_ref = HERef[stain_method]
    max_sat_ref = maxCRef[stain_method]

    # Convert RGB to OD.
    # Note: The original paper used log10, and the reference implementation used the natural log.
    OD = -np.log((x + 1) / light_intensity)  # shape (H*W, C)
    # OD = -np.log10(x / light_intensity + 1e-8)

    # Remove data with OD intensity less than beta.
    # I.e. remove transparent pixels.
    # Note: This needs to be checked per channel, rather than
    # taking an average over all channels for a given pixel.
    OD_thresh = OD[np.logical_not(np.any(OD < beta, axis=1))]  # shape (K, C)

    # Calculate eigenvectors.
    # Note: We can either use eigenvector decomposition, or SVD.
    # eigvals, eigvecs = np.linalg.eig(np.cov(OD_thresh.T))  # np.cov results in inf/nans
    U, s, V = np.linalg.svd(OD_thresh, full_matrices=False)

    # Extract two largest eigenvectors.
    # Note: We swap the sign of the eigvecs here to be consistent
    # with other implementations.  Both +/- eigvecs are valid, with
    # the same eigenvalue, so this is okay.
    # top_eigvecs = eigvecs[:, np.argsort(eigvals)[-2:]] * -1
    top_eigvecs = V[0:2, :].T * -1  # shape (C, 2)

    # Project thresholded optical density values onto plane spanned by
    # 2 largest eigenvectors.
    proj = np.dot(OD_thresh, top_eigvecs)  # shape (K, 2)

    # Calculate angle of each point wrt the first plane direction.
    # Note: the parameters are `np.arctan2(y, x)`
    angles = np.arctan2(proj[:, 1], proj[:, 0])  # shape (K,)

    # Find robust extremes (a and 100-a percentiles) of the angle.
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    # Convert min/max vectors (extremes) back to optimal stains in OD space.
    # This computes a set of axes for each angle onto which we can project
    # the top eigenvectors.  This assumes that the projected values have
    # been normalized to unit length.
    extreme_angles = np.array(
        [
            [np.cos(min_angle), np.cos(max_angle)],
            [np.sin(min_angle), np.sin(max_angle)],
        ]
    )  # shape (2,2)
    stains = np.dot(top_eigvecs, extreme_angles)  # shape (C, 2)

    # Merge vectors with hematoxylin first, and eosin second, as a heuristic.
    if stains[0, 0] < stains[0, 1]:
        stains[:, [0, 1]] = stains[:, [1, 0]]  # swap columns

    # Calculate saturations of each stain.
    # Note: Here, we solve
    #    OD = VS
    #     S = V^{-1}OD
    # where `OD` is the matrix of optical density values of our image,
    # `V` is the matrix of stain vectors, and `S` is the matrix of stain
    # saturations.  Since this is an overdetermined system, we use the
    # least squares solver, rather than a direct solve.
    sats, _, _, _ = np.linalg.lstsq(stains, OD.T, rcond=-1)

    # Normalize stain saturations to have same pseudo-maximum based on
    # a reference max saturation.
    max_sat = np.percentile(sats, 99, axis=1, keepdims=True)
    sats = sats / max_sat * max_sat_ref

    # Compute optimal OD values.
    OD_norm = np.dot(stain_ref, sats)

    # Recreate image.
    # Note: If the image is immediately converted to uint8 with `.astype(np.uint8)`, it will
    # not return the correct values due to the initital values being outside of [0,255].
    # To fix this, we round to the nearest integer, and then clip to [0,255], which is the
    # same behavior as Matlab.
    # x_norm = np.exp(OD_norm) * light_intensity  # natural log approach
    x_norm = 10 ** (-OD_norm) * light_intensity - 1e-8  # log10 approach
    x_norm = np.clip(np.round(x_norm), 0, 255).astype(np.uint8)
    x_norm = x_norm.astype(np.uint8)
    x_norm = x_norm.T.reshape(h, w, c)
    return x_norm


class StainNormalizer:
    def __init__(
        self,
        target=None,
        luminosity=True,
        method="macenko",
        reference_dir=None,
    ):
        """
        Normalize the staining of H&E histology slides.

        This function normalizes the staining of H&E histology slides.

        Args:
        target: Reference image with target staining.
        luminosity: Luminosity normalization.
        method: 'macenko', 'vahadane', 'reinhard', 'custom_macenko'

        Returns:
        Normalized (transformed) image.
        """
        self.normalizer = None
        self.luminosity = luminosity
        self.reference_dir = reference_dir
        self.dataset_maxC_reference = None
        self.stain_matrix_reference = None
        self.target = target
        self.method = method

        if reference_dir is not None:
            dataset_level_ref = reference_dir
            dataset_df = pd.read_csv(
                os.path.join(dataset_level_ref, "stain_vectors_dataset_level_reference.csv")
            )
            dataset_maxC_reference = np.fromstring(
                dataset_df[["saturation_vector_0", "saturation_vector_1"]].to_numpy()[0]
            ).reshape((1, 2))
            dataset_stain_matrix_reference = np.fromstring(
                dataset_df[
                    [
                        "he_matrix_0",
                        "he_matrix_1",
                        "he_matrix_2",
                        "he_matrix_3",
                        "he_matrix_4",
                        "he_matrix_5",
                    ]
                ].to_numpy()[0]
            ).reshape((2, 3))
            self.dataset_maxC_reference = dataset_maxC_reference
            self.stain_matrix_reference = dataset_stain_matrix_reference

        if self.method == "reinhard":
            self.normalizer = stainNorm_Reinhard.Normalizer(
                standardize_brightness=self.luminosity
            )
        elif self.method == "vahadane":
            self.normalizer = stainNorm_Vahadane.Normalizer(
                standardize_brightness=self.luminosity
            )
        elif self.method == "macenko":
            self.normalizer = stainNorm_Macenko.Normalizer(
                standardize_brightness=self.luminosity,
                dataset_stain_matrix_reference=self.stain_matrix_reference,
                dataset_maxC_reference=self.dataset_maxC_reference,
            )

        if target is not None:
            self.fit(target)

    def fit(self, target):
        if self.method == "reinhard":
            self.normalizer.fit(target)
        elif self.method == "vahadane":
            self.normalizer.fit(target)
        elif self.method == "macenko":
            self.normalizer.fit(target)
        elif self.method == "custom_macenko":
            pass
        else:
            self.normalizer.fit(target)

    def get_99_percentile_saturation_vector(self):
        """
        Get 99th percentile of saturation vector.
        Args:
            I: image numpy array
        Returns:
            99th percentile of saturation vector
        """
        return self.normalizer.get_99_percentile_saturation_vector()

    def get_he_vector(self):
        """
        Get H&E stain vectors.
        Args:
            I: image numpy array
        Returns:
            HE stain vectors
        """
        return self.normalizer.get_he_vector()

    def transform(
        self,
        to_transform,
        stain_method="default",
        slide=None,
    ):
        """
        Normalize the staining of H&E histology slides.

        This function normalizes the staining of H&E histology slides.

        Args:
        to_transform: Image to be transformed.
        stain_method: Used only when method is custom_macenko and should be either 'default' or 'matlab_default'

        Returns:
        Normalized (transformed) image.
        """
        slide_stain_matrix_reference = None

        if self.reference_dir is not None:
            dataset_df = pd.read_csv(
                os.path.join(
                    self.reference_dir, "stain_vectors_slide_level_reference.csv"
                )
            )
            if slide is not None:
                dataset_df = dataset_df.loc[dataset_df["slide"] == slide]
            else:
                dataset_df = dataset_df.iloc[0]

            if self.method == "macenko":
                slide_stain_matrix_reference = np.fromstring(
                    dataset_df[
                        [
                            "he_matrix_0",
                            "he_matrix_1",
                            "he_matrix_2",
                            "he_matrix_3",
                            "he_matrix_4",
                            "he_matrix_5",
                        ]
                    ].to_numpy()[0]
                ).reshape((2, 3))

        # Normalize stain
        if self.method == "custom_macenko":
            transformed = custom_macenko(to_transform, stain_method=stain_method)
        elif self.method == "macenko":
            transformed = self.normalizer.transform(
                to_transform, slide_stain_matrix_reference=slide_stain_matrix_reference
            )
        else:
            transformed = self.normalizer.transform(to_transform)

        return transformed
