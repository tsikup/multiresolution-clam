"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import numpy as np
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(os.path.dirname(SCRIPT_DIR)))

# import stain_utils as ut
import he_preprocessing.normalization.stain_utils as ut
from definitions import bcolors


def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = OD[(OD > beta).any(axis=1), :]

    # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))

    # The two principle eigenvectors
    V = V[:, [2, 1]]

    # Make sure vectors are pointing the right way
    if V[0, 0] < 0:
        V[:, 0] *= -1
    if V[0, 1] < 0:
        V[:, 1] *= -1

    # Project on this basis.
    That = np.dot(OD, V)

    # Angular coordinates with repect to the prinicple, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    return ut.normalize_rows(HE)


class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(
        self,
        standardize_brightness=True,
        dataset_maxC_reference=None,
        dataset_stain_matrix_reference=None,
    ):
        self.stain_matrix_target = dataset_stain_matrix_reference
        self.target_concentrations = None
        self.dataset_maxC_reference = dataset_maxC_reference
        self.standardize_brightness = standardize_brightness

        if not (
            all(
                v is not None
                for v in [dataset_maxC_reference, dataset_stain_matrix_reference]
            )
            or all(
                v is None
                for v in [dataset_maxC_reference, dataset_stain_matrix_reference]
            )
        ):
            raise Exception(
                "If dataset_maxC_reference is None, then slide_stain_matrix_reference and dataset_stain_matrix_reference should be None, vice versa and etc."
            )

    def fit(self, target):
        if self.standardize_brightness:
            target, _ = ut.standardize_brightness(target)
        if self.stain_matrix_target is None:
            self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = ut.get_concentrations(
            target, self.stain_matrix_target
        )

    def get_99_percentile_saturation_vector(self):
        maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape(
            (1, 2)
        )
        return maxC_target

    def get_he_vector(self):
        return self.stain_matrix_target

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)

    def transform(self, I, slide_stain_matrix_reference=None):
        if self.standardize_brightness:
            I, _ = ut.standardize_brightness(I)
        I_copy = np.copy(I)
        try:
            if slide_stain_matrix_reference is not None:
                stain_matrix_source = slide_stain_matrix_reference
            else:
                stain_matrix_source = get_stain_matrix(I)
            source_concentrations = ut.get_concentrations(I, stain_matrix_source)
            maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape(
                (1, 2)
            )
            if self.dataset_maxC_reference is not None:
                maxC_target = self.dataset_maxC_reference
            else:
                maxC_target = np.percentile(
                    self.target_concentrations, 99, axis=0
                ).reshape((1, 2))
            source_concentrations *= maxC_target / maxC_source
            return (
                255
                * np.exp(
                    -1
                    * np.dot(source_concentrations, self.stain_matrix_target).reshape(
                        I.shape
                    )
                )
            ).astype(np.uint8)
        except Exception as e:
            print(f"{bcolors.WARNING}Error: {e}. Returning input image.{bcolors.ENDC}")
            return I_copy.astype(np.uint8)

    def hematoxylin(self, I):
        if self.standardize_brightness:
            I, _ = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations(I, stain_matrix_source)
        H = source_concentrations[:, 0].reshape(h, w)
        H = np.exp(-1 * H)
        return H
