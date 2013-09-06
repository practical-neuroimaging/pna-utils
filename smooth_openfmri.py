#!/usr/bin/env python
""" Script to run smoothing on Haxby Open FMRI dataset
"""
from __future__ import division

import os
import sys

import scipy.ndimage as snd
import numpy as np

from nipy import load_image, save_image
from nipy.core.api import as_xyz_image, xyz_affine, Image
from nipy.algorithms.kernel_smooth import fwhm2sigma

# Library to fetch filenames from Open FMRI data layout
from openfmri import get_subjects


def smooth_image(img, fwhm):
    """ Smooth image `img` by FWHM `fwhm`
    """
    # Make sure time axis is last
    img = as_xyz_image(img)
    fwhm = np.asarray(fwhm)
    if fwhm.size == 1:
        fwhm = np.ones(3,) * fwhm
    # 4x4 affine
    affine = xyz_affine(img)
    # Voxel sizes in the three spatial axes
    RZS = affine[:3, :3]
    vox = np.sqrt(np.sum(RZS ** 2))
    # Smoothing in terms of voxels
    vox_fwhm = fwhm / vox
    vox_sd = fwhm2sigma(vox_fwhm)
    # Do the smoothing
    data = img.get_data()
    sm_data = snd.gaussian_filter(data, list(vox_sd) + [0])
    return Image(sm_data, img.coordmap)


def main():
    try:
        DATA_PATH = sys.argv[1]
    except IndexError:
        raise RuntimeError("Pass data path on command line")
    subjects = get_subjects(DATA_PATH)
    for name in sorted(subjects):
        subject = subjects[name]
        print("Smoothing subject " + name)
        for run in subject['functionals']:
            fname = run['filename']
            pth, fpart = os.path.split(fname)
            ra_fname = os.path.join(pth, 'ra' + fpart)
            sra_fname = os.path.join(pth, 'sra' + fpart)
            img = load_image(ra_fname)
            save_image(smooth_image(img, 8.), sra_fname)


if __name__ == '__main__':
    main()
