#!/usr/bin/env python
""" Script to run motion correction / slice timing on Haxby Open FMRI dataset
"""
import os
import sys

import numpy as np

from nipy.algorithms.registration import SpaceTimeRealign
from nipy import load_image, save_image

# Library to fetch filenames from Open FMRI data layout
from openfmri import get_subjects


def time_space_realign(run_fnames, TR, slice_times, slice_axis):
    run_imgs = [load_image(run) for run in run_fnames]
    # Spatio-temporal realigner
    R = SpaceTimeRealign(run_imgs,
                         tr=TR,
                         slice_times=slice_times,
                         slice_info=(slice_axis, 1))
    # Estimate motion within- and between-sessions
    R.estimate(refscan=None)
    # Save back out
    for i, fname in enumerate(run_fnames):
        corr_run = R.resample(i)
        pth, name = os.path.split(fname)
        processed_fname = os.path.join(pth, 'ra' + name)
        save_image(corr_run, processed_fname)


def main():
    try:
        DATA_PATH = sys.argv[1]
    except IndexError:
        raise RuntimeError("Pass data path on command line")
    TR = 2.5
    subjects = get_subjects(DATA_PATH)
    for name, subject in subjects.items():
        run_fnames = []
        for run in subject['functionals']:
            run_fnames.append(run['filename'])
        print("Realigning subject " + name)
        time_space_realign(run_fnames, TR, 'ascending', slice_axis=0)


if __name__ == '__main__':
    main()
