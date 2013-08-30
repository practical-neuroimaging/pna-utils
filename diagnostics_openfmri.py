#!/usr/bin/env python
""" Script to run diagnostics on nifti files in Haxby 2001 dataset

Run with something like::

    python diagnostics_openfmri.py ~/data/ds105

where ``~/data/ds105`` is the path to the Haxby data
"""
import os
import sys

# Library to fetch filenames from Open FMRI data layout
from openfmri import get_subjects


import nipy
from nipy.algorithms.diagnostics import screens

def main():
    try:
        DATA_PATH = sys.argv[1]
    except IndexError:
        raise RuntimeError("Pass data path on command line")
    for name, subject in get_subjects(DATA_PATH).items():
        for run in subject['functionals']:
            fname = run['filename']
            img = nipy.load_image(fname)
            res = screens.screen(img, slice_axis=0)
            pth, fname = os.path.split(fname)
            froot, ext = os.path.splitext(fname)
            screens.write_screen_res(res, pth, froot)


if __name__ == '__main__':
    main()
