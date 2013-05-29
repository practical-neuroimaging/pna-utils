#!/usr/bin/env python
""" Script to clean up nifti files in Haxby 2001 dataset

The script sets correct parameters such as the TR, slice axis, and slice times

Run with something like::

    python fixup_openfmri.py ~/data/ds105

where ``~/data/ds105`` is the path to the Haxby data
"""
import sys

import numpy as np

# Library to fetch filenames from Open FMRI data layout
from openfmri import get_subjects

# Library for fixing nifti files
import  fixnifti
# For interactive work
reload(fixnifti)


def main():
    try:
        DATA_PATH = sys.argv[1]
    except IndexError:
        raise RuntimeError("Pass data path on command line")
    N_SLICES = 40
    TR = 2.5
    SLICE_AXIS=0

    # Figure out slice times.
    slice_order = np.array(range(0, N_SLICES, 2) + range(1, N_SLICES, 2))
    space_to_order = np.argsort(slice_order)
    time_one_slice = TR / N_SLICES
    # This time, don't add half a slice - nifti won't accept it.
    slice_times = space_to_order * time_one_slice

    subjects = get_subjects(DATA_PATH)
    for name, subject in subjects.items():
        for run in subject['functionals']:
            fname = run['filename']
            print("Fixing functional " + fname)
            fixnifti.fixup_nifti_file(filename, 'f', TR, SLICE_AXIS, slice_times)
        for anat_fname in subject['anatomicals']:
            print("Fixing anatomical " + anat_fname)
            fixnifti.fixup_nifti_file(anat_fname)


if __name__ == '__main__':
    main()
