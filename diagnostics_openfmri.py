import os

import nipy
from nipy.algorithms.diagnostics import screens
from openfmri import get_subjects

for name, subject in get_subjects('ds105').items():
    for run in subject['functionals']:
        fname = run['filename']
        img = nipy.load_image(fname)
        res = screens.screen(img, slice_axis=0)
        pth, fname = os.path.split(fname)
        froot, ext = os.path.splitext(fname)
        screens.write_screen_res(res, pth, froot)
