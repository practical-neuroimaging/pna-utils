""" Tests for fixnifti module
"""

import os
import shutil
import tempfile

import numpy as np

import nibabel as nib

from fixnifti import set_nifti_params, fixup_nifti_file

from nose.tools import (assert_true, assert_false, assert_equal, assert_raises)

from numpy.testing import assert_array_equal, assert_almost_equal


def test_fixnifti():
    # Test fixnifti routine
    data = np.arange(24).reshape((2, 3, 4))
    img = nib.Nifti1Image(data, None)
    hdr = img.get_header()
    assert_equal(hdr['sform_code'], 0)
    img2 = set_nifti_params(img)
    # Doesn't change anything
    assert_equal(hdr['sform_code'], 0)
    # But it's not the same image
    assert_false(img is img2)
    # Sform does change sform
    affine = np.diag([3, 4, 5, 1])
    img2 = set_nifti_params(img, affine)
    hdr2 = img2.get_header()
    assert_equal(hdr2['sform_code'], 2)
    assert_array_equal(hdr2.get_sform(), affine)
    # Likewise qform
    assert_equal(hdr['qform_code'], 0)
    img2 = set_nifti_params(img, None, affine)
    hdr2 = img2.get_header()
    assert_equal(hdr2['sform_code'], 0) # Not sform tho'
    assert_equal(hdr2['qform_code'], 2)
    assert_array_equal(hdr2.get_qform(), affine)
    # Likewise TR
    assert_equal(hdr['pixdim'][4], 1.0)
    img2 = set_nifti_params(img, None, None, 2.1)
    hdr2 = img2.get_header()
    assert_equal(hdr2['qform_code'], 0) # Not qform tho'
    # Some error from float32 type of pixdim
    assert_almost_equal(hdr2['pixdim'][4], 2.1, 6)
    # Slice axis
    assert_equal(hdr.get_dim_info(), (None, None, None))
    img2 = set_nifti_params(img, None, None, None, 1)
    hdr2 = img2.get_header()
    assert_equal(hdr2['pixdim'][4], 1)
    assert_equal(hdr2.get_dim_info(), (None, None, 1))
    # Slice times - don't work by default
    assert_raises(nib.spatialimages.HeaderDataError, hdr.get_slice_times)
    # Or with just TR
    img2 = set_nifti_params(img, None, None, 2.0)
    assert_raises(nib.spatialimages.HeaderDataError, hdr.get_slice_times)
    # Or with TR and slice info
    img2 = set_nifti_params(img, None, None, 2.0, 1)
    assert_raises(nib.spatialimages.HeaderDataError, hdr.get_slice_times)
    # But will with slice_times as well
    times = np.array([0, 1, 2]) * 3 / 2.
    img2 = set_nifti_params(img, None, None, 2.0, 1, times)
    hdr2 = img2.get_header()
    assert_array_equal(hdr2.get_slice_times(), times)


def test_fixup_nifti_file():
    # Test fixup_nifti_file function
    data = np.arange(24).reshape((2, 3, 4))
    img = nib.Nifti1Image(data, None)
    hdr = img.get_header()
    assert_equal(hdr['pixdim'][4], 1)
    assert_equal(hdr.get_dim_info(), (None, None, None))
    def_aff = img.get_header().get_best_affine()
    tmpdir = tempfile.mkdtemp()
    try:
        fname = os.path.join(tmpdir, 'test.nii')
        new_fname = os.path.join(tmpdir, 'ftest.nii')
        nib.save(img, fname)
        # Just running without params sets the affine
        fixup_nifti_file(fname)
        img2 = nib.load(new_fname)
        assert_almost_equal(img2.get_affine(), def_aff)
        hdr2 = img2.get_header()
        assert_almost_equal(hdr2.get_sform(), def_aff)
        assert_almost_equal(hdr2.get_qform(), def_aff)
        # set prefix
        fixup_nifti_file(fname, 'pre')
        img2 = nib.load(os.path.join(tmpdir, 'pretest.nii'))
        assert_almost_equal(img2.get_affine(), def_aff)
        # Set TR
        fixup_nifti_file(fname, 'f', 2.1)
        img2 = nib.load(new_fname)
        hdr2 = img2.get_header()
        assert_almost_equal(hdr2['pixdim'][4], 2.1, 6)
        # Set slice axis
        fixup_nifti_file(fname, 'f', None, 1)
        img2 = nib.load(new_fname)
        hdr2 = img2.get_header()
        assert_equal(hdr2.get_dim_info(), (None, None, 1))
        # And slice times
        times = np.array([0, 1, 2]) * 3 / 2.1
        fixup_nifti_file(fname, 'f', 2.1, 1, times)
        img2 = nib.load(new_fname)
        hdr2 = img2.get_header()
        assert_almost_equal(hdr2.get_slice_times(), times)
        del img, img2
    finally:
        shutil.rmtree(tmpdir)
