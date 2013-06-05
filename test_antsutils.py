""" Testing ANTS utilities
"""

import numpy as np

from antsutils import bb2imgdef

from nose.tools import assert_true, assert_equal, assert_raises

from numpy.testing import assert_almost_equal, assert_array_equal


def test_bb2imgdef():
    # Test bounding box to image definition
    bb = [[1, 6], [0, 5], [-2, 7]]
    shape, affine = bb2imgdef(bb, [1, 1, 1])
    assert_equal(shape, (6, 6, 10))
    assert_true(np.array(shape).dtype.type in np.sctypes['int'])
    exp_aff = np.diag([-1, 1, 1, 1])
    exp_aff[:3, -1] = -1, 0, -2
    assert_array_equal(affine, exp_aff)
    # Neurological
    shape, affine = bb2imgdef(bb, [1, 1, 1], radiological=False)
    assert_equal(shape, (6, 6, 10))
    exp_aff_neuro = np.diag([1, 1, 1, 1])
    exp_aff_neuro[:3, -1] = 1, 0, -2
    assert_array_equal(affine, exp_aff_neuro)
    # Values are sorted
    bb_again = [[6, 1], [5, 0], [7, -2]]
    shape, affine = bb2imgdef(bb_again, [1, 1, 1])
    assert_equal(shape, (6, 6, 10))
    assert_array_equal(affine, exp_aff)
    # 2D, 4D also OK
    bb2 = [[1, 6], [0, 5]]
    shape, affine = bb2imgdef(bb2, [1, 1])
    assert_equal(shape, (6, 6))
    assert_array_equal(affine, [[-1, 0, -1], [0, 1, 0], [0, 0, 1]])
    bb4 = [[0, 5], [1, 6], [-2, 7], [-5, -2]]
    shape, affine = bb2imgdef(bb4, [1, 1, 1, 0.5])
    assert_equal(shape, (6, 6, 10, 7))
    assert_array_equal(affine, [[-1, 0, 0, 0, 0],
                                [0, 1, 0, 0, 1],
                                [0, 0, 1, 0, -2],
                                [0, 0, 0, 0.5, -5],
                                [0, 0, 0, 0, 1],
                               ])
    # Different voxel sizes
    shape, affine = bb2imgdef(bb, [1, 0.4, 0.3])
    assert_equal(shape, (6,
                         len(np.arange(0, 5, 0.4))+1,
                         len(np.arange(-2, 7, 0.3))+1))
    assert_true(np.array(shape).dtype.type in np.sctypes['int'])
    exp_aff = np.diag([-1, 0.4, 0.3, 1])
    exp_aff[:3, -1] = -1, 0, -2
    assert_array_equal(affine, exp_aff)
    # Error if voxel sizes are wrong lengths
    assert_raises(ValueError, bb2imgdef, bb, [1, 1])
    assert_raises(ValueError, bb2imgdef, bb, [1, 1, 1, 1])
    # Or any negative
    assert_raises(ValueError, bb2imgdef, bb, [-1, 1, 1])
    assert_raises(ValueError, bb2imgdef, bb, [1, 1, -1])
