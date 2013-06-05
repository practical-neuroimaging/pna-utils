""" Utilities for working with ANTS registrations
"""

import numpy as np

def bb2imgdef(bb, vox_sizes, radiological=True):
    """ Return shape, affine for bounding box `bb`, voxel sizes `vox_sizes`

    Assumes first axis of voxel block is R->L if `radiological` is True, else
    L->R. Assumes second axis is P->A and third is I->S.

    Parameters
    ----------
    bb : (N, 2) array-like
        For each dimension 0..N-1, specifies minumum and maximum point in
        real-world units.  ``N`` will usually be 3
    vox_sizes : (N,) array-like
        For each dimension, specifies size of voxels in real-world units
    radiological : bool, optional
        If True assume radiological storage (R to L) in voxel space.

    Returns
    -------
    shape : (N,) tuple
        Shape of image corresponding to `bb`, `vox_sizes`
    affine : (N+1, N+1) array
        Affine relating array coordinates in corresponding image to units of
        `bb`
    """
    bb = np.asarray(bb)
    vox_sizes = np.asarray(vox_sizes)
    ndim = bb.shape[0]
    if not len(vox_sizes) == ndim:
        raise ValueError('vox_sizes should be same len as bb')
    if np.any(vox_sizes < 0):
        raise ValueError('vox_sizes should all be positive')
    shape = np.zeros(ndim, dtype=np.int)
    affine = np.eye(ndim + 1)
    for i, coord in enumerate(bb):
        coord.sort()
        affine[i, -1] = coord[0]
        affine[i, i] = vox_sizes[i]
        n_gaps = np.ceil((coord[1] - coord[0]) / float(vox_sizes[i]))
        shape[i] = n_gaps + 1
    if radiological:
        affine[0] *= -1
    return tuple(shape), affine
