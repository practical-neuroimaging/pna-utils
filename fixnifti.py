""" Fixup Nifti files """

import os

import nibabel as nib

def set_nifti_params(img,
                     sform=None,
                     qform=None,
                     TR=None,
                     slice_axis=None,
                     slice_times=None):
    """ Set TR and slice times and sform, qform for `img`

    Parameters
    ----------
    img : NiftiImage instance
    sform : None or (4, 4) array-like
        Array to set as ``sform`` matrix in nifti header.  None leaves ``sform``
        unchanged
    qform : None or (4, 4) array-like
        Array to set as ``qform`` matrix in nifti header.  None leaves ``qform``
        unchanged
    TR : None or float, optional
        None means leave TR unchanged.  Float will be set into nifti pixdims as
        time pixdim value
    slice_axis : None or int, optional
        Index of the slice axis in the image data.  None means leave slice
        dimension unchanged.
    slice_times : None or array-like, optional
        A set of slice times to set into the nifti header, or None to leave
        unchanged.  `slice_dim` must be set in order to set `slice_times`

    Returns
    -------
    fixed_img : Nifti1Image instance
        Image with fixes applied, and sform, qform set explicitly
    """
    # NB: We don't normally use attributes starting with _. We can in this case
    # because the author told us it was OK...
    fixed_img = nib.Nifti1Image(img._data, img.get_affine(), img.get_header())
    hdr = fixed_img.get_header()
    if not sform is None:
        hdr.set_sform(sform)
    if not qform is None:
        hdr.set_qform(qform)
    if not TR is None:
        hdr['pixdim'][4] = TR
    if not slice_axis is None:
        hdr.set_dim_info(slice=slice_axis)
    if not slice_times is None:
        hdr.set_slice_times(slice_times)
    return fixed_img


def fixup_nifti_file(fname, prefix='f', TR=None, slice_axis=None, slice_times=None):
    """ Fix parameters for an image filename `fname`

    Saves fixed nifti file with given filename `prefix`

    Will also set estimated affine from image into nifti ``sform`` and
    ``qform``.  This makes sure that other programs can see the estimated
    affine (they may be more picky than we are).

    Parameters
    ----------
    fname : str
        Image filename. We'll load it with ``nibabel.load``
    prefix : str
        Prefix with which to save new fixed file
    TR : None or float, optional
        None means leave TR unchanged.  Float will be set into nifti pixdims as
        time pixdim value
    slice_axis : None or int, optional
        Index of the slice axis in the image data.  None means leave slice
        dimension unchanged.
    slice_times : None or array-like, optional
        A set of slice times to set into the nifti header, or None to leave
        unchanged.  `slice_dim` must be set in order to set `slice_times`
    """
    pth, name = os.path.split(fname)
    new_fname = os.path.join(pth, prefix + name)
    img = nib.load(fname)
    affine = img.get_affine()
    fixed_img = set_nifti_params(img,
                                 sform=affine,
                                 qform=affine,
                                 TR=TR,
                                 slice_axis=slice_axis,
                                 slice_times=slice_times)
    nib.save(fixed_img, new_fname)
