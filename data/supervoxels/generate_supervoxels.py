"""
Modified from Ouyang et al.
https://github.com/cheng-01037/Self-supervised-Fewshot-Medical-Image-Segmentation
"""

import os
import SimpleITK as sitk
import glob
from skimage.measure import label
import scipy.ndimage.morphology as snm
from felzenszwalb_3d import *

base_dir = '<path_to_data>/CHAOST2/chaos_MR_T2_normalized'
# base_dir = '<path_to_data>/CMR/cmr_MR_normalized'

imgs = glob.glob(os.path.join(base_dir, 'image*'))
labels = glob.glob(os.path.join(base_dir, 'label*'))

imgs = sorted(imgs, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))
labels = sorted(labels, key=lambda x: int(x.split('_')[-1].split('.nii.gz')[0]))

fg_thresh = 10

MODE = 'MIDDLE'
n_sv = 5000
# n_sv = 1000

def read_nii_bysitk(input_fid):
    """ read nii to numpy through simpleitk
        peelinfo: taking direction, origin, spacing and metadata out
    """
    img_obj = sitk.ReadImage(input_fid)
    img_np = sitk.GetArrayFromImage(img_obj)
    return img_np

# thresholding the intensity values to get a binary mask of the patient
def fg_mask2d(img_2d, thresh):
    mask_map = np.float32(img_2d > thresh)
    def getLargestCC(segmentation):  # largest connected components
        labels = label(segmentation)
        assert (labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        return largestCC

    if mask_map.max() < 0.999:
        return mask_map
    else:
        post_mask = getLargestCC(mask_map)
        fill_mask = snm.binary_fill_holes(post_mask)
    return fill_mask


# remove supervoxels within the empty regions
def supervox_masking(seg, mask):

    seg[seg == 0] = seg.max() + 1
    seg = np.int32(seg)
    seg[mask == 0] = 0

    return seg

# make supervoxels
for img_path in imgs:
    img = read_nii_bysitk(img_path)
    img = 255 * (img - img.min()) / img.ptp()

    reader = sitk.ImageFileReader()
    reader.SetFileName(img_path)
    reader.LoadPrivateTagsOn()
    reader.ReadImageInformation()

    x = float(reader.GetMetaData('pixdim[1]'))
    y = float(reader.GetMetaData('pixdim[2]'))
    z = float(reader.GetMetaData('pixdim[3]'))

    segments_felzenszwalb = felzenszwalb_3d(img, min_size=n_sv, sigma=0, spacing=(z, x, y))

    # post processing: remove bg (low intensity regions)
    fg_mask_vol = np.zeros(segments_felzenszwalb.shape)
    for ii in range(segments_felzenszwalb.shape[0]):
        _fgm = fg_mask2d(img[ii, ...], fg_thresh)
        fg_mask_vol[ii] = _fgm
    processed_seg_vol = supervox_masking(segments_felzenszwalb, fg_mask_vol)

    # write to nii.gz
    out_seg = sitk.GetImageFromArray(processed_seg_vol)

    idx = os.path.basename(img_path).split("_")[-1].split(".nii.gz")[0]

    # seg_fid = os.path.join(f'<path_to_data>/supervoxels_{n_sv}/', f'superpix-{MODE}_{idx}.nii.gz')
    # sitk.WriteImage(out_seg, seg_fid)
    print(f'image with id {idx} has finished')


