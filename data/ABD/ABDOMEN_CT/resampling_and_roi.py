import numpy as np
import os
import glob
import SimpleITK as sitk
import sys
import niftiio as nio

IMG_FOLDER = "./tmp_normalized/"
SEG_FOLDER = IMG_FOLDER
imgs = glob.glob(IMG_FOLDER + "/image_*.nii.gz")
imgs = [ fid for fid in sorted(imgs) ]
segs = [ fid for fid in sorted(glob.glob(SEG_FOLDER + "/label_*.nii.gz")) ]

pids = [pid.split("_")[-1].split(".")[0] for pid in imgs]

# helper functions copy pasted
def resample_by_res(mov_img_obj, new_spacing, interpolator = sitk.sitkLinear, logging = True):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolator)
    resample.SetOutputDirection(mov_img_obj.GetDirection())
    resample.SetOutputOrigin(mov_img_obj.GetOrigin())
    mov_spacing = mov_img_obj.GetSpacing()

    resample.SetOutputSpacing(new_spacing)
    RES_COE = np.array(mov_spacing) * 1.0 / np.array(new_spacing)
    new_size = np.array(mov_img_obj.GetSize()) *  RES_COE 

    resample.SetSize( [int(sz+1) for sz in new_size] )
    if logging:
        print("Spacing: {} -> {}".format(mov_spacing, new_spacing))
        print("Size {} -> {}".format( mov_img_obj.GetSize(), new_size ))

    return resample.Execute(mov_img_obj)

def resample_lb_by_res(mov_lb_obj, new_spacing, interpolator = sitk.sitkLinear, ref_img = None, logging = True):
    src_mat = sitk.GetArrayFromImage(mov_lb_obj)
    lbvs = np.unique(src_mat)
    if logging:
        print("Label values: {}".format(lbvs))
    for idx, lbv in enumerate(lbvs):
        _src_curr_mat = np.float32(src_mat == lbv) 
        _src_curr_obj = sitk.GetImageFromArray(_src_curr_mat)
        _src_curr_obj.CopyInformation(mov_lb_obj)
        _tar_curr_obj = resample_by_res( _src_curr_obj, new_spacing, interpolator, logging )
        _tar_curr_mat = np.rint(sitk.GetArrayFromImage(_tar_curr_obj)) * lbv
        if idx == 0:
            out_vol = _tar_curr_mat
        else:
            out_vol[_tar_curr_mat == lbv] = lbv
    out_obj = sitk.GetImageFromArray(out_vol)
    out_obj.SetSpacing( _tar_curr_obj.GetSpacing() )
    if ref_img != None:
        out_obj.CopyInformation(ref_img)
    return out_obj
        
## Then crop ROI
def get_label_center(label):
    nnz = np.sum(label > 1e-5)
    return np.int32(np.rint(np.sum(np.nonzero(label), axis = 1) * 1.0 / nnz))

def image_crop(ori_vol, crop_size, referece_ctr_idx, padval = 0., only_2d = True):
    """ crop a 3d matrix given the index of the new volume on the original volume
        Args:
            refernce_ctr_idx: the center of the new volume on the original volume (in indices)
            only_2d: only do cropping on first two dimensions
    """
    _expand_cropsize = [x + 1 for x in crop_size] # to deal with boundary case
    if only_2d:
        assert len(crop_size) == 2, "Actual len {}".format(len(crop_size))
        assert len(referece_ctr_idx) == 2, "Actual len {}".format(len(referece_ctr_idx))
        _expand_cropsize.append(ori_vol.shape[-1])
        
    image_patch = np.ones(tuple(_expand_cropsize)) * padval

    half_size = tuple( [int(x * 1.0 / 2) for x in _expand_cropsize] )
    _min_idx = [0,0,0]
    _max_idx = list(ori_vol.shape)

    # bias of actual cropped size to the beginning and the end of this volume
    _bias_start = [0,0,0]
    _bias_end = [0,0,0]

    for dim,hsize in enumerate(half_size):
        if dim == 2 and only_2d:
            break

        _bias_start[dim] = np.min([hsize, referece_ctr_idx[dim]])
        _bias_end[dim] = np.min([hsize, ori_vol.shape[dim] - referece_ctr_idx[dim]])

        _min_idx[dim] = referece_ctr_idx[dim] - _bias_start[dim]
        _max_idx[dim] = referece_ctr_idx[dim] + _bias_end[dim]
        
    if only_2d:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], ... ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], ... ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], : ]
    # then goes back to original volume
    else:
        image_patch[ half_size[0] - _bias_start[0]: half_size[0] +_bias_end[0], \
                half_size[1] - _bias_start[1]: half_size[1] +_bias_end[1], \
                half_size[2] - _bias_start[2]: half_size[2] +_bias_end[2] ] = \
                ori_vol[ referece_ctr_idx[0] - _bias_start[0]: referece_ctr_idx[0] +_bias_end[0], \
                referece_ctr_idx[1] - _bias_start[1]: referece_ctr_idx[1] +_bias_end[1], \
                referece_ctr_idx[2] - _bias_start[2]: referece_ctr_idx[2] +_bias_end[2] ]

        image_patch = image_patch[ 0: crop_size[0], 0: crop_size[1], 0: crop_size[2] ]
    return image_patch

   
    
def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

import copy
OUT_FOLDER = "./sabs_CT_normalized"
scan_dir = OUT_FOLDER
os.makedirs(scan_dir, exist_ok = True)
BD_BIAS = 32 # cut irrelavent empty boundary to make roi stands out

SPA_FAC = (512 - 2 * BD_BIAS) / 256 # spacing factor

for img_fid, seg_fid, pid in zip(imgs, segs, pids):

    lb_n = nio.read_nii_bysitk(seg_fid)

    img_obj = sitk.ReadImage( img_fid )
    seg_obj = sitk.ReadImage( seg_fid )

    ## image
    array = sitk.GetArrayFromImage(img_obj)
    # cropping
    array = array[:, BD_BIAS: -BD_BIAS, BD_BIAS: -BD_BIAS]
    cropped_img_o = sitk.GetImageFromArray(array)
    cropped_img_o = copy_spacing_ori(img_obj, cropped_img_o)

    # resampling
    img_spa_ori = img_obj.GetSpacing()
    res_img_o = resample_by_res(cropped_img_o, [img_spa_ori[0] * SPA_FAC, img_spa_ori[1] * SPA_FAC, img_spa_ori[-1]], interpolator = sitk.sitkLinear,
                                    logging = True)

    ## label
    lb_arr = sitk.GetArrayFromImage(seg_obj)
    
    # cropping
    lb_arr = lb_arr[:,BD_BIAS: -BD_BIAS, BD_BIAS: -BD_BIAS]
    cropped_lb_o = sitk.GetImageFromArray(lb_arr)
    cropped_lb_o = copy_spacing_ori(seg_obj, cropped_lb_o)

    lb_spa_ori = seg_obj.GetSpacing()

    # resampling
    res_lb_o = resample_lb_by_res(cropped_lb_o,  [lb_spa_ori[0] * SPA_FAC, lb_spa_ori[1] * SPA_FAC, lb_spa_ori[-1] ], interpolator = sitk.sitkLinear,
                                  ref_img = res_img_o, logging = True)

    
    out_img_fid = os.path.join( scan_dir, f'image_{pid}.nii.gz' )
    out_lb_fid  = os.path.join( scan_dir, f'label_{pid}.nii.gz' ) 
    
    # then save
    sitk.WriteImage(res_img_o, out_img_fid, True) 
    sitk.WriteImage(res_lb_o, out_lb_fid, True) 
    print("{} has been saved".format(out_img_fid))
    print("{} has been saved".format(out_lb_fid))
