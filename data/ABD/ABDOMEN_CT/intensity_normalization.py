import numpy as np
import os
import glob
import SimpleITK as sitk

import sys
import niftiio as nio


IMG_FOLDER="./data/SABS/img/"
SEG_FOLDER="./data/SABS/label/"
OUT_FOLDER="./tmp_normalized/"

imgs = glob.glob(IMG_FOLDER + "/*.nii.gz")
imgs = [ fid for fid in sorted(imgs) ]
segs = [ fid for fid in sorted(glob.glob(SEG_FOLDER + "/*.nii.gz")) ]

pids = [   pid.split("img0")[-1].split(".")[0] for pid in imgs]


# helper function
def copy_spacing_ori(src, dst):
    dst.SetSpacing(src.GetSpacing())
    dst.SetOrigin(src.GetOrigin())
    dst.SetDirection(src.GetDirection())
    return dst

import copy
scan_dir = OUT_FOLDER
LIR = -125
HIR = 275
os.makedirs(scan_dir, exist_ok = True)

reindex = 0
for img_fid, seg_fid, pid in zip(imgs, segs, pids):

    img_obj = sitk.ReadImage( img_fid )
    seg_obj = sitk.ReadImage( seg_fid )

    array = sitk.GetArrayFromImage(img_obj)

    array[array > HIR] = HIR
    array[array < LIR] = LIR
    
    array = (array - array.min()) / (array.max() - array.min()) * 255.0
    
    # then normalize this
    
    wined_img = sitk.GetImageFromArray(array)
    wined_img = copy_spacing_ori(img_obj, wined_img)
    
    out_img_fid = os.path.join( scan_dir, f'image_{str(reindex)}.nii.gz' )
    out_lb_fid  = os.path.join( scan_dir, f'label_{str(reindex)}.nii.gz' ) 
    
    # then save
    sitk.WriteImage(wined_img, out_img_fid, True) 
    sitk.WriteImage(seg_obj, out_lb_fid, True) 
    print("{} has been save".format(out_img_fid))
    print("{} has been save".format(out_lb_fid))
    reindex += 1




