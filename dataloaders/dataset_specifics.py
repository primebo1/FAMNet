"""
Dataset Specifics
Extended from ADNet code by Hansen et al.
"""

import torch
import random


def get_label_names(dataset):
    label_names = {}
    if dataset == 'CARDIAC_bssFP':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'
    elif dataset == 'CARDIAC_LGE':
        label_names[0] = 'BG'
        label_names[1] = 'LV-MYO'
        label_names[2] = 'LV-BP'
        label_names[3] = 'RV'
    elif dataset == 'ABDOMEN_MR':
        label_names[0] = 'BG'
        label_names[1] = 'LIVER'
        label_names[2] = 'RIGHT_KIDNEY'
        label_names[3] = 'LEFT_KIDNEY'
        label_names[4] = 'SPLEEN'        
    elif dataset == 'ABDOMEN_CT':
        label_names[0] = 'BG'
        label_names[1] = 'SPLEEN'
        label_names[2] = 'RIGHT_KIDNEY'
        label_names[3] = 'LEFT_KIDNEY'
        label_names[4] = 'GALLBLADDER'
        label_names[5] = 'ESOPHAGUS'
        label_names[6] = 'LIVER'
        label_names[7] = 'STOMACH'
        label_names[8] = 'AORTA'
        label_names[9] = 'INFERIOR_VENA_CAVA'             # Inferior vena cava
        label_names[10] = 'PORTAL_VEIN_AND_SPLENIC_VEIN'  # portal vein and splenic vein
        label_names[11] = 'PANCREAS'
        label_names[12] = 'RIGHT_ADRENAL_GLAND'  # right adrenal gland
        label_names[13] = 'LEFT_ADRENAL_GLAND'   # left adrenal gland
    elif dataset == 'Prostate_UCLH':
        label_names[0] = 'BG'
        label_names[1] = 'Bladder'              
        label_names[2] = 'Bone'                 
        label_names[3] = 'Obturator_Internus'   
        label_names[4] = 'Transition_Zone'      
        label_names[5] = 'Central_Gland'        
        label_names[6] = 'Rectum'               
        label_names[7] = 'Seminal_Vesicle'       
        label_names[8] = 'Neurovascular_Bundle'  
    elif dataset == 'Prostate_TCIA_PD':
        label_names[0] = 'BG'
        label_names[1] = 'Bladder'
        label_names[2] = 'Bone'
        label_names[3] = 'Obturator_Internus'
        label_names[4] = 'Transition_Zone'
        label_names[5] = 'Central_Gland'
        label_names[6] = 'Rectum'
        label_names[7] = 'Seminal_Vesicle'
        label_names[8] = 'Neurovascular_Bundle'
    elif dataset == 'Prostate_NCI':
        label_names[0] = 'BG'
        label_names[1] = 'Bladder'
        label_names[2] = 'Bone'
        label_names[3] = 'Obturator_Internus'
        label_names[4] = 'Transition_Zone'
        label_names[5] = 'Central_Gland'
        label_names[6] = 'Rectum'
        label_names[7] = 'Seminal_Vesicle'
        label_names[8] = 'Neurovascular_Bundle'
        
    return label_names


def get_folds(dataset):
    FOLD = {}
    if dataset == 'CARDIAC_bssFP':
        FOLD[0] = set(range(0, 8))
        FOLD[1] = set(range(9, 17))
        FOLD[2] = set(range(18, 26))
        FOLD[3] = set(range(27, 35))
        FOLD[4] = set(range(36, 44))
        FOLD[4].update([0])
        return FOLD

    elif dataset == 'CARDIAC_LGE':
        FOLD[0] = set(range(0, 8))
        FOLD[1] = set(range(9, 17))
        FOLD[2] = set(range(18, 26))
        FOLD[3] = set(range(27, 35))
        FOLD[4] = set(range(36, 44))
        FOLD[4].update([0])
        return FOLD

    elif dataset == 'ABDOMEN_MR':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'ABDOMEN_CT':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'Prostate_UCLH':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'Prostate_Picture':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'Prostate_NCI':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
    elif dataset == 'Prostate_TCIA_PD':
        FOLD[0] = set(range(0, 5))
        FOLD[1] = set(range(4, 9))
        FOLD[2] = set(range(8, 13))
        FOLD[3] = set(range(12, 17))
        FOLD[4] = set(range(16, 20))
        FOLD[4].update([0])
        return FOLD
        
    else:
        raise ValueError(f'Dataset: {dataset} not found')


def sample_xy(spr, k=0, b=215):
    _, h, v = torch.where(spr)

    if len(h) == 0 or len(v) == 0:
        horizontal = 0
        vertical = 0
    else:

        h_min = min(h)
        h_max = max(h)
        if b > (h_max - h_min):
            kk = min(k, int((h_max - h_min) / 2))
            horizontal = random.randint(max(h_max - b - kk, 0), min(h_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            horizontal = random.randint(max(h_min - kk, 0), min(h_max - b + kk, 256 - b - 1))

        v_min = min(v)
        v_max = max(v)
        if b > (v_max - v_min):
            kk = min(k, int((v_max - v_min) / 2))
            vertical = random.randint(max(v_max - b - kk, 0), min(v_min + kk, 256 - b - 1))
        else:
            kk = min(k, int(b / 2))
            vertical = random.randint(max(v_min - kk, 0), min(v_max - b + kk, 256 - b - 1))

    return horizontal, vertical


