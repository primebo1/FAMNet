"""
Utils for Dataset
Extended from ADNet code by Hansen et al.
"""
import random
import torch
import numpy as np
import operator
import os
import logging


def set_seed(seed):
    """
    Set the random seed
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


CLASS_LABELS = {
    'CHAOST2': {
        'pa_all': set(range(1, 5)),
        0: set([1, 4]),  # upper_abdomen, leaving kidneies as testing classes
        1: set([2, 3]),  # lower_abdomen
    },
}


def get_bbox(fg_mask, inst_mask):
    """
    Get the ground truth bounding boxes
    """

    fg_bbox = torch.zeros_like(fg_mask, device=fg_mask.device)
    bg_bbox = torch.ones_like(fg_mask, device=fg_mask.device)

    inst_mask[fg_mask == 0] = 0
    area = torch.bincount(inst_mask.view(-1))
    cls_id = area[1:].argmax() + 1
    cls_ids = np.unique(inst_mask)[1:]

    mask_idx = np.where(inst_mask[0] == cls_id)
    y_min = mask_idx[0].min()
    y_max = mask_idx[0].max()
    x_min = mask_idx[1].min()
    x_max = mask_idx[1].max()
    fg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 1

    for i in cls_ids:
        mask_idx = np.where(inst_mask[0] == i)
        y_min = max(mask_idx[0].min(), 0)
        y_max = min(mask_idx[0].max(), fg_mask.shape[1] - 1)
        x_min = max(mask_idx[1].min(), 0)
        x_max = min(mask_idx[1].max(), fg_mask.shape[2] - 1)
        bg_bbox[0, y_min:y_max + 1, x_min:x_max + 1] = 0
    return fg_bbox, bg_bbox


def t2n(img_t):
    """
    torch to numpy regardless of whether tensor is on gpu or memory
    """
    if img_t.is_cuda:
        return img_t.data.cpu().numpy()
    else:
        return img_t.data.numpy()


def to01(x_np):
    """
    normalize a numpy to 0-1 for visualize
    """
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-5)


class Scores():

    def __init__(self):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        self.patient_dice = []
        self.patient_iou = []

    def record(self, preds, label):
        assert len(torch.unique(preds)) < 3

        tp = torch.sum((label == 1) * (preds == 1))
        tn = torch.sum((label == 0) * (preds == 0))
        fp = torch.sum((label == 0) * (preds == 1))
        fn = torch.sum((label == 1) * (preds == 0))

        self.patient_dice.append(2 * tp / (2 * tp + fp + fn))
        self.patient_iou.append(tp / (tp + fp + fn))

        self.TP += tp
        self.TN += tn
        self.FP += fp
        self.FN += fn

    def compute_dice(self):
        return 2 * self.TP / (2 * self.TP + self.FP + self.FN)

    def compute_iou(self):
        return self.TP / (self.TP + self.FP + self.FN)


def set_logger(path):
    logger = logging.getLogger()
    logger.handlers = []
    formatter = logging.Formatter('[%(levelname)] - %(name)s - %(message)s')
    logger.setLevel("INFO")

    # log to .txt
    file_handler = logging.FileHandler(path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # log to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger
