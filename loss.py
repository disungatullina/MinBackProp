import cv2
import torch
import torch.nn as nn
from cv_utils import *
from math_utils import *
import numpy as np
from scorings.msac_score import *
from feature_utils import *


class MatchLoss(object):
    """Rewrite Match loss from CLNet, symmetric epipolar distance"""

    def __init__(self):
        self.scoring_fun = MSACScore()

    def forward(
        self, models, gt_E, pts1, pts2, K1, K2, im_size1, im_size2, topk_flag=False, k=1
    ):
        essential_loss = []
        for b in range(gt_E.shape[0]):
            pts1_1 = pts1[b].clone()
            pts2_2 = pts2[b].clone()
            Es = models[b]

            _, gt_R_1, gt_t_1, gt_inliers = cv2.recoverPose(
                gt_E[b].astype(np.float64),
                pts1_1.unsqueeze(1).cpu().detach().numpy(),
                pts2_2.unsqueeze(1).cpu().detach().numpy(),
                np.eye(3, dtype=gt_E.dtype),
            )

            # find the ground truth inliers
            gt_mask = np.where(gt_inliers.ravel() > 0, 1.0, 0.0).astype(bool)
            gt_mask = torch.from_numpy(gt_mask).to(pts1_1.device)

            # symmetric epipolar errors based on gt inliers
            geod = batch_episym(
                pts1_1[gt_mask].repeat(Es.shape[0], 1, 1),
                pts2_2[gt_mask].repeat(Es.shape[0], 1, 1),
                Es,
            )
            e_l = torch.min(geod, geod.new_ones(geod.shape))
            if torch.isnan(e_l.mean()).any():
                print("nan values in pose loss")  # .1*

            if topk_flag:
                topk_indices = torch.topk(e_l.mean(1), k=k, largest=False).indices
                essential_loss.append(e_l[topk_indices].mean())
            else:
                essential_loss.append(e_l.mean())
        # average
        return sum(essential_loss) / gt_E.shape[0]
