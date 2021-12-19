import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .utils import bbox_overlaps

def _loc_loss_func(pos_boxes, gt_boxes):
    pre_widths = pos_boxes[:, 2] - pos_boxes[:, 0] + 1.0
    pre_heights = pos_boxes[:, 3] - pos_boxes[:, 1] + 1.0
    pre_ctr_x = pos_boxes[:, 0] + 0.5 * pre_widths
    pre_ctr_y = pos_boxes[:, 1] + 0.5 * pre_heights

    pre_boxes_ctr = torch.cat([pre_ctr_x, pre_ctr_y, pre_widths, pre_heights], axis=1)

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_boxes[:, 1] + 0.5 * gt_heights

    gt_boxes_ctr = torch.cat([gt_ctr_x, gt_ctr_y, gt_widths, gt_heights], axis=1)

    loss = F.smooth_l1_loss(pre_boxes_ctr, gt_boxes_ctr)

    return loss

def _cls_loss_func(pos_scores, neg_scores):
    pos_label = torch.ones(pos_scores.shape)
    neg_label = torch.zeros(neg_scores)

    return F.cross_entropy(pos_scores, pos_label) + F.cross_entropy(neg_scores, neg_label)


class ProposalTarget(nn.Module):
    def __init__(self, positive_thresh=0.7, negtive_thresh=0.3, num_fg=128):
        self.positive_thresh = positive_thresh
        self.negtive_thresh = negtive_thresh
        self.num_fg = num_fg

    def forward(self, rois, fg_scores, gts):
        '''
        :param rois: [*,4]
        :param fg_scores: [*,1]
        :param gts: [*, 5]
        :return:
        '''

        gt_boxes = gts[:, :4]

        labels = torch.full(rois.shape[0], -1).float()

        # [achors, gts]
        overlaps = bbox_overlaps(rois.detach().numpy(), gt_boxes.detach().numpy())

        # anchor target anchor : target
        max_overlaps, argmax_overlaps = torch.max(overlaps, dim=1)

        # target : anchor
        gt_max_overlaps, _ = torch.max(overlaps, dim=0)

        # < 0.3 为负样本
        labels[max_overlaps < 0.3] = 0
        gt_max_overlaps[gt_max_overlaps == 0] = 1e-5

        # 与标签具有最大 overlaps 的 anchor index
        keep = torch.sum(overlaps.eq(gt_max_overlaps.reshape(-1,1).expand_as(overlaps)), 1)

        if torch.sum(keep) > 0:
            labels[keep > 0] = 1

        labels[max_overlaps > 0.7] = 1

        pos_inds = labels[labels == 1]
        neg_inds = labels[labels == 2]

        if torch.sum(labels == 1) > self.num_fg:
            perm = torch.randperm(pos_inds.numel())[:self.num_fg]
            pos_inds = pos_inds[perm]

        if torch.sum(labels == 0.3) > (256 - self.num_fg):
            perm = torch.randperm(neg_inds.numel())[:self.num_fg]
            neg_inds = neg_inds[perm]

        pos_boxes = rois[pos_inds]
        target_boxes = gts[argmax_overlaps[pos_inds]]
        pos_labels = fg_scores[pos_inds]
        neg_labels = fg_scores[neg_inds]

        # loc loss
        loc_loss = _loc_loss_func(pos_boxes, target_boxes)
        cls_loss = _cls_loss_func(pos_labels, neg_labels)

        return loc_loss + cls_loss
















