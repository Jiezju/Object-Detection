import torch
from torch import nn
from torchvision.ops import RoIPool

class ROIRCNN(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, in_channels):
        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale
        self.in_channels = in_channels

        self.roipool = RoIPool(roi_size, spatial_scale)
        self.post_fc0 = nn.Linear(self.in_channels, 2048)
        self.post_fc1 = nn.Linear(2048, 2048)
        self.cls_fc = nn.Linear(2048, self.n_class)
        self.loc_fc = nn.Linear(2048, self.n_class*4)

    def forward(self, features, rois):
        roi_out = self.roipool(features, rois)
        pool = roi_out.reshape(256, -1)
        fc = self.post_fc1(self.post_fc0(pool))

        roi_cls_locs = self.cls_loc_fcloc(fc)
        roi_scores = self.cls_fc(fc)

        return roi_cls_locs, roi_scores

