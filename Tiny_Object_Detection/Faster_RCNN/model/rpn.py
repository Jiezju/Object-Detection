import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat
from Faster_RCNN.utils.anchors import FCNAnchors, nms

class RPN(nn.Module):
    '''
    根据backbone输入 feature 产生 roi

    anchor 坐标： [x_c,y_c,w,h] 中心点 + 宽高

    '''
    def __init__(self, in_channels=1024, feature_size=26, stride=16, training=True):
        super(RPN, self).__init__()
        # ratio 表明 长宽 比例 （面积不变）
        self._ratio = [0.5, 1, 2]  # h / w
        # 基本 anchor 的 放大倍数
        self._scale = [8, 16, 32]
        self._base_size = 16
        self.n_pre_nms = 3000
        self.n_post_nms = 300
        self.nms_thresh = 0.5
        self._feature_size = feature_size # 输入 特征图大小 [26,26]
        self._rpn_stride = stride
        self.training = training

        # Anchor 生成
        self._anchors = FCNAnchors()
        self.n_anchors = len(self._scale) * len(self._ratio)

        # rpn 网络
        self.base_conv = nn.Conv2d(in_channels, 512, kernel_size=(3,3), padding=(1,1))
        self.cls_conv = nn.Conv2d(512, out_channels=self.n_anchors*2, kernel_size=(1,1))
        self.reg_conv = nn.Conv2d(512, out_channels=self.n_anchors*4, kernel_size=(1, 1))

    def _gen_proposals(self, anchors, locs):
        '''
        anchors: [x_l, y_l, x_r, y_r]
        locs: [t_c, t_c, t_w, t_h]
        '''
        bbxes_width = anchors[:, 2] - anchors[:, 0]
        boxes_height = anchors[:, 3] - anchors[:, 1]
        boxes_xc = anchors[:, 0] + 0.5 * bbxes_width
        boxes_yc = anchors[:, 1] + 0.5 * boxes_height

        tx = locs[:, 0::4].reshape(-1)
        ty = locs[:, 1::4].reshape(-1)
        tw = locs[:, 2::4].reshape(-1)
        th = locs[:, 3::4].reshape(-1)

        ctr_x = tx * bbxes_width + boxes_xc
        ctr_y = ty * boxes_height + boxes_yc
        w = torch.exp(tw) * bbxes_width
        h = torch.exp(th) * boxes_height

        ctr_x = ctr_x.reshape(-1,1)
        ctr_y = ctr_y.reshape(-1, 1)
        w = w.reshape(-1, 1)
        h = h.reshape(-1, 1)

        dst_bbox = torch.zeros(locs.shape, dtype=locs.dtype)
        dst_bbox[:, 0::4] = ctr_x - 0.5 * w
        dst_bbox[:, 1::4] = ctr_y - 0.5 * h
        dst_bbox[:, 2::4] = ctr_x + 0.5 * w
        dst_bbox[:, 3::4] = ctr_y + 0.5 * h

        return dst_bbox

    def _filter_boxes(self, boxes, scores):
        img_width = self._feature_size * self._rpn_stride
        img_height = self._feature_size * self._rpn_stride

        # proposals 的坐标截取
        boxes[:, slice(0, 4, 2)] = torch.clip(boxes[:, slice(0, 4, 2)], 0, img_width)
        boxes[:, slice(1, 4, 2)] = torch.clip(boxes[:, slice(1, 4, 2)], 0, img_height)

        # 宽高的最小值不可以小于16
        min_size = 16
        # 计算高宽
        ws = boxes[:, 2] - boxes[:, 0]
        hs = boxes[:, 3] - boxes[:, 1]
        # 防止建议框过小
        keep = torch.where((hs >= min_size) & (ws >= min_size))[0]
        boxes = boxes[keep, :]
        scores = scores[:, keep]

        # 排序 从小到大 逆序获得最大的那些score
        order = torch.argsort(scores.reshape(-1), descending=True)

        # 第一次筛选proposal 只选前n_pre_nms（3000）个
        if self.n_pre_nms > 0:
            order = order[:self.n_pre_nms]

        # NMS 
        boxes = boxes[order, :]
        scores = scores[:, order]

        proposals = nms(boxes.detach().numpy(), self.nms_thresh)

        proposals = torch.Tensor(proposals)

        return proposals[:self.n_post_nms], scores[:self.n_post_nms]

    def forward(self, x):
        batch = x.shape[0]

        assert batch == 1

        # 获取网络类别预测输出
        base = self.base_conv(x)
        rpn_conv = F.relu(base)

        rpn_reg_offset = self.reg_conv(rpn_conv)
        rpn_locs = rpn_reg_offset.permute(0, 2, 3, 1).contiguous().reshape(batch, -1, 4)

        rpn_cls = self.cls_conv(rpn_conv)
        rpn_cls_reshape = rpn_cls.permute(0,2,3,1).contiguous().reshape(batch, -1, 2)
        rpn_scores = F.softmax(rpn_cls_reshape, -1)

        # 获取前景 score
        rpn_fg_scores = rpn_scores[:,:,1]

        # generate anchors
        anchors = self._anchors()
        
        anchors = anchors.reshape(1,-1,4).expand(batch,-1,4)

        # generate proposals  anchor -> proposal
        proposals = self._gen_proposals(anchors[0], rpn_locs[0])

        # 筛选 proposal
        rois, fg_scores = self._filter_boxes(proposals, rpn_fg_scores)

        return rois, fg_scores
        
if __name__ == '__main__':
    rpn = RPN()
    x = torch.randn((1,1024,26,26))
    rpn(x)
    print('Success')





