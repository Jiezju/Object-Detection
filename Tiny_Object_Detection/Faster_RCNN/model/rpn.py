import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pat

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
        self._feature_size = feature_size # 输入 特征图大小 [26,26]
        self._rpn_stride = stride
        self.training = training

        # Anchor 生成
        self._anchors = self._generate_anchors()

        self.base_conv = nn.Conv2d(in_channels, 512, kernel_size=(3,3), padding=(1,1))
        self.cls_conv = nn.Conv2d(512, out_channels=18, kernel_size=(1,1))
        self.reg_conv = nn.Conv2d(512, out_channels=36, kernel_size=(1, 1))

    def _generate_anchors(self):
        # # 1. 产生 base anchor
        # 生成 多尺度 anchor 的 w h
        size = self._base_size * self._base_size
        size_ratios = size / np.array(self._ratio)
        # round()方法返回x的四舍五入的数字，sqrt()方法返回数字x的平方根
        ws = np.round(np.sqrt(size_ratios))  # ws:[23 16 11]
        hs = np.round(ws * self._ratio)  # hs:[12 16 22],ws和hs一一对应

        base_anchors = []

        for w,h in zip(ws, hs):
            for scale in self._scale:
                anchor = [0, 0] + [w*scale, h*scale]
                base_anchors.append(anchor)

        # # 1. 产生 所有 anchor = base anchor + offset
        x_offsets = np.arange(0, self._feature_size)
        y_offsets = np.arange(0, self._feature_size)

        # 返回的 anchor 是映射在原图上的坐标 [x_c, y_c, w, h]
        anchors = []
        for x_o in x_offsets:
            for y_o in y_offsets:
                for anchor in base_anchors:
                    anchors.append([x_o * self._rpn_stride, y_o * self._rpn_stride] + anchor[2:])

        anchors = np.array(anchors)

        # # 显示 anchors
        # plt.figure(figsize=(10, 10))
        # img = np.ones((416, 416, 3))
        # plt.imshow(img)
        # Axs = plt.gca()
        # for i in range(anchors.shape[0]):
        #     anchor = anchors[i]
        #     rec = pat.Rectangle((anchor[0] - anchor[2] // 2, anchor[1] - anchor[3] // 2),
        #                         anchor[2], anchor[3],
        #                         edgecolor='r',
        #                         facecolor='none')
        #     Axs.add_patch(rec)
        # plt.show()

        return anchors

    def _filter_boxes(self, boxes, scores):
        sorted_scores, sorted_boxes_idx = torch.sort(scores, dim=0, descending=True)
        proposals = boxes[sorted_boxes_idx]


    def forward(self, x):
        # 获取网络类别预测输出
        base = self.base_conv(x)
        rpn_conv = F.relu(base)

        rpn_cls = self.cls_conv(rpn_conv)
        rpn_cls_reshape = rpn_cls.reshape(1, 2, 9*self._feature_size, self._feature_size)
        rpn_cls_prob_reshape = F.softmax(rpn_cls_reshape, 1)

        rpn_cls_prob = rpn_cls_prob_reshape.reshape(1, 18, self._feature_size, self._feature_size)
        rpn_reg_offset = self.reg_conv(rpn_conv)

        # generate proposals
        rpn_cls = rpn_cls_prob.permute(0,2,3,1).reshape(1, -1, 2)
        rpn_reg = rpn_reg_offset.permute(0,2,3,1).reshape(1, -1, 4)
        proposals = torch.zeros((self._feature_size*self._feature_size*9, 4))
        scores = torch.zeros((self._feature_size*self._feature_size*9))
        # anchor -> proposal
        for i in range(proposals.shape[0]):
            proposals[i, 0] = self._anchors[i, 0] + self._anchors[i, 2] * rpn_reg[0, i, 0]
            proposals[i, 1] = self._anchors[i, 1] + self._anchors[i, 3] * rpn_reg[0, i, 1]
            proposals[i, 2] = self._anchors[i, 2] + torch.exp(rpn_reg[0, i, 2])
            proposals[i, 3] = self._anchors[i, 3] + torch.exp(rpn_reg[0, i, 3])
            scores[i] = rpn_cls[0, i, 1]

        # 筛选 proposal
        self._filter_boxes(proposals, scores)





            








        
if __name__ == '__main__':
    rpn = RPN()
    x = torch.randn((1,1024,26,26))
    rpn(x)
    print('Success')





