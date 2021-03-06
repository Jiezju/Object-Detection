import numpy as np
import torch

def iou(bbox_a, bbox_b):
    lmax = max(bbox_a[0], bbox_b[0])
    tmax = max(bbox_a[1], bbox_b[1])

    lmin = max(bbox_a[2], bbox_b[2])
    tmin = max(bbox_a[3], bbox_b[3])

    insec = max(0, lmin - lmax) * max(0, tmin - tmax)

    unia = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1])
    unib = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1])

    return insec / (unia + unib - insec)

def nms(bboxes, thresh):
    results = []
    
    boxes = []
    
    for box in bboxes:
        boxes.append(box)

    while boxes:
        candidate = boxes.pop(0)

        remains = []
        for box in boxes:
            if (box != candidate).all() and iou(box, candidate) < thresh:
                remains.append(box)
        
        boxes = remains
        results.append(candidate)

    return results

def bbox_overlaps(anchors, gt_boxes):
    ious = np.zeros((anchors.shape[0], gt_boxes.shape[0])).astype(np.float32)

    for i in range(anchors.shape[0]):
        for j in range(gt_boxes.shape[0]):
            ious[i,j] = iou(anchors[i], gt_boxes[j])

    return torch.Tensor(ious)


class FCNAnchors():
    def __init__(self, base_size=16,  feat_stride=16, f_size=(26,26), 
                        ratios=[0.5, 1, 2],
                        scales=[8, 16, 32]):
        self._base_size = base_size
        self._feat_stride = feat_stride
        self._f_size = f_size
        self._ratios = ratios
        self._scales = scales

    def _generate_anchor_base(self):
        # [x_l,y_l, x_r,y_r]
        anchor_base = np.zeros((len(self._ratios) * len(self._scales), 4),
                            dtype=np.float32)
        for i in range(len(self._ratios)):
            for j in range(len(self._scales)):
                h = self._base_size * self._scales[j] * np.sqrt(self._ratios[i])
                w = self._base_size * self._scales[j] * np.sqrt(1. / self._ratios[i])

                index = i * len(self._scales) + j
                anchor_base[index, 0] = - h / 2.
                anchor_base[index, 1] = - w / 2.
                anchor_base[index, 2] = h / 2.
                anchor_base[index, 3] = w / 2.
        return anchor_base

    def _enumerate_shifted_anchor(self, anchor_base):
        # ????????????????????? ??????416,416???????????????26*26????????????
        shift_x = np.arange(0, self._f_size[0] * self._feat_stride, self._feat_stride)
        shift_y = np.arange(0, self._f_size[1] * self._feat_stride, self._feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        # ??????????????????????????????
        shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                        shift_x.ravel(),shift_y.ravel(),), axis=1)

        # ?????????????????????9????????????
        A = anchor_base.shape[0] # ????????????????????????
        K = shift.shape[0]  # ????????????
        anchors = anchor_base.reshape((1, A, 4)) + \
                shift.reshape((K, 1, 4))  # ???K???A???4???
        # ??????????????????
        anchors = anchors.reshape((K * A, 4)).astype(np.float32) #???9*26*26,4???
        return torch.Tensor(anchors)
    
    def __call__(self):
        base_anchors = self._generate_anchor_base()
        return self._enumerate_shifted_anchor(base_anchors)
