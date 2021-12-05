import numpy as np

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

    while bboxes:
        candidate = bboxes.pop(0)

        remains = []
        for box in bboxes:
            if box != candidate and iou(box, candidate) < thresh:
                remains.append(box)
        
        bboxes = remains
        results.append(candidate)

    return results

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
        # 计算网格中心点 将（416,416）划分为（26*26）个网格
        shift_x = np.arange(0, self._f_size[0] * self._feat_stride, self._feat_stride)
        shift_y = np.arange(0, self._f_size[1] * self._feat_stride, self._feat_stride)
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)

        # 上下左右坐标的偏移量
        shift = np.stack((shift_x.ravel(),shift_y.ravel(),
                        shift_x.ravel(),shift_y.ravel(),), axis=1)

        # 每个网格点上的9个先验框
        A = anchor_base.shape[0] # 每个网格先验框数
        K = shift.shape[0]  # 网格个数
        anchors = anchor_base.reshape((1, A, 4)) + \
                shift.reshape((K, 1, 4))  # （K，A，4）
        # 所有的先验框
        anchors = anchors.reshape((K * A, 4)).astype(np.float32) #（9*26*26,4）
        return anchors
    
    def __call__(self):
        base_anchors = self._generate_anchor_base()
        return self._enumerate_shifted_anchor(base_anchors)




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    anchor_gen = FCNAnchors()
    nine_anchors = anchor_gen()
    print(nine_anchors)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108,109,110,111,112,113,114,115,116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    
    plt.show()