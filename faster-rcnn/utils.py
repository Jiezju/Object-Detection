import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def anchor_gen(feat_w, feat_h, scales, ratios, rpn_stride):
    # generate scales
    scales, ratios = np.meshgrid(scales, ratios)
    scales, ratios = scales.flatten(), ratios.flatten()
    scales_h = scales * np.sqrt(ratios)
    scales_w = scales / np.sqrt(ratios)

    # generate centers
    shift_x = np.arange(0, feat_w) * rpn_stride
    shift_y = np.arange(0, feat_h) * rpn_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift_x, shift_y = shift_x.flatten(), shift_y.flatten()

    center_x, anchor_w = np.meshgrid(shift_x, scales_w)
    center_y, anchor_h = np.meshgrid(shift_y, scales_h)

    anchor_center = np.stack([center_x, center_y], axis=-1).reshape(-1, 2)
    anchor_size = np.stack([anchor_w, anchor_h], axis=-1).reshape(-1, 2)
    boxes = np.concatenate([anchor_center - 0.5 * anchor_size, anchor_center + 0.5 * anchor_size], axis=-1)
    return boxes


def test_anchor_gen():
    feat_w = feat_h = 16
    rpn_stride = 8
    scales = [2, 4, 8]
    ratios = [0.5, 1, 2]
    boxes = anchor_gen(feat_w, feat_h, scales, ratios, rpn_stride)
    print(boxes.shape)

    plt.figure(figsize=(10, 10))
    img = np.ones((128,128,3))
    plt.imshow(img)
    axes = plt.gca()  # get current aces

    for i in range(boxes.shape[0]):
        anchor = boxes[i]
        rec = patches.Rectangle((anchor[0], anchor[1]), anchor[2] - anchor[0], anchor[3] - anchor[1], edgecolor='r', facecolor='none')
        axes.add_patch(rec)

    plt.show()



if __name__ == "__main__":
    test_anchor_gen()
