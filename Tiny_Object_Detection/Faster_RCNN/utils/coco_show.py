# show image and labels
import cv2
import numpy as np

CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

def coco_show(image, anns):
    imgIdx = 1
    if anns.shape[0] == 0:
        return -1
    for ann in anns:
        # 通过类型索引获得类型
        p1 = (int(round(ann[0])), int(round(ann[1])))
        p2 = (int(round(ann[2])), int(round(ann[3])))
        cv2.rectangle(image, p1, p2, (255, 0, 0), 2)
        # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
        cv2.putText(image, CLASSES[int(ann[-1])], (p2[0] - 40, p2[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, 8)
    cv2.imshow('test', image)
    cv2.waitKey(0)
    imgIdx += 1
