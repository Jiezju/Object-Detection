from pycocotools.coco import COCO
from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import utils


# image process
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


class CocoDataset(Dataset):
    def __init__(self, ann_file, img_path, input_size=(416,416), transform=None):
        self.coco = COCO(ann_file) # json
        self.transform = transform
        self.image_path = img_path
        self.image_ids = self.coco.getImgIds()
        self.img_size = input_size
        self.labels = {}
        self.load_classes()

    def load_classes(self):
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        # coco ids is not from 1, and not continue
        # make a new index from 0 to 79, continuely

        # classes:             {names:      new_index}
        # coco_labels:         {new_index:  coco_index}
        # coco_labels_inverse: {coco_index: new_index}
        self.classes, self.coco_labels, self.coco_labels_inverse = {}, {}, {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # labels:              {new_index:  names}
        self.labels = {}
        for k, v in self.classes.items():
            self.labels[v] = k

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        img = self.load_image(index)
        ann = self.load_anns(index)
        img, ratio, pad = letterbox(img, self.img_size)
        boxes = np.zeros_like(ann)
        boxes[:, 0] = ratio[0] * ann[:, 0] + pad[0]  # pad width
        boxes[:, 1] = ratio[1] * ann[:, 1] + pad[1]  # pad height
        boxes[:, 2] = ratio[0] * ann[:, 2] + pad[0]
        boxes[:, 3] = ratio[1] * ann[:, 3] + pad[1]
        boxes[:, 4] = ann[:, 4]
        sample = {'img': img, 'bbox': boxes}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, index):
        image_info = self.coco.loadImgs(self.image_ids[index])[0]
        imgpath = os.path.join(self.image_path,
                               image_info['file_name'])
        img = cv2.imread(imgpath)
        return img.astype(np.float32)

    def load_anns(self, index):
        annotation_ids = self.coco.getAnnIds(self.image_ids[index], iscrowd=False)
        # anns is num_anns x 5, (x1, x2, y1, y2, new_idx)
        anns = np.zeros((0, 5))

        # skip the image without annoations
        if len(annotation_ids) == 0:
            return anns

        coco_anns = self.coco.loadAnns(annotation_ids)
        for a in coco_anns:
            # skip the annotations with width or height < 1
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            ann = np.zeros((1, 5))
            ann[0, :4] = a['bbox']
            ann[0, 4] = self.coco_labels_inverse[a['category_id']]
            anns = np.append(anns, ann, axis=0)

        # (x1, y1, width, height) --> (x1, y1, x2, y2)
        anns[:, 2] += anns[:, 0]
        anns[:, 3] += anns[:, 1]

        return anns

    def image_aspect_ratio(self, index):
        image = self.coco.loadImgs(self.image_ids[index])[0]
        return float(image['width']) / float(image['height'])


# test
if __name__ == '__main__':
    ann_files = '/home/bright/pycharm_project/mmdetection/tools/data/coco/annotations/instances_val2017.json'
    imgs_path = '/home/bright/pycharm_project/mmdetection/tools/data/coco/val2017'
    coco = CocoDataset(ann_file=ann_files, img_path=imgs_path)
    sample = coco[0]
    utils.coco_show(sample['img'].astype(np.uint8), sample['bbox'])
    print('success!')
