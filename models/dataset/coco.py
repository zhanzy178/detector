import torch.utils.data.Dataset as Dataset
from pycocotools.coco import COCO
import os
import numpy as np
import cv2

class COCODataset(Dataset):
    """COCO Dataset for faster training"""

    def __init__(self, ann_file, img_root):
        super(COCODataset, self).__init__()

        # initialize COCO api for instance annotations
        self.coco = COCO(ann_file)
        self.img_root = img_root

        self.cats = self.coco.loadCats(self.coco.getCatIds()) # background is 0
        self.img_ids = self.coco.getImgIds()
        self.annotations = self.load_annotations(self.img_ids)

        self.img_size = (1333, 800)

    def __len__(self):
        pass


    def __getitem__(self, index):
        img_id = self.img_ids[index]

        img_path = os.path.join(self.img_root, '%012d.jpg'%img_id)
        bboxes, labels = self.annotations[index]

        # TODO: read image and apply transform


    def load_annotations(self, img_ids):
        annotations = []
        for img_id in img_ids:
            anns = self.coco.loadAnns(imgIds=img_id)
            bboxes = np.array([a['bbox'] for a in anns], dtype=np.float64)
            labels = np.array([a['category_id'] for a in anns], dtype=np.int32)

            annotations.append((bboxes, labels))
        return annotations


