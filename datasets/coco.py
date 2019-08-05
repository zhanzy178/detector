from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import numpy as np
import cv2


class COCODataset(Dataset):
    """COCO Dataset for faster training"""

    def __init__(self, ann_file, img_root, test_mode=False):
        super(COCODataset, self).__init__()

        # initialize COCO api for instance annotations
        self.coco = COCO(ann_file)
        self.img_root = img_root
        self.test_mode = test_mode

        self.cats = self.coco.loadCats(self.coco.getCatIds()) # background is 0
        self.cats2label = {c['id']:i+1 for i, c in enumerate(self.cats)}
        self.img_ids = self.coco.getImgIds()
        if not self.test_mode:
            self.annotations = self.load_annotations(self.img_ids)

        self.img_size = (1000, 600) # (W, H)

    def __len__(self):
        return len(self.img_ids)


    def __getitem__(self, index):
        img_id = self.img_ids[index]

        img_path = os.path.join(self.img_root, '%012d.jpg'%img_id)

        # read image and apply transform
        if not os.path.exists(img_path):
            raise FileNotFoundError('file %s'%img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # read in rgb
        img, img_meta = self.img_transform(img)

        data = dict(img=img)
        if not self.test_mode:
            bboxes, labels = self.annotations[index]
            bboxes, labels = self.anno_transform(bboxes, labels, img_meta)
            data.update(dict(gt_bboxes=bboxes, gt_labels=labels))

        img_meta['img_id'] = img_id
        data.update(dict(img_meta=img_meta))

        return data

    def img_transform(self, img):
        img_meta = dict()

        # rescale, img.shape = H, W, C
        rescale_h = float(min(self.img_size))/min(img.shape) # H
        rescale_w = float(max(self.img_size))/max(img.shape) # W
        img_meta['scale_ratio'] = scale_ratio = min(rescale_h, rescale_w)
        new_size = (int(scale_ratio*img.shape[1]+0.5), int(scale_ratio*img.shape[0]+0.5))  # new_size=(w, h)
        img = cv2.resize(img, new_size)
        img_meta['img_size'] = np.array(new_size, dtype=np.float32)

        # normalize
        img = img.astype(np.float32)
        img_norm_mean = [123.675, 116.28, 103.53]
        img_norm_std = [58.395, 57.12, 57.375]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - img_norm_mean) / img_norm_std
        img = img.transpose((2, 0, 1))

        img = img.astype(np.float32)
        return img, img_meta


    def anno_transform(self, bboxes, labels, img_meta):
        # rescale
        bboxes = bboxes * img_meta['scale_ratio']
        return bboxes, labels

    def load_annotations(self, img_ids):
        annotations = []
        for img_id in img_ids:
            anns_id = self.coco.getAnnIds(imgIds=img_id)
            anns = self.coco.loadAnns(ids=anns_id)

            # convert coco bbox format from (x_left, y_top, w, h) to (x_center, y_center, w, h)
            bboxes = np.array([a['bbox'] for a in anns if a['iscrowd']==0], dtype=np.float32)
            if bboxes.shape[0] != 0:
                bboxes[:, [0, 1]] += bboxes[:, [2, 3]] / 2
            labels = np.array([self.cats2label[a['category_id']] for a in anns if a['iscrowd']==0], dtype=np.int64)

            annotations.append((bboxes, labels))
        return annotations


    def coco_result(self, det_bboxes_list, det_labels_list, img_id_list):
        annotations = []
        label2cats = {v:k for k, v in self.cats2label.items()}
        for det_bboxes, det_labels, image_id in zip(det_bboxes_list, det_labels_list, img_id_list):
            for bi in range(len(det_bboxes)):
                bbox = det_bboxes[bi]
                score = float(bbox[-1])
                x1 = float(bbox[0] - bbox[2]/2)
                y1 = float(bbox[1] - bbox[3]/2)
                w = float(bbox[2])
                h = float(bbox[3])
                category_id = label2cats[int(det_labels[bi])]

                annotations.append(dict(
                    image_id=image_id,
                    bbox=[x1, y1, w, h],
                    score=score,
                    category_id=category_id
                ))

        return annotations

    def evaluate(self, result_file):
        cocoDt = self.coco.loadRes(result_file)

        # running evaluation
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        # cocoEval.params.imgIds = [self.img_ids[i] for i in range(8)]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
