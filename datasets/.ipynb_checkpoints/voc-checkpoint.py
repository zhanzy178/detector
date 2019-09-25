from torch.utils.data import Dataset
from cvtools.bbox import xyxy2xywh
import os
import numpy as np
import cv2
import json
from cvtools.evaluate import VOC_CLASS

class VOCDataset(Dataset):
    def __init__(self, ann_file, img_root, flip_rate=0.0, valid_mode=False):
        super(VOCDataset, self).__init__()
        self.class_name = VOC_CLASS

        self.img_root = img_root
        self.valid_mode = valid_mode
        self.annotations = self.load_annotations(ann_file)

        self.flip_rate=flip_rate
        self.img_size = (1000, 600)  # (W, H)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        anno = self.annotations[index]
        img_path = os.path.join(self.img_root, anno['filename'])

        # read image and apply transform
        if not os.path.exists(img_path):
            raise FileNotFoundError('file %s'%img_path)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img, img_meta = self.img_transform(img)

        data = dict(img=img)
        bboxes_ori, labels_ori = anno['bboxes'], anno['labels']
        if self.valid_mode:
            data.update(dict(gt_bboxes=bboxes_ori, gt_labels=labels_ori))
        else:
            bboxes, labels = self.anno_transform(bboxes_ori, labels_ori, img_meta)
            data.update(dict(gt_bboxes=bboxes, gt_labels=labels))

        data.update(dict(difficults=anno['difficults']))

        img_meta['img_id'] = anno['img_id']
        img_meta['filename'] = os.path.splitext(anno['filename'])[0]
        data.update(dict(img_meta=img_meta))

        return data

    def load_annotations(self, ann_file):
        with open(ann_file, 'r') as f:
            annotations = json.load(f)

        annotations_list = []
        for ann in annotations:
            bboxes = np.array([b['bbox'] for b in ann['bboxes']], dtype=np.float32)
            difficults = np.array([b['difficult'] for b in ann['bboxes']], dtype=np.int32)
            labels = np.array([self.class_name.index(b['name'])+1 for b in ann['bboxes']], dtype=np.int64)
            if bboxes.shape[0] != 0:
                # convert to format of xywh
                bboxes = xyxy2xywh(bboxes)

            for b in ann['bboxes']:
                assert self.class_name.index(b['name']) != -1

            annotations_list.append(dict(
                img_id=int(ann['filename'].split('.')[0]),
                filename=ann['filename'],
                width=ann['width'],
                height=ann['height'],
                bboxes=bboxes,
                difficults=difficults,
                labels=labels
                # backgroud as 0 label
            ))

        return annotations_list

    def img_transform(self, img):
        img_meta = dict()

        # rescale, img.shape = H, W, C
        rescale_h = float(min(self.img_size))/min(img.shape[0], img.shape[1]) # H
        rescale_w = float(max(self.img_size))/max(img.shape[0], img.shape[1]) # W
        img_meta['scale_ratio'] = scale_ratio = min(rescale_h, rescale_w)
        new_size = (int(scale_ratio*img.shape[1]+0.5), int(scale_ratio*img.shape[0]+0.5))  # new_size=(w, h)
        img = cv2.resize(img, new_size)
        img_meta['img_size'] = np.array(new_size, dtype=np.float32)
        
        # random flip
        if np.random.rand() < self.flip_rate:
            img = cv2.flip(img, 1)
            img_meta['flipped'] = True
        else:
            img_meta['flipped'] = False
            

        # normalize
        img = img.astype(np.float32)
        img_norm_mean = [0.485, 0.456, 0.406]
        img_norm_std = [0.229, 0.224, 0.225]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # to rgb
        img = (img/255.0 - img_norm_mean) / img_norm_std
        img = img.transpose((2, 0, 1))

        img = img.astype(np.float32)
        return img, img_meta


    def anno_transform(self, bboxes, labels, img_meta):
        # rescale
        bboxes = bboxes * img_meta['scale_ratio']
        if img_meta['flipped']:
            bboxes[:, 0] = img_meta['img_size'][0] - bboxes[:, 0]
    
        return bboxes, labels
