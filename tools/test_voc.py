import sys
sys.path.append('.')
from cvtools.utils import add_args, Config
from cvtools.runner import load_checkpoint
from cvtools.utils import progressbar
from models import build_detector
from datasets import build_dataset

from torch.utils.data import DataLoader
import os
import numpy as np
import torch

from voc_eval import voc_eval
from cvtools.evaluate.voc_eval import VOC_CLASS


def test(cfg, checkpoint, result, eval):
    class_name = VOC_CLASS
    if not os.path.exists(result):
        # compute result and save to result path

        detector = build_detector(cfg.detector)
        detector.cuda()
        assert os.path.exists(checkpoint)
        load_checkpoint(detector, checkpoint)

        dataset = build_dataset(cfg.dataset.test)
        dataloader = DataLoader(dataset)

        cls_results = {name: [] for name in dataset.class_name}
        bar = progressbar.ProgressBar(len(dataloader))
        for batch_i, batch in enumerate(dataloader):
            with torch.no_grad():
                detector.eval()
                img, img_meta = batch['img'].cuda(), batch['img_meta']
                det_bboxes, det_labels = detector(img, img_meta)
            for b in range(img.size(0)):
                for bi, bbox in enumerate(det_bboxes[b]):
                    name = dataset.class_name[det_labels[b][bi]-1]
                    cls_results[name].append(dict(
                        bbox=[bbox[0]-bbox[2]/2+0.5, bbox[1]-bbox[3]/2+0.5, bbox[0]+bbox[2]/2-0.5, bbox[1]+bbox[3]/2-0.5, bbox[-1]],
                        filename=img_meta['filename'][b]
                    ))
            bar.update()

        # save in voc format
        os.mkdir(result)
        for cls in cls_results:
            rs = cls_results[cls]
            with open(os.path.join(result, cls+'.txt'), 'wt') as f:
                for r in rs:
                    f.write('{:s} {:.8f} {:.8f} {:.8f} {:.8f} {:.8f}\n'.
                          format(r['filename'], r['bbox'][-1],
                                 r['bbox'][0], r['bbox'][1],
                                 r['bbox'][2], r['bbox'][3]))

    if eval:
        # eval result
        aps = []
        imagesetfile = os.path.join(cfg.data_root, 'ImageSets/Main', 'test.txt')
        annopath = os.path.join(cfg.data_root, 'Annotations', '{}.xml')
        class_result=dict()
        for cls in class_name:
            rec, prec, ap = voc_eval(os.path.join(result, cls+'.txt'),
                     annopath,
                     imagesetfile,
                     cls,
                     os.path.join(result, '.cache'),
                     use_07_metric=False)

            aps.append(ap)

            print('AP for {} = {:.4f}'.format(cls, ap))
        print('Mean AP = {:.4f}'.format(np.mean(aps)))


if __name__ == '__main__':
    parser, cfg = Config.auto_argparser('Config file for detector, dataset')
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('result', type=str)
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
    test(cfg, args.checkpoint, args.result, args.eval)
