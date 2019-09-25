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
from cvtools.evaluate import VOC_CLASS


def test(cfg, checkpoint, result, eval):
    class_name = VOC_CLASS
    if not os.path.exists(result):
        # compute result and save to result path

        detector = build_detector(cfg.detector)
        detector.cuda()
        # assert os.path.exists(checkpoint)
        # load_checkpoint(detector, checkpoint)
        state_dict = torch.load('/home/zzy/Downloads/faster_rcnn_1_6_10021.pth')
        dstate_dict = detector.state_dict()

        key_mapper = {
            'backbone.features.0.weight':'RCNN_base.0.weight', 'backbone.features.0.bias':'RCNN_base.0.bias', 'backbone.features.2.weight':'RCNN_base.2.weight',
            'backbone.features.2.bias':'RCNN_base.2.bias',   'backbone.features.5.weight':'RCNN_base.5.weight', 'backbone.features.5.bias':'RCNN_base.5.bias',
            'backbone.features.7.weight':'RCNN_base.7.weight',   'backbone.features.7.bias':'RCNN_base.7.bias', 'backbone.features.10.weight':'RCNN_base.10.weight',
            'backbone.features.10.bias':'RCNN_base.10.bias', 'backbone.features.12.weight':'RCNN_base.12.weight', 'backbone.features.12.bias':'RCNN_base.12.bias',
            'backbone.features.14.weight':'RCNN_base.14.weight', 'backbone.features.14.bias':'RCNN_base.14.bias', 'backbone.features.17.weight':'RCNN_base.17.weight',
            'backbone.features.17.bias':'RCNN_base.17.bias', 'backbone.features.19.weight':'RCNN_base.19.weight', 'backbone.features.19.bias':'RCNN_base.19.bias',
            'backbone.features.21.weight':'RCNN_base.21.weight', 'backbone.features.21.bias':'RCNN_base.21.bias', 'backbone.features.24.weight':'RCNN_base.24.weight',
            'backbone.features.24.bias':'RCNN_base.24.bias', 'backbone.features.26.weight':'RCNN_base.26.weight', 'backbone.features.26.bias':'RCNN_base.26.bias',
            'backbone.features.28.weight':'RCNN_base.28.weight', 'backbone.features.28.bias':'RCNN_base.28.bias', 'rpn_head.conv.weight':'RCNN_rpn.RPN_Conv.weight', 'rpn_head.conv.bias':'RCNN_rpn.RPN_Conv.bias',
            'rpn_head.obj_cls.weight':'RCNN_rpn.RPN_cls_score.weight', 'rpn_head.obj_cls.bias':'RCNN_rpn.RPN_cls_score.bias', 'rpn_head.obj_reg.weight':'RCNN_rpn.RPN_bbox_pred.weight', 'rpn_head.obj_reg.bias':'RCNN_rpn.RPN_bbox_pred.bias',
            'bbox_head.shared_layers.0.weight':'RCNN_top.0.weight', 'bbox_head.shared_layers.0.bias':'RCNN_top.0.bias', 'bbox_head.shared_layers.3.weight':'RCNN_top.3.weight',
            'bbox_head.shared_layers.3.bias':'RCNN_top.3.bias', 'bbox_head.cls_fc.weight':'RCNN_cls_score.weight', 'bbox_head.cls_fc.bias':'RCNN_cls_score.bias', 'bbox_head.reg_fc.weight':'RCNN_bbox_pred.weight',
            'bbox_head.reg_fc.bias':'RCNN_bbox_pred.bias'
        }
        for k in list(dstate_dict.keys()):
            dstate_dict[k] = state_dict['model'][key_mapper[k]]
        detector.load_state_dict(dstate_dict)

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
        for ci, cls in enumerate(class_name):
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
