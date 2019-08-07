from torch.utils.data import DataLoader

from mmcv.utils import Config
from mmcv.runner import Runner

from models import build_detector
from datasets import build_dataset


def batch_processor(model, data, train_mode):
    """mmcv.Runner process data api"""
    img, img_meta, gt_bboxes, gt_labels = \
        data['img'].cuda(), data['img_meta'], data['gt_bboxes'].cuda(), data['gt_labels'].cuda()

    if train_mode:
        obj_cls_losses, obj_reg_losses, cls_losses, reg_losses \
            = model(img, img_meta, gt_bboxes, gt_labels)
        return dict(obj_cls_losses=obj_cls_losses,
                    obj_reg_losses=obj_reg_losses,
                    cls_losses=cls_losses,
                    reg_losses=reg_losses)
    else:
        det_bboxes, det_labels = model(img, img_meta, gt_bboxes, gt_labels)
        return dict(det_bboxes=det_bboxes, det_labels=det_labels)


def train_detector(cfg):
    # model
    detector = build_detector(cfg.detector)

    # data
    train_dataset = build_dataset(cfg.dataset.train)
    val_dataset = build_dataset(cfg.dataset.val)
    train_dataloader = DataLoader(train_dataset, batch_size=1)
    val_dataloader = DataLoader(val_dataset, batch_size=1)

    # runner
    runner = Runner(detector, batch_processor)
    runner.register_training_hooks()

    # start training
    runner.run([train_dataloader, val_dataloader], [('train', 1), ('val', 1)], 120)


if __name__ == '__main__':
    parser, cfg = Config.auto_argparser('Config file for detector, dataset, training setting')
    train_detector(cfg)
