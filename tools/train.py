import logging
from torch.utils.data import DataLoader

import sys
sys.path.append('.')
from models import build_detector
from datasets import build_dataset
from cvtools.utils import Config
from cvtools.runner import Runner


def batch_processor(model, data, train_mode):
    """cvtools.Runner process data api"""
    img, img_meta, gt_bboxes, gt_labels = \
        data['img'].cuda(), data['img_meta'], data['gt_bboxes'].cuda(), data['gt_labels'].cuda()

    if train_mode:
        rpn_cls_loss, rpn_reg_loss, cls_loss, reg_loss \
            = model(img, img_meta, gt_bboxes, gt_labels)

        loss = 0

        # log
        log_vars = dict()
        if rpn_cls_loss is not None:
            log_vars.update(dict(rpn_cls_loss=float(rpn_cls_loss)))
            loss += rpn_cls_loss
        if rpn_reg_loss is not None:
            log_vars.update(dict(rpn_reg_loss=float(rpn_reg_loss)))
            loss += rpn_cls_loss
        if cls_loss is not None:
            log_vars.update(dict(cls_loss=float(cls_loss)))
            loss += rpn_cls_loss
        if reg_loss is not None:
            log_vars.update(dict(reg_loss=float(reg_loss)))
            loss += rpn_cls_loss

        if loss is not None:
            log_vars.update(dict(loss=float(loss)))

        return dict(loss=loss,
                    log_vars=log_vars,
                    num_samples=img.size(0))
    else:
        det_bboxes, det_labels = model(img, img_meta, gt_bboxes, gt_labels)
        return dict(det_bboxes=det_bboxes, det_labels=det_labels, gt_bboxes=gt_bboxes, gt_labels=gt_labels)


def train_detector(cfg):
    # model
    detector = build_detector(cfg.detector)
    detector.cuda()

    # data
    train_dataset = build_dataset(cfg.dataset.train)
    val_dataset = build_dataset(cfg.dataset.val)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.img_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.img_batch_size)

    # runner
    runner = Runner(detector, batch_processor, cfg.optimizer, cfg.work_dir)
    runner.register_training_hooks(
        lr_config=cfg.lr_hook_cfg,
        optimizer_config=cfg.optimizer_hook_cfg,
        checkpoint_config=cfg.checkpoint_hook_cfg,
        log_config=cfg.log_hooks_cfg
    )

    # start training
    runner.run([train_dataloader, val_dataloader], [('train', 1), ('val', 1)], cfg.epoch)


if __name__ == '__main__':
    parser, cfg = Config.auto_argparser('Config file for detector, dataset, training setting')
    train_detector(cfg)
