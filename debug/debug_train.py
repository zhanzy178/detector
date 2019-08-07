from models.detector import FasterRCNN
from datasets import COCODataset
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
import json
import mmcv


if __name__ == '__main__':
    train_cocodataset = COCODataset('/home/zzy/Projects/Datasets/coco/annotations/instances_train2017.json', '/home/zzy/Projects/Datasets/coco/images/train2017')
    train_loader = DataLoader(train_cocodataset, batch_size=1)
    val_cocodataset = COCODataset('/home/zzy/Projects/Datasets/coco/annotations/instances_val2017.json', '/home/zzy/Projects/Datasets/coco/images/val2017')
    val_loader = DataLoader(val_cocodataset, batch_size=1)


    faster_rcnn = FasterRCNN(num_classes=81).cuda()
    sgd_opt = SGD(faster_rcnn.parameters(), 0.01)

    # faster_rcnn.eval()
    for ep in range(1000):

        # train
        train_bar = mmcv.ProgressBar(len(train_loader))
        faster_rcnn.train()
        for iter, b in enumerate(train_loader):
            obj_cls_losses, obj_reg_losses, cls_losses, reg_losses \
                = faster_rcnn(b['img'].cuda(), b['img_meta'], b['gt_bboxes'].cuda(), b['gt_labels'].cuda())
            (obj_cls_losses + obj_reg_losses + cls_losses + reg_losses).backward()
            sgd_opt.step()
            sgd_opt.zero_grad()

            if iter % 100 == 0:
                print('\nobj_cls: %.3f, obj_reg: %.3f, cls: %.3f, reg: %.3f'
                      %(obj_cls_losses.item(), obj_reg_losses.item(), cls_losses.item(), reg_losses.item()))
            train_bar.update()

        torch.save(faster_rcnn.state_dict(), 'debug_faster.pth')

        # val
        val_bar = mmcv.ProgressBar(len(val_loader))
        faster_rcnn.eval()
        det_bboxes_results = []
        det_labels_results = []
        img_id_list = []
        for iter, b in enumerate(val_loader):
            with torch.no_grad():
                det_bboxes, det_labels = faster_rcnn(b['img'].cuda(), b['img_meta'])
            det_bboxes_results += det_bboxes
            det_labels_results += det_labels
            img_id_list += [iid.item() for iid in b['img_meta']['img_id']]

            val_bar.update()

        # save result
        coco_result = val_loader.dataset.coco_result(det_bboxes_results, det_labels_results, img_id_list)
        with open('debug_det_result.json', 'w') as f:
            json.dump(coco_result, f)

        # eval
        val_loader.dataset.evaluate('debug_det_result.json')
