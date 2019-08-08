import sys
sys.path.append('.')

from models.detector import FasterRCNN
from datasets import COCODataset
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from time import time


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    faster_rcnn = FasterRCNN(num_classes=81).cuda()
    cocodataset = COCODataset('/home/zzy/Datasets/coco/annotations/instances_val2017.json', '/home/zzy/Datasets/coco/images/val2017')
    loader = DataLoader(cocodataset, batch_size=1)
    sgd_opt = SGD([p for p in faster_rcnn.parameters() if p.requires_grad], 0.001)

    faster_rcnn.train()
    # faster_rcnn.eval()
    for ep in range(1000):
        for iter, b in enumerate(loader):
            start = time()
            obj_cls_losses, obj_reg_losses, cls_losses, reg_losses \
                = faster_rcnn(b['img'].cuda(), b['img_meta'], b['gt_bboxes'].cuda(), b['gt_labels'].cuda())

            (obj_cls_losses + obj_reg_losses + cls_losses + reg_losses).backward()

            # if (iter+1) % 1 == 0:
            torch.save(faster_rcnn.state_dict(), 'debug_faster.pth')
            sgd_opt.step()
            sgd_opt.zero_grad()

            end = time()
            print(b['img'].size())
            print((obj_cls_losses + obj_reg_losses + cls_losses + reg_losses).item())
            print('step', ep)
            print('time', end-start)

            # if (iter+1) % 8 == 0:
            #     break



