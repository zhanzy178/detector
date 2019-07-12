from models.detector import FasterRCNN
from datasets import COCODataset
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD


if __name__ == '__main__':
    faster_rcnn = FasterRCNN(num_classes=81).cuda()
    cocodataset = COCODataset('/home/zzy/Projects/Datasets/coco/annotations/instances_val2017.json', '/home/zzy/Projects/Datasets/coco/images/val2017')
    loader = DataLoader(cocodataset, batch_size=1)
    sgd_opt = SGD(faster_rcnn.parameters(), 0.01)

    faster_rcnn.train()
    # faster_rcnn.eval()
    for ep in range(1000):
        for iter, b in enumerate(loader):
            obj_cls_losses, obj_reg_losses, cls_losses, reg_losses \
                = faster_rcnn(b['img'].cuda(), b['img_meta'], b['gt_bboxes'].cuda(), b['gt_labels'].cuda())

            (obj_cls_losses + obj_reg_losses + cls_losses + reg_losses).backward()

            print((obj_cls_losses + obj_reg_losses + cls_losses + reg_losses).item())

            if (iter+1) % 8 == 0:
                torch.save(faster_rcnn.state_dict(), 'debug_rpn.pth')
                sgd_opt.step()
                sgd_opt.zero_grad()
                print('step', ep)
                break



