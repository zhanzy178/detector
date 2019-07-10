from models.detector import FasterRCNN
from datasets import COCODataset
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD


if __name__ == '__main__':
    faster_rcnn = FasterRCNN().cuda()
    cocodataset = COCODataset('/home/zzy/Projects/Datasets/coco/annotations/instances_val2017.json', '/home/zzy/Projects/Datasets/coco/images/val2017')
    loader = DataLoader(cocodataset, batch_size=1)
    sgd_opt = SGD(faster_rcnn.parameters(), 0.01)

    faster_rcnn.train()
    # faster_rcnn.eval()
    for ep in range(100):
        for iter, b in enumerate(loader):
            proposals, pscores, obj_cls_losses, obj_reg_losses \
                = faster_rcnn(b['img'].cuda(), b['img_meta'], b['gt_bboxes'].cuda(), b['gt_labels'].cuda())

            (obj_cls_losses+obj_reg_losses*10).backward()
            print('id:%d, p:%d, cls_loss:%.5f, reg_loss:%.5f, loss:%.5f' %
                (b['img_meta']['img_id'][0], proposals[0].size(0), obj_cls_losses.item(), obj_reg_losses.item(), (obj_cls_losses+obj_reg_losses*10).item()))
            if (iter+1) % 16 == 0:
                sgd_opt.step()
                sgd_opt.zero_grad()
                print('step')
                torch.save(faster_rcnn.state_dict(), 'debug_rpn.pth')
                break



