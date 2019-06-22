from models.backbone import vgg16_bn
from models.rpn_head import RPNHead
import torch.nn as nn

class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = vgg16_bn(pretrained=True)
        self.rpn_head = RPNHead()
        #TODO: roi feat extractor
        #TODO: bbox head

    def forward(self, imgs, gt_bboxes=None, gt_labels=None):
        feats = self.backbone(imgs)
        proposals, obj_cls_scores, obj_reg_scores, \
        obj_cls_losses, obj_reg_losses = self.rpn_head(feats, gt_bboxes, gt_labels)
