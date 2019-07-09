from models.backbone import vgg16_bn
from models.rpn_head import RPNHead
from models.utils import nms
import torch.nn as nn

class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self):
        super(FasterRCNN, self).__init__()
        self.backbone = vgg16_bn(pretrained=True)
        self.rpn_head = RPNHead()
        #TODO: bbox head

        self.rpn_proposal_num = 2000

    def forward(self, imgs, gt_bboxes=None, gt_labels=None):
        feats = self.backbone(imgs)
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses = self.rpn_head(feats, gt_bboxes)
        obj_scores = nn.functional.softmax(obj_cls_scores, dim=2)
        proposals, obj_scores = nms(proposals, obj_scores, nms_iou_thr=0.7)
        return proposals, obj_scores, obj_cls_losses, obj_reg_losses
