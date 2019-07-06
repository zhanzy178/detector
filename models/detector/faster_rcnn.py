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
        #TODO: bbox head

        self.rpn_proposal_num = 2000

    def forward(self, imgs, gt_bboxes=None, gt_labels=None):
        feats = self.backbone(imgs)
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses = self.rpn_head(feats, gt_bboxes)
        obj_scores = nn.functional.softmax(obj_cls_scores)
        _, sorted_ind = obj_scores[:, 0].sort()

        return proposals[sorted_ind[:self.rpn_proposal_num]], obj_scores[sorted_ind[:self.rpn_proposal_num]]
