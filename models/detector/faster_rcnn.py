from models.backbone import vgg16_bn
from models.rpn_head import RPNHead
from models.utils import nms
import torch.nn as nn
import torch
from models.roi_pool import RoIPool
from models.bbox_head import BBoxHead

class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.strides = [16]

        self.backbone = vgg16_bn(pretrained=True)
        self.rpn_head = RPNHead(self.strides)

        self.roi_pool = RoIPool(out_size=7, spatial_scale=1.0/self.strides[0])
        self.bbox_head = BBoxHead(num_classes=num_classes)

        self.rpn_proposal_num = 2000

    def forward(self, img, img_meta, gt_bboxes=None, gt_labels=None):
        feat = self.backbone(img)
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses = self.rpn_head(feat, img_meta, gt_bboxes)
        obj_cls_scores = nn.functional.softmax(obj_cls_scores, dim=2)
        proposals, obj_scores = nms(proposals, obj_cls_scores, nms_iou_thr=0.7)

        # extract 2000 proposals
        for b in range(len(proposals)):
            if proposals[b].size(0) > self.rpn_proposal_num:
                proposals[b] = proposals[b][:self.rpn_proposal_num]

        # TODO: bbox head

        return proposals, obj_scores, obj_cls_losses, obj_reg_losses
