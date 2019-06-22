import torch.nn as nn
import torch
from models.utils import anchor2bbox

class RPNHead(nn.Module):
    """RPNHead Region Proposal Network to predict region proposal"""
    def __init__(self):
        super(RPNHead, self).__init__()

        self.anchor_ratio = [0.5, 1.0, 1.5] # ratio is h / w
        self.anchor_scale = [128.0, 256.0, 512.0]
        self.anchor_stride = [16]
        self.anchor_template_len = len(self.anchor_ratio) * len(self.anchor_scale)
        self.anchor_template = torch.zeros(size=(self.anchor_template_len, 4), dtype=torch.float64)
        for ri, r in enumerate(self.anchor_ratio):
            for si, s in enumerate(self.anchor_scale):
                i = ri*len(self.anchor_ratio) + si
                #  w = s / sqrt(r); h = s * sqrt(r);
                sqrt_r = sqrt(r)
                w = s/sqrt_r
                h = s*sqrt_r
                self.anchor_template[i, :] = torch.Tensor([w/2, h/2, w, h])

        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.obj_cls = nn.Conv2d(512, 2*self.anchor_template_len, kernel_size=1, stride=1)
        self.obj_reg = nn.Conv2d(512, 4*self.anchor_template_len, kernel_size=1, stride=1)

    def forward(self, feature_maps, gt_bboxes, gt_labels):
        f = self.conv(feature_maps)
        obj_cls_scores = self.obj_cls(f)
        obj_reg_scores = self.obj_reg(f)

        # TODO: compute region proposals for bbox prediction and recognition
        proposals = self.convert_bboxes(obj_reg_scores)

        obj_cls_losses, obj_reg_losses = None, None

        # train mode compute loss
        if self.training:
            # TODO: compute RPN loss and split train forward and test forward
            pass

        return proposals, obj_cls_scores, obj_reg_scores, obj_cls_losses, obj_reg_losses

    def convert_bboxes(self, obj_reg_scores):
        """generate bboxes xywh from predict scores

        Args:
            obj_reg_scores: size=(B, W, H, 4 * anchor_ratio * anchor_scale)

        """

        # generate anchors from ratio scale and stride
        # anchors: size=(B, W, H, anchor_ratio * anchor_scale, 4)
        B, W, H, N = obj_reg_scores.size()
        assert self.anchor_template_len == (N//4)
        anchors = obj_reg_scores.new_zeros(size=(B, W, H, self.anchor_template_len, 4))
        anchors[:, :, :] = self.anchor_template
        for wi in range(W):
            for hi in range(H):
                anchors[:, wi, hi, :, :] += torch.Tensor([wi*self.anchor_stride[0], h*self.anchor_stride[0], 0, 0])

        return anchor2bbox(anchors, obj_reg_scores)


