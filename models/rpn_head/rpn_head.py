import torch.nn as nn
import torch
from math import *
from models.utils.anchor import anchor2bbox

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
                self.anchor_template[i, :] = torch.Tensor([self.anchor_stride[0]/2, self.anchor_stride[0]/2, w, h])

        self.conv = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.obj_cls = nn.Conv2d(512, 2*self.anchor_template_len, kernel_size=1, stride=1)
        self.obj_reg = nn.Conv2d(512, 4*self.anchor_template_len, kernel_size=1, stride=1)

    def forward(self, feature_maps, gt_bboxes=None):
        # network forward
        f = self.conv(feature_maps)
        obj_cls_scores = self.obj_cls(f).transpose(2, 3).transpose(1, 3)
        obj_reg_scores = self.obj_reg(f).transpose(2, 3).transpose(1, 3)
        obj_cls_scores = obj_cls_scores.view(*obj_cls_scores.size()[0:3], -1, 2)
        obj_reg_scores = obj_reg_scores.view(*obj_reg_scores.size()[0:3], -1, 4) # size B, W, H, 9, 4

        # generate anchors
        anchors, anchors_ignore = self.generate_anchors(obj_reg_scores)

        # gernerate regresion results, bbox proposals
        batch_size = obj_reg_scores.size(0)
        anchors = anchors.contiguous().view(batch_size, -1, 4)
        anchors_ignore = anchors_ignore.contiguous().view(batch_size, -1)
        obj_cls_scores = obj_cls_scores.contiguous().view(batch_size, -1, 2)
        obj_reg_scores = obj_reg_scores.contiguous().view(batch_size, -1, 4)
        proposals = anchor2bbox(anchors, obj_reg_scores)

        # clip bbox outliers in test
        if not self.training:
            proposals_corner = proposals.clone()
            proposals_corner[:, :, [0, 1]] -= proposals[:, :, [2, 3]]
            proposals_corner[:, :, [2, 3]] += proposals[:, :, [0, 1]]
            proposals_corner[:, [0, 2]] = proposals_corner[:, [0, 2]].clamp(0, feature_maps.size(2)*self.anchor_stride[0])
            proposals_corner[:, [1, 3]] = proposals_corner[:, [1, 3]].clamp(0, feature_maps.size(3)*self.anchor_stride[0])

            proposals[:, :, [2, 3]] = (proposals_corner[:, :, [2, 3]] - proposals_corner[:, :, [0, 1]])/2
            proposals[:, :, [0, 1]] += proposals[:, :, [2, 3]]

        # compute loss in train
        obj_cls_losses, obj_reg_losses = None, None
        if self.training:
            # TODO: compute RPN loss
            pass
        import pdb; pdb.set_trace(); pass

        return proposals, obj_cls_scores, obj_cls_losses, obj_reg_losses


    def generate_anchors(self, obj_reg_scores):
        """generate anchor xywh from predict scores

        Args:
            obj_reg_scores: size=(B, W, H, 4 * anchor_ratio * anchor_scale)

        """

        # generate anchors from ratio scale and stride
        # anchors: size=(B, W, H, anchor_ratio * anchor_scale, 4)
        B, W, H, N, _ = obj_reg_scores.size()
        assert self.anchor_template_len == N
        anchors = obj_reg_scores.new_zeros(size=(B, W, H, self.anchor_template_len, 4))
        anchors[:, :, :] = self.anchor_template
        anchors_ignore = obj_reg_scores.new_zeros(size=(B, W, H, self.anchor_template_len), dtype=torch.long)
        for wi in range(W):
            for hi in range(H):
                anchors[:, wi, hi, :, :] += obj_reg_scores.new_tensor([wi*self.anchor_stride[0], hi*self.anchor_stride[0], 0, 0])

        if self.training: # ignore outliers
            anchors_corner = anchors.clone()
            imgs_size = anchors_corner.new_tensor([W*self.anchor_stride[0], H*self.anchor_stride[0]])
            anchors_corner[..., [0, 1]] -= anchors[..., [2, 3]]
            anchors_corner[..., [2, 3]] = imgs_size - anchors[..., [0, 1]] - anchors_corner[..., [2, 3]]
            anchors_ignore[(anchors_corner[..., 0]<0) | (anchors_corner[..., 1]<0) | (anchors_corner[..., 2]<0) | (anchors_corner[..., 3]<0)] = 1

        return anchors, anchors_ignore
