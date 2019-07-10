import torch.nn as nn
import torch
from math import *
from models.utils.anchor import anchor2bbox
from models.assigner import assign_bbox
from models.sampler import random_sample_pos_neg
import torch.nn.functional as F

class RPNHead(nn.Module):
    """RPNHead Region Proposal Network to predict region proposal"""
    def __init__(self, strides):
        super(RPNHead, self).__init__()

        self.anchor_ratio = [0.5, 1.0, 2.0] # ratio is h / w
        self.anchor_scale = [128.0, 256.0, 512.0]
        self.anchor_stride = strides
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

        self.pos_iou_thr = 0.7
        self.neg_iou_thr = 0.3

        self.sample_num = 256
        self.pos_sample_rate = 0.5

    def forward(self, feature_maps, img_meta, gt_bboxes=None):
        # network forward
        f = F.relu(self.conv(feature_maps), inplace=True)
        obj_cls_scores = self.obj_cls(f).transpose(2, 3).transpose(1, 3)
        obj_reg_scores = self.obj_reg(f).transpose(2, 3).transpose(1, 3)
        obj_cls_scores = obj_cls_scores.view(*obj_cls_scores.size()[0:3], -1, 2)
        obj_reg_scores = obj_reg_scores.view(*obj_reg_scores.size()[0:3], -1, 4) # size B, H, W, 9, 4

        # generate anchors
        anchors, anchors_ignore = self.generate_anchors(obj_reg_scores, img_meta)
        # DEBUG: torch.save(dict(anchors=anchors, anchors_ignore=anchors_ignore), 'anchor.pth')

        # generate regresion results, bbox proposals
        batch_size = obj_reg_scores.size(0)
        anchors = anchors.contiguous().view(batch_size, -1, 4)
        anchors_ignore = anchors_ignore.contiguous().view(batch_size, -1)
        obj_cls_scores = obj_cls_scores.contiguous().view(batch_size, -1, 2)
        obj_reg_scores = obj_reg_scores.contiguous().view(batch_size, -1, 4)
        proposals = anchor2bbox(anchors, obj_reg_scores)

        # clip bbox outliers in test
        if not self.training:
            for b, im_size in enumerate(img_meta['img_size']):  # (w, h)
                proposals_corner = proposals[b, ...].clone()
                proposals_corner[..., [0, 1]] -= proposals[b, ..., [2, 3]]/2
                proposals_corner[..., [2, 3]] += proposals_corner[..., [0, 1]]
                proposals_corner[..., [0, 2]] = proposals_corner[..., [0, 2]].clamp(0, im_size[0])  # w
                proposals_corner[..., [1, 3]] = proposals_corner[..., [1, 3]].clamp(0, im_size[1])  # h
                proposals[b, ..., [2, 3]] = (proposals_corner[..., [2, 3]] - proposals_corner[..., [0, 1]])
                proposals[b, ..., [0, 1]] = proposals_corner[..., [0, 1]] + proposals[..., [2, 3]]/2

        # compute loss in train
        obj_cls_losses, obj_reg_losses = None, None
        if self.training:
            assign_results = None
            base = 0
            for b in range(anchors.size(0)):
                assign_result = assign_bbox(anchors[b], anchors_ignore[b], gt_bboxes[b], self.pos_iou_thr, self.neg_iou_thr)
                if assign_results is None:
                    assign_results = assign_result.view(1, -1)
                else:
                    assign_result[(assign_result > 0).nonzero()] += base
                    assign_results = torch.cat([assign_results, assign_result.view(1, -1)])
                base += gt_bboxes[b].size(0)

            # DEBUG: torch.save(dict(anchors=anchors, assign_results=assign_results), 'assign_results.pth')
            pos_ind, neg_ind = random_sample_pos_neg(assign_results.view(-1), self.sample_num, self.pos_sample_rate)
            # DEBUG: torch.save(dict(anchors=anchors, assign_results=assign_results, pos_ind=pos_ind, neg_ind=neg_ind), 'sample_results.pth')
            obj_cls_losses, obj_reg_losses = self.loss(obj_cls_scores.view(-1, 2), obj_reg_scores.view(-1, 4), anchors.view(-1, 4), gt_bboxes.view(-1, 4), assign_results.view(-1), pos_ind, neg_ind)

        return proposals, obj_cls_scores, obj_cls_losses, obj_reg_losses


    def generate_anchors(self, obj_reg_scores, img_meta):
        """generate anchor xywh from predict scores

        Args:
            obj_reg_scores: size=(B, H, W, 4 * anchor_ratio * anchor_scale)

        """

        # generate anchors from ratio scale and stride
        # anchors: size=(B, H, W, anchor_ratio * anchor_scale, 4)

        B, H, W, N, _ = obj_reg_scores.size()
        assert self.anchor_template_len == N
        anchors = obj_reg_scores.new_zeros(size=(B, H, W, self.anchor_template_len, 4))
        anchors[:, :, :] = self.anchor_template
        anchors_ignore = obj_reg_scores.new_zeros(size=(B, H, W, self.anchor_template_len), dtype=torch.long)
        for hi in range(H):
            for wi in range(W):
                anchors[:, hi, wi, :, :] += obj_reg_scores.new_tensor([wi*self.anchor_stride[0], hi*self.anchor_stride[0], 0, 0])  # x+w_delta, y+h_delta, w, h

        if self.training: # ignore outliers
            for b, im_size in enumerate(img_meta['img_size'].cuda()):
                anchors_corner = anchors[b, ...].clone()
                anchors_corner[..., [0, 1]] -= anchors[b, ..., [2, 3]]/2
                anchors_corner[..., [2, 3]] += anchors_corner[..., [0, 1]]
                anchors_ignore[b, (anchors_corner[..., 0]<0) | (anchors_corner[..., 1]<0)
                               | (anchors_corner[..., 2]>im_size[0]) | (anchors_corner[..., 3]>im_size[1])] = 1

        return anchors, anchors_ignore

    def loss(self, obj_cls_score, obj_reg_score, anchor, gt_bbox, assign_results, pos_ind, neg_ind):
        # object classify loss
        cls_num = pos_ind.size(0) + neg_ind.size(0)
        sam_ind = torch.cat([pos_ind, neg_ind]).view(-1)
        cls_target = obj_reg_score.new_zeros([cls_num], dtype=torch.long)
        cls_target[:pos_ind.size(0)] = 1
        cls_losser = nn.CrossEntropyLoss()
        cls_losses = cls_losser(obj_cls_score[sam_ind], cls_target)


        # proposal bounding box regression loss
        pos_ind = pos_ind.view(-1)
        pos_anchor = anchor[pos_ind]
        pos_gt = gt_bbox[assign_results[pos_ind]-1]
        pos_reg_score = obj_reg_score[pos_ind]
        gt_score = pos_gt.clone()
        gt_score[:, [0, 1]] = (gt_score[:, [0, 1]] - pos_anchor[:, [0, 1]]) / pos_anchor[:, [2, 3]]
        gt_score[:, [2, 3]] = (gt_score[:, [2, 3]] / pos_anchor[:, [2, 3]]).log()

        reg_losser = nn.SmoothL1Loss()
        reg_losses = reg_losser(pos_reg_score, gt_score)
        # when gt_score.size(0) = 0, reg_losses will be nan


        return cls_losses, reg_losses
