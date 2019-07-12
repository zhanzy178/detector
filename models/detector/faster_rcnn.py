from models.backbone import vgg16_bn
from models.rpn_head import RPNHead
from models.utils import nms
import torch.nn as nn
import torch
import torch.nn.functional as F
from models.roi_pool import RoIPool
from models.bbox_head import BBoxHead
from models.assigner import assign_bbox
from models.utils import proposal2bbox
from models.sampler import random_sample_pos_neg

class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.strides = [16]
        self.frozen_layer_num = 2

        self.backbone = vgg16_bn(pretrained=True, frozen_layer_num=self.frozen_layer_num)
        self.rpn_head = RPNHead(self.strides)

        self.roi_pool = RoIPool(out_size=7, spatial_scale=1.0/self.strides[0])
        self.bbox_head = BBoxHead(num_classes=num_classes)

        self.rpn_proposal_num = 2000

        self.pos_iou_thr = 0.5
        self.neg_iou_thr = 0.5

        self.sample_num = 512
        self.pos_sample_rate = 0.25

        self.rpn_nms_thr_iou = 0.7

        self.bbox_nms_thr_iou = 0.5
        self.bbox_nms_score_thr = 0.05



    def forward(self, img, img_meta, gt_bboxes=None, gt_labels=None):
        feat = self.backbone(img)

        # rpn predict proposals
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses = self.rpn_head(feat, img_meta, gt_bboxes)
        obj_cls_scores = nn.functional.softmax(obj_cls_scores, dim=2)
        proposals, obj_scores = nms(proposals, obj_cls_scores[..., 1], nms_iou_thr=self.rpn_nms_thr_iou)

        # extract 2000 proposals
        for b in range(len(proposals)):
            if proposals[b].size(0) > self.rpn_proposal_num:
                proposals[b] = proposals[b][:self.rpn_proposal_num]

        if self.training:  # train
            # assign and sample proposals
            cls_losses, reg_losses = 0, 0
            for b in range(len(proposals)):
                assign_result = assign_bbox(proposals[b], None, gt_bboxes[b], self.pos_iou_thr, self.neg_iou_thr)
                pos_ind, neg_ind = random_sample_pos_neg(assign_result.view(-1), self.sample_num, self.pos_sample_rate)
                sam_ind = torch.cat([pos_ind, neg_ind]).view(-1)

                rois = proposals[b].clone()
                rois[:, [0, 1]] -= rois[:, [2, 3]]
                rois[:, [2, 3]] += rois[:, [0, 1]]
                batch_ind = rois.new_zeros((rois.size(0), 1))
                rois = torch.cat([batch_ind, rois], dim=1)
                rois_feat = self.roi_pool(feat[b, None], rois[sam_ind])
                cls_scores, reg_scores = self.bbox_head(rois_feat)

                # compute cls loss
                cls_target = gt_labels.new_zeros((sam_ind.size(0), ))
                cls_target[:pos_ind.size(0)] = gt_labels[b, assign_result[pos_ind].view(-1)-1]
                cls_losser = nn.CrossEntropyLoss()
                cls_losses += cls_losser(cls_scores, cls_target)

                # copmute reg loss
                if pos_ind.size(0) > 0:
                    pos_gt = gt_bboxes[b, assign_result[pos_ind].view(-1)-1]
                    pos_proposal = proposals[b][pos_ind.view(-1)]
                    reg_target = pos_gt.clone()
                    reg_target[:, [0, 1]] = (reg_target[:, [0, 1]] - pos_proposal[:, [0, 1]]) / pos_proposal[:, [2, 3]]
                    reg_target[:, [2, 3]] = (reg_target[:, [2, 3]] / pos_proposal[:, [2, 3]]).log()
                    pos_gt_labels = gt_labels[b, assign_result[pos_ind].view(-1)-1]
                    pos_gt_labels = pos_gt_labels.view(-1, 1, 1).repeat(1, 1, 4)
                    reg_class_scores = reg_scores[:pos_ind.size(0)].view(pos_ind.size(0), -1, 4).gather(1, pos_gt_labels).view(-1, 4)
                    reg_losser = nn.SmoothL1Loss()
                    reg_losses += reg_losser(reg_class_scores, reg_target)

            return obj_cls_losses, obj_reg_losses, cls_losses, reg_losses

        else:  # test
            # convert to rois
            det_results = []
            for b in range(1, len(proposals)):
                rois = proposals[b].clone()
                rois[:, [0, 1]] -= rois[:, [2, 3]]
                rois[:, [2, 3]] += rois[:, [0, 1]]
                batch_ind = rois.new_zeros((rois.size(0), 1))
                rois = torch.cat([batch_ind, rois], dim=1)

                # inference
                rois_feat = self.roi_pool(feat[b], rois)
                cls_scores, reg_scores = self.bbox_head(rois_feat)

                # compute bbox xywh
                cls_bboxes = proposal2bbox(proposals[b], cls_scores)
                softmax_cls_scores = F.softmax(cls_scores, dim=1)

                # multiclass nms
                img_det_results = []
                for cls in range(len(cls_bboxes.size(1) // 4)):
                    ind = softmax_cls_scores[:, cls] > self.nms_score_thr
                    scores = softmax_cls_scores[ind, cls]
                    bboxes = cls_bboxes[ind, cls*4:(cls+1)*4]
                    det_bboxes, det_scores = nms(bboxes, scores)
                    img_det_results.append(torch.cat((det_bboxes, det_scores.view(-1, 1)), dim=1))

                det_results.append(img_det_results)

            return det_results
