from models.backbone import vgg16_bn
from models.rpn_head import RPNHead
from cvtools.bbox import nms_wrapper
import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.roi_pool import RoIPool
from torchvision.ops import RoIPool
from models.bbox_head import BBoxHead
from models.assigner import assign_bbox
from models.utils import proposal2bbox
from models.sampler import random_sample_pos_neg
from cvtools.bbox import xywh2xyxy


class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.strides = [16]
        self.frozen_layer_num = 4

        self.backbone = vgg16_bn(pretrained=True, frozen_layer_num=self.frozen_layer_num)
        self.rpn_head = RPNHead(self.strides)

        self.roi_pool = RoIPool(output_size=(7, 7), spatial_scale=1.0/self.strides[0])
        self.bbox_head = BBoxHead(num_classes=num_classes)

        self.rpn_proposal_num = 2000

        self.pos_iou_thr = 0.5
        self.neg_iou_thr = 0.5

        self.roi_num_per_img = 128  # 512
        self.pos_sample_rate = 0.25

        self.rpn_nms_thr_iou = 0.7

        self.bbox_nms_thr_iou = 0.5
        self.bbox_nms_score_thr = 0.05
        self.bbox_nms_max_num = 300


    def forward(self, img, img_meta, gt_bboxes=None, gt_labels=None):
        feat = self.backbone(img)

        # rpn predict proposals
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses, proposals_ignore = self.rpn_head(feat, img_meta, gt_bboxes)
        obj_cls_scores = nn.functional.softmax(obj_cls_scores, dim=2)

        proposals, obj_scores = nms_wrapper(proposals, obj_cls_scores[..., 1], proposals_ignore, nms_iou_thr=self.rpn_nms_thr_iou)

        # extract 2000 proposals
        for b in range(len(proposals)):
            if proposals[b].size(0) > self.rpn_proposal_num:
                proposals[b] = proposals[b][:self.rpn_proposal_num]

        if self.training:  # train
            # assign and sample proposals
            cls_losses, reg_losses = 0, 0
            for b in range(len(proposals)):
                assign_result = assign_bbox(proposals[b], None, gt_bboxes[b], self.pos_iou_thr, self.neg_iou_thr)
                if assign_result is None: continue

                pos_ind, neg_ind = random_sample_pos_neg(assign_result.view(-1), self.roi_num_per_img, self.pos_sample_rate)
                sam_ind = torch.cat([pos_ind, neg_ind]).view(-1)

                rois = proposals[b].clone()
                rois = xywh2xyxy(rois)
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
            det_bboxes_results = []
            det_labels_results = []
            for b in range(len(proposals)):
                rois = proposals[b].clone()
                rois = xywh2xyxy(rois)
                batch_ind = rois.new_zeros((rois.size(0), 1))
                rois = torch.cat([batch_ind, rois], dim=1)

                # inference
                rois_feat = self.roi_pool(feat[b, None], rois)
                cls_scores, reg_scores = self.bbox_head(rois_feat)

                # compute bbox xywh
                cls_bboxes = proposal2bbox(proposals[b], reg_scores)
                softmax_cls_scores = F.softmax(cls_scores, dim=1)

                # multiclass nms
                img_det_bboxes = []
                img_det_labels = []
                val_ind = softmax_cls_scores > self.bbox_nms_score_thr
                for cls in range(1, cls_bboxes.size(1) // 4):
                    ind = val_ind[:, cls]
                    if not any(ind):
                        continue

                    scores = softmax_cls_scores[ind, cls]
                    bboxes = cls_bboxes[ind, cls*4:(cls+1)*4]
                    det_bboxes, det_scores = nms_wrapper(bboxes[None, ...], scores[None, ...], None, self.bbox_nms_thr_iou)
                    det_bboxes, det_scores = det_bboxes[0], det_scores[0]
                    img_det_bboxes.append(torch.cat((det_bboxes, det_scores.view(-1, 1)), dim=1))
                    img_det_labels.append(torch.zeros(size=(det_bboxes.size(0), ), dtype=torch.long) + cls)

                if len(img_det_bboxes) == 0:
                    det_bboxes_results.append(torch.Tensor(img_det_bboxes))
                    det_labels_results.append(torch.Tensor(img_det_labels))
                    continue

                img_det_bboxes = torch.cat(img_det_bboxes)
                img_det_labels = torch.cat(img_det_labels)

                _, sort_ind = (-img_det_bboxes[:, -1]).sort()
                if img_det_bboxes.size(0) > self.bbox_nms_max_num:
                    sort_ind = sort_ind[:self.bbox_nms_max_num]
                img_det_bboxes = img_det_bboxes[sort_ind].cpu().numpy()
                img_det_labels = img_det_labels[sort_ind].cpu().numpy()

                # bbox transform
                img_det_bboxes = img_det_bboxes / float(img_meta['scale_ratio'][b])

                det_bboxes_results.append(img_det_bboxes)
                det_labels_results.append(img_det_labels)

            return det_bboxes_results, det_labels_results
