from models.backbone import vgg16_bn, vgg16, load_vgg_fc
from models.rpn_head import RPNHead
from cvtools.bbox import nms_wrapper
import torch.nn as nn
import torch
import torch.nn.functional as F
# from models.roi_pool import RoIPool
from torchvision.ops import RoIAlign
from models.bbox_head import BBoxHead
from models.assigner import assign_bbox
from models.utils import proposal2bbox
from models.sampler import random_sample_pos_neg
from cvtools.bbox import xywh2xyxy
from models.loss import SmoothL1Loss

class FasterRCNN(nn.Module):
    """Faster RCNN detector
    """

    def __init__(self, num_classes):
        super(FasterRCNN, self).__init__()
        self.strides = [16]
        self.frozen_layer_num = 4

#         self.backbone = vgg16(pretrained=True, frozen_layer_num=self.frozen_layer_num)
        self.rpn_head = RPNHead(self.strides)

        self.roi_pool = RoIAlign(output_size=(7, 7), spatial_scale=1.0/self.strides[0], sampling_ratio=-1)
        self.bbox_head = BBoxHead(num_classes=num_classes, pretrained='vgg16')
        
        ### DEBUG ####
        import torchvision.models as models
        vgg = models.vgg16()
        model_path = '/home/zzy/Projects/faster-rcnn.pytorch/data/pretrained_model/vgg16_caffe.pth'
        print("Loading pretrained weights from %s" %(model_path))
        state_dict = torch.load(model_path)
        vgg.load_state_dict({k:v for k,v in state_dict.items() if k in vgg.state_dict()})
        self.backbone = nn.Sequential(*list(vgg.features._modules.values())[:-1])
        # Fix the layers before conv3:
        for layer in range(10):
          for p in self.backbone[layer].parameters(): p.requires_grad = False
        self.bbox_head.shared_layers = nn.Sequential(*list(vgg.classifier._modules.values())[:-1])
        ####
        
        #### RPN ####
        self.train_before_rpn_proposal_num = 12000
        self.train_after_rpn_proposal_num = 2000
        self.test_before_rpn_proposal_num = 6000  # 6000
        self.test_after_rpn_proposal_num = 300  # 300

        self.pos_iou_thr = 0.5
        self.neg_iou_thr = 0.5

        self.roi_num_per_img = 128  # 512
        self.pos_sample_rate = 0.25

        self.rpn_nms_thr_iou = 0.7
        self.rpn_min_size = 16
        

        #### RCNN ####
        self.bbox_nms_thr_iou = 0.5
        self.bbox_nms_score_thr = 0.05
        # target normalize
        self.target_mean = [0.0, 0.0, 0.0, 0.0]
        self.target_std = [0.1, 0.1, 0.2, 0.2]
        
        # load pretrained
#         load_vgg_fc(self, 'vgg16')
        self._init_weights()
    

    def forward(self, img, img_meta, gt_bboxes=None, gt_labels=None):
        feat = self.backbone(img)
        
        # rpn predict proposals
        proposals, obj_cls_scores, \
        obj_cls_losses, obj_reg_losses, proposals_ignore = self.rpn_head(feat, img_meta, gt_bboxes)
        obj_cls_scores = nn.functional.softmax(obj_cls_scores, dim=2)

        # filter out small bbox size
        for pi, proposal in enumerate(proposals):
            small_bbox_ind = ((proposal[..., [2]] < self.rpn_min_size) | (proposal[..., [3]] < self.rpn_min_size))
            proposals_ignore[pi][small_bbox_ind.view(-1)] = 1

        if self.training:  # train
            proposals, obj_scores = nms_wrapper(proposals,
                                                obj_cls_scores[..., 1],
                                                proposals_ignore,
                                                nms_iou_thr=self.rpn_nms_thr_iou,
                                                num_before=self.train_before_rpn_proposal_num,
                                                num_after=self.train_after_rpn_proposal_num)

            # assign and sample proposals
            cls_losses, reg_losses = 0, 0
            for b in range(len(proposals)):
                if gt_bboxes[b].size(0) > 0:
                    proposals[b] = torch.cat((proposals[b], gt_bboxes[b]))  # add gt to train
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

                # compute reg loss
                if pos_ind.size(0) > 0:
                    pos_gt = gt_bboxes[b, assign_result[pos_ind].view(-1)-1]
                    pos_proposal = proposals[b][pos_ind.view(-1)]
                    reg_target = pos_gt.clone()
                    reg_target[:, [0, 1]] = (reg_target[:, [0, 1]] - pos_proposal[:, [0, 1]]) / pos_proposal[:, [2, 3]]
                    reg_target[:, [2, 3]] = (reg_target[:, [2, 3]] / pos_proposal[:, [2, 3]]).log()
                    pos_gt_labels = gt_labels[b, assign_result[pos_ind].view(-1)-1]
                    pos_gt_labels = pos_gt_labels.view(-1, 1, 1).repeat(1, 1, 4)
                    reg_class_scores = reg_scores[:pos_ind.size(0)].view(pos_ind.size(0), -1, 4).gather(1, pos_gt_labels).view(-1, 4)
                    
                    target_mean, target_std = reg_target.new_tensor(self.target_mean), reg_target.new_tensor(self.target_std)
                    reg_target = (reg_target - target_mean) / target_std
                    reg_losses += SmoothL1Loss(reg_class_scores, reg_target, weights=1.0/sam_ind.size(0), sigma=1.0, reduction='sum')

            return obj_cls_losses, obj_reg_losses, cls_losses, reg_losses

        else:  # test
            proposals, obj_scores = nms_wrapper(proposals,
                                                obj_cls_scores[..., 1],
                                                proposals_ignore,
                                                nms_iou_thr=self.rpn_nms_thr_iou,
                                                num_before=self.test_before_rpn_proposal_num,
                                                num_after=self.test_after_rpn_proposal_num)

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
                target_mean, target_std = reg_target.new_tensor(self.target_mean), reg_target.new_tensor(self.target_std)
                reg_scores = reg_scores*target_std + target_mean
                cls_bboxes = proposal2bbox(proposals[b], reg_scores, img_meta['img_size'][b])
                softmax_cls_scores = F.softmax(cls_scores, dim=1)

                # multiclass nms
                img_det_bboxes = []
                img_det_labels = []
                val_ind = softmax_cls_scores > self.bbox_nms_score_thr
                valid_flag = val_ind.any(0)
                for cls in range(1, cls_bboxes.size(1) // 4):
                    if not valid_flag[cls]:
                        continue

                    ind = val_ind[:, cls]
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
                img_det_bboxes = img_det_bboxes[sort_ind].cpu().numpy()
                img_det_labels = img_det_labels[sort_ind].cpu().numpy()

                # bbox transform
                img_det_bboxes[:, :4] = img_det_bboxes[:, :4] / float(img_meta['scale_ratio'][b])

                det_bboxes_results.append(img_det_bboxes)
                det_labels_results.append(img_det_labels)

            return det_bboxes_results, det_labels_results

        
    def _init_weights(self):
        
        # random init weights
        self._normal_init_module_weights(self.rpn_head.conv)
        self._normal_init_module_weights(self.rpn_head.obj_cls)
        self._normal_init_module_weights(self.rpn_head.obj_reg)
        self._normal_init_module_weights(self.bbox_head.cls_fc)
        self._normal_init_module_weights(self.bbox_head.reg_fc, std=0.001)

    def _normal_init_module_weights(self, m,  mean=0.0, std=0.01):
        torch.nn.init.normal_(m.weight, mean, std)
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.constant_(m.bias, 0.0)