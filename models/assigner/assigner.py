from models.utils import bbox_overlap
import torch

# @profile
def assign_bbox(proposals, proposals_ignore, gts, pos_iou_thr, neg_iou_thr):
    """assign proposal positive, negative and ignore according to gt.
    """

    if proposals_ignore is not None:
        nignore_ind = (proposals_ignore==0).nonzero().view(-1)
        inside_proposals = proposals[nignore_ind]
    else:
        inside_proposals = proposals

    iou = bbox_overlap(inside_proposals, gts)
    if iou is None:
        assign_result = proposals.new_zeros((proposals.size(0), ), dtype=torch.long)
        if proposals_ignore is not None:
            assign_result[nignore_ind] = -1
        else:
            assign_result -= 1
        return assign_result

    anchor_max_iou, anchor_max_ind = iou.max(dim=1)

    assign_result = anchor_max_ind.clone() + 1
    assign_result[(anchor_max_iou < pos_iou_thr).nonzero().view(-1)] = 0
    assign_result[(anchor_max_iou < neg_iou_thr).nonzero().view(-1)] = -1

    # assign gt nearest bbox, also is the best match bbox
    gt_max_iou, gt_max_ind = iou.max(dim=0)
    for i in range(len(gt_max_ind)):
        # avoid all zeros iou assign, this may result in many positive sample iou less than 0.3
        # min_pos_iou = 0
        if gt_max_iou[i] > neg_iou_thr:
            max_iou_ind = iou[:, i] == gt_max_iou[i]

            # One gt may match multi nearest anchors, and we assign this anchor to its nearest gt,
            # so every gt at least has one anchor.
            # But in mmdet, these anchors is assign to this gt, with:
            # assign_result[max_iou_ind] = i+1
            assign_result[max_iou_ind] = anchor_max_ind[max_iou_ind]+1

    # map assign result to origin proposals size
    if proposals_ignore is not None:
        assign_result_origin = assign_result.new_zeros(size=(proposals.size(0), ))
        assign_result_origin[nignore_ind] = assign_result
    else:
        assign_result_origin = assign_result

    return assign_result_origin
