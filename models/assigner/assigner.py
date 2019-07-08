from models.utils import bbox_overlap
import torch

def assign_bbox(proposals, proposals_ignore, gts, pos_iou_thr, neg_iou_thr):
    """assign proposal positive, negative and ignore according to gt.
    """
    nignore_ind = (proposals_ignore==0).nonzero().view(-1)
    inside_proposals = proposals[nignore_ind]

    iou = bbox_overlap(inside_proposals, gts)
    if iou is None:
        assign_result = proposals.new_zeros(size=(proposals.size(0),), dtype=torch.long)-1
        assign_result[nignore_ind] = 0
        return assign_result

    anchor_max_iou, anchor_max_ind = iou.max(dim=1)

    assign_result = anchor_max_ind.clone() + 1
    for i in range(len(anchor_max_ind)):
        if assign_result[i] != 0:
            if iou[i][anchor_max_ind[i]] < neg_iou_thr:
                assign_result[i] = -1
            elif iou[i][anchor_max_ind[i]] < pos_iou_thr:
                assign_result[i] = 0


    # assign gt highest iou bbox
    gt_max_iou, gt_max_ind = iou.max(dim=0)
    for i in range(len(gt_max_ind)):
        # avoid all zeros iou assign, this may result in many positive sample iou less than 0.3
        # min_pos_iou = 0
        if gt_max_iou[i] > 0:
            max_iou_ind = iou[:, i] == gt_max_iou[i]

            # One gt may match multi nearest anchors, and we assign this anchor to its nearest gt,
            # so every gt at least has one anchor.
            # But in mmdet, these anchors is assign to this gt, with:
            # assign_result[max_iou_ind] = i+1
            assign_result[max_iou_ind] = anchor_max_ind[max_iou_ind]+1

    # map assign result to origin proposals size
    assign_result_origin = assign_result.new_zeros(size=(proposals.size(0), ))
    assign_result_origin[nignore_ind] = assign_result

    print(anchor_max_iou.max(), end=' ')
    return assign_result_origin
