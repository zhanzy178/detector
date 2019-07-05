from models.utils import bbox_overlap

def assign_bbox(proposals, proposals_ignore, gts, pos_iou_thr, neg_iou_thr):
    """assign proposal positive, negative and ignore according to gt.
    """

    iou = bbox_overlap(proposals, gts)
    max_value, max_ind = iou.max(dim=1)

    assign_result = max_ind.clone() + 1
    assign_result[(proposals_ignore==1).nonzero()] = 0
    for i in range(len(max_ind)):
        if assign_result[i] != 0:
            if iou[i][max_ind[i]] < neg_iou_thr:
                assign_result[i] = -1
            elif iou[i][max_ind[i]] < pos_iou_thr:
                assign_result[i] = 0

    # assign gt highest iou bbox
    gt_max_value, gt_max_ind = iou.max(dim=0)
    for i in range(len(gt_max_ind)):
        if gt_max_value[i] >= pos_iou_thr and proposals_ignore[gt_max_ind[i]] == 0:
            assign_result[gt_max_ind[i]] = i + 1

    return assign_result
