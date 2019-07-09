from .geometry import bbox_overlap
import torch

def nms(bboxes, scores, nms_iou_thr = 0.7):
    """
    Input bboxes array: batch_size x num x 4
    Output bboxes list: every item in list is nms result
    """

    _, sorted_ind = scores[..., 0].sort()
    bboxes = bboxes.gather(1, sorted_ind[..., None].repeat((1, 1, 4)))
    scores = scores[..., 1].gather(1, sorted_ind)

    nms_bboxes = []
    nms_scores = []
    for b in range(len(bboxes)):
        bbox_list = bboxes[b]
        score_list = scores[b]
        suppression = torch.zeros(size=(bbox_list.size(0), ), dtype=torch.long)

        iou = bbox_overlap(bbox_list, bbox_list)

        for i, iou_row in enumerate(iou):
            if suppression[i] == 1: continue
            suppression[(iou_row > nms_iou_thr).nonzero()] = 1
            suppression[i] = 0

        left_ind = (suppression==0).nonzero().view(-1)
        nms_bboxes.append(bbox_list[left_ind])
        nms_scores.append(score_list[left_ind])

    return nms_bboxes, nms_scores
