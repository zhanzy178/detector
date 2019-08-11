import numpy as np

def anchor2bbox(anchors, obj_reg_scores):
    wh_ratio_clip = 16 / 1000  # from mmdet
    max_ratio = np.abs(np.log(wh_ratio_clip))
    obj_reg_scores[..., [2, 3]] = obj_reg_scores[..., [2, 3]].clamp(-max_ratio, max_ratio)

    assert anchors.size() == obj_reg_scores.size()
    proposals_bbox = anchors.clone()
    proposals_bbox[..., [0, 1]] += obj_reg_scores[..., [0, 1]]*proposals_bbox[..., [2, 3]]
    proposals_bbox[..., [2, 3]] *= obj_reg_scores[..., [2, 3]].exp()

    return proposals_bbox

