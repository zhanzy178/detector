import numpy as np
from cvtools.bbox import xywh2xyxy, xyxy2xywh

def anchor2bbox(anchors, obj_reg_scores, img_sizes=None):
    wh_ratio_clip = 16 / 1000  # from mmdet
    max_ratio = np.abs(np.log(wh_ratio_clip))
    obj_reg_scores[..., [2, 3]] = obj_reg_scores[..., [2, 3]].clamp(-max_ratio, max_ratio)

    assert anchors.size() == obj_reg_scores.size()
    proposals_bbox = anchors.clone()
    proposals_bbox[..., [0, 1]] += obj_reg_scores[..., [0, 1]]*proposals_bbox[..., [2, 3]]
    proposals_bbox[..., [2, 3]] *= obj_reg_scores[..., [2, 3]].exp()

    # clip bbox outliers
    if img_sizes is not None:
        for b, im_size in enumerate(img_sizes):  # (w, h)
            proposals_corner = xywh2xyxy(proposals_bbox[b, ...])
            proposals_corner[..., [0, 2]] = proposals_corner[..., [0, 2]].clamp(0, im_size[0])  # w
            proposals_corner[..., [1, 3]] = proposals_corner[..., [1, 3]].clamp(0, im_size[1])  # h
            proposals_bbox[b, ...] = xyxy2xywh(proposals_corner)

    return proposals_bbox

