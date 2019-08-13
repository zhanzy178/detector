import numpy as np
from cvtools.bbox import xyxy2xywh, xywh2xyxy

def proposal2bbox(proposals, reg_scores, img_size=None):
    """
    use for fast rcnn to compute bbox from scores and proposals
    :param rois:
    :type rois:
    :param reg_scores:  num_classes * 4
    :type reg_scores:
    :return: proposals_bbox: num_classes * xywh
    :rtype:
    """
    wh_ratio_clip = 16 / 1000  # from mmdet
    max_ratio = np.abs(np.log(wh_ratio_clip))
    reg_scores[:, 2::4] = reg_scores[:, 2::4].clamp(-max_ratio, max_ratio)
    reg_scores[:, 3::4] = reg_scores[:, 3::4].clamp(-max_ratio, max_ratio)

    assert proposals.size(0) == reg_scores.size(0)
    proposals_bbox = reg_scores.clone()
    proposals_bbox[:, 0::4] = proposals_bbox[:, 0::4]*proposals[:, 2, None] + proposals[:, 0, None]
    proposals_bbox[:, 1::4] = proposals_bbox[:, 1::4]*proposals[:, 3, None] + proposals[:, 1, None]
    proposals_bbox[:, 2::4] = proposals_bbox[:, 2::4].exp() * proposals[:, 2, None]
    proposals_bbox[:, 3::4] = proposals_bbox[:, 3::4].exp() * proposals[:, 3, None]

    if img_size is not None:
        proposals_corner = xywh2xyxy(proposals_bbox)
        proposals_corner[..., [0, 2]] = proposals_corner[..., [0, 2]].clamp(0, img_size[0])  # w
        proposals_corner[..., [1, 3]] = proposals_corner[..., [1, 3]].clamp(0, img_size[1])  # h
        proposals_bbox = xyxy2xywh(proposals_corner)

    return proposals_bbox
