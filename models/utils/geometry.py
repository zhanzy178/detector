import torch
from .transform import xywh2xyxy

def bbox_overlap(bbox1, bbox2):
    """Computing iou between two bboxes.
        The input bbox in format xywh.
    """

    if bbox1.size(0) == 0 or bbox2.size(0) == 0:
        return None

    # convert format to ltrb
    bbox_corner1 = xywh2xyxy(bbox1)
    bbox_corner2 = xywh2xyxy(bbox2)

    area1 = bbox1[:, 2]*bbox1[:, 3]
    area2 = bbox2[:, 2]*bbox2[:, 3]
    w = torch.min(bbox_corner1[:, None, 2], bbox_corner2[:, 2])-torch.max(bbox_corner1[:, None, 0], bbox_corner2[:, 0])
    h = torch.min(bbox_corner1[:, None, 3], bbox_corner2[:, 3])-torch.max(bbox_corner1[:, None, 1], bbox_corner2[:, 1])
    area_overlap = w.clamp(0)*h.clamp(0)

    iou = area_overlap / (area1[:, None] + area2 - area_overlap)

    return iou
