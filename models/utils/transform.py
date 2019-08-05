def win2feat():
    """This function maps windows on image to feature map."""
    pass

def feat2win():
    """This function maps feature map to windows on image."""
    pass

def xywh2xyxy(bbox_t):
    """This function maps feature map bbox xywh format to x1y1x2y2 format."""
    bbox = bbox_t.clone()
    bbox[..., [0, 1]] -= bbox[..., [2, 3]] / 2
    bbox[..., [2, 3]] += bbox[..., [0, 1]]

    return bbox


def xyxy2xywh(bbox_t):
    """This function maps feature map bbox x1y1x2y2 format to xywh format."""
    bbox = bbox_t.clone()
    bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[:, [0, 1]]
    bbox[..., [0, 1]] += bbox[..., [2, 3]] / 2

    return bbox
