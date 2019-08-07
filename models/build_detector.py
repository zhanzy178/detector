import models.detector as detector_module
from mmcv.runner import obj_from_dict


def build_detector(cfg):
    detector_args = cfg.args
    detector_args['type'] = cfg.type
    return obj_from_dict(detector_args, detector_module)
