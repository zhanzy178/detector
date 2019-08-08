import models.detector as detector_module
from cvtools.runner import obj_from_dict


def build_detector(cfg):
    return obj_from_dict(cfg, detector_module)
