import datasets
from cvtools.runner import obj_from_dict


def build_dataset(cfg):
    return obj_from_dict(cfg, datasets)
