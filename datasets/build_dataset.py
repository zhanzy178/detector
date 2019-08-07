import datasets
from mmcv.runner import obj_from_dict


def build_dataset(cfg):
    dataset_args = cfg.args
    dataset_args['type'] = cfg.type
    return obj_from_dict(dataset_args, datasets)
