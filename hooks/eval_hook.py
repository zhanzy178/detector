from mmcv.runner import Hook

class EvalHook(Hook):
    def __init__(self, **kargs):

    def after_val_epoch(self, runner):
