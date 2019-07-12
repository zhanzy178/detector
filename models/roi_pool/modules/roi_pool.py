from torch.nn.modules.module import Module
from ..functions.roi_pool import roi_pool


class RoIPool(Module):

    def __init__(self, out_size, spatial_scale):
        super(RoIPool, self).__init__()

        self.out_size = out_size
        self.spatial_scale = float(spatial_scale)

    def forward(self, features, rois):
        """

        :param features: (batch, c, h, w)
        :type features:
        :param rois: rois in format of (batch_ind, x1, y1, x2, y2)
        :type rois:
        :return:
        :rtype:
        """

        return roi_pool(features, rois, self.out_size, self.spatial_scale)
