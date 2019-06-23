from models.rpn_head import RPNHead
import torch


if __name__ == '__main__':
    rpn = RPNHead().cuda()

    # B, C, W, H
    feat_maps = torch.zeros(size=(4, 512, 40, 60), dtype=torch.float32).cuda()
    output = rpn(feat_maps, torch.Tensor([150, 640]))