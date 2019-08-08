import sys
sys.path.append('.')

from models.assigner import assign_bbox
import torch


proposals = torch.Tensor([[1, 1, 2, 2], [2, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [5, 5, 1, 1]])
proposals_ignore = torch.Tensor([0, 0, 0, 1, 0])
gts =  torch.Tensor([[1, 1, 2, 2], [2, 1, 2, 2], [1, 2, 2, 2], [1.5, 1.5, 3, 3]])
print(assign_bbox(proposals, proposals_ignore, gts, 0.4, 0.3))