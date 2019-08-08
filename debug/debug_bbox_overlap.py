import sys
sys.path.append('.')

from models.utils import bbox_overlap
import torch
k = torch.Tensor([[1, 1, 2, 2], [2, 1, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [1.5, 1.5, 3, 3]])
p = torch.Tensor([[1, 1, 2, 2], [2, 1, 2, 2], [1, 2, 2, 2], [5, 5, 1, 1]])
print(bbox_overlap(k, p))
