import torch
import numpy as np

def random_sample_pos_neg(assign_result, num, pos_rate):
    pos_num = int(pos_rate*num)
    pos_ind = ((assign_result!=0) & (assign_result!=-1)).nonzero()
    if len(pos_ind) > pos_num:
        randind = np.array(pos_ind.cpu())
        np.random.shuffle(randind)
        pos_ind = pos_ind.new_tensor(randind[:pos_num])

    neg_num = num - pos_ind.size(0)
    neg_ind = (assign_result==-1).nonzero()
    if len(neg_ind) > neg_num:
        randind = np.array(neg_ind.cpu())
        np.random.shuffle(randind)
        neg_ind = neg_ind.new_tensor(randind[:neg_num])

    return pos_ind, neg_ind

