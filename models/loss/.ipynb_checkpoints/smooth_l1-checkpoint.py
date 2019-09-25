import torch


def SmoothL1Loss(pred_score, gt_score, weights, sigma=1.0, reduction='mean'):
    assert reduction in ['mean', 'sum']
    assert len(pred_score.size()) == 2 and len(gt_score.size()) == 2
    diff = pred_score - gt_score
    abs_diff = torch.abs(diff)
    sigma_2 = sigma**2

    smoothl1_sign = (abs_diff < (1.0/sigma_2)).detach().float()
    loss = torch.pow(diff, 2)*(sigma_2/2.0)*smoothl1_sign + (abs_diff - (0.5 / sigma_2)) * (1-smoothl1_sign)
    loss = loss.sum(1)

    if reduction == 'mean':
        loss = loss.mean()
    else:
        loss = loss.sum()

    loss = loss*weights

    return loss

