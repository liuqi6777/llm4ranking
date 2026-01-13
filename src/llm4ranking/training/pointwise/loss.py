import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def rank_net(y_pred, y_true, weighted=False, use_rank=False, weight_by_diff=False, weight_by_diff_powed=False):
    if use_rank is None:
        y_true = torch.tensor([[1 / (np.argsort(y_true)[::-1][i] + 1) for i in range(y_pred.size(1))]] * y_pred.size(0)).cuda()

    document_pairs_candidates = list(itertools.product(range(y_true.shape[1]), repeat=2))

    pairs_true = y_true[:, document_pairs_candidates]
    selected_pred = y_pred[:, document_pairs_candidates]

    true_diffs = pairs_true[:, :, 0] - pairs_true[:, :, 1]
    pred_diffs = selected_pred[:, :, 0] - selected_pred[:, :, 1]

    the_mask = (true_diffs > 0) & (~torch.isinf(true_diffs))

    pred_diffs = pred_diffs[the_mask]

    weight = None
    if weighted:
        values, indices = torch.sort(y_true, descending=True)
        ranks = torch.zeros_like(indices)
        ranks.scatter_(1, indices, torch.arange(1, y_true.numel() + 1).to(y_true.device).view_as(indices))
        pairs_ranks = ranks[:, document_pairs_candidates] 
        rank_sum = pairs_ranks.sum(-1)
        weight = 1 / rank_sum[the_mask]    
    else:
        if weight_by_diff:
            abs_diff = torch.abs(true_diffs)
            weight = abs_diff[the_mask]
        elif weight_by_diff_powed:
            true_pow_diffs = torch.pow(pairs_true[:, :, 0], 2) - torch.pow(pairs_true[:, :, 1], 2)
            abs_diff = torch.abs(true_pow_diffs)
            weight = abs_diff[the_mask]

    true_diffs = (true_diffs > 0).type(torch.float32)
    true_diffs = true_diffs[the_mask]

    return nn.BCEWithLogitsLoss(weight=weight)(pred_diffs, true_diffs)