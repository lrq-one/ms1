import os

import torch


def build_setcover_rerank_target_from_quality(
    selector_quality,
    pool_idx,
    formulae_mask,
    eps=1e-8,
):
    """
    Simplified setcover target: normalize candidate-local quality within pool.
    """
    B, K = pool_idx.shape
    pool_quality = torch.gather(selector_quality, 1, pool_idx)
    pool_mask = torch.gather(formulae_mask.float(), 1, pool_idx)

    target = pool_quality.clamp_min(0.0) * pool_mask
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)

    pos = torch.zeros_like(target)
    try:
        rerank_pos_k = int(os.environ.get("QUALITY_SETCOVER_TOPK", "24"))
    except Exception:
        rerank_pos_k = 24
    kk = max(1, min(rerank_pos_k, K))

    top_idx = torch.topk(target, k=kk, dim=1).indices
    pos.scatter_(1, top_idx, 1.0)
    pos = pos * pool_mask

    return target, pos, pool_mask


