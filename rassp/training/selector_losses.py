import os
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value


def compute_selector_false_support_loss(
    selector_logits,
    batch,
    topk=64,
    eps=1e-8,
):
    if selector_logits is None or not torch.is_tensor(selector_logits):
        return torch.tensor(0.0)

    formulae_mask = batch.get("formulae_mask", None) if isinstance(batch, dict) else None
    cand_idx = batch.get("formulae_peaks_official_idx", None) if isinstance(batch, dict) else None
    cand_int = batch.get("formulae_peaks_official_intensity", None) if isinstance(batch, dict) else None

    true_idx_obj = batch.get("true_all_official_idx", None) if isinstance(batch, dict) else None
    if true_idx_obj is None and isinstance(batch, dict):
        true_idx_obj = batch.get("true_official_idx", None)

    if cand_idx is None or cand_int is None or true_idx_obj is None:
        return selector_logits.new_tensor(0.0)

    if not torch.is_tensor(cand_idx) or not torch.is_tensor(cand_int):
        return selector_logits.new_tensor(0.0)

    if cand_idx.dim() != 3 or cand_int.dim() != 3:
        return selector_logits.new_tensor(0.0)

    B, M = selector_logits.shape[:2]
    K = min(int(topk), int(M))
    if K <= 0:
        return selector_logits.new_tensor(0.0)

    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        formulae_mask = torch.ones_like(selector_logits, dtype=torch.float32)
    else:
        formulae_mask = formulae_mask.float().to(selector_logits.device)

    logits = selector_logits.masked_fill(formulae_mask <= 0, _neg_mask_fill_value(selector_logits))
    top_idx = torch.topk(logits, k=K, dim=1).indices

    losses = []

    for b in range(B):
        if torch.is_tensor(true_idx_obj):
            t = true_idx_obj[b].detach().to(selector_logits.device).long().reshape(-1)
        else:
            try:
                t = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=selector_logits.device).reshape(-1)
            except Exception:
                continue

        t = t[t >= 0]
        if t.numel() == 0:
            continue

        selected = top_idx[b]

        idx_b = cand_idx[b, selected].to(selector_logits.device).long()
        int_b = cand_int[b, selected].to(selector_logits.device).float()

        valid = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)
        total_mass = torch.where(valid, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit = valid & torch.isin(idx_b, t)
        true_mass = torch.where(hit, int_b, torch.zeros_like(int_b)).sum(dim=1)

        false_ratio = 1.0 - true_mass / total_mass.clamp_min(eps)
        false_ratio = torch.clamp(false_ratio, 0.0, 1.0)

        selected_logits = logits[b, selected]
        selected_prob = torch.softmax(selected_logits, dim=0)

        losses.append((selected_prob * false_ratio.detach()).sum())

    if len(losses) == 0:
        return selector_logits.new_tensor(0.0)

    return torch.stack(losses).mean()
def build_selector_utility_tensors(
    selector_logits,
    batch,
    eps=1e-8,
):
    """
    Build per-candidate utility target for selector training.

    Returns:
        utility: [B, M], normalized 0-1 utility for pairwise ranking
        utility_dist: [B, M], row-normalized soft target for KL/CE
        valid_mask: [B, M]
        stats: dict
    """
    if selector_logits is None or not torch.is_tensor(selector_logits):
        return None, None, None, {}

    formulae_mask = batch.get("formulae_mask", None)
    cand_idx = batch.get("formulae_peaks_official_idx", None)
    cand_int = batch.get("formulae_peaks_official_intensity", None)

    true_idx_obj = batch.get("true_all_official_idx", None)
    if true_idx_obj is None:
        true_idx_obj = batch.get("true_official_idx", None)

    if cand_idx is None or cand_int is None or true_idx_obj is None:
        return None, None, None, {}

    if not torch.is_tensor(cand_idx) or not torch.is_tensor(cand_int):
        return None, None, None, {}

    if cand_idx.dim() != 3 or cand_int.dim() != 3:
        return None, None, None, {}

    device = selector_logits.device
    B, M = selector_logits.shape[:2]

    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        formulae_mask = torch.ones((B, M), dtype=torch.float32, device=device)
    else:
        formulae_mask = formulae_mask.float().to(device)
        if formulae_mask.dim() > 2:
            formulae_mask = formulae_mask.reshape(formulae_mask.shape[0], -1)
        formulae_mask = formulae_mask[:B, :M]

    cand_idx = cand_idx[:B, :M].to(device).long()
    cand_int = cand_int[:B, :M].to(device).float()

    true_hit_mass = torch.zeros((B, M), dtype=torch.float32, device=device)
    false_mass = torch.zeros((B, M), dtype=torch.float32, device=device)
    total_mass = torch.zeros((B, M), dtype=torch.float32, device=device)

    for b in range(B):
        if torch.is_tensor(true_idx_obj):
            true_idx = true_idx_obj[b].detach().to(device).long().reshape(-1)
        else:
            try:
                true_idx = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=device).reshape(-1)
            except Exception:
                continue

        true_idx = true_idx[true_idx >= 0]
        if true_idx.numel() == 0:
            continue

        idx_b = cand_idx[b]
        int_b = cand_int[b]

        valid_peak = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)
        hit = valid_peak & torch.isin(idx_b, true_idx)

        total = torch.where(valid_peak, int_b, torch.zeros_like(int_b)).sum(dim=1)
        hit_m = torch.where(hit, int_b, torch.zeros_like(int_b)).sum(dim=1)
        false_m = torch.clamp(total - hit_m, min=0.0)

        total_mass[b] = total
        true_hit_mass[b] = hit_m
        false_mass[b] = false_m

    hit_share = true_hit_mass / total_mass.clamp_min(eps)
    false_share = false_mass / total_mass.clamp_min(eps)

    try:
        lambda_false = float(os.environ.get("SELECTOR_UTILITY_FALSE_LAMBDA", "0.8"))
    except Exception:
        lambda_false = 0.8

    try:
        temp = float(os.environ.get("SELECTOR_UTILITY_TEMP", "0.7"))
    except Exception:
        temp = 0.7

    valid_mask = (formulae_mask > 0.5) & (total_mass > eps)

    raw_utility = hit_share - lambda_false * false_share

    # 不要直接全部 clamp 到 0 后再做排序，否则 pairwise 的低质量区分会消失
    raw_utility = raw_utility.masked_fill(~valid_mask, -1e9)

    row_min = raw_utility.masked_fill(~valid_mask, float("inf")).amin(dim=1, keepdim=True)
    row_max = raw_utility.masked_fill(~valid_mask, float("-inf")).amax(dim=1, keepdim=True)

    row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))
    row_max = torch.where(torch.isfinite(row_max), row_max, torch.ones_like(row_max))

    utility = (raw_utility - row_min) / (row_max - row_min + eps)
    utility = utility.masked_fill(~valid_mask, 0.0)

    dist_logits = raw_utility / max(temp, 1e-6)
    dist_logits = dist_logits.masked_fill(~valid_mask, -1e9)
    utility_dist = torch.softmax(dist_logits, dim=1)
    utility_dist = utility_dist * valid_mask.float()
    utility_dist = utility_dist / utility_dist.sum(dim=1, keepdim=True).clamp_min(eps)

    stats = {
        "utility_mean": utility[valid_mask].mean().detach() if valid_mask.any() else selector_logits.new_tensor(0.0),
        "utility_hit_share": hit_share[valid_mask].mean().detach() if valid_mask.any() else selector_logits.new_tensor(0.0),
        "utility_false_share": false_share[valid_mask].mean().detach() if valid_mask.any() else selector_logits.new_tensor(0.0),
    }

    return utility.detach(), utility_dist.detach(), valid_mask.detach(), stats

def compute_selector_utility_target_loss(
    selector_logits,
    batch,
    eps=1e-8,
):
    utility, utility_dist, valid_mask, stats = build_selector_utility_tensors(
        selector_logits,
        batch,
        eps=eps,
    )

    if utility_dist is None or valid_mask is None:
        return selector_logits.new_tensor(0.0)

    logits = selector_logits.float()
    logits = logits.masked_fill(~valid_mask, _neg_mask_fill_value(logits))

    log_prob = F.log_softmax(logits, dim=1)

    valid_rows = utility_dist.sum(dim=1) > eps
    if not bool(valid_rows.any().item()):
        return selector_logits.new_tensor(0.0)

    loss = -(utility_dist.detach() * log_prob).sum(dim=1)
    return loss[valid_rows].mean()