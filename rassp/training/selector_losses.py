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


def compute_selector_soft_false_support_loss(
    selector_logits,
    batch,
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

    device = selector_logits.device
    B, M = selector_logits.shape[:2]

    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        fm = torch.ones_like(selector_logits, dtype=torch.float32, device=device)
    else:
        fm = formulae_mask.float().to(device)
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:B, :M]

    cand_idx = cand_idx[:B, :M].to(device).long()
    cand_int = cand_int[:B, :M].to(device).float()

    false_ratio_all = torch.zeros((B, M), dtype=torch.float32, device=device)

    for b in range(B):
        if torch.is_tensor(true_idx_obj):
            t = true_idx_obj[b].detach().to(device).long().reshape(-1)
        else:
            try:
                t = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=device).reshape(-1)
            except Exception:
                continue

        t = t[t >= 0]
        if t.numel() == 0:
            continue

        idx_b = cand_idx[b]
        int_b = cand_int[b]

        valid_peak = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)
        total_mass = torch.where(valid_peak, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit = valid_peak & torch.isin(idx_b, t)
        true_mass = torch.where(hit, int_b, torch.zeros_like(int_b)).sum(dim=1)

        false_ratio = 1.0 - true_mass / total_mass.clamp_min(eps)
        false_ratio_all[b] = false_ratio.clamp(0.0, 1.0)

    logits = selector_logits.masked_fill(fm <= 0.5, _neg_mask_fill_value(selector_logits))
    prob = torch.softmax(logits, dim=1)

    return (prob * false_ratio_all.detach() * (fm > 0.5).float()).sum(dim=1).mean()


def compute_selector_utility_target_loss(
    selector_logits,
    batch,
    eps=1e-8,
):
    if selector_logits is None or not torch.is_tensor(selector_logits):
        return torch.tensor(0.0)

    formulae_mask = batch.get("formulae_mask", None)
    cand_idx = batch.get("formulae_peaks_official_idx", None)
    cand_int = batch.get("formulae_peaks_official_intensity", None)

    true_idx_obj = batch.get("true_all_official_idx", None)
    if true_idx_obj is None:
        true_idx_obj = batch.get("true_official_idx", None)

    if cand_idx is None or cand_int is None or true_idx_obj is None:
        return selector_logits.new_tensor(0.0)

    if not torch.is_tensor(cand_idx) or not torch.is_tensor(cand_int):
        return selector_logits.new_tensor(0.0)

    if cand_idx.dim() != 3 or cand_int.dim() != 3:
        return selector_logits.new_tensor(0.0)

    device = selector_logits.device
    B, _ = selector_logits.shape[:2]

    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        formulae_mask = torch.ones_like(selector_logits, dtype=torch.float32, device=device)
    else:
        formulae_mask = formulae_mask.float().to(device)

    logits = selector_logits.masked_fill(formulae_mask <= 0, _neg_mask_fill_value(selector_logits))

    try:
        lambda_false = float(os.environ.get("SELECTOR_UTILITY_FALSE_LAMBDA", "0.25"))
    except Exception:
        lambda_false = 0.25

    try:
        gamma = float(os.environ.get("SELECTOR_UTILITY_GAMMA", "1.0"))
    except Exception:
        gamma = 1.0

    losses = []

    for b in range(B):
        if torch.is_tensor(true_idx_obj):
            t = true_idx_obj[b].detach().to(device).long().reshape(-1)
        else:
            try:
                t = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=device).reshape(-1)
            except Exception:
                continue

        t = t[t >= 0]
        if t.numel() == 0:
            continue

        idx_b = cand_idx[b].to(device).long()
        int_b = cand_int[b].to(device).float()

        valid = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)

        total_mass = torch.where(valid, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit = valid & torch.isin(idx_b, t)
        hit_mass = torch.where(hit, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit_share = hit_mass / total_mass.clamp_min(eps)
        false_share = 1.0 - hit_share

        utility = hit_share - float(lambda_false) * false_share
        utility = torch.clamp(utility, min=0.0)

        if gamma != 1.0:
            utility = torch.pow(utility.clamp_min(0.0), gamma)

        utility = utility * torch.log1p(hit_mass)

        utility = utility * formulae_mask[b].float()

        s = utility.sum()
        if not torch.isfinite(s) or float(s.detach().cpu().item()) <= eps:
            continue

        target = utility / s.clamp_min(eps)
        log_prob = F.log_softmax(logits[b], dim=0)

        losses.append(-(target.detach() * log_prob).sum())

    if len(losses) == 0:
        return selector_logits.new_tensor(0.0)

    return torch.stack(losses).mean()


def build_selector_utility_tensors(
    selector_logits,
    batch,
    eps=1e-8,
):
    """
    Compatibility shim for older imports.

    Returns:
        utility: [B, M]
        target_dist: [B, M]
    """
    if selector_logits is None or not torch.is_tensor(selector_logits):
        return None, None

    formulae_mask = batch.get("formulae_mask", None)
    cand_idx = batch.get("formulae_peaks_official_idx", None)
    cand_int = batch.get("formulae_peaks_official_intensity", None)

    true_idx_obj = batch.get("true_all_official_idx", None)
    if true_idx_obj is None:
        true_idx_obj = batch.get("true_official_idx", None)

    if cand_idx is None or cand_int is None or true_idx_obj is None:
        return None, None

    if not torch.is_tensor(cand_idx) or not torch.is_tensor(cand_int):
        return None, None

    if cand_idx.dim() != 3 or cand_int.dim() != 3:
        return None, None

    device = selector_logits.device
    B, M = selector_logits.shape[:2]

    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        formulae_mask = torch.ones_like(selector_logits, dtype=torch.float32, device=device)
    else:
        formulae_mask = formulae_mask.float().to(device)

    try:
        lambda_false = float(os.environ.get("SELECTOR_UTILITY_FALSE_LAMBDA", "0.25"))
    except Exception:
        lambda_false = 0.25

    try:
        gamma = float(os.environ.get("SELECTOR_UTILITY_GAMMA", "1.0"))
    except Exception:
        gamma = 1.0

    utility = torch.zeros((B, M), dtype=torch.float32, device=device)

    for b in range(B):
        if torch.is_tensor(true_idx_obj):
            t = true_idx_obj[b].detach().to(device).long().reshape(-1)
        else:
            try:
                t = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=device).reshape(-1)
            except Exception:
                continue

        t = t[t >= 0]
        if t.numel() == 0:
            continue

        idx_b = cand_idx[b].to(device).long()
        int_b = cand_int[b].to(device).float()

        valid = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)
        total_mass = torch.where(valid, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit = valid & torch.isin(idx_b, t)
        hit_mass = torch.where(hit, int_b, torch.zeros_like(int_b)).sum(dim=1)

        hit_share = hit_mass / total_mass.clamp_min(eps)
        false_share = 1.0 - hit_share

        util_b = hit_share - float(lambda_false) * false_share
        util_b = torch.clamp(util_b, min=0.0)
        if gamma != 1.0:
            util_b = torch.pow(util_b.clamp_min(0.0), gamma)
        util_b = util_b * torch.log1p(hit_mass)
        utility[b] = util_b * formulae_mask[b].float()

    target_dist = utility.clamp_min(0.0)
    target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(eps)

    return utility, target_dist
