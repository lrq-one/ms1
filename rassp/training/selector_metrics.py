import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value
from rassp.training.formula_targets import (
    _build_true_official_dense_from_cached_sparse_batch,
    get_formulae_official_intensity_from_batch,
)
from rassp.training.spectrum_targets import build_true_official_dense_from_raw


def compute_selector_quality_metrics(selector_logits, selector_quality, formulae_mask, ks=(32, 64, 128, 256)):
    out = {}
    valid_quality = selector_quality * formulae_mask.float()

    for k in ks:
        kk = min(k, selector_logits.shape[1])
        idx = torch.topk(
            selector_logits.masked_fill(formulae_mask <= 0.5, -1e9),
            k=kk,
            dim=1,
        ).indices

        q_top = torch.gather(valid_quality, 1, idx)
        pos_top = (q_top > 0.3).float()

        out[f"selector_quality_mean_at_{k}"] = q_top.mean().detach()
        out[f"selector_precision_at_{k}"] = pos_top.mean().detach()

    return out


def _zero_precursor_bin_dense_batch(dense_spect, precursor_mz, bin_width):
    if (not torch.is_tensor(dense_spect)) or precursor_mz is None:
        return dense_spect

    out = dense_spect.clone()
    if not torch.is_tensor(precursor_mz):
        precursor_mz = torch.as_tensor(precursor_mz)

    pmz = precursor_mz.to(device=out.device, dtype=torch.float32).reshape(-1)
    if pmz.shape[0] < int(out.shape[0]):
        pad = torch.zeros((int(out.shape[0]) - int(pmz.shape[0]),), dtype=torch.float32, device=out.device)
        pmz = torch.cat([pmz, pad], dim=0)
    pmz = pmz[: int(out.shape[0])]

    bin_idx = torch.floor(pmz / float(bin_width) + 1e-8).long()
    valid = torch.isfinite(pmz) & (bin_idx >= 0) & (bin_idx < int(out.shape[1]))
    if bool(valid.any().item()):
        row_idx = torch.arange(int(out.shape[0]), device=out.device)
        out[row_idx[valid], bin_idx[valid]] = 0.0
    return out


def build_true_official_dense_from_batch(
    batch,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    exclude_precursor=True,
    device=None,
):
    """Build dense official-bin targets from either cached sparse targets or raw spectra."""
    if not isinstance(batch, dict):
        return None
    if device is None:
        device = torch.device("cpu")

    if torch.is_tensor(batch.get("vect_feat", None)):
        batch_n = int(batch["vect_feat"].shape[0])
    else:
        idx_src = batch.get("true_official_idx", None)
        if torch.is_tensor(idx_src):
            batch_n = int(idx_src.shape[0]) if idx_src.dim() > 0 else 1
        elif isinstance(idx_src, (list, tuple)):
            batch_n = len(idx_src)
        else:
            return None

    official_bin_width = float(max(1e-6, float(official_bin_width)))
    official_max_mz = float(max(official_bin_width, float(official_max_mz)))
    official_bin_n = int(math.floor(float(official_max_mz) / official_bin_width)) + 1

    dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )
    if not used_cache:
        dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get("spect_raw", None),
            precursor_mz=batch.get("precursor_mz", None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=exclude_precursor,
            batch_n=batch_n,
            device=device,
        )

    if exclude_precursor:
        dense = _zero_precursor_bin_dense_batch(
            dense,
            batch.get("precursor_mz", None),
            official_bin_width,
        )

    return dense


def _build_true_official_dense_for_batch(batch, official_metric_cfg, device):
    if not isinstance(batch, dict):
        return None
    if not torch.is_tensor(batch.get("vect_feat", None)):
        return None

    batch_n = int(batch["vect_feat"].shape[0])
    official_bin_width = float(official_metric_cfg.get("bin_width", 0.01))
    official_max_mz = float(official_metric_cfg.get("max_mz", 1005.0))
    official_bin_n = int(math.floor(float(official_max_mz) / float(official_bin_width))) + 1

    true_official_dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )

    if not used_cache:
        true_official_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get("spect_raw", None),
            precursor_mz=batch.get("precursor_mz", None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=official_metric_cfg.get("exclude_precursor", True),
            batch_n=batch_n,
            device=device,
        )

    if official_metric_cfg.get("exclude_precursor", True):
        true_official_dense = _zero_precursor_bin_dense_batch(
            true_official_dense,
            batch.get("precursor_mz", None),
            official_bin_width,
        )

    return true_official_dense


def compute_candidate_support_stats(
    batch,
    cand_probs_or_mask,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    eps=1e-8,
):
    """Project candidate probabilities or masks to official bins and summarize support quality."""
    off_idx = batch.get("formulae_peaks_official_idx_agg", None)
    off_int = batch.get("formulae_peaks_official_intensity_agg", None)
    if off_idx is None:
        off_idx = batch.get("formulae_peaks_official_idx", None)
    if off_int is None:
        off_int = get_formulae_official_intensity_from_batch(batch)

    if not (torch.is_tensor(off_idx) and torch.is_tensor(off_int) and torch.is_tensor(cand_probs_or_mask)):
        return {}

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    device = cand_probs_or_mask.device
    off_idx = off_idx.to(device=device).long()
    off_int = off_int.to(device=device).float()
    probs = cand_probs_or_mask.to(device=device).float()
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    elif probs.dim() > 2:
        probs = probs.reshape(probs.shape[0], -1)

    B = min(int(probs.shape[0]), int(off_idx.shape[0]), int(off_int.shape[0]))
    M = min(int(probs.shape[1]), int(off_idx.shape[1]), int(off_int.shape[1]))
    K = min(int(off_idx.shape[2]), int(off_int.shape[2]))
    if B <= 0 or M <= 0 or K <= 0:
        return {}

    probs = probs[:B, :M]
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    try:
        official_bin_n = int(np.floor(float(official_max_mz) / float(official_bin_width))) + 1
    except Exception:
        official_bin_n = 1
    official_bin_n = max(1, int(official_bin_n))

    valid = (off_idx >= 0) & (off_idx < official_bin_n) & torch.isfinite(off_int) & (off_int > 0)
    probs_eff = probs
    if probs_eff.dtype != off_int.dtype:
        probs_eff = probs_eff.to(dtype=off_int.dtype)
    contrib = probs_eff.unsqueeze(-1) * off_int * valid.float()

    pred_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    flat_idx = off_idx.clamp(0, max(0, official_bin_n - 1)).reshape(B, -1)
    flat_val = contrib.reshape(B, -1)
    flat_val = flat_val * valid.reshape(B, -1).float()
    pred_dense.scatter_add_(1, flat_idx, flat_val)

    true_dense = build_true_official_dense_from_batch(
        batch,
        official_bin_width=official_bin_width,
        official_max_mz=official_max_mz,
        exclude_precursor=True,
        device=device,
    )
    if true_dense is None:
        return {}
    true_dense = true_dense[:B].to(device=device)

    pred_support = pred_dense > float(eps)
    true_support = true_dense > float(eps)
    overlap = pred_support & true_support
    false = pred_support & (~true_support)

    pred_int_sum = pred_dense.sum(dim=-1).clamp_min(float(eps))
    pred_int_on_true = (pred_dense * true_support.float()).sum(dim=-1) / pred_int_sum
    false_support = (pred_dense * (~true_support).float()).sum(dim=-1) / pred_int_sum
    cos = F.cosine_similarity(pred_dense, true_dense, dim=-1, eps=float(eps))

    return {
        "pred_n": float(pred_support.float().sum(dim=-1).mean().detach().cpu().item()),
        "false_pred_n": float(false.float().sum(dim=-1).mean().detach().cpu().item()),
        "overlap_n": float(overlap.float().sum(dim=-1).mean().detach().cpu().item()),
        "pred_int_on_true": float(pred_int_on_true.mean().detach().cpu().item()),
        "false_support": float(false_support.mean().detach().cpu().item()),
        "official_cos": float(cos.mean().detach().cpu().item()),
    }


def _build_topk_mask_from_scores(scores_tensor, formulae_mask=None, topk=64, candidate_mask=None):
    if not torch.is_tensor(scores_tensor):
        return None

    sc = scores_tensor.float()
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)
    elif sc.dim() > 2:
        sc = sc.reshape(sc.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(sc.shape[0]), int(fm.shape[0]))
        use_m = min(int(sc.shape[1]), int(fm.shape[1]))
        sc = sc[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        fm = (fm > 0.5).float()
    else:
        fm = torch.ones_like(sc)

    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.float()
        if cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)

        use_b2 = min(int(sc.shape[0]), int(cm.shape[0]))
        use_m2 = min(int(sc.shape[1]), int(cm.shape[1]))
        sc = sc[:use_b2, :use_m2]
        fm = fm[:use_b2, :use_m2]
        cm = cm[:use_b2, :use_m2]

        fm_full = fm
        fm_active = fm * (cm > 0.5).float()

        row_has_active = fm_active.sum(dim=-1, keepdim=True) > 0
        fm = torch.where(row_has_active, fm_active, fm_full)

    if int(sc.shape[1]) <= 0:
        return None

    kk = max(1, min(int(topk), int(sc.shape[1])))

    masked_scores = sc.masked_fill(fm <= 0, _neg_mask_fill_value(sc))
    top_idx = torch.topk(masked_scores, k=kk, dim=-1).indices

    keep = torch.zeros_like(sc, dtype=torch.float32)
    keep.scatter_(1, top_idx, 1.0)
    keep = keep * fm
    return keep


def _build_mask_from_topk_indices(topk_idx, scores_tensor, formulae_mask=None, candidate_mask=None):
    if not torch.is_tensor(topk_idx) or not torch.is_tensor(scores_tensor):
        return None

    sc = scores_tensor.float()
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)
    elif sc.dim() > 2:
        sc = sc.reshape(sc.shape[0], -1)

    B = int(sc.shape[0])
    M = int(sc.shape[1])

    idx = topk_idx
    if idx.dim() == 1:
        idx = idx.unsqueeze(0)
    elif idx.dim() > 2:
        idx = idx.reshape(idx.shape[0], -1)

    idx = idx.to(device=sc.device, dtype=torch.long)
    use_b = min(B, int(idx.shape[0]))

    keep = torch.zeros((B, M), dtype=torch.float32, device=sc.device)
    if use_b > 0 and M > 0:
        idx_clamped = idx[:use_b].clamp(0, max(0, M - 1))
        keep[:use_b].scatter_(1, idx_clamped, 1.0)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:use_b, :M]
    else:
        fm = torch.ones((use_b, M), dtype=torch.float32, device=sc.device)

    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.float()
        if cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)
        cm = cm[:use_b, :M]

        fm_full = fm
        fm_active = fm * (cm > 0.5).float()

        row_has_active = fm_active.sum(dim=-1, keepdim=True) > 0
        fm = torch.where(row_has_active, fm_active, fm_full)

    keep[:use_b] = keep[:use_b] * fm
    return keep


def _build_group_unique_topk_mask_from_scores(
    scores,
    formulae_mask=None,
    group_id=None,
    topk=256,
    candidate_mask=None,
):
    """
    Build topK mask with at most one candidate per formula group.
    """
    if not torch.is_tensor(scores):
        return None

    s = scores.detach()
    if s.dim() == 1:
        s = s.unsqueeze(0)
    elif s.dim() > 2:
        s = s.reshape(s.shape[0], -1)

    B, M = int(s.shape[0]), int(s.shape[1])
    device = s.device

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.to(device=device)
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        valid = torch.zeros((B, M), dtype=torch.bool, device=device)
        use_b = min(B, int(fm.shape[0]))
        use_m = min(M, int(fm.shape[1]))
        valid[:use_b, :use_m] = fm[:use_b, :use_m] > 0.5
    else:
        valid = torch.ones((B, M), dtype=torch.bool, device=device)

    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.to(device=device)
        if cm.dim() == 1:
            cm = cm.unsqueeze(0)
        elif cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)
        use_b = min(B, int(cm.shape[0]))
        use_m = min(M, int(cm.shape[1]))
        cm_full = torch.zeros((B, M), dtype=torch.bool, device=device)
        cm_full[:use_b, :use_m] = cm[:use_b, :use_m] > 0.5
        valid = valid & cm_full

    if torch.is_tensor(group_id):
        gid = group_id.to(device=device, dtype=torch.long)
        if gid.dim() == 1:
            gid = gid.unsqueeze(0)
        elif gid.dim() > 2:
            gid = gid.reshape(gid.shape[0], -1)
        use_b = min(B, int(gid.shape[0]))
        use_m = min(M, int(gid.shape[1]))
        gid_full = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1).clone()
        gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
        gid = gid_full
    else:
        gid = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1)

    out = torch.zeros((B, M), dtype=torch.float32, device=device)
    kk = max(1, min(int(topk), M))

    for b in range(B):
        valid_idx = torch.nonzero(valid[b], as_tuple=False).reshape(-1)
        if valid_idx.numel() <= 0:
            continue

        valid_scores = s[b, valid_idx]
        order = torch.argsort(valid_scores, descending=True)

        seen = set()
        chosen = []

        for oi in order.detach().cpu().tolist():
            idx = int(valid_idx[oi].detach().cpu().item())
            g = int(gid[b, idx].detach().cpu().item())

            if g in seen:
                continue

            seen.add(g)
            chosen.append(idx)

            if len(chosen) >= kk:
                break

        if len(chosen) > 0:
            chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=device)
            out[b, chosen_t] = 1.0

    return out


def _mask_recall(pred_mask, true_mask):
    if (not torch.is_tensor(pred_mask)) or (not torch.is_tensor(true_mask)):
        return float("nan")

    pm = pred_mask.float()
    tm = true_mask.float()

    if pm.dim() == 1:
        pm = pm.unsqueeze(0)
    elif pm.dim() > 2:
        pm = pm.reshape(pm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(pm.shape[0]), int(tm.shape[0]))
    use_m = min(int(pm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float("nan")

    pm = pm[:use_b, :use_m] > 0.5
    tm = tm[:use_b, :use_m] > 0.5

    denom = tm.sum(dim=-1).float().clamp_min(1.0)
    hit = (pm & tm).sum(dim=-1).float()
    recall = hit / denom
    return float(recall.mean().detach().cpu().item())


def _mask_precision(pred_mask, true_mask):
    if (not torch.is_tensor(pred_mask)) or (not torch.is_tensor(true_mask)):
        return float("nan")

    pm = pred_mask.float()
    tm = true_mask.float()

    if pm.dim() == 1:
        pm = pm.unsqueeze(0)
    elif pm.dim() > 2:
        pm = pm.reshape(pm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(pm.shape[0]), int(tm.shape[0]))
    use_m = min(int(pm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float("nan")

    pm = pm[:use_b, :use_m] > 0.5
    tm = tm[:use_b, :use_m] > 0.5

    denom = pm.sum(dim=-1).float().clamp_min(1.0)
    hit = (pm & tm).sum(dim=-1).float()
    precision = hit / denom
    return float(precision.mean().detach().cpu().item())


def _mask_ratio_in_topk(source_mask, topk_mask):
    """Ratio of source_mask within selected topK candidates."""
    if (not torch.is_tensor(source_mask)) or (not torch.is_tensor(topk_mask)):
        return float("nan")

    sm = source_mask.float()
    tm = topk_mask.float()

    if sm.dim() == 1:
        sm = sm.unsqueeze(0)
    elif sm.dim() > 2:
        sm = sm.reshape(sm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(sm.shape[0]), int(tm.shape[0]))
    use_m = min(int(sm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float("nan")

    sm = sm[:use_b, :use_m]
    tm = tm[:use_b, :use_m]

    denom = tm.sum(dim=-1).clamp_min(1.0)
    ratio = ((sm > 0.5).float() * (tm > 0.5).float()).sum(dim=-1) / denom

    return float(ratio.mean().detach().cpu().item())
