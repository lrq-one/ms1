import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value
from rassp.model.selector_topk import coverage_aware_topk, group_unique_topk, plain_topk
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


build_group_unique_topk_mask_from_scores = _build_group_unique_topk_mask_from_scores
build_mask_from_topk_indices = _build_mask_from_topk_indices
build_topk_mask_from_scores = _build_topk_mask_from_scores
build_true_official_dense_for_batch = _build_true_official_dense_for_batch
mask_precision = _mask_precision
mask_ratio_in_topk = _mask_ratio_in_topk
mask_recall = _mask_recall


def select_model_topk_indices(
    selector_logits,
    batch,
    k,
    use_coverage=False,
    use_group_unique=False,
    candidate_mask=None,
):
    formulae_mask = batch.get("formulae_mask", None)
    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.float()
        if cm.dim() == 1:
            cm = cm.unsqueeze(0)
        elif cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)

        if torch.is_tensor(formulae_mask):
            fm = formulae_mask.float()
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            elif fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)
            use_b = min(int(fm.shape[0]), int(cm.shape[0]))
            use_m = min(int(fm.shape[1]), int(cm.shape[1]))
            fm = fm[:use_b, :use_m]
            cm = cm[:use_b, :use_m]
            fm_active = fm * (cm > 0.5).float()
            row_has_active = fm_active.sum(dim=-1, keepdim=True) > 0
            formulae_mask = torch.where(row_has_active, fm_active, fm)
        else:
            formulae_mask = cm

    group_id = batch.get("formulae_instance_group_id", None)

    if use_coverage:
        return coverage_aware_topk(
            selector_logits,
            batch.get("formulae_peaks_official_idx", None),
            batch.get("formulae_peaks_official_intensity", None),
            formulae_mask=formulae_mask,
            group_id=group_id,
            k=k,
            duplicate_penalty=float(os.environ.get("COVERAGE_TOPK_DUP_PENALTY", "0.35")),
            novelty_bonus=float(os.environ.get("COVERAGE_TOPK_NOVELTY_BONUS", "0.10")),
        )

    if use_group_unique:
        return group_unique_topk(
            selector_logits,
            group_id,
            k=k,
            mask=formulae_mask,
        )

    return plain_topk(
        selector_logits,
        k=k,
        mask=formulae_mask,
    )


def compute_selected_support_metrics(topk_idx, batch, eps=1e-8):
    cand_idx = batch.get("formulae_peaks_official_idx", None)
    cand_int = batch.get("formulae_peaks_official_intensity", None)
    if cand_idx is None or cand_int is None:
        cand_idx = batch.get("formulae_peaks_mass_idx", None)
        cand_int = batch.get("formulae_peaks_intensity", None)

    true_idx_obj = batch.get("true_all_official_idx", None)
    if true_idx_obj is None:
        true_idx_obj = batch.get("true_official_idx", None)

    if not (torch.is_tensor(topk_idx) and torch.is_tensor(cand_idx) and torch.is_tensor(cand_int)):
        return {}

    true_mass_list = []
    false_mass_list = []
    batch_n = int(min(topk_idx.shape[0], cand_idx.shape[0], cand_int.shape[0]))

    for b in range(batch_n):
        if torch.is_tensor(true_idx_obj):
            true_idx = true_idx_obj[b].detach().to(cand_idx.device).long().reshape(-1)
        else:
            try:
                true_idx = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=cand_idx.device).reshape(-1)
            except Exception:
                continue

        true_idx = true_idx[true_idx >= 0]
        if true_idx.numel() == 0:
            continue

        sel = topk_idx[b].long().clamp(0, max(0, int(cand_idx.shape[1]) - 1))
        idx_b = cand_idx[b, sel].long()
        int_b = cand_int[b, sel].float()

        valid = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)
        total = torch.where(valid, int_b, torch.zeros_like(int_b)).sum()
        if float(total.detach().item()) <= eps:
            continue

        hit = valid & torch.isin(idx_b, true_idx)
        true_mass = torch.where(hit, int_b, torch.zeros_like(int_b)).sum()
        ratio_true = true_mass / total.clamp_min(eps)

        true_mass_list.append(ratio_true)
        false_mass_list.append(1.0 - ratio_true)

    if len(true_mass_list) == 0:
        return {}

    return {
        "selected_true_hit_mass": float(torch.stack(true_mass_list).mean().detach().cpu().item()),
        "selected_false_mass": float(torch.stack(false_mass_list).mean().detach().cpu().item()),
    }
def compute_topk_pool_recall_metrics(
    topk_idx,
    batch,
    teacher_mask=None,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    eps=1e-8,
):
    """
    Pool-capacity diagnostics.

    Different from selected_true_hit_mass:
      selected_true_hit_mass = purity among selected candidates.
      true_peak_recall = how much true spectrum support is covered by selected candidates.
      teacher_recall = how many teacher-selected candidates are included in selected topK.

    These metrics answer:
      Does selector retrieve a useful candidate pool before reranking?
    """
    cand_idx = batch.get("formulae_peaks_official_idx_agg", None)
    cand_int = batch.get("formulae_peaks_official_intensity_agg", None)

    if cand_idx is None:
        cand_idx = batch.get("formulae_peaks_official_idx", None)
    if cand_int is None:
        cand_int = batch.get("formulae_peaks_official_intensity", None)

    true_idx_obj = batch.get("true_all_official_idx", None)
    if true_idx_obj is None:
        true_idx_obj = batch.get("true_official_idx", None)

    true_int_obj = batch.get("true_all_official_intensity", None)
    if true_int_obj is None:
        true_int_obj = batch.get("true_official_intensity", None)

    if not (
        torch.is_tensor(topk_idx)
        and torch.is_tensor(cand_idx)
        and torch.is_tensor(cand_int)
    ):
        return {}

    if cand_idx.dim() == 2:
        cand_idx = cand_idx.unsqueeze(0)
    if cand_int.dim() == 2:
        cand_int = cand_int.unsqueeze(0)

    device = cand_idx.device
    batch_n = int(min(topk_idx.shape[0], cand_idx.shape[0], cand_int.shape[0]))

    try:
        official_bin_n = int(np.floor(float(official_max_mz) / float(official_bin_width))) + 1
    except Exception:
        official_bin_n = 1
    official_bin_n = max(1, int(official_bin_n))

    true_peak_recall_list = []
    true_int_recall_list = []
    teacher_recall_list = []

    for b in range(batch_n):
        sel = topk_idx[b].long().clamp(0, max(0, int(cand_idx.shape[1]) - 1))

        idx_b = cand_idx[b, sel].long()
        int_b = cand_int[b, sel].float()

        valid = (
            (idx_b >= 0)
            & (idx_b < official_bin_n)
            & torch.isfinite(int_b)
            & (int_b > 0)
        )

        selected_bins = torch.zeros(
            (official_bin_n,),
            dtype=torch.bool,
            device=device,
        )

        if bool(valid.any().item()):
            selected_bins[idx_b[valid]] = True

        if torch.is_tensor(true_idx_obj):
            true_idx = true_idx_obj[b].detach().to(device).long().reshape(-1)
        else:
            try:
                true_idx = torch.as_tensor(true_idx_obj[b], dtype=torch.long, device=device).reshape(-1)
            except Exception:
                true_idx = None

        if true_idx is not None:
            true_idx = true_idx[(true_idx >= 0) & (true_idx < official_bin_n)]

            if true_idx.numel() > 0:
                true_hit = selected_bins[true_idx].float()
                true_peak_recall_list.append(true_hit.mean())

                if torch.is_tensor(true_int_obj):
                    true_int = true_int_obj[b].detach().to(device).float().reshape(-1)
                    use_n = min(int(true_idx.numel()), int(true_int.numel()))
                    if use_n > 0:
                        ti = true_int[:use_n].clamp_min(0.0)
                        th = true_hit[:use_n]
                        denom = ti.sum().clamp_min(float(eps))
                        true_int_recall_list.append((ti * th).sum() / denom)

        if torch.is_tensor(teacher_mask):
            tm = teacher_mask
            if tm.dim() == 1:
                tm = tm.unsqueeze(0)
            elif tm.dim() > 2:
                tm = tm.reshape(tm.shape[0], -1)

            if b < int(tm.shape[0]):
                tmb = tm[b].to(device).float()
                teacher_pos = tmb > 0.5
                denom = teacher_pos.float().sum().clamp_min(1.0)

                pred_mask = torch.zeros_like(tmb, dtype=torch.bool)
                sel2 = sel.clamp(0, max(0, int(tmb.shape[0]) - 1))
                pred_mask[sel2] = True

                hit = (pred_mask & teacher_pos).float().sum()
                teacher_recall_list.append(hit / denom)

    out = {}

    if len(true_peak_recall_list) > 0:
        out["true_peak_recall"] = float(torch.stack(true_peak_recall_list).mean().detach().cpu().item())

    if len(true_int_recall_list) > 0:
        out["true_int_recall"] = float(torch.stack(true_int_recall_list).mean().detach().cpu().item())

    if len(teacher_recall_list) > 0:
        out["teacher_recall"] = float(torch.stack(teacher_recall_list).mean().detach().cpu().item())

    return out

def compute_selector_eval_pack(
    selector_logits,
    batch,
    formulae_mask=None,
    teacher_mask=None,
    active_mask=None,
    topk_list=(32, 64, 128, 256),
    use_group_unique=False,
    use_coverage=False,
):
    """
    High-level selector eval wrapper.

    This keeps train_ms_subsetnet.py from depending on private mask helpers.
    """
    out = {}

    if selector_logits is None or not torch.is_tensor(selector_logits):
        return out

    if formulae_mask is None:
        formulae_mask = batch.get("formulae_mask", None)

    group_id = batch.get("formulae_instance_group_id", None)

    for k in topk_list:
        topk_idx = None
        if use_coverage:
            topk_idx = select_model_topk_indices(
                selector_logits=selector_logits,
                batch=batch,
                k=int(k),
                use_coverage=True,
                use_group_unique=False,
                candidate_mask=active_mask,
            )
            pred_mask = _build_mask_from_topk_indices(
                topk_idx,
                selector_logits,
                formulae_mask=formulae_mask,
                candidate_mask=active_mask,
            )
        elif use_group_unique:
            topk_idx = select_model_topk_indices(
                selector_logits=selector_logits,
                batch=batch,
                k=int(k),
                use_coverage=False,
                use_group_unique=True,
                candidate_mask=active_mask,
            )

            pred_mask = _build_mask_from_topk_indices(
                topk_idx,
                selector_logits,
                formulae_mask=formulae_mask,
                candidate_mask=active_mask,
            )
        else:
            topk_idx = select_model_topk_indices(
                selector_logits=selector_logits,
                batch=batch,
                k=int(k),
                use_coverage=False,
                use_group_unique=False,
                candidate_mask=active_mask,
            )

            pred_mask = _build_mask_from_topk_indices(
                topk_idx,
                selector_logits,
                formulae_mask=formulae_mask,
                candidate_mask=active_mask,
            )

        if pred_mask is None:
            continue

        if torch.is_tensor(teacher_mask):
            out[f"selector_recall@{k}"] = _mask_recall(pred_mask, teacher_mask)
            out[f"selector_precision@{k}"] = _mask_precision(pred_mask, teacher_mask)

        if topk_idx is None:
            if torch.is_tensor(formulae_mask):
                masked_logits = selector_logits.masked_fill(formulae_mask <= 0.5, -1e9)
            else:
                masked_logits = selector_logits
            topk_idx = torch.topk(
                masked_logits,
                k=min(int(k), int(selector_logits.shape[1])),
                dim=1,
            ).indices

        support_metrics = compute_selected_support_metrics(topk_idx, batch)
        for kk, vv in support_metrics.items():
            out[f"{kk}@{k}"] = vv

    return out
