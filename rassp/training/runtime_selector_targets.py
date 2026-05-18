import os
import math

import numpy as np
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value
from rassp.training.formula_targets import (
    _build_cached_true_top20_tensors,
    _build_true_official_dense_from_cached_sparse_batch,
    get_formulae_official_intensity_from_batch,
)


def normalize_per_sample(x, mask=None, eps=1e-8):
    if mask is None:
        mask = torch.ones_like(x, dtype=torch.bool)
    else:
        mask = mask > 0.5

    x_masked = x.masked_fill(~mask, 0.0)
    x_min = x.masked_fill(~mask, float("inf")).amin(dim=1, keepdim=True)
    x_max = x.masked_fill(~mask, float("-inf")).amax(dim=1, keepdim=True)

    x_min = torch.where(torch.isfinite(x_min), x_min, torch.zeros_like(x_min))
    x_max = torch.where(torch.isfinite(x_max), x_max, torch.ones_like(x_max))

    out = (x_masked - x_min) / (x_max - x_min + eps)
    out = out.masked_fill(~mask, 0.0)
    return out


def build_selector_utility_target(
    true_hit_mass,
    false_mass,
    valid_mask=None,
    false_lambda=1.2,
    temp=0.7,
):
    if valid_mask is None:
        valid_mask = torch.ones_like(true_hit_mass, dtype=torch.bool)
    else:
        valid_mask = valid_mask > 0.5

    raw_utility = true_hit_mass.float() - float(false_lambda) * false_mass.float()
    raw_utility = raw_utility.masked_fill(~valid_mask, -1e9)

    utility = normalize_per_sample(raw_utility, valid_mask)

    logits = raw_utility / max(float(temp), 1e-6)
    logits = logits.masked_fill(~valid_mask, -1e9)
    utility_dist = torch.softmax(logits, dim=1)
    utility_dist = utility_dist * valid_mask.float()
    utility_dist = utility_dist / utility_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

    return utility.detach(), utility_dist.detach()


def build_selector_teacher_dist_from_official_overlap(
    batch,
    formulae_mask,
    official_bin_n,
    eps=1e-8,
):
    """
    Runtime selector teacher.

    Purpose:
      Current cache has no valid teacher_formula_probs, so teacher KL is zero.
      This function builds selector_teacher_dist on the fly from:
        - candidate official-bin peaks
        - true official sparse spectrum

    Output:
      teacher_dist: [B, M], non-negative row-normalized distribution.
    """
    device = formulae_mask.device
    B, M = formulae_mask.shape

    off_idx = batch.get('formulae_peaks_official_idx', None)
    off_int = batch.get('formulae_peaks_official_intensity', None)

    if off_idx is None:
        off_idx = batch.get('formulae_peaks_mass_idx', None)
    if off_int is None:
        off_int = batch.get('formulae_peaks_intensity', None)

    if off_idx is None or off_int is None:
        return torch.zeros((B, M), dtype=torch.float32, device=device)

    off_idx = off_idx.to(device=device, dtype=torch.long)
    off_int = off_int.to(device=device, dtype=torch.float32)

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    K = min(off_idx.shape[-1], off_int.shape[-1])
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )

    off_int = torch.where(
        valid_peak,
        off_int.clamp_min(0.0),
        torch.zeros_like(off_int),
    )

    off_int_norm = off_int / off_int.sum(dim=-1, keepdim=True).clamp_min(eps)

    true_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)

    true_idx_list = batch.get('true_all_official_idx', None)
    true_int_list = batch.get('true_all_official_intensity', None)

    if isinstance(true_idx_list, (list, tuple)) and isinstance(true_int_list, (list, tuple)):
        for b in range(min(B, len(true_idx_list), len(true_int_list))):
            ti = true_idx_list[b]
            tv = true_int_list[b]

            if ti is None or tv is None:
                continue

            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv)

            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            tv = tv.to(device=device, dtype=torch.float32).reshape(-1)

            n = min(ti.numel(), tv.numel())
            if n <= 0:
                continue

            ti = ti[:n]
            tv = tv[:n]

            keep = (
                (ti >= 0)
                & (ti < official_bin_n)
                & torch.isfinite(tv)
                & (tv > 0)
            )

            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, ti[keep], tv[keep].clamp_min(0.0))

    elif torch.is_tensor(true_idx_list) and torch.is_tensor(true_int_list):
        ti = true_idx_list.to(device=device, dtype=torch.long)
        tv = true_int_list.to(device=device, dtype=torch.float32)

        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        if tv.dim() == 1:
            tv = tv.unsqueeze(0)

        for b in range(min(B, ti.shape[0], tv.shape[0])):
            idx = ti[b].reshape(-1)
            val = tv[b].reshape(-1)

            n = min(idx.numel(), val.numel())
            if n <= 0:
                continue

            idx = idx[:n]
            val = val[:n]

            keep = (
                (idx >= 0)
                & (idx < official_bin_n)
                & torch.isfinite(val)
                & (val > 0)
            )

            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, idx[keep], val[keep].clamp_min(0.0))

    true_dense = true_dense / true_dense.sum(dim=-1, keepdim=True).clamp_min(eps)
    true_support = (true_dense > 0).float()

    idx_safe = off_idx.clamp(0, official_bin_n - 1)

    true_at_candidate_bins = torch.gather(
        true_dense.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    support_at_candidate_bins = torch.gather(
        true_support.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    overlap_intensity = (
        off_int_norm
        * true_at_candidate_bins
        * valid_peak.float()
    ).sum(dim=-1)

    hit_support_mass = (
        off_int_norm
        * support_at_candidate_bins
        * valid_peak.float()
    ).sum(dim=-1)

    false_support_mass = (
        off_int_norm
        * (1.0 - support_at_candidate_bins)
        * valid_peak.float()
    ).sum(dim=-1)

    # Optional top20 bonus.
    top20_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    top20_idx_list = batch.get('true_top20_official_idx', None)

    if isinstance(top20_idx_list, (list, tuple)):
        for b in range(min(B, len(top20_idx_list))):
            ti = top20_idx_list[b]
            if ti is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            keep = (ti >= 0) & (ti < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, ti[keep]] = 1.0

    elif torch.is_tensor(top20_idx_list):
        ti = top20_idx_list.to(device=device, dtype=torch.long)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)

        for b in range(min(B, ti.shape[0])):
            idx = ti[b].reshape(-1)
            keep = (idx >= 0) & (idx < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, idx[keep]] = 1.0

    top20_at_candidate_bins = torch.gather(
        top20_dense.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    hit_top20_mass = (
        off_int_norm
        * top20_at_candidate_bins
        * valid_peak.float()
    ).sum(dim=-1)

    # Runtime teacher score.
    # This is not full set-cover yet, but it is a real dense teacher distribution.
    try:
        w_overlap = float(os.environ.get("RUNTIME_TEACHER_W_OVERLAP", "6.0"))
    except Exception:
        w_overlap = 6.0

    try:
        w_support = float(os.environ.get("RUNTIME_TEACHER_W_SUPPORT", "0.5"))
    except Exception:
        w_support = 0.5

    try:
        w_top20 = float(os.environ.get("RUNTIME_TEACHER_W_TOP20", "2.0"))
    except Exception:
        w_top20 = 2.0

    try:
        w_false = float(os.environ.get("RUNTIME_TEACHER_W_FALSE", "2.0"))
    except Exception:
        w_false = 2.0

    score = (
        w_overlap * overlap_intensity
        + w_support * hit_support_mass
        + w_top20 * hit_top20_mass
        - w_false * false_support_mass
    )

    score = torch.where(
        formulae_mask > 0.5,
        score,
        torch.full_like(score, -1e9),
    )

    try:
        teacher_topk = int(os.environ.get("RUNTIME_SELECTOR_TEACHER_TOPK", "64"))
    except Exception:
        teacher_topk = 64

    tk = max(1, min(int(teacher_topk), M))

    top_idx = torch.topk(score, k=tk, dim=1).indices

    teacher_mask = torch.zeros_like(score)
    teacher_mask.scatter_(1, top_idx, 1.0)

    # Only keep useful positives.
    teacher_mask = teacher_mask * (score > 0).float() * formulae_mask.float()

    try:
        temp = float(os.environ.get("RUNTIME_SELECTOR_TEACHER_TEMP", "0.25"))
    except Exception:
        temp = 0.25

    masked_score = score.masked_fill(teacher_mask <= 0.5, -1e9)

    teacher_dist = F.softmax(masked_score / max(float(temp), 1e-6), dim=1)
    teacher_dist = teacher_dist * teacher_mask
    teacher_dist = teacher_dist / teacher_dist.sum(dim=1, keepdim=True).clamp_min(eps)

    teacher_dist = torch.where(
        teacher_mask.sum(dim=1, keepdim=True) > 0,
        teacher_dist,
        torch.zeros_like(teacher_dist),
    )

    return teacher_dist.detach()

def build_selector_teacher_dist_setcover(
    batch,
    formulae_mask,
    official_bin_n,
    eps=1e-8,
):
    """
    Runtime vectorized set-cover selector teacher.

    Important:
      This avoids Python loop over candidates.
      It loops over batch and greedy steps only.
    """
    device = formulae_mask.device
    B, M = formulae_mask.shape

    off_idx = batch.get('formulae_peaks_official_idx', None)
    off_int = batch.get('formulae_peaks_official_intensity', None)

    if off_idx is None:
        off_idx = batch.get('formulae_peaks_mass_idx', None)
    if off_int is None:
        off_int = batch.get('formulae_peaks_intensity', None)

    if off_idx is None or off_int is None:
        return torch.zeros((B, M), dtype=torch.float32, device=device)

    off_idx = off_idx.to(device=device, dtype=torch.long)
    off_int = off_int.to(device=device, dtype=torch.float32)

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    K = min(off_idx.shape[-1], off_int.shape[-1])
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )

    off_int = torch.where(valid_peak, off_int.clamp_min(0.0), torch.zeros_like(off_int))
    off_int_norm = off_int / off_int.sum(dim=-1, keepdim=True).clamp_min(eps)
    idx_safe_all = off_idx.clamp(0, official_bin_n - 1)

    # -----------------------------
    # Build true dense spectrum
    # -----------------------------
    true_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)

    true_idx_list = batch.get('true_all_official_idx', None)
    true_int_list = batch.get('true_all_official_intensity', None)

    if isinstance(true_idx_list, (list, tuple)) and isinstance(true_int_list, (list, tuple)):
        for b in range(min(B, len(true_idx_list), len(true_int_list))):
            ti = true_idx_list[b]
            tv = true_int_list[b]
            if ti is None or tv is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv)

            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            tv = tv.to(device=device, dtype=torch.float32).reshape(-1)

            n = min(ti.numel(), tv.numel())
            if n <= 0:
                continue

            ti = ti[:n]
            tv = tv[:n]

            keep = (
                (ti >= 0)
                & (ti < official_bin_n)
                & torch.isfinite(tv)
                & (tv > 0)
            )

            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, ti[keep], tv[keep].clamp_min(0.0))

    elif torch.is_tensor(true_idx_list) and torch.is_tensor(true_int_list):
        ti = true_idx_list.to(device=device, dtype=torch.long)
        tv = true_int_list.to(device=device, dtype=torch.float32)

        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        if tv.dim() == 1:
            tv = tv.unsqueeze(0)

        for b in range(min(B, ti.shape[0], tv.shape[0])):
            idx = ti[b].reshape(-1)
            val = tv[b].reshape(-1)

            n = min(idx.numel(), val.numel())
            if n <= 0:
                continue

            idx = idx[:n]
            val = val[:n]

            keep = (
                (idx >= 0)
                & (idx < official_bin_n)
                & torch.isfinite(val)
                & (val > 0)
            )

            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, idx[keep], val[keep].clamp_min(0.0))

    true_dense = true_dense / true_dense.sum(dim=-1, keepdim=True).clamp_min(eps)
    true_support = (true_dense > 0).float()

    # -----------------------------
    # Build top20 support
    # -----------------------------
    top20_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    top20_idx_list = batch.get('true_top20_official_idx', None)

    if isinstance(top20_idx_list, (list, tuple)):
        for b in range(min(B, len(top20_idx_list))):
            ti = top20_idx_list[b]
            if ti is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            keep = (ti >= 0) & (ti < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, ti[keep]] = 1.0

    elif torch.is_tensor(top20_idx_list):
        ti = top20_idx_list.to(device=device, dtype=torch.long)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)

        for b in range(min(B, ti.shape[0])):
            idx = ti[b].reshape(-1)
            keep = (idx >= 0) & (idx < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, idx[keep]] = 1.0

    # -----------------------------
    # Hyperparams
    # -----------------------------
    try:
        setcover_steps = int(os.environ.get("RUNTIME_SETCOVER_STEPS", "16"))
    except Exception:
        setcover_steps = 16

    try:
        candidate_prefilter_k = int(os.environ.get("RUNTIME_SETCOVER_PREFILTER_TOPK", "128"))
    except Exception:
        candidate_prefilter_k = 128

    try:
        w_gain = float(os.environ.get("RUNTIME_SETCOVER_W_GAIN", "3.0"))
    except Exception:
        w_gain = 3.0

    try:
        w_top20 = float(os.environ.get("RUNTIME_SETCOVER_W_TOP20", "1.5"))
    except Exception:
        w_top20 = 1.5

    try:
        w_false = float(os.environ.get("RUNTIME_SETCOVER_W_FALSE", "0.8"))
    except Exception:
        w_false = 0.8

    try:
        w_redun = float(os.environ.get("RUNTIME_SETCOVER_W_REDUN", "0.3"))
    except Exception:
        w_redun = 0.3
    try:
        min_steps = int(os.environ.get("RUNTIME_SETCOVER_MIN_STEPS", "16"))
    except Exception:
        min_steps = 16

    force_min_steps = os.environ.get("RUNTIME_SETCOVER_FORCE_MIN_STEPS", "1") == "1"

    try:
        stop_eps = float(os.environ.get("RUNTIME_SETCOVER_STOP_EPS", "0.0"))
    except Exception:
        stop_eps = 0.0
    teacher_dist = torch.zeros((B, M), dtype=torch.float32, device=device)

    # -----------------------------
    # Per sample greedy, vectorized over candidates
    # -----------------------------
    for b in range(B):
        valid_m = formulae_mask[b].float()
        if valid_m.sum() <= 0:
            continue

        true_b = true_dense[b]
        support_b = true_support[b]
        top20_b = top20_dense[b]

        idx_b = idx_safe_all[b]          # [M, K]
        int_b = off_int_norm[b]          # [M, K]
        valid_peak_b = valid_peak[b].float()

        # Base prefilter score, vectorized over all M.
        true_at = true_b[idx_b]
        support_at = support_b[idx_b]
        top20_at = top20_b[idx_b]

        overlap = (int_b * true_at * valid_peak_b).sum(dim=-1)
        support_hit = (int_b * support_at * valid_peak_b).sum(dim=-1)
        top20_hit = (int_b * top20_at * valid_peak_b).sum(dim=-1)
        false_mass = (int_b * (1.0 - support_at) * valid_peak_b).sum(dim=-1)

        base_score = (
            4.0 * overlap
            + 1.0 * support_hit
            + 2.0 * top20_hit
            - 1.0 * false_mass
        )

        base_score = base_score.masked_fill(valid_m <= 0.5, -1e9)

        pk = max(1, min(int(candidate_prefilter_k), M))
        pref_idx = torch.topk(base_score, k=pk, dim=0).indices

        pref_bins = idx_b[pref_idx]              # [P, K]
        pref_int = int_b[pref_idx]               # [P, K]
        pref_valid = valid_peak_b[pref_idx]      # [P, K]

        pref_true = true_b[pref_bins]            # [P, K]
        pref_support = support_b[pref_bins]      # [P, K]
        pref_top20 = top20_b[pref_bins]          # [P, K]

        pref_true_mass = pref_true * pref_int * pref_valid
        pref_top20_mass = pref_top20 * pref_int * pref_valid
        pref_false_mass = (1.0 - pref_support) * pref_int * pref_valid

        available = torch.ones((pk,), dtype=torch.bool, device=device)

        covered_true = torch.zeros((official_bin_n,), dtype=torch.float32, device=device)
        covered_top20 = torch.zeros((official_bin_n,), dtype=torch.float32, device=device)

        selected_pref_positions = []

        steps = max(1, min(int(setcover_steps), pk))

        for _step in range(steps):
            already_true = covered_true[pref_bins].clamp(0.0, 1.0)
            already_top20 = covered_top20[pref_bins].clamp(0.0, 1.0)

            new_true_gain = (pref_true_mass * (1.0 - already_true)).sum(dim=-1)
            new_top20_gain = (pref_top20_mass * (1.0 - already_top20)).sum(dim=-1)
            redun_true = (pref_true_mass * already_true).sum(dim=-1)
            false_add = pref_false_mass.sum(dim=-1)

            gain_score = (
                w_gain * new_true_gain
                + w_top20 * new_top20_gain
                - w_false * false_add
                - w_redun * redun_true
            )

            gain_score = gain_score.masked_fill(~available, -1e9)

            best_pos = torch.argmax(gain_score)
            best_score = gain_score[best_pos]

            selected_n = len(selected_pref_positions)

            # Do not stop too early. A too-sparse teacher cannot train top256 recall.
            if best_score <= stop_eps:
                if (not force_min_steps) or (selected_n >= int(min_steps)):
                    break

            selected_pref_positions.append(best_pos)

            available[best_pos] = False

            bins_sel = pref_bins[best_pos]
            true_sel = pref_true_mass[best_pos]
            top20_sel = pref_top20_mass[best_pos]

            covered_true.scatter_add_(0, bins_sel, true_sel)
            covered_top20.scatter_add_(0, bins_sel, top20_sel)

        if len(selected_pref_positions) == 0:
            continue

        selected_pref_positions = torch.stack(selected_pref_positions).long()
        selected_global_idx = pref_idx[selected_pref_positions]

        # Use positive greedy gains approximated by base_score for probability.
        selected_scores = base_score[selected_global_idx].clamp_min(0.0)

        if selected_scores.sum() <= 0:
            selected_scores = torch.ones_like(selected_scores)

        selected_probs = selected_scores / selected_scores.sum().clamp_min(eps)

        teacher_dist[b, selected_global_idx] = selected_probs

    return teacher_dist.detach()

def build_candidate_local_quality_target(
    batch,
    formulae_mask,
    official_bin_n,
    true_key_idx='true_all_official_idx',
    true_key_int='true_all_official_intensity',
    top20_key_idx='true_top20_official_idx',
    eps=1e-8,
):
    """
    Returns:
      quality: [B, M], float in [0, 1]
      pos_label: [B, M], 0/1
      valid_mask: [B, M], 0/1
    """
    device = formulae_mask.device
    B, M = formulae_mask.shape

    off_idx = batch.get('formulae_peaks_official_idx', None)
    off_int = batch.get('formulae_peaks_official_intensity', None)

    if off_idx is None:
        off_idx = batch.get('formulae_peaks_mass_idx', None)
    if off_int is None:
        off_int = batch.get('formulae_peaks_intensity', None)

    if off_idx is None or off_int is None:
        quality = torch.zeros((B, M), dtype=torch.float32, device=device)
        return quality, quality, formulae_mask.float(), {}

    off_idx = off_idx.to(device=device, dtype=torch.long)
    off_int = off_int.to(device=device, dtype=torch.float32)

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    K = min(off_idx.shape[-1], off_int.shape[-1])
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )

    off_int = torch.where(valid_peak, off_int.clamp_min(0.0), torch.zeros_like(off_int))

    off_int_norm = off_int / off_int.sum(dim=-1, keepdim=True).clamp_min(eps)

    true_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    true_idx_list = batch.get(true_key_idx, None)
    true_int_list = batch.get(true_key_int, None)

    if isinstance(true_idx_list, (list, tuple)) and isinstance(true_int_list, (list, tuple)):
        for b in range(min(B, len(true_idx_list), len(true_int_list))):
            ti = true_idx_list[b]
            tv = true_int_list[b]
            if ti is None or tv is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            tv = tv.to(device=device, dtype=torch.float32).reshape(-1)
            n = min(ti.numel(), tv.numel())
            if n <= 0:
                continue
            ti = ti[:n]
            tv = tv[:n]
            keep = (ti >= 0) & (ti < official_bin_n) & torch.isfinite(tv) & (tv > 0)
            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, ti[keep], tv[keep].clamp_min(0.0))
    elif torch.is_tensor(true_idx_list) and torch.is_tensor(true_int_list):
        ti = true_idx_list.to(device=device, dtype=torch.long)
        tv = true_int_list.to(device=device, dtype=torch.float32)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        if tv.dim() == 1:
            tv = tv.unsqueeze(0)
        for b in range(min(B, ti.shape[0], tv.shape[0])):
            idx = ti[b].reshape(-1)
            val = tv[b].reshape(-1)
            n = min(idx.numel(), val.numel())
            idx = idx[:n]
            val = val[:n]
            keep = (idx >= 0) & (idx < official_bin_n) & torch.isfinite(val) & (val > 0)
            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, idx[keep], val[keep].clamp_min(0.0))

    true_dense = true_dense / true_dense.sum(dim=-1, keepdim=True).clamp_min(eps)
    true_support = (true_dense > 0).float()

    # Allow small official-bin mismatch when building selector target.
    # Exact 0.01-bin matching is too strict for candidate-local supervision.
    try:
        selector_target_bin_tol = int(os.environ.get("SELECTOR_TARGET_BIN_TOL", "1"))
    except Exception:
        selector_target_bin_tol = 1

    selector_target_bin_tol = max(0, int(selector_target_bin_tol))

    if selector_target_bin_tol > 0:
        ksz = 2 * selector_target_bin_tol + 1

        # For intensity overlap, max-pool is intentionally used:
        # a candidate peak gets credit if it falls near a true bin.
        true_dense_for_match = F.max_pool1d(
            true_dense.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)

        true_support_for_match = F.max_pool1d(
            true_support.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)
    else:
        true_dense_for_match = true_dense
        true_support_for_match = true_support

    idx_safe = off_idx.clamp(0, official_bin_n - 1)
    true_at_candidate_bins_exact = torch.gather(
        true_dense.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    support_at_candidate_bins_exact = torch.gather(
        true_support.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    true_at_candidate_bins_tol = torch.gather(
        true_dense_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    support_at_candidate_bins_tol = torch.gather(
        true_support_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    overlap_intensity_exact = (
        off_int_norm * true_at_candidate_bins_exact * valid_peak.float()
    ).sum(dim=-1)
    overlap_intensity_tol = (
        off_int_norm * true_at_candidate_bins_tol * valid_peak.float()
    ).sum(dim=-1)
    hit_support_mass_tol = (
        off_int_norm * support_at_candidate_bins_tol * valid_peak.float()
    ).sum(dim=-1)
    false_support_mass_exact = (
        off_int_norm * (1.0 - support_at_candidate_bins_exact) * valid_peak.float()
    ).sum(dim=-1)

    top20_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    top20_idx_list = batch.get(top20_key_idx, None)
    if isinstance(top20_idx_list, (list, tuple)):
        for b in range(min(B, len(top20_idx_list))):
            ti = top20_idx_list[b]
            if ti is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            keep = (ti >= 0) & (ti < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, ti[keep]] = 1.0
    elif torch.is_tensor(top20_idx_list):
        ti = top20_idx_list.to(device=device, dtype=torch.long)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        for b in range(min(B, ti.shape[0])):
            idx = ti[b].reshape(-1)
            keep = (idx >= 0) & (idx < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, idx[keep]] = 1.0

    if selector_target_bin_tol > 0:
        ksz = 2 * selector_target_bin_tol + 1
        top20_dense_for_match = F.max_pool1d(
            top20_dense.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)
    else:
        top20_dense_for_match = top20_dense
    top20_at_candidate_bins = torch.gather(
        top20_dense_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )
    hit_top20_mass = (off_int_norm * top20_at_candidate_bins * valid_peak.float()).sum(dim=-1)

    try:
        w_overlap = float(os.environ.get("SELECTOR_QUALITY_W_OVERLAP", "1.50"))
    except Exception:
        w_overlap = 1.50

    try:
        w_support = float(os.environ.get("SELECTOR_QUALITY_W_SUPPORT", "0.35"))
    except Exception:
        w_support = 0.35

    try:
        w_top20 = float(os.environ.get("SELECTOR_QUALITY_W_TOP20", "0.75"))
    except Exception:
        w_top20 = 0.75

    try:
        w_false = float(os.environ.get("SELECTOR_QUALITY_W_FALSE", "1.20"))
    except Exception:
        w_false = 1.20

    # ------------------------------------------------------------------
    # Absolute clean selector target.
    # Do NOT row-minmax bad candidates into positives.
    # ------------------------------------------------------------------

    exact_support_mass = (1.0 - false_support_mass_exact).clamp(0.0, 1.0)

    quality_raw = (
        w_overlap * overlap_intensity_exact
        + 0.40 * overlap_intensity_tol
        + w_support * hit_support_mass_tol
        + w_top20 * hit_top20_mass
        - w_false * false_support_mass_exact
    )

    # A multiplicative clean gate is much stronger than only subtracting false mass.
    try:
        clean_gamma = float(os.environ.get("SELECTOR_CLEAN_GAMMA", "2.0"))
    except Exception:
        clean_gamma = 2.0

    clean_gate = exact_support_mass.clamp(0.0, 1.0) ** clean_gamma

    # Final score: must be positive and clean.
    quality_score = quality_raw.clamp_min(0.0) * clean_gate

    # Absolute positive filters.
    try:
        min_exact_support = float(os.environ.get("SELECTOR_POS_MIN_EXACT_SUPPORT", "0.20"))
    except Exception:
        min_exact_support = 0.20

    try:
        min_tol_support = float(os.environ.get("SELECTOR_POS_MIN_TOL_SUPPORT", "0.25"))
    except Exception:
        min_tol_support = 0.25

    try:
        max_false_support = float(os.environ.get("SELECTOR_POS_MAX_FALSE_SUPPORT", "0.80"))
    except Exception:
        max_false_support = 0.80

    strict_keep = (
        (formulae_mask > 0.5)
        & (quality_score > 0.0)
        & (exact_support_mass >= min_exact_support)
        & (hit_support_mass_tol >= min_tol_support)
        & (false_support_mass_exact <= max_false_support)
    )

    quality_score = torch.where(
        strict_keep,
        quality_score,
        torch.zeros_like(quality_score),
    )

    # Normalize by row max only. No row-min subtraction.
    row_max = quality_score.max(dim=1, keepdim=True).values
    quality = quality_score / row_max.clamp_min(eps)
    quality = torch.where(
        row_max > eps,
        quality,
        torch.zeros_like(quality),
    )
    quality = quality * formulae_mask.float()

    try:
        target_support_topk = int(os.environ.get("TARGET_SUPPORT_TOPK", "64"))
    except Exception:
        target_support_topk = 64

    try:
        target_min_pos = int(os.environ.get("TARGET_MIN_POS", "8"))
    except Exception:
        target_min_pos = 8

    k = max(1, min(target_support_topk, M))

    # First choose topK among clean candidates.
    masked_quality = quality.masked_fill(formulae_mask <= 0.5, -1e9)
    top_idx = torch.topk(masked_quality, k=k, dim=1).indices

    pos_label = torch.zeros_like(quality)
    pos_label.scatter_(1, top_idx, 1.0)
    pos_label = pos_label * strict_keep.float()

    # Fallback: if a row has too few strict positives, add the best fallback candidates.
    # This avoids empty KL rows, but still prevents top64 garbage positives.
    row_pos_n = pos_label.sum(dim=1, keepdim=True)

    fallback_score = (
        0.50 * exact_support_mass
        + 0.35 * hit_support_mass_tol
        + 0.15 * hit_top20_mass
    ) * formulae_mask.float()

    try:
        fb_min_exact_support = float(os.environ.get("SELECTOR_FB_MIN_EXACT_SUPPORT", "0.05"))
    except Exception:
        fb_min_exact_support = 0.05

    try:
        fb_max_false_support = float(os.environ.get("SELECTOR_FB_MAX_FALSE_SUPPORT", "0.95"))
    except Exception:
        fb_max_false_support = 0.95

    fallback_keep = (
        (formulae_mask > 0.5)
        & (fallback_score > 0.0)
        & (exact_support_mass >= fb_min_exact_support)
        & (false_support_mass_exact <= fb_max_false_support)
    )

    fallback_score = fallback_score.masked_fill(~fallback_keep, -1e9)

    fb_k = max(1, min(target_min_pos, M))
    fb_idx = torch.topk(fallback_score, k=fb_k, dim=1).indices
    fb_label = torch.zeros_like(pos_label)
    fb_label.scatter_(1, fb_idx, 1.0)
    fb_label = fb_label * fallback_keep.float()

    need_fb = (row_pos_n < float(target_min_pos)).float()
    fallback_used = need_fb.float()
    pos_label = torch.where(
        need_fb > 0.5,
        torch.maximum(pos_label, fb_label),
        pos_label,
    )

    # But final pos should never exceed target_support_topk.
    if target_support_topk < M:
        pos_quality = quality.masked_fill(pos_label <= 0.5, -1e9)
        keep_idx = torch.topk(pos_quality, k=k, dim=1).indices
        keep_label = torch.zeros_like(pos_label)
        keep_label.scatter_(1, keep_idx, 1.0)
        pos_label = pos_label * keep_label

    clean_pos_label = pos_label.clone()

    # ------------------------------------------------------------------
    # Pool-level selector target.
    # Clean positives are too few for topK recall.
    # Pool positives teach selector which candidates should enter top256.
    # ------------------------------------------------------------------
    try:
        pool_pos_topk = int(os.environ.get("SELECTOR_POOL_POS_TOPK", "96"))
    except Exception:
        pool_pos_topk = 96

    try:
        pool_min_overlap_tol = float(os.environ.get("SELECTOR_POOL_MIN_OVERLAP_TOL", "0.0005"))
    except Exception:
        pool_min_overlap_tol = 0.0005

    try:
        pool_min_overlap_exact = float(os.environ.get("SELECTOR_POOL_MIN_OVERLAP_EXACT", "0.0001"))
    except Exception:
        pool_min_overlap_exact = 0.0001

    try:
        pool_min_top20 = float(os.environ.get("SELECTOR_POOL_MIN_TOP20", "0.0005"))
    except Exception:
        pool_min_top20 = 0.0005

    try:
        pool_max_false_support = float(os.environ.get("SELECTOR_POOL_MAX_FALSE_SUPPORT", "0.97"))
    except Exception:
        pool_max_false_support = 0.97

    # Pool target should be intensity-overlap driven.
    # Do not let exact_support_mass dominate, because it rewards weak support hits.
    pool_score = (
        3.00 * overlap_intensity_tol
        + 1.50 * overlap_intensity_exact
        + 2.00 * hit_top20_mass
        + 0.25 * hit_support_mass_tol
        - 0.35 * false_support_mass_exact
    )

    pool_signal = (
        (overlap_intensity_tol >= pool_min_overlap_tol)
        | (overlap_intensity_exact >= pool_min_overlap_exact)
        | (hit_top20_mass >= pool_min_top20)
    )

    pool_keep = (
        (formulae_mask > 0.5)
        & pool_signal
        & (false_support_mass_exact <= pool_max_false_support)
        & torch.isfinite(pool_score)
    )

    pool_score = torch.where(
        pool_keep,
        pool_score,
        torch.full_like(pool_score, -1e9),
    )

    pk = max(1, min(int(pool_pos_topk), M))
    pool_idx = torch.topk(pool_score, k=pk, dim=1).indices

    pool_pos_label = torch.zeros_like(pos_label)
    pool_pos_label.scatter_(1, pool_idx, 1.0)
    pool_pos_label = pool_pos_label * pool_keep.float()

    # ------------------------------------------------------------------
    # Cached teacher / set-cover teacher target.
    # This is the real spectrum-level supervision. Hand-crafted pool target
    # is only a fallback; teacher target should dominate selector recall.
    # ------------------------------------------------------------------
    teacher_pos_label = torch.zeros_like(pos_label)
    teacher_dist = torch.zeros_like(pos_label)

    use_cached_teacher = os.environ.get("SELECTOR_USE_CACHED_TEACHER_TARGET", "1") == "1"
    if use_cached_teacher:
        teacher = batch.get("selector_teacher_dist", None)
        if teacher is None:
            teacher = batch.get("teacher_formula_probs", None)

        if torch.is_tensor(teacher):
            teacher = teacher.to(device=device, dtype=torch.float32)

            if teacher.dim() == 1:
                teacher = teacher.unsqueeze(0)
            elif teacher.dim() > 2:
                teacher = teacher.reshape(teacher.shape[0], -1)

            if teacher.shape[0] < B:
                pad = torch.zeros(
                    (B - teacher.shape[0], teacher.shape[1]),
                    device=device,
                    dtype=torch.float32,
                )
                teacher = torch.cat([teacher, pad], dim=0)
            teacher = teacher[:B]

            if teacher.shape[1] < M:
                pad = torch.zeros(
                    (B, M - teacher.shape[1]),
                    device=device,
                    dtype=torch.float32,
                )
                teacher = torch.cat([teacher, pad], dim=1)
            teacher = teacher[:, :M]

            teacher = torch.nan_to_num(teacher, nan=0.0, posinf=0.0, neginf=0.0)
            teacher = teacher.clamp_min(0.0) * formulae_mask.float()

            teacher_sum = teacher.sum(dim=1, keepdim=True)
            teacher_dist = teacher / teacher_sum.clamp_min(eps)
            teacher_dist = torch.where(
                teacher_sum > eps,
                teacher_dist,
                torch.zeros_like(teacher_dist),
            )

            try:
                teacher_pos_topk = int(os.environ.get("SELECTOR_TEACHER_POS_TOPK", "32"))
            except Exception:
                teacher_pos_topk = 32

            try:
                teacher_min_prob = float(os.environ.get("SELECTOR_TEACHER_MIN_PROB", "1e-8"))
            except Exception:
                teacher_min_prob = 1e-8

            tk = max(1, min(int(teacher_pos_topk), M))
            teacher_idx = torch.topk(
                teacher_dist.masked_fill(formulae_mask <= 0.5, -1e9),
                k=tk,
                dim=1,
            ).indices

            teacher_pos_label.scatter_(1, teacher_idx, 1.0)
            teacher_pos_label = teacher_pos_label * (teacher_dist > teacher_min_prob).float()
            teacher_pos_label = teacher_pos_label * formulae_mask.float()
            if os.environ.get("DEBUG_SELECTOR_TEACHER", "0") == "1":
                try:
                    print(
                        "[SELECTOR_TEACHER_DEBUG]",
                        "teacher_sum_mean=", float(teacher_sum.detach().mean().cpu().item()),
                        "teacher_dist_n=", float((teacher_dist > 0).float().sum(dim=1).mean().detach().cpu().item()),
                        "teacher_pos_rate=", float((teacher_pos_label.sum() / formulae_mask.float().sum().clamp_min(1.0)).detach().cpu().item()),
                        flush=True,
                    )
                except Exception:
                    pass
    # Final BCE positives:
    # clean = very high precision
    # teacher = real spectrum-level/set-cover supervision
    # pool = fallback local target
    if os.environ.get("SELECTOR_DISABLE_HAND_POOL_TARGET", "1") == "1":
        pos_label = torch.maximum(clean_pos_label, teacher_pos_label)
    else:
        pos_label = torch.maximum(torch.maximum(clean_pos_label, teacher_pos_label), pool_pos_label)

    has_signal = (pos_label.sum(dim=1, keepdim=True) > 0).float()
    valid_mask = formulae_mask.float() * has_signal
    teacher_added_label = teacher_pos_label * (1.0 - clean_pos_label)
    utility, utility_dist = build_selector_utility_target(
        true_hit_mass=overlap_intensity_exact,
        false_mass=false_support_mass_exact,
        valid_mask=formulae_mask > 0.5,
        false_lambda=float(os.environ.get("SELECTOR_UTILITY_FALSE_LAMBDA", "1.2")),
        temp=float(os.environ.get("SELECTOR_UTILITY_TEMP", "0.7")),
    )
    return quality, pos_label, valid_mask, {
        'overlap_intensity_exact': overlap_intensity_exact.detach(),
        'overlap_intensity_tol': overlap_intensity_tol.detach(),
        'hit_support_mass_tol': hit_support_mass_tol.detach(),
        'hit_top20_mass': hit_top20_mass.detach(),
        'false_support_mass_exact': false_support_mass_exact.detach(),
        'exact_support_mass': exact_support_mass.detach(),
        'strict_keep': strict_keep.float().detach(),
        'quality_score': quality_score.detach(),
        'fallback_used': fallback_used.detach(),
        'fallback_keep': fallback_keep.float().detach(),
        'clean_pos_label': clean_pos_label.detach(),
        'pool_pos_label': pool_pos_label.detach(),
        'pool_keep': pool_keep.float().detach(),
        'pool_score': pool_score.detach(),
        'teacher_pos_label': teacher_pos_label.detach(),
        'teacher_dist': teacher_dist.detach(),
        'teacher_added_label': teacher_added_label.detach(),
        'true_hit_mass': overlap_intensity_exact.detach(),
        'false_mass': false_support_mass_exact.detach(),
        'utility': utility.detach(),
        'utility_dist': utility_dist.detach(),
    }


def _get_first_tensor_from_batch(batch, names):
    for name in names:
        x = batch.get(name, None)
        if torch.is_tensor(x):
            return x
    feat = batch.get("features", None)
    if isinstance(feat, dict):
        for name in names:
            x = feat.get(name, None)
            if torch.is_tensor(x):
                return x
    return None


def build_selector_teacher_dist_template_oracle(
    batch,
    formulae_mask,
    official_bin_n,
):
    """
    Build a direct template-oracle teacher.

    Teacher score for each candidate formula is based on:
      candidate official peak template vs true official spectrum.

    This is different from local quality / setcover:
    it directly supervises the selector to find the few candidates whose
    cached peak templates actually explain the target spectrum.
    """
    device = formulae_mask.device
    B, M = formulae_mask.shape

    true_idx = _get_first_tensor_from_batch(
        batch,
        [
            "true_all_official_idx",
            "true_official_idx",
            "target_official_idx",
        ],
    )
    true_int = _get_first_tensor_from_batch(
        batch,
        [
            "true_all_official_intensity",
            "true_official_intensity",
            "target_official_intensity",
        ],
    )

    cand_idx = _get_first_tensor_from_batch(
        batch,
        [
            "formulae_peaks_official_idx",
            "formulae_peaks_official_idx_agg",
            "formulae_peaks_mass_idx",
        ],
    )
    cand_int = _get_first_tensor_from_batch(
        batch,
        [
            "formulae_peaks_official_intensity",
            "formulae_peaks_official_intensity_agg",
            "formulae_peaks_intensity",
        ],
    )

    if not (
        torch.is_tensor(true_idx)
        and torch.is_tensor(true_int)
        and torch.is_tensor(cand_idx)
        and torch.is_tensor(cand_int)
    ):
        zero = formulae_mask.float() * 0.0
        return zero, zero, formulae_mask.float(), {
            "template_oracle_missing": torch.ones((B,), device=device),
        }

    true_idx = true_idx.to(device=device, dtype=torch.long)
    true_int = true_int.to(device=device, dtype=torch.float32)
    cand_idx = cand_idx.to(device=device, dtype=torch.long)
    cand_int = cand_int.to(device=device, dtype=torch.float32)

    if true_idx.dim() == 1:
        true_idx = true_idx.unsqueeze(0)
    if true_int.dim() == 1:
        true_int = true_int.unsqueeze(0)

    if true_idx.dim() > 2:
        true_idx = true_idx.reshape(true_idx.shape[0], -1)
    if true_int.dim() > 2:
        true_int = true_int.reshape(true_int.shape[0], -1)

    if cand_idx.dim() == 2:
        cand_idx = cand_idx.unsqueeze(-1)
    if cand_int.dim() == 2:
        cand_int = cand_int.unsqueeze(-1)

    if cand_idx.dim() > 3:
        cand_idx = cand_idx.reshape(cand_idx.shape[0], cand_idx.shape[1], -1)
    if cand_int.dim() > 3:
        cand_int = cand_int.reshape(cand_int.shape[0], cand_int.shape[1], -1)

    use_b = min(B, int(true_idx.shape[0]), int(true_int.shape[0]), int(cand_idx.shape[0]), int(cand_int.shape[0]))
    use_m = min(M, int(cand_idx.shape[1]), int(cand_int.shape[1]))
    use_p = min(int(cand_idx.shape[2]), int(cand_int.shape[2]))

    if use_b <= 0 or use_m <= 0 or use_p <= 0:
        zero = formulae_mask.float() * 0.0
        return zero, zero, formulae_mask.float(), {}

    true_idx = true_idx[:use_b]
    true_int = true_int[:use_b]
    cand_idx = cand_idx[:use_b, :use_m, :use_p]
    cand_int = cand_int[:use_b, :use_m, :use_p]

    true_dense = torch.zeros(
        (use_b, int(official_bin_n)),
        dtype=torch.float32,
        device=device,
    )

    true_valid = (
        (true_idx >= 0)
        & (true_idx < int(official_bin_n))
        & torch.isfinite(true_int)
        & (true_int > 0)
    )

    true_safe_idx = true_idx.clamp(min=0, max=int(official_bin_n) - 1)
    true_dense.scatter_add_(1, true_safe_idx, true_int * true_valid.float())
    true_dense = true_dense.clamp_min(0.0)
    true_dense = true_dense / true_dense.sum(dim=1, keepdim=True).clamp_min(1e-12)

    cand_valid = (
        (cand_idx >= 0)
        & (cand_idx < int(official_bin_n))
        & torch.isfinite(cand_int)
        & (cand_int > 0)
    )

    cand_safe_idx = cand_idx.clamp(min=0, max=int(official_bin_n) - 1)
    cand_mass = cand_int * cand_valid.float()
    cand_mass = cand_mass / cand_mass.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    flat_idx = cand_safe_idx.reshape(use_b, -1)
    true_at_peak = torch.gather(true_dense, 1, flat_idx).reshape(use_b, use_m, use_p)

    true_hit = (true_at_peak > 1e-12).float() * cand_valid.float()

    overlap_mass = (cand_mass * true_hit).sum(dim=-1)
    false_mass = 1.0 - overlap_mass

    true_dot = (cand_mass * true_at_peak * cand_valid.float()).sum(dim=-1)

    # Same spirit as cache_template_oracle_probe.py:
    # true intensity dot + overlap reward - light false penalty.
    oracle_score = true_dot + 0.10 * overlap_mass - 0.02 * false_mass
    oracle_score = torch.where(
        torch.isfinite(oracle_score),
        oracle_score,
        torch.zeros_like(oracle_score),
    )
    oracle_score = oracle_score.clamp_min(0.0)

    full_score = formulae_mask.float() * 0.0
    full_score[:use_b, :use_m] = oracle_score
    full_score = full_score * formulae_mask.float()

    try:
        topk = int(os.environ.get("TEMPLATE_ORACLE_TOPK", "64"))
    except Exception:
        topk = 64
    topk = max(1, min(topk, M))

    # Keep only strongest oracle candidates as teacher positives.
    masked_score = full_score.masked_fill(formulae_mask <= 0.5, -1e9)
    top_idx = torch.topk(masked_score, k=topk, dim=1).indices

    pos_label = torch.zeros_like(full_score)
    pos_label.scatter_(1, top_idx, 1.0)
    pos_label = pos_label * (full_score > 0).float() * formulae_mask.float()

    teacher_dist = full_score * pos_label
    row_sum = teacher_dist.sum(dim=1, keepdim=True)
    valid_row = row_sum > 1e-12
    teacher_dist = teacher_dist / row_sum.clamp_min(1e-12)

    valid_mask = formulae_mask.float() * valid_row.float()

    extra = {
        "template_oracle_score": full_score,
        "template_oracle_pos_label": pos_label,
        "template_oracle_overlap_mass": torch.zeros_like(full_score),
        "template_oracle_false_mass": torch.zeros_like(full_score),
    }
    extra["template_oracle_overlap_mass"][:use_b, :use_m] = overlap_mass
    extra["template_oracle_false_mass"][:use_b, :use_m] = false_mass

    return teacher_dist, pos_label, valid_mask, extra