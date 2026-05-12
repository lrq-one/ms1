import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value
from rassp.training.spectrum_targets import build_true_official_dense_from_raw


def get_formulae_official_intensity_from_batch(batch):
    """Prefer official-bin intensity tensor when present; fallback to legacy intensity."""
    if not isinstance(batch, dict):
        return None
    off_int = batch.get("formulae_peaks_official_intensity", None)
    if torch.is_tensor(off_int) and off_int.dim() == 3:
        return off_int
    return batch.get("formulae_peaks_intensity", None)


def _build_true_official_dense_from_cached_sparse_batch(batch, batch_n, device, official_bin_n):
    out = torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device)
    idx_src = batch.get("true_official_idx", None)
    val_src = batch.get("true_official_intensity", None)
    used_cache = False

    if torch.is_tensor(idx_src) and torch.is_tensor(val_src):
        idx_t = idx_src.long()
        val_t = val_src.float()
        if idx_t.dim() == 1:
            idx_t = idx_t.unsqueeze(0)
        if val_t.dim() == 1:
            val_t = val_t.unsqueeze(0)
        if idx_t.dim() == 2 and val_t.dim() == 2:
            use_b = min(int(batch_n), int(idx_t.shape[0]), int(val_t.shape[0]))
            use_k = min(int(idx_t.shape[1]), int(val_t.shape[1]))
            if use_b > 0 and use_k > 0:
                idx_t = idx_t[:use_b, :use_k].to(device=device)
                val_t = val_t[:use_b, :use_k].to(device=device)
                valid = (
                    (idx_t >= 0)
                    & (idx_t < int(official_bin_n))
                    & torch.isfinite(val_t)
                    & (val_t > 0)
                )
                if bool(valid.any().item()):
                    idx_safe = idx_t.clamp(0, max(0, int(official_bin_n) - 1))
                    out[:use_b].scatter_add_(1, idx_safe, val_t * valid.float())
                    used_cache = True
        return out, used_cache

    if isinstance(idx_src, (list, tuple)) and isinstance(val_src, (list, tuple)):
        use_b = min(int(batch_n), len(idx_src), len(val_src))
        for bi in range(use_b):
            try:
                idx_i = np.asarray(idx_src[bi], dtype=np.int64).reshape(-1)
                val_i = np.asarray(val_src[bi], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if idx_i.size <= 0 or val_i.size <= 0:
                continue
            use_k = min(int(idx_i.shape[0]), int(val_i.shape[0]))
            idx_i = idx_i[:use_k]
            val_i = val_i[:use_k]
            valid_i = (
                (idx_i >= 0)
                & (idx_i < int(official_bin_n))
                & np.isfinite(val_i)
                & (val_i > 0)
            )
            if not np.any(valid_i):
                continue
            idx_t = torch.as_tensor(idx_i[valid_i], dtype=torch.long, device=device)
            val_t = torch.as_tensor(val_i[valid_i], dtype=torch.float32, device=device)
            out[bi].scatter_add_(0, idx_t, val_t)
            used_cache = True
    return out, used_cache


build_true_official_dense_from_cached_sparse_batch = _build_true_official_dense_from_cached_sparse_batch


def _build_cached_true_top20_tensors(batch, batch_n, official_bin_n, device, default_k=20):
    k = max(1, int(default_k))
    out_idx = torch.full((int(batch_n), k), -1, dtype=torch.long, device=device)
    out_val = torch.zeros((int(batch_n), k), dtype=torch.float32, device=device)
    out_valid = torch.zeros((int(batch_n), k), dtype=torch.bool, device=device)
    used_cache = False

    idx_src = batch.get("true_top20_official_idx", None)
    val_src = batch.get("true_top20_official_intensity", None)
    if idx_src is None or val_src is None:
        return out_idx, out_val, out_valid, used_cache

    if torch.is_tensor(idx_src) and torch.is_tensor(val_src):
        idx_t = idx_src.long()
        val_t = val_src.float()
        if idx_t.dim() == 1:
            idx_t = idx_t.unsqueeze(0)
        if val_t.dim() == 1:
            val_t = val_t.unsqueeze(0)
        if idx_t.dim() != 2 or val_t.dim() != 2:
            return out_idx, out_val, out_valid, used_cache

        use_b = min(int(batch_n), int(idx_t.shape[0]), int(val_t.shape[0]))
        use_k = min(int(idx_t.shape[1]), int(val_t.shape[1]))
        if use_b <= 0 or use_k <= 0:
            return out_idx, out_val, out_valid, used_cache

        idx_t = idx_t[:use_b, :use_k].to(device=device)
        val_t = val_t[:use_b, :use_k].to(device=device)

        valid = (
            (idx_t >= 0)
            & (idx_t < int(official_bin_n))
            & torch.isfinite(val_t)
            & (val_t > 0)
        )
        if bool(valid.any().item()):
            order = torch.argsort(val_t, dim=-1, descending=True)
            idx_sorted = torch.gather(idx_t, 1, order)
            val_sorted = torch.gather(val_t, 1, order)
            valid_sorted = torch.gather(valid, 1, order)
            keep = min(k, int(idx_sorted.shape[1]))
            out_idx[:use_b, :keep] = idx_sorted[:, :keep]
            out_val[:use_b, :keep] = val_sorted[:, :keep]
            out_valid[:use_b, :keep] = valid_sorted[:, :keep]
            used_cache = True
        return out_idx, out_val, out_valid, used_cache

    if isinstance(idx_src, (list, tuple)) and isinstance(val_src, (list, tuple)):
        use_b = min(int(batch_n), len(idx_src), len(val_src))
        for bi in range(use_b):
            try:
                idx_i = np.asarray(idx_src[bi], dtype=np.int64).reshape(-1)
                val_i = np.asarray(val_src[bi], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if idx_i.size <= 0 or val_i.size <= 0:
                continue
            use_n = min(int(idx_i.shape[0]), int(val_i.shape[0]))
            idx_i = idx_i[:use_n]
            val_i = val_i[:use_n]
            valid_i = (
                (idx_i >= 0)
                & (idx_i < int(official_bin_n))
                & np.isfinite(val_i)
                & (val_i > 0)
            )
            if not np.any(valid_i):
                continue
            idx_v = idx_i[valid_i]
            val_v = val_i[valid_i]
            order = np.argsort(-val_v, kind="stable")
            take = min(k, int(order.shape[0]))
            if take <= 0:
                continue
            sel = order[:take]
            out_idx[bi, :take] = torch.as_tensor(idx_v[sel], dtype=torch.long, device=device)
            out_val[bi, :take] = torch.as_tensor(val_v[sel], dtype=torch.float32, device=device)
            out_valid[bi, :take] = True
            used_cache = True
    return out_idx, out_val, out_valid, used_cache


def _get_teacher_formula_target_from_batch(batch):
    tq = batch.get("teacher_formula_probs", None)
    if not torch.is_tensor(tq):
        return None

    tq = tq.float()

    if tq.dim() == 1:
        tq = tq.unsqueeze(0)
    elif tq.dim() > 2:
        tq = tq.reshape(tq.shape[0], -1)

    fm = batch.get("formulae_mask", None)
    if torch.is_tensor(fm):
        fm = fm.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(tq.shape[0]), int(fm.shape[0]))
        use_m = min(int(tq.shape[1]), int(fm.shape[1]))
        if use_b <= 0 or use_m <= 0:
            return None

        tq = tq[:use_b, :use_m]
        fm = fm[:use_b, :use_m]

        tq = torch.where(torch.isfinite(tq), tq, torch.zeros_like(tq))
        tq = tq.clamp_min(0.0)
        tq = tq * (fm > 0.5).float()

        row_sum = tq.sum(dim=-1, keepdim=True)
        bad = row_sum <= 1e-12

        if bool(bad.any().item()):
            fallback = (fm > 0.5).float()
            fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            tq = torch.where(
                bad.expand_as(tq),
                fallback,
                tq / row_sum.clamp_min(1e-12),
            )
        else:
            tq = tq / row_sum.clamp_min(1e-12)

        return tq

    tq = torch.where(torch.isfinite(tq), tq, torch.zeros_like(tq))
    tq = tq.clamp_min(0.0)
    row_sum = tq.sum(dim=-1, keepdim=True)
    valid = row_sum > 1e-12

    if bool(valid.any().item()):
        tq = torch.where(
            valid.expand_as(tq),
            tq / row_sum.clamp_min(1e-12),
            torch.full_like(tq, 1.0 / max(1, int(tq.shape[1]))),
        )
        return tq

    return None


def apply_teacher_topk_to_target(target_probs, formulae_mask=None, topk=0):
    if (not torch.is_tensor(target_probs)) or int(topk) <= 0:
        return target_probs, formulae_mask

    tp = target_probs.float()
    if tp.dim() == 1:
        tp = tp.unsqueeze(0)
    elif tp.dim() > 2:
        tp = tp.reshape(tp.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(tp.shape[0]), int(fm.shape[0]))
        use_m = min(int(tp.shape[1]), int(fm.shape[1]))
        tp = tp[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        fm = (fm > 0.5).float()
    else:
        fm = torch.ones_like(tp)

    kk = max(1, min(int(topk), int(tp.shape[1])))

    score = tp.masked_fill(fm <= 0.5, -1e9)
    top_idx = torch.topk(score, k=kk, dim=-1).indices

    keep = torch.zeros_like(tp)
    keep.scatter_(1, top_idx, 1.0)
    keep = keep * fm

    try:
        pos_eps = float(os.environ.get("TEACHER_TOPK_POS_EPS", "1e-12"))
    except Exception:
        pos_eps = 1e-12

    positive = (tp > float(pos_eps)).float() * fm

    if os.environ.get("TEACHER_TOPK_MASK_POSITIVE_ONLY", "1") == "1":
        eff_mask = keep * positive

        row_has_pos = eff_mask.sum(dim=-1, keepdim=True) > 0
        eff_mask = torch.where(row_has_pos, eff_mask, keep)
    else:
        eff_mask = keep

    tp_masked = tp * eff_mask

    row_sum = tp_masked.sum(dim=-1, keepdim=True)

    fallback = eff_mask / eff_mask.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    tp_out = torch.where(
        row_sum > 1e-12,
        tp_masked / row_sum.clamp_min(1e-12),
        fallback,
    )

    return tp_out, eff_mask


def compute_formula_target_probs_from_batch(
    batch,
    bin_width=0.1,
    max_mz=1005.0,
    target_mode="exact_overlap",
    support_temperature=1.0,
    support_topk=0,
):
    use_teacher = str(os.environ.get("USE_TEACHER_FORMULA_TARGET", "1")).strip().lower() not in (
        "0",
        "false",
        "no",
        "off",
    )

    if use_teacher:
        teacher_target = _get_teacher_formula_target_from_batch(batch)
        if torch.is_tensor(teacher_target):
            return teacher_target

    del target_mode

    off_idx = batch.get("formulae_peaks_official_idx_agg", None)
    off_int = batch.get("formulae_peaks_official_intensity_agg", None)
    if off_idx is None:
        off_idx = batch.get("formulae_peaks_official_idx", None)
    if off_int is None:
        off_int = get_formulae_official_intensity_from_batch(batch)

    if not (torch.is_tensor(off_idx) and torch.is_tensor(off_int)):
        return None

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)
    if off_idx.dim() != 3 or off_int.dim() != 3:
        return None

    batch_n = min(int(off_idx.shape[0]), int(off_int.shape[0]))
    formula_n = min(int(off_idx.shape[1]), int(off_int.shape[1]))
    peak_n = min(int(off_idx.shape[2]), int(off_int.shape[2]))
    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None

    device = off_idx.device
    off_idx = off_idx[:batch_n, :formula_n, :peak_n].long()
    off_int = off_int[:batch_n, :formula_n, :peak_n].float()

    formulae_mask = batch.get("formulae_mask", None)
    if torch.is_tensor(formulae_mask):
        mask = formulae_mask.float()
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], -1)
        use_b = min(batch_n, int(mask.shape[0]))
        use_m = min(formula_n, int(mask.shape[1]))
        off_idx = off_idx[:use_b, :use_m, :]
        off_int = off_int[:use_b, :use_m, :]
        mask = mask[:use_b, :use_m]
        batch_n = use_b
        formula_n = use_m
        peak_n = min(peak_n, int(off_idx.shape[2]), int(off_int.shape[2]))
    else:
        mask = torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)

    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None

    try:
        temp = float(support_temperature)
    except Exception:
        temp = 1.0
    if (not np.isfinite(temp)) or temp <= 0:
        temp = 1.0

    try:
        topk = int(support_topk)
    except Exception:
        topk = 0
    topk = max(0, topk)

    bwidth = float(max(1e-6, bin_width))
    max_mz = float(max(bwidth, max_mz))
    official_bin_n = int(math.floor(max_mz / bwidth)) + 1

    true_dense, used_cached_true = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )

    if not used_cached_true:
        raw_batch = batch.get("spect_raw", None)
        if not getattr(compute_formula_target_probs_from_batch, "_warned_raw_fallback", False):
            print(
                "[target] missing cached true_official_idx/intensity, fallback to spect_raw target build; "
                "rebuild cache via cache_featurizer_condv2 for official cache mode.",
                flush=True,
            )
            compute_formula_target_probs_from_batch._warned_raw_fallback = True
        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=raw_batch,
            precursor_mz=batch.get("precursor_mz", None),
            official_bin_width=bwidth,
            official_max_mz=max_mz,
            exclude_precursor=True,
            batch_n=batch_n,
            device=device,
        )

    true_norm = torch.sqrt((true_dense ** 2).sum(dim=-1).clamp_min(1e-12))
    true_support = true_dense > 0

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )
    safe_idx = off_idx.clamp(0, max(0, int(official_bin_n) - 1))

    true_at_peak = torch.gather(
        true_dense,
        1,
        safe_idx.reshape(batch_n, -1),
    ).reshape(batch_n, formula_n, peak_n)

    candidate_dot = (true_at_peak * off_int * valid_peak.float()).sum(dim=-1)
    cand_norm = torch.sqrt(((off_int * valid_peak.float()) ** 2).sum(dim=-1).clamp_min(1e-12))
    candidate_overlap = candidate_dot / (cand_norm * true_norm.unsqueeze(-1) + 1e-12)

    support_at_peak = torch.gather(
        true_support.float(),
        1,
        safe_idx.reshape(batch_n, -1),
    ).reshape(batch_n, formula_n, peak_n)
    overlap_support_score = (support_at_peak * valid_peak.float()).sum(dim=-1) / valid_peak.float().sum(
        dim=-1
    ).clamp_min(1.0)

    candidate_int_precision_score = (
        support_at_peak * off_int * valid_peak.float()
    ).sum(dim=-1) / (
        off_int * valid_peak.float()
    ).sum(dim=-1).clamp_min(1e-8)
    top20_idx, top20_val, top20_valid, used_cached_top20 = _build_cached_true_top20_tensors(
        batch=batch,
        batch_n=batch_n,
        official_bin_n=official_bin_n,
        device=device,
        default_k=20,
    )
    if not used_cached_top20:
        if not getattr(compute_formula_target_probs_from_batch, "_warned_top20_fallback", False):
            print(
                "[target] missing cached true_top20_official_idx/intensity, fallback to dense topk(true_official).",
                flush=True,
            )
            compute_formula_target_probs_from_batch._warned_top20_fallback = True
        k20_dense = min(20, int(true_dense.shape[-1]))
        if k20_dense > 0:
            dense_top_val, dense_top_idx = torch.topk(true_dense, k=k20_dense, dim=-1)
            top20_idx[:, :k20_dense] = dense_top_idx
            top20_val[:, :k20_dense] = dense_top_val
            top20_valid[:, :k20_dense] = dense_top_val > 0

    try:
        weak_thr = float(os.environ.get("QUALITY_HYBRID_WEAK_INTENSITY_MAX", "0.05"))
    except Exception:
        weak_thr = 0.05
    if (not np.isfinite(weak_thr)) or weak_thr <= 0:
        weak_thr = 0.05

    k20 = int(top20_idx.shape[1])
    hit_top20_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    weak_hit_top20_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    hit_top20_intensity_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    weak_top20_valid = top20_valid & (top20_val <= float(weak_thr))

    if k20 > 0:
        for kk in range(k20):
            tk = top20_idx[:, kk].view(batch_n, 1, 1)
            tk_valid = top20_valid[:, kk].view(batch_n, 1)
            hit_k = ((safe_idx == tk) & valid_peak).any(dim=-1).float()
            hit_top20_score += hit_k * tk_valid.float()
            hit_top20_intensity_score += hit_k * tk_valid.float() * top20_val[:, kk].view(batch_n, 1)
            tk_weak = weak_top20_valid[:, kk].view(batch_n, 1)
            weak_hit_top20_score += hit_k * tk_weak.float()
        hit_top20_score = hit_top20_score / float(max(1, k20))
        top20_int_den = (top20_val * top20_valid.float()).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hit_top20_intensity_score = hit_top20_intensity_score / top20_int_den
        weak_den = weak_top20_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        weak_hit_top20_score = weak_hit_top20_score / weak_den
        no_weak = weak_top20_valid.float().sum(dim=-1, keepdim=True) <= 0
        weak_hit_top20_score = torch.where(no_weak, hit_top20_score, weak_hit_top20_score)

    q1 = candidate_overlap * mask
    q2 = hit_top20_score * mask
    q3 = overlap_support_score * mask
    q4 = weak_hit_top20_score * mask
    q5 = hit_top20_intensity_score * mask
    q6 = candidate_int_precision_score * mask

    if os.environ.get("QUALITY_HYBRID_NORMALIZE", "1") == "1":

        def _row_minmax(x):
            x_safe = torch.where(mask > 0, x, torch.full_like(x, float("inf")))
            row_min = torch.amin(x_safe, dim=-1, keepdim=True)
            row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))

            x_safe_max = torch.where(mask > 0, x, torch.full_like(x, float("-inf")))
            row_max = torch.amax(x_safe_max, dim=-1, keepdim=True)
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

            denom = (row_max - row_min).clamp_min(1e-8)
            out = (x - row_min) / denom
            return torch.where(mask > 0, out, torch.zeros_like(out))

        q1 = _row_minmax(q1)
        q2 = _row_minmax(q2)
        q3 = _row_minmax(q3)
        q4 = _row_minmax(q4)
        q5 = _row_minmax(q5)
        q6 = _row_minmax(q6)

    w1 = float(os.environ.get("QUALITY_HYBRID_W_COS", "0.7"))
    w2 = float(os.environ.get("QUALITY_HYBRID_W_HIT20", "0.2"))
    w3 = float(os.environ.get("QUALITY_HYBRID_W_OVERLAP", "0.1"))
    w4 = float(os.environ.get("QUALITY_HYBRID_W_WEAK20", "0.0"))
    w5 = float(os.environ.get("QUALITY_HYBRID_W_HIT20_INT", "0.0"))
    w6 = float(os.environ.get("QUALITY_HYBRID_W_PREC_INT", "0.0"))

    candidate_overlap = (
        w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4 + w5 * q5 + w6 * q6
    ) * mask

    try:
        source_instance_teacher_bonus = float(os.environ.get("SOURCE_INSTANCE_TEACHER_BONUS", "0.0"))
    except Exception:
        source_instance_teacher_bonus = 0.0

    if source_instance_teacher_bonus > 0.0:
        inst_src = batch.get("formulae_instance_is_source", None)
        if torch.is_tensor(inst_src):
            inst_src = inst_src.to(device=candidate_overlap.device, dtype=candidate_overlap.dtype)
            if inst_src.dim() == 1:
                inst_src = inst_src.unsqueeze(0)
            if inst_src.dim() > 2:
                inst_src = inst_src.reshape(inst_src.shape[0], -1)

            use_b = min(int(candidate_overlap.shape[0]), int(inst_src.shape[0]))
            use_m = min(int(candidate_overlap.shape[1]), int(inst_src.shape[1]))

            bonus = torch.zeros_like(candidate_overlap)
            bonus[:use_b, :use_m] = inst_src[:use_b, :use_m] * float(source_instance_teacher_bonus)
            candidate_overlap = candidate_overlap + bonus * mask.float()

    independent_teacher_scores = candidate_overlap.float().clamp_min(0.0) * mask.float()
    independent_sum = independent_teacher_scores.sum(dim=-1, keepdim=True)

    independent_fallback = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)

    independent_teacher_probs = torch.where(
        independent_sum > 1e-12,
        independent_teacher_scores / independent_sum.clamp_min(1e-8),
        independent_fallback,
    )

    if str(os.environ.get("QUALITY_USE_SETCOVER_TEACHER", "0")).strip() == "1":
        with torch.no_grad():
            try:
                sc_topk = int(os.environ.get("QUALITY_SETCOVER_TOPK", "16"))
            except Exception:
                sc_topk = 16

            try:
                pool_k = int(os.environ.get("QUALITY_SETCOVER_POOL_TOPK", "1024"))
            except Exception:
                pool_k = 1024

            try:
                lambda_false = float(os.environ.get("QUALITY_SETCOVER_LAMBDA_FALSE", "0.5"))
            except Exception:
                lambda_false = 0.5

            try:
                lambda_redun = float(os.environ.get("QUALITY_SETCOVER_LAMBDA_REDUN", "0.2"))
            except Exception:
                lambda_redun = 0.2

            try:
                min_gain = float(os.environ.get("QUALITY_SETCOVER_MIN_GAIN", "1e-8"))
            except Exception:
                min_gain = 1e-8

            pool_k = max(1, min(int(pool_k), int(formula_n)))
            sc_topk = max(1, min(int(sc_topk), int(pool_k)))

            neg_val = _neg_mask_fill_value(candidate_overlap)
            rank_score = candidate_overlap.masked_fill(mask <= 0, neg_val)

            pool_idx_all = torch.topk(rank_score, k=pool_k, dim=-1).indices
            sel_probs = torch.zeros_like(candidate_overlap)

            for bi in range(int(batch_n)):
                pool_idx = pool_idx_all[bi]
                pool_valid = mask[bi, pool_idx] > 0

                idx_pool = off_idx[bi, pool_idx].long()
                int_pool = off_int[bi, pool_idx].float()
                valid_pool = valid_peak[bi, pool_idx].bool()

                valid_pool = (
                    valid_pool
                    & (idx_pool >= 0)
                    & (idx_pool < int(official_bin_n))
                    & torch.isfinite(int_pool)
                    & (int_pool > 0)
                    & pool_valid.unsqueeze(-1)
                )

                idx_safe = idx_pool.clamp(0, max(0, int(official_bin_n) - 1))
                int_pool = int_pool * valid_pool.float()

                cand_tot = int_pool.sum(dim=-1)
                true_hit = (true_at_peak[bi, pool_idx].float() * int_pool).sum(dim=-1)
                cand_false_mass = (cand_tot - true_hit).clamp_min(0.0)

                residual = true_dense[bi].float().clone()
                selected_bins_dense = torch.zeros_like(residual)

                selected_local = []
                selected_gain_vals = []
                selected_mask = torch.zeros((pool_k,), dtype=torch.bool, device=device)

                for step in range(sc_topk):
                    res_vals = residual[idx_safe]
                    gain = torch.minimum(res_vals, int_pool).sum(dim=-1)

                    sel_vals = selected_bins_dense[idx_safe]
                    redun = (sel_vals * int_pool).sum(dim=-1)

                    score = (
                        gain
                        - float(lambda_false) * cand_false_mass
                        - float(lambda_redun) * redun
                        + 1e-4 * rank_score[bi, pool_idx].clamp_min(0.0)
                    )

                    score = score.masked_fill(~pool_valid, neg_val)
                    score = score.masked_fill(selected_mask, neg_val)
                    score = score.masked_fill(cand_tot <= 0, neg_val)

                    best_score, best_local_t = torch.max(score, dim=0)

                    if not bool(torch.isfinite(best_score).item()):
                        break

                    best_local = int(best_local_t.detach().item())
                    best_gain = float(gain[best_local].detach().item())
                    if step > 0 and best_gain <= float(min_gain):
                        break

                    selected_mask[best_local] = True
                    selected_local.append(best_local)
                    selected_gain_vals.append(gain[best_local].clamp_min(0.0))

                    sel_valid = valid_pool[best_local]
                    if bool(sel_valid.any().item()):
                        sel_idxs = idx_safe[best_local, sel_valid]
                        sel_ints = int_pool[best_local, sel_valid]

                        delta = torch.zeros_like(residual)
                        delta.scatter_add_(0, sel_idxs, sel_ints)
                        residual = torch.clamp(residual - delta, min=0.0)

                        selected_bins_dense.scatter_add_(0, sel_idxs, sel_ints)

                if len(selected_local) > 0:
                    selected_orig_idx = pool_idx[
                        torch.as_tensor(selected_local, dtype=torch.long, device=device)
                    ]

                    gains = torch.stack(selected_gain_vals).float()
                    if float(gains.sum().detach().item()) <= 1e-12:
                        gains = candidate_overlap[bi, selected_orig_idx].float().clamp_min(1e-8)

                    weights = gains / gains.sum().clamp_min(1e-8)
                    sel_probs[bi, selected_orig_idx] = weights

            sel_sum = sel_probs.sum(dim=-1, keepdim=True)

            if os.environ.get("QUALITY_SETCOVER_HYBRID", "0").strip() == "1":
                try:
                    independent_w = float(os.environ.get("QUALITY_SETCOVER_HYBRID_Q6_WEIGHT", "0.7"))
                except Exception:
                    independent_w = 0.7
                independent_w = float(np.clip(independent_w, 0.0, 1.0))

                setcover_probs = torch.where(
                    sel_sum > 0,
                    sel_probs / sel_sum.clamp_min(1e-8),
                    independent_teacher_probs,
                )

                hybrid_probs = independent_w * independent_teacher_probs + (1.0 - independent_w) * setcover_probs
                hybrid_probs = hybrid_probs * mask.float()
                hybrid_probs = hybrid_probs / hybrid_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                candidate_overlap = hybrid_probs
            else:
                candidate_overlap = torch.where(sel_sum > 0, sel_probs, candidate_overlap)

    try:
        target_gamma = float(os.environ.get("QUALITY_TARGET_GAMMA", "2.0"))
    except Exception:
        target_gamma = 2.0
    if (not np.isfinite(target_gamma)) or target_gamma <= 0:
        target_gamma = 1.5

    eff_gamma = float(target_gamma) * float(1.0 / temp)

    if abs(eff_gamma - 1.0) > 1e-8:
        positive = candidate_overlap > 0
        candidate_overlap = torch.where(
            positive,
            torch.pow(candidate_overlap.clamp_min(1e-12), eff_gamma),
            candidate_overlap * 0.0,
        )
    if topk > 0 and int(candidate_overlap.shape[1]) > topk:
        k = int(min(topk, int(candidate_overlap.shape[1])))
        rank_score = candidate_overlap.masked_fill(mask <= 0, _neg_mask_fill_value(candidate_overlap))
        top_idx = torch.topk(rank_score, k=k, dim=-1).indices
        keep = torch.zeros_like(candidate_overlap)
        keep.scatter_(1, top_idx, 1.0)
        candidate_overlap = candidate_overlap * keep

    overlap_sum = candidate_overlap.sum(dim=-1, keepdim=True)
    valid_sum = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    target_probs = torch.where(
        overlap_sum > 0,
        candidate_overlap / overlap_sum.clamp_min(1e-8),
        mask / valid_sum,
    )
    return target_probs
