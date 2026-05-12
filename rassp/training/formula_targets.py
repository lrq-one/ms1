import math
import os

import torch

from rassp.training.spectrum_targets import build_true_official_dense_from_raw


def get_formulae_official_intensity_from_batch(batch):
    """
    Prefer official-bin intensity tensor when present; fallback to legacy intensity.
    """
    if not isinstance(batch, dict):
        return None

    off_int = batch.get("formulae_peaks_official_intensity_agg", None)
    if torch.is_tensor(off_int) and off_int.dim() == 3:
        return off_int

    off_int = batch.get("formulae_peaks_official_intensity", None)
    if torch.is_tensor(off_int) and off_int.dim() == 3:
        return off_int

    return batch.get("formulae_peaks_intensity", None)


def _get_formulae_official_idx_from_batch(batch):
    if not isinstance(batch, dict):
        return None

    off_idx = batch.get("formulae_peaks_official_idx_agg", None)
    if torch.is_tensor(off_idx) and off_idx.dim() == 3:
        return off_idx

    off_idx = batch.get("formulae_peaks_official_idx", None)
    if torch.is_tensor(off_idx) and off_idx.dim() == 3:
        return off_idx

    return batch.get("formulae_peaks_mass_idx", None)


def _infer_batch_n(batch):
    if not isinstance(batch, dict):
        return 0

    for key in [
        "vect_feat",
        "formulae_features",
        "formulae_peaks_official_idx",
        "formulae_peaks_official_idx_agg",
        "true_official_idx",
        "true_all_official_idx",
    ]:
        v = batch.get(key, None)
        if torch.is_tensor(v) and v.dim() >= 1:
            return int(v.shape[0])
        if isinstance(v, (list, tuple)):
            return len(v)

    return 0


def _to_2d_sparse_tensor(x, batch_n, device, dtype):
    """
    Convert cached sparse target idx/intensity to [B, K] tensor.
    Supports tensor or list-of-arrays/list-of-tensors.
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        t = x.to(device=device, dtype=dtype)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        elif t.dim() > 2:
            t = t.reshape(t.shape[0], -1)

        if int(t.shape[0]) < int(batch_n):
            pad_shape = (int(batch_n) - int(t.shape[0]), int(t.shape[1]))
            pad = torch.zeros(pad_shape, dtype=dtype, device=device)
            t = torch.cat([t, pad], dim=0)

        return t[: int(batch_n)]

    if not isinstance(x, (list, tuple)):
        return None

    rows = []
    max_len = 0

    for i in range(min(int(batch_n), len(x))):
        item = x[i]
        try:
            if torch.is_tensor(item):
                row = item.detach().to(device=device, dtype=dtype).reshape(-1)
            else:
                row = torch.as_tensor(item, dtype=dtype, device=device).reshape(-1)
        except Exception:
            row = torch.zeros((0,), dtype=dtype, device=device)

        rows.append(row)
        max_len = max(max_len, int(row.numel()))

    if max_len <= 0:
        return torch.zeros((int(batch_n), 0), dtype=dtype, device=device)

    out = torch.zeros((int(batch_n), max_len), dtype=dtype, device=device)

    for i, row in enumerate(rows):
        if row.numel() > 0:
            out[i, : int(row.numel())] = row

    return out


def _build_true_official_dense_from_cached_sparse_batch(
    batch,
    batch_n,
    device,
    official_bin_n,
):
    """
    Build dense official-bin target from cached sparse official target.

    Returns:
        dense: [B, official_bin_n]
        used_cache: bool
    """
    if not isinstance(batch, dict):
        return torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device), False

    idx_obj = batch.get("true_all_official_idx", None)
    val_obj = batch.get("true_all_official_intensity", None)

    if idx_obj is None or val_obj is None:
        idx_obj = batch.get("true_official_idx", None)
        val_obj = batch.get("true_official_intensity", None)

    if idx_obj is None or val_obj is None:
        idx_obj = batch.get("true_top20_official_idx", None)
        val_obj = batch.get("true_top20_official_intensity", None)

    if idx_obj is None or val_obj is None:
        return torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device), False

    idx = _to_2d_sparse_tensor(idx_obj, batch_n, device=device, dtype=torch.long)
    val = _to_2d_sparse_tensor(val_obj, batch_n, device=device, dtype=torch.float32)

    if not (torch.is_tensor(idx) and torch.is_tensor(val)):
        return torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device), False

    use_b = min(int(batch_n), int(idx.shape[0]), int(val.shape[0]))
    use_k = min(int(idx.shape[1]), int(val.shape[1])) if idx.dim() >= 2 and val.dim() >= 2 else 0

    dense = torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device)

    if use_b <= 0 or use_k <= 0:
        return dense, False

    idx = idx[:use_b, :use_k].long()
    val = val[:use_b, :use_k].float()

    valid = (
        (idx >= 0)
        & (idx < int(official_bin_n))
        & torch.isfinite(val)
        & (val > 0)
    )

    safe_idx = idx.clamp(0, max(0, int(official_bin_n) - 1))
    dense[:use_b].scatter_add_(1, safe_idx, torch.where(valid, val, torch.zeros_like(val)))

    return dense, bool(valid.any().item())


build_true_official_dense_from_cached_sparse_batch = _build_true_official_dense_from_cached_sparse_batch


def _build_cached_true_top20_tensors(batch, batch_n=None, device=None):
    """
    Return cached true top20 sparse official target tensors if available.
    """
    if not isinstance(batch, dict):
        return None, None

    if batch_n is None:
        batch_n = _infer_batch_n(batch)

    if device is None:
        for v in batch.values():
            if torch.is_tensor(v):
                device = v.device
                break
        if device is None:
            device = torch.device("cpu")

    idx_obj = batch.get("true_top20_official_idx", None)
    val_obj = batch.get("true_top20_official_intensity", None)

    if idx_obj is None or val_obj is None:
        idx_obj = batch.get("true_official_idx", None)
        val_obj = batch.get("true_official_intensity", None)

    if idx_obj is None or val_obj is None:
        return None, None

    idx = _to_2d_sparse_tensor(idx_obj, int(batch_n), device=device, dtype=torch.long)
    val = _to_2d_sparse_tensor(val_obj, int(batch_n), device=device, dtype=torch.float32)

    return idx, val


build_cached_true_top20_tensors = _build_cached_true_top20_tensors


def _normalize_target_probs(target, formulae_mask=None, eps=1e-12):
    if not torch.is_tensor(target):
        return None

    out = torch.where(torch.isfinite(target), target, torch.zeros_like(target)).float()
    out = out.clamp_min(0.0)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float().to(device=out.device)
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(out.shape[0]), int(fm.shape[0]))
        use_m = min(int(out.shape[1]), int(fm.shape[1]))

        out = out[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        out = out * (fm > 0.5).float()
    else:
        fm = torch.ones_like(out)

    row_sum = out.sum(dim=-1, keepdim=True)
    fallback = fm / fm.sum(dim=-1, keepdim=True).clamp_min(float(eps))

    out = torch.where(row_sum > float(eps), out / row_sum.clamp_min(float(eps)), fallback)
    return out


def _get_teacher_formula_target_from_batch(batch, formulae_mask=None):
    """
    Read cached teacher_formula_probs if present.
    """
    if not isinstance(batch, dict):
        return None

    teacher = batch.get("teacher_formula_probs", None)
    if not torch.is_tensor(teacher):
        return None

    if teacher.dim() == 1:
        teacher = teacher.unsqueeze(0)
    elif teacher.dim() > 2:
        teacher = teacher.reshape(teacher.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float().to(device=teacher.device)
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(teacher.shape[0]), int(fm.shape[0]))
        use_m = min(int(teacher.shape[1]), int(fm.shape[1]))

        teacher = teacher[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
    else:
        fm = None

    return _normalize_target_probs(teacher, fm)


def apply_teacher_topk_to_target(
    target_probs,
    formulae_mask=None,
    topk=64,
    eps=1e-12,
):
    """
    Keep only target topK and renormalize.
    """
    if not torch.is_tensor(target_probs):
        return target_probs

    target = target_probs.float()

    if target.dim() == 1:
        target = target.unsqueeze(0)
    elif target.dim() > 2:
        target = target.reshape(target.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float().to(device=target.device)
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(target.shape[0]), int(fm.shape[0]))
        use_m = min(int(target.shape[1]), int(fm.shape[1]))

        target = target[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        target = target * (fm > 0.5).float()
    else:
        fm = torch.ones_like(target)

    k = max(1, min(int(topk), int(target.shape[1])))

    masked = target.masked_fill(fm <= 0.5, 0.0)
    idx = torch.topk(masked, k=k, dim=1).indices

    keep = torch.zeros_like(masked)
    keep.scatter_(1, idx, 1.0)

    out = masked * keep
    return _normalize_target_probs(out, fm, eps=eps)


def compute_formula_target_probs_from_batch(
    batch,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    support_topk=64,
    exclude_precursor=True,
    eps=1e-12,
    **kwargs,
):
    """
    Build formula-level target probability distribution.

    Priority:
      1. If FORMULA_TARGET_MODE=teacher and teacher_formula_probs exists, use cached teacher.
      2. Otherwise compute overlap quality between candidate official peaks and true official spectrum.
    """
    if not isinstance(batch, dict):
        return None

    formulae_mask = batch.get("formulae_mask", None)
    teacher = _get_teacher_formula_target_from_batch(batch, formulae_mask=formulae_mask)

    mode = os.environ.get("FORMULA_TARGET_MODE", "overlap").strip().lower()
    if mode in {"teacher", "cached_teacher", "teacher_probs"} and torch.is_tensor(teacher):
        try:
            tk = int(os.environ.get("TARGET_SUPPORT_TOPK", str(support_topk)))
        except Exception:
            tk = int(support_topk)
        return apply_teacher_topk_to_target(teacher, formulae_mask=formulae_mask, topk=tk, eps=eps)

    off_idx = _get_formulae_official_idx_from_batch(batch)
    off_int = get_formulae_official_intensity_from_batch(batch)

    if not (torch.is_tensor(off_idx) and torch.is_tensor(off_int)):
        if torch.is_tensor(teacher):
            return teacher
        return None

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    device = off_idx.device
    batch_n = min(int(off_idx.shape[0]), int(off_int.shape[0]))
    formula_n = min(int(off_idx.shape[1]), int(off_int.shape[1]))
    peak_n = min(int(off_idx.shape[2]), int(off_int.shape[2]))

    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None

    off_idx = off_idx[:batch_n, :formula_n, :peak_n].long()
    off_int = off_int[:batch_n, :formula_n, :peak_n].float()

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float().to(device=device)
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(batch_n, int(fm.shape[0]))
        use_m = min(formula_n, int(fm.shape[1]))

        off_idx = off_idx[:use_b, :use_m, :]
        off_int = off_int[:use_b, :use_m, :]
        fm = fm[:use_b, :use_m]
        batch_n = use_b
        formula_n = use_m
    else:
        fm = torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)

    official_bin_width = float(max(1e-6, float(official_bin_width)))
    official_max_mz = float(max(official_bin_width, float(official_max_mz)))
    official_bin_n = int(math.floor(official_max_mz / official_bin_width)) + 1

    true_dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )

    if not used_cache:
        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get("spect_raw", None),
            precursor_mz=batch.get("precursor_mz", None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=bool(exclude_precursor),
            batch_n=batch_n,
            device=device,
        )

    true_dense = true_dense[:batch_n].to(device=device)

    valid = (
        (off_idx >= 0)
        & (off_idx < official_bin_n)
        & torch.isfinite(off_int)
        & (off_int > 0)
        & (fm > 0.5).unsqueeze(-1)
    )

    safe_idx = off_idx.clamp(0, max(0, official_bin_n - 1))
    true_at_peak = torch.gather(
        true_dense,
        1,
        safe_idx.reshape(batch_n, -1),
    ).reshape(batch_n, formula_n, peak_n)

    candidate_mass = torch.where(valid, off_int.clamp_min(0.0), torch.zeros_like(off_int)).sum(dim=-1)
    hit_mass = torch.where(valid, off_int.clamp_min(0.0) * true_at_peak.clamp_min(0.0), torch.zeros_like(off_int)).sum(dim=-1)

    # Normalize candidate score to avoid favoring candidates with many peaks only.
    quality = hit_mass / candidate_mass.clamp_min(float(eps))
    quality = torch.where(torch.isfinite(quality), quality, torch.zeros_like(quality))
    quality = quality * (fm > 0.5).float()

    # If everything is zero, fallback to teacher if available; otherwise uniform over valid candidates.
    if torch.is_tensor(teacher):
        teacher = teacher.to(device=quality.device, dtype=quality.dtype)
        use_b = min(int(teacher.shape[0]), int(quality.shape[0]))
        use_m = min(int(teacher.shape[1]), int(quality.shape[1]))
        quality[:use_b, :use_m] = torch.where(
            quality[:use_b, :use_m] > 0,
            quality[:use_b, :use_m],
            teacher[:use_b, :use_m],
        )

    target = _normalize_target_probs(quality, fm, eps=eps)

    try:
        topk = int(os.environ.get("TARGET_SUPPORT_TOPK", str(support_topk)))
    except Exception:
        topk = int(support_topk)

    if topk > 0:
        target = apply_teacher_topk_to_target(target, formulae_mask=fm, topk=topk, eps=eps)

    return target
