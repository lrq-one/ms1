import math
import os

import numpy as np
import torch


def _extract_precursor_mz_at(precursor_mz, i):
    if precursor_mz is None:
        return None
    try:
        if torch.is_tensor(precursor_mz):
            flat = precursor_mz.detach().reshape(-1)
            if i < int(flat.shape[0]):
                v = float(flat[i].cpu().item())
                return v if np.isfinite(v) else None
            return None
        if isinstance(precursor_mz, np.ndarray):
            flat = precursor_mz.reshape(-1)
            if i < int(flat.shape[0]):
                v = float(flat[i])
                return v if np.isfinite(v) else None
            return None
        if isinstance(precursor_mz, (list, tuple)):
            if i < len(precursor_mz):
                v = float(precursor_mz[i])
                return v if np.isfinite(v) else None
            return None
        v = float(precursor_mz)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def build_true_official_dense_from_raw(
    spect_raw_list,
    precursor_mz,
    official_bin_width,
    official_max_mz,
    exclude_precursor,
    batch_n,
    device,
):
    """Build dense official-bin target spectra from raw sparse peaks (target-only, non-differentiable path)."""
    bin_width = float(max(1e-6, official_bin_width))
    max_mz = float(max(bin_width, official_max_mz))
    official_bin_n = int(math.floor(max_mz / bin_width)) + 1

    out = torch.zeros((int(batch_n), official_bin_n), dtype=torch.float32, device=device)
    if not isinstance(spect_raw_list, (list, tuple)):
        return out

    use_n = min(int(batch_n), len(spect_raw_list))
    for i in range(use_n):
        peaks = _to_sparse_peak_array(spect_raw_list[i], spect_bin_centers=None, min_intensity=0.0)
        if peaks.size == 0:
            continue

        mz = peaks[:, 0].astype(np.float64)
        intensity = peaks[:, 1].astype(np.float64)

        valid = (
            np.isfinite(mz)
            & np.isfinite(intensity)
            & (intensity > 0)
            & (mz >= 0.0)
            & (mz < max_mz)
        )
        mz = mz[valid]
        intensity = intensity[valid]
        if mz.size == 0:
            continue

        idx = _official_bin_indices_np(mz, bin_width).astype(np.int64)
        if exclude_precursor:
            pmz = _extract_precursor_mz_at(precursor_mz, i)
            keep = _precursor_keep_mask_np(
                mz=mz,
                precursor_mz=pmz,
                bin_width=bin_width,
                exclude_precursor=True,
            )
            idx = idx[keep]
            intensity = intensity[keep]
        if idx.size == 0:
            continue

        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device).clamp(0, max(0, official_bin_n - 1))
        val_t = torch.as_tensor(intensity.astype(np.float32), dtype=torch.float32, device=device)
        out[i].scatter_add_(0, idx_t, val_t)

    return out


def _to_sparse_peak_array(x, spect_bin_centers=None, min_intensity=0.0):
    """Convert mixed spectrum representations to sparse (mz, intensity) peaks."""
    if x is None:
        return np.zeros((0, 2), dtype=np.float32)

    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(e, dtype=np.float32) for e in arr], axis=0)
        except Exception:
            arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        if spect_bin_centers is None:
            return np.zeros((0, 2), dtype=np.float32)
        vec = np.asarray(arr, dtype=np.float32)
        n = min(vec.shape[0], spect_bin_centers.shape[0])
        vec = vec[:n]
        idx = np.nonzero(vec > float(min_intensity))[0]
        if idx.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([
            spect_bin_centers[idx].astype(np.float32),
            vec[idx].astype(np.float32),
        ], axis=-1)

    if arr.ndim == 2 and arr.shape[1] == 2:
        out = np.asarray(arr, dtype=np.float32)
        valid = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & (out[:, 1] > float(min_intensity))
        out = out[valid]
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return out

    try:
        out = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        valid = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & (out[:, 1] > float(min_intensity))
        out = out[valid]
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return out
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)



def _official_bin_indices_np(mz, bin_width):
    bw = float(max(1e-6, float(bin_width)))
    mode = str(os.environ.get("OFFICIAL_BIN_MODE", "floor")).strip().lower()
    arr = np.asarray(mz, dtype=np.float64)
    if mode in ("round", "nearest", "nominal"):
        return np.rint(arr / bw).astype(np.int64)
    return np.floor(arr / bw + 1e-8).astype(np.int64)


def _precursor_keep_mask_np(mz, precursor_mz, bin_width, exclude_precursor):
    mz = np.asarray(mz, dtype=np.float64)
    keep = np.ones((mz.shape[0],), dtype=bool)
    if not bool(exclude_precursor):
        return keep

    try:
        pmz = float(precursor_mz)
    except Exception:
        return keep

    if not np.isfinite(pmz) or pmz <= 0:
        return keep

    try:
        tol_da = float(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.0"))
    except Exception:
        tol_da = 0.0
    try:
        isotope_n = int(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "0"))
    except Exception:
        isotope_n = 0

    if tol_da > 0.0:
        for iso_k in range(max(0, isotope_n) + 1):
            keep &= (np.abs(mz - (pmz + float(iso_k))) > float(tol_da))
    else:
        idx = _official_bin_indices_np(mz, bin_width)
        p_idx = int(_official_bin_indices_np(np.asarray([pmz], dtype=np.float64), bin_width)[0])
        keep &= (idx != p_idx)

    return keep
