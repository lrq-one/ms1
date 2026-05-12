#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cache quality probe for formula/MS cache.

Purpose
-------
Sample ~30 cache .pkl files, inspect whether candidate formula official peaks
can explain the true spectrum, and whether formulae_aux_feat carries useful
signal for candidate quality.

Outputs
-------
1) sample_summary.csv        per-sample oracle/cache-quality summary
2) candidate_topk.csv        per-sample top candidate rows by cosine
3) summary.png               global distribution figure
4) gallery.png               multi-sample overlay gallery
5) sample_XXXX_overlay.png   per-sample overlay plots
6) run_config.json           resolved arguments / chosen files

Notes
-----
- Main cache-quality score uses aggregated official peaks if available
  (formulae_peaks_official_idx_agg / intensity_agg), else falls back to
  non-aggregated official peaks.
- Auxiliary features are not spectra. We therefore assess them indirectly by
  their correlation with candidate quality scores (cosine / coverage), not by
  direct peakwise similarity.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.optimize import nnls  # type: ignore
except Exception:
    nnls = None


# ---------------------------- basic helpers ----------------------------

def _to_numpy(x, dtype=None):
    if x is None:
        return None
    try:
        import torch
        if torch.is_tensor(x):
            arr = x.detach().cpu().numpy()
        else:
            arr = np.asarray(x)
    except Exception:
        arr = np.asarray(x)
    if dtype is not None:
        try:
            arr = arr.astype(dtype, copy=False)
        except Exception:
            arr = np.asarray(arr, dtype=dtype)
    return arr


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _rankdata_average(a: np.ndarray) -> np.ndarray:
    """Simple average-rank implementation; enough for Spearman on small dims."""
    a = np.asarray(a)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size <= 1 or y.size <= 1:
        return np.nan
    x = x - x.mean()
    y = y - y.mean()
    dx = float(np.linalg.norm(x))
    dy = float(np.linalg.norm(y))
    if dx <= 1e-12 or dy <= 1e-12:
        return np.nan
    return float(np.dot(x, y) / (dx * dy + 1e-12))


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1 or len(y) <= 1:
        return np.nan
    rx = _rankdata_average(np.asarray(x, dtype=float))
    ry = _rankdata_average(np.asarray(y, dtype=float))
    return _pearson(rx, ry)


# ---------------------------- spectrum helpers ----------------------------

def sparse_to_dense(idx: np.ndarray, val: np.ndarray, bin_n: int) -> np.ndarray:
    out = np.zeros((int(bin_n),), dtype=np.float32)
    if idx is None or val is None:
        return out
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    val = np.asarray(val, dtype=np.float32).reshape(-1)
    if idx.size == 0 or val.size == 0:
        return out
    use_n = min(idx.size, val.size)
    idx = idx[:use_n]
    val = val[:use_n]
    valid = np.isfinite(val) & (val > 0) & (idx >= 0) & (idx < int(bin_n))
    if not np.any(valid):
        return out
    np.add.at(out, idx[valid], val[valid])
    return out


def dense_to_sparse(d: np.ndarray, thr: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    d = np.asarray(d, dtype=np.float32).reshape(-1)
    valid = np.isfinite(d) & (d > float(thr))
    idx = np.nonzero(valid)[0].astype(np.int32)
    val = d[valid].astype(np.float32)
    return idx, val

def normalize_to_100(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    mx = float(np.max(v)) if v.size > 0 else 0.0
    if mx <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v / mx * 100.0).astype(np.float32)


def _active_xlim(*dense_list, bin_width: float, pad_mz: float = 15.0, min_lo: float = 40.0, max_hi: float = 1005.0):
    idx_all = []
    for d in dense_list:
        if d is None:
            continue
        d = np.asarray(d, dtype=np.float32).reshape(-1)
        idx = np.nonzero(d > 0)[0]
        if idx.size > 0:
            idx_all.append(idx)
    if len(idx_all) == 0:
        return (40.0, 200.0)

    idx_cat = np.concatenate(idx_all)
    mz = _idx_to_mz(idx_cat, bin_width)
    lo = max(float(min_lo), float(np.min(mz) - pad_mz))
    hi = min(float(max_hi), float(np.max(mz) + pad_mz))
    if hi - lo < 80:
        mid = 0.5 * (lo + hi)
        lo = max(float(min_lo), mid - 40.0)
        hi = min(float(max_hi), mid + 40.0)
    return (lo, hi)
def l1_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    s = float(v.sum())
    if s <= 1e-12:
        return np.zeros_like(v, dtype=np.float32)
    return (v / s).astype(np.float32)


def cosine_dense(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb + 1e-12))


def js_similarity_dense(a: np.ndarray, b: np.ndarray) -> float:
    a = l1_normalize(a).astype(np.float64)
    b = l1_normalize(b).astype(np.float64)
    sa = float(a.sum())
    sb = float(b.sum())
    if sa <= 1e-12 or sb <= 1e-12:
        return 0.0
    m = 0.5 * (a + b)
    eps = 1e-12
    kl_am = np.sum(np.where(a > 0, a * np.log((a + eps) / (m + eps)), 0.0))
    kl_bm = np.sum(np.where(b > 0, b * np.log((b + eps) / (m + eps)), 0.0))
    js = 0.5 * (kl_am + kl_bm)
    return float(np.clip(1.0 - js / np.log(2.0), 0.0, 1.0))


def matched_intensity_coverage_dense(candidate: np.ndarray, true_dense: np.ndarray) -> float:
    candidate = np.asarray(candidate, dtype=np.float32)
    true_dense = np.asarray(true_dense, dtype=np.float32)
    true_sum = float(true_dense.sum())
    if true_sum <= 1e-12:
        return np.nan
    support = candidate > 0
    return float(true_dense[support].sum() / (true_sum + 1e-12))


def topk_peak_recall_dense(candidate: np.ndarray, true_dense: np.ndarray, k: int = 20) -> float:
    candidate = np.asarray(candidate, dtype=np.float32)
    true_dense = np.asarray(true_dense, dtype=np.float32)
    true_idx = np.nonzero(true_dense > 0)[0]
    if true_idx.size == 0:
        return np.nan
    k = max(1, min(int(k), int(true_idx.size)))
    top_true = np.argsort(-true_dense, kind="mergesort")[:k]
    top_true = set(int(i) for i in top_true if true_dense[int(i)] > 0)
    cand_set = set(int(i) for i in np.nonzero(candidate > 0)[0].tolist())
    if not top_true:
        return np.nan
    return float(len(top_true.intersection(cand_set)) / float(len(top_true)))


def support_precision_recall(candidate: np.ndarray, true_dense: np.ndarray) -> Tuple[float, float, float]:
    c = candidate > 0
    t = true_dense > 0
    inter = float(np.logical_and(c, t).sum())
    c_n = float(c.sum())
    t_n = float(t.sum())
    prec = inter / c_n if c_n > 0 else np.nan
    rec = inter / t_n if t_n > 0 else np.nan
    iou = inter / (float(np.logical_or(c, t).sum()) + 1e-12)
    return float(prec), float(rec), float(iou)


def nnls_mixture_cos(cands_dense: np.ndarray, true_dense: np.ndarray, k: int) -> Tuple[float, np.ndarray]:
    """Fit a nonnegative mixture over first k columns of cands_dense (selected beforehand)."""
    if cands_dense.ndim != 2 or cands_dense.shape[0] == 0 or cands_dense.shape[1] == 0:
        return np.nan, np.zeros((0,), dtype=np.float32)
    k = max(1, min(int(k), int(cands_dense.shape[0])))
    A = np.asarray(cands_dense[:k], dtype=np.float64).T  # [bin, k]
    b = np.asarray(true_dense, dtype=np.float64).reshape(-1)
    if A.size == 0 or b.size == 0:
        return np.nan, np.zeros((0,), dtype=np.float32)
    try:
        if nnls is not None:
            w, _ = nnls(A, b)
        else:
            w, *_ = np.linalg.lstsq(A, b, rcond=None)
            w = np.clip(w, 0.0, None)
    except Exception:
        return np.nan, np.zeros((k,), dtype=np.float32)
    pred = A @ w
    return cosine_dense(pred, b), w.astype(np.float32)


# ---------------------------- cache loading ----------------------------

@dataclass
class SampleData:
    sample_id: str
    source_file: str
    mol_id: Optional[str]
    precursor_mz: float

    collision_energy: float
    collision_energy_type: Optional[str]
    instrument_type: Optional[str]
    fragmentation_method: Optional[str]

    true_dense: np.ndarray

    # formula candidates
    cand_dense_main: np.ndarray
    cand_dense_nonagg: np.ndarray
    cand_valid_mask: np.ndarray


    # formula metadata
    formulae_aux_feat: Optional[np.ndarray]
    formulae_features: Optional[np.ndarray]
    formulae_source_flag: Optional[np.ndarray]
    formulae_break_depth: Optional[np.ndarray]
    formulae_ring_cut_flag: Optional[np.ndarray]
def _pick_sample_id(meta: dict, pkl_path: Path) -> str:
    for k in ["identifier", "mol_id", "inchikey", "smiles", "row_idx"]:
        if k in meta and meta[k] is not None and str(meta[k]) != "":
            return str(meta[k])
    return pkl_path.stem


def _build_true_dense_from_features(features: dict, bin_n: int) -> np.ndarray:
    idx = features.get("true_official_idx", None)
    val = features.get("true_official_intensity", None)
    idx = _to_numpy(idx, np.int64)
    val = _to_numpy(val, np.float32)
    if idx is None or val is None:
        return np.zeros((bin_n,), dtype=np.float32)
    return sparse_to_dense(idx.reshape(-1), val.reshape(-1), bin_n)


def _build_candidate_dense(
    idx_arr,
    val_arr,
    formulae_mask,
    bin_n: int,
    max_candidates: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:    
    idx = _to_numpy(idx_arr, np.int64)
    val = _to_numpy(val_arr, np.float32)
    mask = _to_numpy(formulae_mask, np.float32)
    if idx is None or val is None:
        return np.zeros((0, bin_n), dtype=np.float32), np.zeros((0,), dtype=bool)
    if idx.ndim != 2 and idx.ndim != 3:
        idx = np.asarray(idx).reshape(-1, idx.shape[-1])
    if val.ndim != 2 and val.ndim != 3:
        val = np.asarray(val).reshape(-1, val.shape[-1])
    if idx.ndim == 3:
        idx = idx[0]
    if val.ndim == 3:
        val = val[0]
    cand_n = min(int(idx.shape[0]), int(val.shape[0]))

    if int(max_candidates or 0) > 0:
        cand_n = min(cand_n, int(max_candidates))

    if cand_n <= 0:
        return np.zeros((0, bin_n), dtype=np.float32), np.zeros((0,), dtype=bool)

    idx = idx[:cand_n]
    val = val[:cand_n]
    if mask is None:
        valid_mask = np.ones((cand_n,), dtype=bool)
    else:
        if mask.ndim > 1:
            mask = mask.reshape(-1)
        valid_mask = mask[:cand_n] > 0.5
        if valid_mask.shape[0] < cand_n:
            pad = np.zeros((cand_n - valid_mask.shape[0],), dtype=bool)
            valid_mask = np.concatenate([valid_mask, pad], axis=0)
    out = np.zeros((cand_n, bin_n), dtype=np.float32)
    for i in range(cand_n):
        if not valid_mask[i]:
            continue
        out[i] = sparse_to_dense(idx[i], val[i], bin_n)
    return out, valid_mask.astype(bool)

def load_one_cache_pkl(
    pkl_path: Path,
    bin_n: int,
    max_candidates: int = 0,
    skip_nonagg: bool = False,
) -> SampleData:
    with open(pkl_path, "rb") as f:
        meta = pickle.load(f)
    features = meta.get("features", {}) if isinstance(meta, dict) else {}

    sample_id = _pick_sample_id(meta, pkl_path)
    mol_id = meta.get("mol_id", None)
    precursor_mz = _safe_float(meta.get("precursor_mz", np.nan), default=np.nan)
    true_dense = _build_true_dense_from_features(features, bin_n=bin_n)

    formulae_mask = features.get("formulae_mask", None)

    # main candidate representation: prefer aggregated official peaks
    idx_main = features.get("formulae_peaks_official_idx_agg", None)
    val_main = features.get("formulae_peaks_official_intensity_agg", None)
    if idx_main is None or val_main is None:
        idx_main = features.get("formulae_peaks_official_idx", None)
        val_main = features.get("formulae_peaks_official_intensity", None)

    cand_dense_main, cand_valid_mask = _build_candidate_dense(
        idx_main,
        val_main,
        formulae_mask,
        bin_n,
        max_candidates=max_candidates,
    )
    if skip_nonagg:
        cand_dense_nonagg = np.zeros((0, bin_n), dtype=np.float32)
    else:
        cand_dense_nonagg, _ = _build_candidate_dense(
            features.get("formulae_peaks_official_idx", None),
            features.get("formulae_peaks_official_intensity", None),
            formulae_mask,
            bin_n,
            max_candidates=max_candidates,
        )

    formulae_aux_feat = _to_numpy(features.get("formulae_aux_feat", None), np.float32)
    if formulae_aux_feat is not None and formulae_aux_feat.ndim == 3:
        formulae_aux_feat = formulae_aux_feat[0]

    formulae_features = _to_numpy(features.get("formulae_features", None), np.float32)
    if formulae_features is not None and formulae_features.ndim == 3:
        formulae_features = formulae_features[0]

    formulae_source_flag = _to_numpy(features.get("formulae_source_flag", None), np.int64)
    if formulae_source_flag is not None:
        formulae_source_flag = formulae_source_flag.reshape(-1)

    formulae_break_depth = _to_numpy(features.get("formulae_break_depth", None), np.int64)
    if formulae_break_depth is not None:
        formulae_break_depth = formulae_break_depth.reshape(-1)

    formulae_ring_cut_flag = _to_numpy(features.get("formulae_ring_cut_flag", None), np.int64)
    if formulae_ring_cut_flag is not None:
        formulae_ring_cut_flag = formulae_ring_cut_flag.reshape(-1)
    return SampleData(
        sample_id=str(sample_id),
        source_file=str(pkl_path),
        mol_id=None if mol_id is None else str(mol_id),
        precursor_mz=precursor_mz,

        collision_energy=_safe_float(meta.get("collision_energy", np.nan), default=np.nan),
        collision_energy_type=meta.get("collision_energy_type", None),
        instrument_type=meta.get("instrument_type", None),
        fragmentation_method=meta.get("fragmentation_method", None),

        true_dense=true_dense,

        cand_dense_main=cand_dense_main,
        cand_dense_nonagg=cand_dense_nonagg,
        cand_valid_mask=cand_valid_mask,

        formulae_aux_feat=formulae_aux_feat,
        formulae_features=formulae_features,
        formulae_source_flag=formulae_source_flag,
        formulae_break_depth=formulae_break_depth,
        formulae_ring_cut_flag=formulae_ring_cut_flag,
    )


def dense_set_union_metrics(
    dense_set: np.ndarray,
    valid_mask: np.ndarray,
    true_dense: np.ndarray,
    name: str,
) -> dict:
    out = {
        f"{name}_candidate_n": 0,
        f"{name}_valid_n": 0,
        f"{name}_union_top20_recall": np.nan,
        f"{name}_union_coverage": np.nan,
        f"{name}_union_precision": np.nan,
        f"{name}_union_support_recall": np.nan,
        f"{name}_union_iou": np.nan,
        f"{name}_best_cos": np.nan,
        f"{name}_best_idx": -1,
    }

    if dense_set is None or dense_set.ndim != 2 or dense_set.shape[0] <= 0:
        return out

    valid_mask = np.asarray(valid_mask, dtype=bool).reshape(-1)
    n = min(int(dense_set.shape[0]), int(valid_mask.shape[0]))
    dense_set = dense_set[:n]
    valid_mask = valid_mask[:n]

    valid_idx = np.where(valid_mask)[0]
    out[f"{name}_candidate_n"] = int(n)
    out[f"{name}_valid_n"] = int(valid_idx.size)

    if valid_idx.size <= 0:
        return out

    union = np.maximum.reduce(dense_set[valid_idx], axis=0)

    out[f"{name}_union_top20_recall"] = topk_peak_recall_dense(union, true_dense, k=20)
    out[f"{name}_union_coverage"] = matched_intensity_coverage_dense(union, true_dense)

    p, r, iou = support_precision_recall(union, true_dense)
    out[f"{name}_union_precision"] = p
    out[f"{name}_union_support_recall"] = r
    out[f"{name}_union_iou"] = iou

    cos = np.asarray(
        [
            cosine_dense(dense_set[i], true_dense) if valid_mask[i] else np.nan
            for i in range(n)
        ],
        dtype=np.float32,
    )
    finite = np.where(np.isfinite(cos))[0]
    if finite.size > 0:
        best = int(finite[np.argmax(cos[finite])])
        out[f"{name}_best_cos"] = float(cos[best])
        out[f"{name}_best_idx"] = int(best)

    return out
# ---------------------------- analysis core ----------------------------

def evaluate_one_sample(sd: SampleData, topk_plot: int = 3, mix_pool: int = 20, mix_k: int = 5) -> Tuple[dict, List[dict], dict]:
    true_dense = sd.true_dense
    cand = sd.cand_dense_main
    cand_nonagg = sd.cand_dense_nonagg
    valid = sd.cand_valid_mask
    cand_n = int(cand.shape[0])
    if cand_n <= 0:
        row = {
            "sample_id": sd.sample_id,
            "source_file": sd.source_file,
            "mol_id": sd.mol_id,
            "precursor_mz": sd.precursor_mz,
            "candidate_n": 0,
            "valid_candidate_n": 0,
        }
        return row, [], {"plot_candidates": []}

    cos = np.full((cand_n,), np.nan, dtype=np.float32)
    js = np.full((cand_n,), np.nan, dtype=np.float32)
    cov = np.full((cand_n,), np.nan, dtype=np.float32)
    hit20 = np.full((cand_n,), np.nan, dtype=np.float32)
    sup_prec = np.full((cand_n,), np.nan, dtype=np.float32)
    sup_rec = np.full((cand_n,), np.nan, dtype=np.float32)
    sup_iou = np.full((cand_n,), np.nan, dtype=np.float32)
    peak_n = np.zeros((cand_n,), dtype=np.int32)

    for i in range(cand_n):
        if not valid[i]:
            continue
        d = cand[i]
        peak_n[i] = int((d > 0).sum())
        cos[i] = cosine_dense(d, true_dense)
        js[i] = js_similarity_dense(d, true_dense)
        cov[i] = matched_intensity_coverage_dense(d, true_dense)
        hit20[i] = topk_peak_recall_dense(d, true_dense, k=20)
        p, r, iou = support_precision_recall(d, true_dense)
        sup_prec[i] = p
        sup_rec[i] = r
        sup_iou[i] = iou

    valid_idx = np.where(valid)[0]
    finite_cos_idx = valid_idx[np.isfinite(cos[valid_idx])]
    order = finite_cos_idx[np.argsort(-cos[finite_cos_idx], kind="mergesort")] if finite_cos_idx.size > 0 else np.asarray([], dtype=np.int64)

    top_rows: List[dict] = []
    for rank, idx in enumerate(order[: max(1, topk_plot * 3)], start=1):
        top_rows.append({
            "sample_id": sd.sample_id,
            "source_file": sd.source_file,
            "rank_by_cos": int(rank),
            "candidate_idx": int(idx),
            "cosine": _safe_float(cos[idx]),
            "js_similarity": _safe_float(js[idx]),
            "matched_intensity_coverage": _safe_float(cov[idx]),
            "top20_recall": _safe_float(hit20[idx]),
            "support_precision": _safe_float(sup_prec[idx]),
            "support_recall": _safe_float(sup_rec[idx]),
            "support_iou": _safe_float(sup_iou[idx]),
            "candidate_peak_n_agg": int((cand[idx] > 0).sum()),
            "candidate_peak_n_nonagg": int((cand_nonagg[idx] > 0).sum()) if idx < cand_nonagg.shape[0] else 0,
        })

    # union oracle support among top-k candidates (ranked by single-candidate cosine)
    def union_metrics(k: int) -> Tuple[float, float, float, float]:
        if order.size <= 0:
            return np.nan, np.nan, np.nan, np.nan
        sel = order[: max(1, min(k, order.size))]
        union = np.maximum.reduce(cand[sel], axis=0)

        recall = topk_peak_recall_dense(union, true_dense, k=20)
        coverage = matched_intensity_coverage_dense(union, true_dense)
        precision, support_recall, iou = support_precision_recall(union, true_dense)

        # 这里 support_recall 和 top20 recall 不是一个概念：
        # - recall: true top20 peak recall
        # - support_recall: 全 support bin 的 recall
        # 我们保留 precision + iou 更有用
        return recall, coverage, precision, iou
    union5_recall, union5_cov, union5_prec, union5_iou = union_metrics(5)
    union10_recall, union10_cov, union10_prec, union10_iou = union_metrics(10)
    union100_recall, union100_cov, union100_prec, union100_iou = union_metrics(100)
    union1000_recall, union1000_cov, union1000_prec, union1000_iou = union_metrics(1000)

    # oracle nonnegative mixture over top candidates
    mix_sel = order[: max(1, min(int(mix_pool), order.size))] if order.size > 0 else np.asarray([], dtype=np.int64)
    mix_cos = np.nan
    mix_weights = np.zeros((0,), dtype=np.float32)
    mix_pred = None
    if mix_sel.size > 0:
        mix_cos, mix_weights = nnls_mixture_cos(cand[mix_sel], true_dense, k=min(int(mix_k), int(mix_sel.size)))
        k_fit = min(int(mix_k), int(mix_sel.size))
        if k_fit > 0 and mix_weights.size >= k_fit:
            mix_pred = np.sum(cand[mix_sel[:k_fit]] * mix_weights[:k_fit, None], axis=0)

    union1000_pred = None
    if order.size > 0:
        sel1000 = order[: max(1, min(1000, order.size))]
        union1000_pred = np.maximum.reduce(cand[sel1000], axis=0)

    # auxiliary feature signal
    aux = sd.formulae_aux_feat
    aux_best_abs_spear = np.nan
    aux_mean_abs_spear = np.nan
    aux_best_dim = -1
    aux_best_abs_pear = np.nan
    if aux is not None:
        aux = np.asarray(aux, dtype=np.float32)
        if aux.ndim == 3:
            aux = aux[0]
        if aux.ndim == 2:
            use_n = min(int(aux.shape[0]), int(cand_n))
            aux = aux[:use_n]
            score = cos[:use_n]
            mask = valid[:use_n] & np.isfinite(score)
            if mask.sum() >= 5:
                spear_vals = []
                pear_vals = []
                for d in range(aux.shape[1]):
                    x = aux[mask, d]
                    y = score[mask]
                    if np.nanstd(x) <= 1e-12:
                        spear_vals.append(np.nan)
                        pear_vals.append(np.nan)
                    else:
                        spear_vals.append(_spearman(x, y))
                        pear_vals.append(_pearson(x, y))
                spear_arr = np.asarray(spear_vals, dtype=float)
                pear_arr = np.asarray(pear_vals, dtype=float)
                if np.isfinite(np.abs(spear_arr)).any():
                    aux_best_dim = int(np.nanargmax(np.abs(spear_arr)))
                    aux_best_abs_spear = float(np.nanmax(np.abs(spear_arr)))
                    aux_mean_abs_spear = float(np.nanmean(np.abs(spear_arr)))
                if np.isfinite(np.abs(pear_arr)).any():
                    aux_best_abs_pear = float(np.nanmax(np.abs(pear_arr)))

    best_idx = int(order[0]) if order.size > 0 else -1
    best_cos = _safe_float(cos[best_idx]) if best_idx >= 0 else np.nan
    best_js = _safe_float(js[best_idx]) if best_idx >= 0 else np.nan
    best_cov = _safe_float(cov[best_idx]) if best_idx >= 0 else np.nan
    best_hit20 = _safe_float(hit20[best_idx]) if best_idx >= 0 else np.nan
    best_support_recall = _safe_float(sup_rec[best_idx]) if best_idx >= 0 else np.nan

    row = {
        "sample_id": sd.sample_id,
        "source_file": sd.source_file,
        "mol_id": sd.mol_id,
        "precursor_mz": sd.precursor_mz,
        "candidate_n": int(cand_n),
        "valid_candidate_n": int(valid.sum()),
        "candidate_peak_n_mean_agg": float(np.nanmean(peak_n[valid])) if valid.any() else np.nan,
        "true_peak_n": int((true_dense > 0).sum()),
        "oracle_best_candidate_idx": int(best_idx),
        "oracle_best_cos": best_cos,
        "oracle_best_js": best_js,
        "oracle_best_matched_intensity_coverage": best_cov,
        "oracle_best_top20_recall": best_hit20,
        "oracle_best_support_recall": best_support_recall,
        "oracle_mix_topk_cos": mix_cos,
        "aux_best_abs_spearman_to_cos": aux_best_abs_spear,
        "aux_mean_abs_spearman_to_cos": aux_mean_abs_spear,
        "aux_best_abs_pearson_to_cos": aux_best_abs_pear,
        "aux_best_dim": int(aux_best_dim),
        "oracle_union_top5_recall": union5_recall,
        "oracle_union_top5_coverage": union5_cov,
        "oracle_union_top5_precision": union5_prec,
        "oracle_union_top5_iou": union5_iou,

        "oracle_union_top10_recall": union10_recall,
        "oracle_union_top10_coverage": union10_cov,
        "oracle_union_top10_precision": union10_prec,
        "oracle_union_top10_iou": union10_iou,

        "oracle_union_top100_recall": union100_recall,
        "oracle_union_top100_coverage": union100_cov,
        "oracle_union_top100_precision": union100_prec,
        "oracle_union_top100_iou": union100_iou,

        "oracle_union_top1000_recall": union1000_recall,
        "oracle_union_top1000_coverage": union1000_cov,
        "oracle_union_top1000_precision": union1000_prec,
        "oracle_union_top1000_iou": union1000_iou,
    }

    # ---------------- metadata / CE bucket ----------------
    row["collision_energy"] = sd.collision_energy
    row["collision_energy_type"] = sd.collision_energy_type
    row["instrument_type"] = sd.instrument_type
    row["fragmentation_method"] = sd.fragmentation_method

    if np.isfinite(sd.collision_energy):
        if sd.collision_energy < 20:
            row["ce_bucket"] = "low"
        elif sd.collision_energy < 45:
            row["ce_bucket"] = "mid"
        else:
            row["ce_bucket"] = "high"
    else:
        row["ce_bucket"] = "missing"

    # ---------------- formula source ablation ----------------
    if sd.formulae_source_flag is not None:
        sf = np.asarray(sd.formulae_source_flag).reshape(-1)
        use_n = min(
            int(sf.shape[0]),
            int(sd.cand_dense_main.shape[0]),
            int(sd.cand_valid_mask.shape[0]),
        )

        if use_n > 0:
            sf = sf[:use_n]
            dense_use = sd.cand_dense_main[:use_n]
            valid_use = sd.cand_valid_mask[:use_n]

            source_groups = {
                "formula_only": sf == 0,
                "structural_formula": (sf == 1) | (sf == 3),
                "common_loss_formula": (sf == 2) | (sf == 3),
                "any_source_formula": sf > 0,
            }

            for group_name, group_mask in source_groups.items():
                group_valid = valid_use & group_mask
                row.update(
                    dense_set_union_metrics(
                        dense_use,
                        group_valid,
                        true_dense,
                        name=group_name,
                    )
                )


    plot_payload = {
        "plot_candidates": order[: max(1, topk_plot)].tolist(),
        "mix_sel": mix_sel[: max(1, min(int(mix_k), len(mix_sel)))].tolist(),
        "mix_weights": mix_weights[: max(1, min(int(mix_k), len(mix_weights)))].tolist() if mix_weights.size > 0 else [],
        "mix_pred": mix_pred,
        "cos": cos,
        "top_rows": top_rows,
        "union1000_pred": union1000_pred,
    }
    return row, top_rows, plot_payload


# ---------------------------- plotting ----------------------------

def _idx_to_mz(idx: np.ndarray, bin_width: float) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    return (idx.astype(np.float64) + 0.5) * float(bin_width)


def plot_sample_overlay(sd: SampleData, plot_payload: dict, out_path: Path, bin_width: float, topk_plot: int = 3):
    true_dense = normalize_to_100(sd.true_dense)
    true_idx, true_val = dense_to_sparse(true_dense, thr=0.0)

    order = plot_payload.get("plot_candidates", [])
    mix_pred = plot_payload.get("mix_pred", None)

    best_dense = None
    if len(order) > 0:
        best_dense = normalize_to_100(sd.cand_dense_main[int(order[0])])

    mix_dense = None
    if mix_pred is not None:
        mix_dense = normalize_to_100(np.asarray(mix_pred, dtype=np.float32))

    union1000_dense = None
    union1000_pred = plot_payload.get("union1000_pred", None)
    if union1000_pred is not None:
        union1000_dense = normalize_to_100(np.asarray(union1000_pred, dtype=np.float32))

    fig, ax = plt.subplots(figsize=(12, 4.8))

    # 真实谱：上半轴
    ax.vlines(
        _idx_to_mz(true_idx, bin_width),
        0.0,
        true_val,
        linewidth=0.8,
        alpha=0.95,
        color="black",
        label="True",
        zorder=3,
    )

    # top candidates：下半轴（镜像）
    cand_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for rank, cand_idx in enumerate(order[: max(1, topk_plot)], start=1):
        cand = normalize_to_100(sd.cand_dense_main[int(cand_idx)])
        c_idx, c_val = dense_to_sparse(cand, thr=0.0)
        if c_idx.size == 0:
            continue
        ax.vlines(
            _idx_to_mz(c_idx, bin_width),
            0.0,
            -c_val,
            linewidth=0.7,
            alpha=0.65 if rank > 1 else 0.85,
            color=cand_colors[(rank - 1) % len(cand_colors)],
            label=f"Cand#{cand_idx} rank{rank}",
            zorder=2,
        )

    if union1000_dense is not None:
        u_idx, u_val = dense_to_sparse(union1000_dense, thr=0.0)
        if u_idx.size > 0:
            ax.vlines(
                _idx_to_mz(u_idx, bin_width),
                0.0,
                -u_val,
                linewidth=0.8,
                alpha=0.55,
                color="#9467bd",
                label="Union top1000",
                zorder=1,
            )

    # oracle mix：下半轴，更醒目
    if mix_dense is not None:
        m_idx, m_val = dense_to_sparse(mix_dense, thr=0.0)
        if m_idx.size > 0:
            ax.vlines(
                _idx_to_mz(m_idx, bin_width),
                0.0,
                -m_val,
                linewidth=1.0,
                alpha=0.90,
                color="#d62728",
                label="Oracle mix",
                zorder=4,
            )

    xlo, xhi = _active_xlim(true_dense, best_dense, mix_dense, union1000_dense, bin_width=bin_width, pad_mz=12.0)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(-105, 105)

    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity")
    ax.set_title(f"Cache quality overlay | {sd.sample_id}")

    # y 轴显示成镜像质谱风格
    yticks = [-100, -50, 0, 50, 100]
    yticklabels = ["100", "50", "0", "50", "100"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    text_lines = []
    top_rows = plot_payload.get("top_rows", [])
    if top_rows:
        best = top_rows[0]
        text_lines.append(f"best cos={best['cosine']:.3f}")
        text_lines.append(f"best recall@20={best['top20_recall']:.3f}")
        text_lines.append(f"best cov={best['matched_intensity_coverage']:.3f}")
    mix_weights = plot_payload.get("mix_weights", [])
    if mix_weights:
        text_lines.append("mix w=" + ", ".join(f"{float(w):.2f}" for w in mix_weights))

    if text_lines:
        ax.text(
            0.01,
            0.98,
            "\n".join(text_lines),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

def plot_summary(df: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    ax = axes.ravel()

    cols = [
        ("oracle_best_cos", "Best formula candidate cosine"),
        ("oracle_mix_topk_cos", "Oracle mix cosine"),
        ("oracle_union_top100_precision", "Formula top100 union precision"),
        ("oracle_union_top100_coverage", "Formula top100 union coverage"),
        ("formula_only_union_precision", "Formula-only union precision"),
    ]

    for i, (col, title) in enumerate(cols):
        if col not in df.columns:
            ax[i].set_title(title + " (missing)")
            ax[i].axis("off")
            continue

        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]

        if vals.size <= 0:
            ax[i].set_title(title + " (empty)")
            ax[i].axis("off")
            continue

        ax[i].hist(vals, bins=12)
        ax[i].set_title(title)
        ax[i].set_xlabel(col)
        ax[i].set_ylabel("count")

    x = df["oracle_best_cos"].astype(float).to_numpy()
    y = df["oracle_mix_topk_cos"].astype(float).to_numpy()
    valid = np.isfinite(x) & np.isfinite(y)
    ax[5].scatter(x[valid], y[valid], s=24)
    ax[5].set_title("Best single vs oracle mix")
    ax[5].set_xlabel("best single cosine")
    ax[5].set_ylabel("mix cosine")
    if np.any(valid):
        lo = min(float(np.nanmin(x[valid])), float(np.nanmin(y[valid])))
        hi = max(float(np.nanmax(x[valid])), float(np.nanmax(y[valid])))
        ax[5].plot([lo, hi], [lo, hi], linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_gallery(sample_entries: Sequence[Tuple[SampleData, dict]], out_path: Path, bin_width: float, max_panels: int = 12):
    sample_entries = list(sample_entries[:max_panels])
    n = len(sample_entries)
    if n <= 0:
        return

    cols = 2
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.6 * rows), squeeze=False)
    axes = axes.ravel()

    for ax, (sd, payload) in zip(axes, sample_entries):
        true_dense = normalize_to_100(sd.true_dense)
        true_idx, true_val = dense_to_sparse(true_dense, thr=0.0)

        best_dense = None
        order = payload.get("plot_candidates", [])
        if len(order) > 0:
            best_idx = int(order[0])
            best_dense = normalize_to_100(sd.cand_dense_main[best_idx])

        mix_dense = None
        mix_pred = payload.get("mix_pred", None)
        if mix_pred is not None:
            mix_dense = normalize_to_100(np.asarray(mix_pred, dtype=np.float32))

        ax.vlines(
            _idx_to_mz(true_idx, bin_width),
            0.0,
            true_val,
            linewidth=0.7,
            alpha=0.95,
            color="black",
            label="True",
        )

        if best_dense is not None:
            c_idx, c_val = dense_to_sparse(best_dense, thr=0.0)
            ax.vlines(
                _idx_to_mz(c_idx, bin_width),
                0.0,
                -c_val,
                linewidth=0.7,
                alpha=0.80,
                color="#1f77b4",
                label="Best cand",
            )

        if mix_dense is not None:
            m_idx, m_val = dense_to_sparse(mix_dense, thr=0.0)
            ax.vlines(
                _idx_to_mz(m_idx, bin_width),
                0.0,
                -m_val,
                linewidth=0.9,
                alpha=0.90,
                color="#d62728",
                label="Oracle mix",
            )

        xlo, xhi = _active_xlim(true_dense, best_dense, mix_dense, bin_width=bin_width, pad_mz=10.0)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-105, 105)
        ax.axhline(0.0, color="gray", linewidth=0.7, alpha=0.6)

        ax.set_title(str(sd.sample_id), fontsize=10)
        ax.set_xlabel("m/z")
        ax.set_ylabel("Rel. int.")
        ax.set_yticks([-100, 0, 100])
        ax.set_yticklabels(["100", "0", "100"])

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=7, frameon=False, loc="upper right")

    for ax in axes[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
# ---------------------------- main ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe cache quality on ~30 sampled cache files")
    p.add_argument("--cache-dir", required=True, help="cache directory containing per-sample .pkl files")
    p.add_argument("--sample-n", type=int, default=30, help="number of cache files to inspect")
    p.add_argument("--seed", type=int, default=123, help="random seed for file sampling")
    p.add_argument("--official-bin-width", type=float, default=0.01)
    p.add_argument("--official-max-mz", type=float, default=1005.0)
    p.add_argument("--topk-plot-candidates", type=int, default=3)
    p.add_argument("--mix-pool", type=int, default=20, help="fit oracle mix from top-N single-candidate cos candidates")
    p.add_argument("--mix-k", type=int, default=5, help="number of candidates used in oracle nonnegative mix")
    p.add_argument("--out-dir", required=True, help="output directory")
    p.add_argument("--max-candidates", type=int, default=0,
                   help="limit candidates per sample for dense probe; 0 means no limit")
    p.add_argument("--skip-nonagg", action="store_true",
                   help="skip non-aggregated candidate dense matrix to save memory")
    return p.parse_args()


def main():
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not cache_dir.is_dir():
        raise FileNotFoundError(f"cache dir not found: {cache_dir}")

    all_pkls = sorted(cache_dir.glob("*.pkl"))
    if len(all_pkls) <= 0:
        raise RuntimeError(f"no .pkl files found under {cache_dir}")

    rng = random.Random(int(args.seed))
    sample_n = min(int(args.sample_n), len(all_pkls))
    chosen = rng.sample(all_pkls, sample_n)
    chosen = sorted(chosen)

    bin_n = int(math.floor(float(args.official_max_mz) / float(args.official_bin_width))) + 1

    sample_rows: List[dict] = []
    candidate_rows: List[dict] = []
    gallery_entries: List[Tuple[SampleData, dict]] = []

    for ii, pkl_path in enumerate(chosen, start=1):
        print(f"[probe] {ii}/{len(chosen)} {pkl_path.name}", flush=True)
        try:
            sd = load_one_cache_pkl(
                pkl_path,
                bin_n=bin_n,
                max_candidates=int(args.max_candidates),
                skip_nonagg=bool(args.skip_nonagg),
            )
            row, top_rows, payload = evaluate_one_sample(
                sd,
                topk_plot=int(args.topk_plot_candidates),
                mix_pool=int(args.mix_pool),
                mix_k=int(args.mix_k),
            )
            sample_rows.append(row)
            candidate_rows.extend(top_rows)
            gallery_entries.append((sd, payload))

            overlay_path = out_dir / f"{Path(pkl_path).stem}_overlay.png"
            plot_sample_overlay(
                sd,
                payload,
                overlay_path,
                bin_width=float(args.official_bin_width),
                topk_plot=int(args.topk_plot_candidates),
            )
        except Exception as e:
            sample_rows.append({
                "sample_id": pkl_path.stem,
                "source_file": str(pkl_path),
                "error": str(e),
            })

    df = pd.DataFrame(sample_rows)
    cand_df = pd.DataFrame(candidate_rows)
    df.to_csv(out_dir / "sample_summary.csv", index=False)
    cand_df.to_csv(out_dir / "candidate_topk.csv", index=False)

    if len(df) > 0:
        plot_summary(df, out_dir / "summary.png")
        plot_gallery(gallery_entries, out_dir / "gallery.png", bin_width=float(args.official_bin_width), max_panels=12)

    # short text report
    report = {}
    if len(df) > 0:
        def _mean(col):
            vals = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            return None if vals.size == 0 else float(np.mean(vals))

        report = {
            "n_samples": int(len(df)),
            "oracle_best_cos_mean": _mean("oracle_best_cos"),
            "oracle_mix_topk_cos_mean": _mean("oracle_mix_topk_cos"),

            "oracle_union_top10_recall_mean": _mean("oracle_union_top10_recall"),
            "oracle_union_top10_coverage_mean": _mean("oracle_union_top10_coverage"),
            "oracle_union_top10_precision_mean": _mean("oracle_union_top10_precision"),
            "oracle_union_top10_iou_mean": _mean("oracle_union_top10_iou"),

            "oracle_union_top100_recall_mean": _mean("oracle_union_top100_recall"),
            "oracle_union_top100_coverage_mean": _mean("oracle_union_top100_coverage"),
            "oracle_union_top100_precision_mean": _mean("oracle_union_top100_precision"),
            "oracle_union_top100_iou_mean": _mean("oracle_union_top100_iou"),

            "oracle_union_top1000_recall_mean": _mean("oracle_union_top1000_recall"),
            "oracle_union_top1000_coverage_mean": _mean("oracle_union_top1000_coverage"),
            "oracle_union_top1000_precision_mean": _mean("oracle_union_top1000_precision"),
            "oracle_union_top1000_iou_mean": _mean("oracle_union_top1000_iou"),

            "oracle_best_matched_intensity_coverage_mean": _mean("oracle_best_matched_intensity_coverage"),
            "aux_best_abs_spearman_to_cos_mean": _mean("aux_best_abs_spearman_to_cos"),
            

            "formula_only_union_precision_mean": _mean("formula_only_union_precision"),
            "formula_only_union_coverage_mean": _mean("formula_only_union_coverage"),
            "structural_formula_union_precision_mean": _mean("structural_formula_union_precision"),
            "structural_formula_union_coverage_mean": _mean("structural_formula_union_coverage"),

            "common_loss_formula_union_precision_mean": _mean("common_loss_formula_union_precision"),
            "common_loss_formula_union_coverage_mean": _mean("common_loss_formula_union_coverage"),

            "any_source_formula_union_precision_mean": _mean("any_source_formula_union_precision"),
            "any_source_formula_union_coverage_mean": _mean("any_source_formula_union_coverage"),
        }

        if "ce_bucket" in df.columns:
            ce_report = {}

            for bucket, sub in df.groupby("ce_bucket"):

                def _submean(col):
                    vals = pd.to_numeric(
                        sub.get(col, pd.Series(dtype=float)),
                        errors="coerce",
                    ).to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    return None if vals.size == 0 else float(np.mean(vals))

                ce_report[str(bucket)] = {
                    "n": int(len(sub)),

                    "formula_top100_precision": _submean("oracle_union_top100_precision"),
                    "formula_top100_coverage": _submean("oracle_union_top100_coverage"),

                    "formula_only_precision": _submean("formula_only_union_precision"),
                    "formula_only_coverage": _submean("formula_only_union_coverage"),

                    "structural_formula_precision": _submean("structural_formula_union_precision"),
                    "structural_formula_coverage": _submean("structural_formula_union_coverage"),

                    "any_source_formula_precision": _submean("any_source_formula_union_precision"),
                    "any_source_formula_coverage": _submean("any_source_formula_union_coverage"),
                }

            report["ce_bucket_report"] = ce_report
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "args": vars(args),
            "cache_dir": str(cache_dir),
            "chosen_files": [str(x) for x in chosen],
            "report": report,
        }, f, ensure_ascii=False, indent=2)

    print("[done] wrote:")
    print(f"  - {out_dir / 'sample_summary.csv'}")
    print(f"  - {out_dir / 'candidate_topk.csv'}")
    print(f"  - {out_dir / 'summary.png'}")
    print(f"  - {out_dir / 'gallery.png'}")
    print(f"  - {out_dir / 'run_config.json'}")
    if report:
        print("[report]")
        for k, v in report.items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
