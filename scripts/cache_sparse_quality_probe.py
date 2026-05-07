#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, pickle, random, math, json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from scipy.optimize import nnls
except Exception:
    nnls = None


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
        arr = arr.astype(dtype, copy=False)
    return arr


def sparse_to_dense(idx, val, bin_n):
    out = np.zeros((bin_n,), dtype=np.float32)
    if idx is None or val is None:
        return out
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    val = np.asarray(val, dtype=np.float32).reshape(-1)
    n = min(idx.size, val.size)
    idx = idx[:n]
    val = val[:n]
    valid = (idx >= 0) & (idx < bin_n) & np.isfinite(val) & (val > 0)
    if np.any(valid):
        np.add.at(out, idx[valid], val[valid])
    return out


def load_cache(path, bin_n):
    obj = pickle.load(open(path, "rb"))
    f = obj["features"]

    true_idx = _to_numpy(f.get("true_official_idx"), np.int64)
    true_val = _to_numpy(f.get("true_official_intensity"), np.float32)
    true_dense = sparse_to_dense(true_idx, true_val, bin_n)

    idx = f.get("formulae_peaks_official_idx_agg", None)
    val = f.get("formulae_peaks_official_intensity_agg", None)
    if idx is None or val is None:
        idx = f.get("formulae_peaks_official_idx", None)
        val = f.get("formulae_peaks_official_intensity", None)

    idx = _to_numpy(idx, np.int64)
    val = _to_numpy(val, np.float32)

    if idx.ndim == 3:
        idx = idx[0]
    if val.ndim == 3:
        val = val[0]

    mask = _to_numpy(f.get("formulae_mask"), np.float32)
    if mask is None:
        mask = np.ones((idx.shape[0],), dtype=np.float32)
    mask = mask.reshape(-1)[:idx.shape[0]] > 0.5

    source_flag = _to_numpy(f.get("formulae_source_flag"), np.int64)
    if source_flag is not None:
        source_flag = source_flag.reshape(-1)[:idx.shape[0]]

    return obj, true_dense, idx, val, mask, source_flag


def candidate_cosines(true_dense, idx, val, mask, bin_n):
    true_norm = float(np.linalg.norm(true_dense))
    cand_n = idx.shape[0]
    cos = np.full((cand_n,), np.nan, dtype=np.float32)

    if true_norm <= 1e-12:
        return cos

    for i in range(cand_n):
        if not mask[i]:
            continue
        ii = idx[i]
        vv = val[i]
        valid = (ii >= 0) & (ii < bin_n) & np.isfinite(vv) & (vv > 0)
        if not np.any(valid):
            cos[i] = 0.0
            continue
        ii = ii[valid]
        vv = vv[valid].astype(np.float64)
        dot = float(np.sum(true_dense[ii].astype(np.float64) * vv))
        cn = float(np.linalg.norm(vv))
        cos[i] = dot / (cn * true_norm + 1e-12) if cn > 1e-12 else 0.0
    return cos


def union_metrics(true_dense, idx, val, selected, bin_n, topk_peak=20):
    support = np.zeros((bin_n,), dtype=bool)

    for i in selected:
        ii = idx[int(i)]
        vv = val[int(i)]
        valid = (ii >= 0) & (ii < bin_n) & np.isfinite(vv) & (vv > 0)
        if np.any(valid):
            support[ii[valid]] = True

    true_support = true_dense > 0
    true_sum = float(true_dense.sum())

    if true_sum <= 1e-12:
        coverage = np.nan
    else:
        coverage = float(true_dense[support].sum() / (true_sum + 1e-12))

    support_n = int(support.sum())
    inter_n = int(np.logical_and(support, true_support).sum())
    true_n = int(true_support.sum())
    union_n = int(np.logical_or(support, true_support).sum())

    precision = inter_n / support_n if support_n > 0 else np.nan
    support_recall = inter_n / true_n if true_n > 0 else np.nan
    iou = inter_n / union_n if union_n > 0 else np.nan

    true_idx = np.nonzero(true_dense > 0)[0]
    if true_idx.size > 0:
        order = np.argsort(-true_dense)[:min(topk_peak, true_idx.size)]
        top_true = set(int(x) for x in order if true_dense[int(x)] > 0)
        top20_recall = len([x for x in top_true if support[x]]) / max(1, len(top_true))
    else:
        top20_recall = np.nan

    return {
        "coverage": coverage,
        "precision": precision,
        "support_recall": support_recall,
        "iou": iou,
        "top20_recall": top20_recall,
        "support_n": support_n,
        "inter_n": inter_n,
        "true_n": true_n,
    }


def nnls_mix_cos(true_dense, idx, val, order, mix_pool, mix_k, bin_n):
    if len(order) == 0:
        return np.nan
    sel = order[:min(int(mix_pool), len(order))]
    k = min(int(mix_k), len(sel))
    if k <= 0:
        return np.nan

    A = np.zeros((bin_n, k), dtype=np.float64)
    for col, ci in enumerate(sel[:k]):
        ii = idx[int(ci)]
        vv = val[int(ci)]
        valid = (ii >= 0) & (ii < bin_n) & np.isfinite(vv) & (vv > 0)
        if np.any(valid):
            np.add.at(A[:, col], ii[valid], vv[valid].astype(np.float64))

    b = true_dense.astype(np.float64)
    if nnls is not None:
        w, _ = nnls(A, b)
    else:
        w, *_ = np.linalg.lstsq(A, b, rcond=None)
        w = np.clip(w, 0, None)

    pred = A @ w
    den = float(np.linalg.norm(pred) * np.linalg.norm(b))
    return float(np.dot(pred, b) / (den + 1e-12)) if den > 1e-12 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--official-bin-width", type=float, default=0.01)
    ap.add_argument("--official-max-mz", type=float, default=1005.0)
    ap.add_argument("--mix-pool", type=int, default=20)
    ap.add_argument("--mix-k", type=int, default=8)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pkls = sorted(cache_dir.glob("*.pkl"))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(pkls, min(args.sample_n, len(pkls))))

    bin_n = int(math.floor(args.official_max_mz / args.official_bin_width)) + 1
    rows = []

    for si, p in enumerate(chosen, 1):
        print(f"[sparse-probe] {si}/{len(chosen)} {p.name}", flush=True)
        obj, true_dense, idx, val, mask, source_flag = load_cache(p, bin_n)

        cos = candidate_cosines(true_dense, idx, val, mask, bin_n)
        valid_idx = np.where(mask & np.isfinite(cos))[0]
        order = valid_idx[np.argsort(-cos[valid_idx], kind="mergesort")] if valid_idx.size else np.asarray([], dtype=np.int64)

        row = {
            "file": str(p),
            "sample_id": str(obj.get("identifier", p.stem)),
            "candidate_n": int(idx.shape[0]),
            "valid_candidate_n": int(mask.sum()),
            "true_peak_n": int((true_dense > 0).sum()),
            "oracle_best_cos": float(cos[order[0]]) if order.size else np.nan,
            "oracle_mix_topk_cos": nnls_mix_cos(true_dense, idx, val, order, args.mix_pool, args.mix_k, bin_n),
        }

        for k in [10, 100, 1000]:
            m = union_metrics(true_dense, idx, val, order[:min(k, len(order))], bin_n)
            for kk, vv in m.items():
                row[f"oracle_union_top{k}_{kk}"] = vv

        all_valid = np.where(mask)[0]
        m = union_metrics(true_dense, idx, val, all_valid, bin_n)
        for kk, vv in m.items():
            row[f"all_formula_union_{kk}"] = vv

        if source_flag is not None:
            groups = {
                "formula_only": source_flag == 0,
                "structural_formula": (source_flag == 1) | (source_flag == 3),
                "common_loss_formula": (source_flag == 2) | (source_flag == 3),
                "any_source_formula": source_flag > 0,
            }
            for name, gm in groups.items():
                sel = np.where(mask & gm[:len(mask)])[0]
                mm = union_metrics(true_dense, idx, val, sel, bin_n)
                for kk, vv in mm.items():
                    row[f"{name}_union_{kk}"] = vv

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sparse_sample_summary.csv", index=False)

    def mean(col):
        x = pd.to_numeric(df.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy(float)
        x = x[np.isfinite(x)]
        return None if x.size == 0 else float(x.mean())

    report = {
        "n_samples": int(len(df)),
        "oracle_best_cos_mean": mean("oracle_best_cos"),
        "oracle_mix_topk_cos_mean": mean("oracle_mix_topk_cos"),
        "oracle_union_top100_coverage_mean": mean("oracle_union_top100_coverage"),
        "oracle_union_top100_precision_mean": mean("oracle_union_top100_precision"),
        "oracle_union_top1000_coverage_mean": mean("oracle_union_top1000_coverage"),
        "oracle_union_top1000_precision_mean": mean("oracle_union_top1000_precision"),
        "all_formula_union_coverage_mean": mean("all_formula_union_coverage"),
        "all_formula_union_precision_mean": mean("all_formula_union_precision"),
        "structural_formula_union_coverage_mean": mean("structural_formula_union_coverage"),
        "structural_formula_union_precision_mean": mean("structural_formula_union_precision"),
        "any_source_formula_union_coverage_mean": mean("any_source_formula_union_coverage"),
        "any_source_formula_union_precision_mean": mean("any_source_formula_union_precision"),
    }

    with open(out_dir / "sparse_run_config.json", "w") as f:
        json.dump({"args": vars(args), "report": report}, f, indent=2)

    print("[sparse-report]")
    for k, v in report.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
