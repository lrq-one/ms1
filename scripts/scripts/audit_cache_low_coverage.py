#!/usr/bin/env python3
import os
import glob
import csv
import pickle
import argparse
import numpy as np

def as1d(x, dtype):
    if x is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=dtype)

def load_feat(obj):
    if isinstance(obj, dict) and isinstance(obj.get("features", None), dict):
        return obj["features"]
    return obj if isinstance(obj, dict) else {}

def get_meta(obj, key, default=None):
    if isinstance(obj, dict):
        if key in obj:
            return obj.get(key)
        meta = obj.get("meta", None)
        if isinstance(meta, dict) and key in meta:
            return meta.get(key)
    return default

def union_support(feat):
    mask = as1d(feat.get("formulae_mask", None), np.float32) > 0.5
    idx = np.asarray(feat.get("formulae_peaks_official_idx", []), dtype=np.int64)
    inten = np.asarray(
        feat.get("formulae_peaks_official_intensity", feat.get("formulae_peaks_intensity", [])),
        dtype=np.float32,
    )

    if idx.ndim != 2:
        return set()

    n = min(idx.shape[0], mask.shape[0])
    idx = idx[:n]
    mask = mask[:n]

    support = set()
    for i in range(n):
        if not mask[i]:
            continue
        row = idx[i].reshape(-1)
        for b in row[row >= 0]:
            support.add(int(b))
    return support

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--official-bin-width", type=float, default=0.01)
    ap.add_argument("--sample-n", type=int, default=0)
    ap.add_argument("--precursor-tol-da", type=float, default=0.03)
    ap.add_argument("--isotope-n", type=int, default=2)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    if args.sample_n and args.sample_n > 0:
        paths = paths[:args.sample_n]

    rows = []
    bw = float(args.official_bin_width)

    for p in paths:
        try:
            obj = pickle.load(open(p, "rb"))
        except Exception:
            continue

        feat = load_feat(obj)

        true_idx = as1d(feat.get("true_official_idx", None), np.int64)
        true_int = as1d(feat.get("true_official_intensity", None), np.float32)

        n = min(true_idx.size, true_int.size)
        true_idx = true_idx[:n]
        true_int = true_int[:n]

        valid = (true_idx >= 0) & np.isfinite(true_int) & (true_int > 0)
        true_idx = true_idx[valid]
        true_int = true_int[valid]

        total = float(np.sum(true_int))
        if total <= 1e-12:
            continue

        support = union_support(feat)
        hit = np.asarray([int(x) in support for x in true_idx], dtype=bool)

        covered_int = float(np.sum(true_int[hit]))
        missing_int = float(np.sum(true_int[~hit]))
        coverage = covered_int / max(1e-12, total)

        pmz = get_meta(obj, "precursor_mz", None)
        try:
            pmz = float(pmz)
        except Exception:
            pmz = np.nan

        missing_precursor_like_int = 0.0
        missing_top = []

        if np.isfinite(pmz) and pmz > 0:
            missing_idx = true_idx[~hit]
            missing_val = true_int[~hit]
            missing_mz = missing_idx.astype(np.float64) * bw

            near = np.zeros_like(missing_mz, dtype=bool)
            for k in range(max(0, int(args.isotope_n)) + 1):
                near |= np.abs(missing_mz - (pmz + float(k))) <= float(args.precursor_tol_da)

            missing_precursor_like_int = float(np.sum(missing_val[near]))

        order = np.argsort(-true_int)
        for j in order[:5]:
            b = int(true_idx[j])
            mz = b * bw
            missing_top.append(
                f"{mz:.4f}:{float(true_int[j]):.4g}:{'hit' if b in support else 'MISS'}"
            )

        formula_mask = as1d(feat.get("formulae_mask", None), np.float32)
        active_mask = as1d(feat.get("formulae_active_mask", None), np.float32)

        rows.append({
            "file": os.path.basename(p),
            "identifier": get_meta(obj, "identifier", ""),
            "collision_energy": get_meta(obj, "collision_energy", ""),
            "precursor_mz": pmz,
            "coverage": coverage,
            "missing_intensity_ratio": missing_int / max(1e-12, total),
            "missing_precursor_like_ratio": missing_precursor_like_int / max(1e-12, total),
            "true_peak_n": int(true_idx.size),
            "formula_valid_n": int(np.sum(formula_mask > 0.5)) if formula_mask.size else 0,
            "active_n": int(np.sum(active_mask > 0.5)) if active_mask.size else 0,
            "top_true_peaks": ";".join(missing_top),
        })

    rows = sorted(rows, key=lambda r: float(r["coverage"]))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "file", "identifier", "collision_energy", "precursor_mz",
            "coverage", "missing_intensity_ratio", "missing_precursor_like_ratio",
            "true_peak_n", "formula_valid_n", "active_n", "top_true_peaks",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    cov = np.asarray([float(r["coverage"]) for r in rows], dtype=float)
    miss_prec = np.asarray([float(r["missing_precursor_like_ratio"]) for r in rows], dtype=float)

    print("n:", len(rows))
    if cov.size:
        print("coverage mean/p10/p50/p90:", float(np.mean(cov)), float(np.percentile(cov, 10)), float(np.percentile(cov, 50)), float(np.percentile(cov, 90)))
        print("missing_precursor_like_ratio mean/p90:", float(np.mean(miss_prec)), float(np.percentile(miss_prec, 90)))
        print("worst 20:")
        for r in rows[:20]:
            print(r)

if __name__ == "__main__":
    main()