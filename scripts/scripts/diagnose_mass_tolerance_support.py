#!/usr/bin/env python3
import argparse, glob, os, pickle
import numpy as np

def to_peaks(x):
    arr = np.asarray(x, dtype=np.float64)
    if arr.size <= 0:
        return np.zeros((0, 2), dtype=np.float64)
    arr = arr.reshape(-1, 2)
    mz = arr[:, 0]
    it = arr[:, 1]
    m = np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz >= 0)
    return arr[m]

def exclude_precursor(peaks, pmz, tol=0.05, isotope_n=2):
    if peaks.shape[0] <= 0:
        return peaks
    if not np.isfinite(pmz) or pmz <= 0:
        return peaks
    mz = peaks[:, 0]
    keep = np.ones((peaks.shape[0],), dtype=bool)
    for k in range(max(0, int(isotope_n)) + 1):
        keep &= np.abs(mz - (float(pmz) + float(k))) > float(tol)
    return peaks[keep]

def nearest_distances(true_mz, cand_mz):
    true_mz = np.asarray(true_mz, dtype=np.float64)
    cand_mz = np.asarray(cand_mz, dtype=np.float64)
    cand_mz = cand_mz[np.isfinite(cand_mz)]
    cand_mz = np.unique(np.sort(cand_mz))
    if true_mz.size <= 0 or cand_mz.size <= 0:
        return np.full((true_mz.size,), np.inf, dtype=np.float64)

    pos = np.searchsorted(cand_mz, true_mz)
    out = np.full((true_mz.size,), np.inf, dtype=np.float64)

    m = pos < cand_mz.size
    out[m] = np.minimum(out[m], np.abs(cand_mz[pos[m]] - true_mz[m]))

    m = pos > 0
    out[m] = np.minimum(out[m], np.abs(cand_mz[pos[m] - 1] - true_mz[m]))

    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--precursor-tol", type=float, default=0.05)
    ap.add_argument("--precursor-isotope-n", type=int, default=2)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    if args.sample_n > 0:
        paths = paths[:args.sample_n]

    tolerances = [0.002, 0.005, 0.01, 0.02, 0.05, 0.10]
    rows = []

    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        pmz = float(obj.get("precursor_mz", np.nan))
        true_peaks = exclude_precursor(
            to_peaks(obj.get("spect", [])),
            pmz,
            tol=args.precursor_tol,
            isotope_n=args.precursor_isotope_n,
        )
        if true_peaks.shape[0] <= 0:
            continue

        feat = obj.get("features", {})
        fmask = np.asarray(feat.get("formulae_mask", []), dtype=np.float32).reshape(-1) > 0.5
        peaks = np.asarray(feat.get("formulae_peaks", []), dtype=np.float64)

        if peaks.ndim != 3 or peaks.shape[-1] < 2:
            continue

        n = min(peaks.shape[0], fmask.shape[0])
        peaks = peaks[:n]
        fmask = fmask[:n]

        mz = peaks[..., 0]
        it = peaks[..., 1]
        valid = fmask[:, None] & np.isfinite(mz) & np.isfinite(it) & (it > 0) & (mz >= 0)
        cand_mz = mz[valid]

        true_mz = true_peaks[:, 0]
        true_int = true_peaks[:, 1]
        d = nearest_distances(true_mz, cand_mz)

        row = {"file": os.path.basename(p), "true_n": int(true_mz.size)}
        total_int = float(np.sum(true_int))
        for tol in tolerances:
            hit = d <= tol
            row[f"tol_{tol:g}_recall"] = float(np.mean(hit)) if hit.size else np.nan
            row[f"tol_{tol:g}_intcov"] = float(np.sum(true_int[hit]) / max(1e-12, total_int))
        row["nearest_p50"] = float(np.percentile(d[np.isfinite(d)], 50)) if np.isfinite(d).any() else np.inf
        row["nearest_p90"] = float(np.percentile(d[np.isfinite(d)], 90)) if np.isfinite(d).any() else np.inf
        rows.append(row)

    print(f"[mass-tol] n_samples={len(rows)} cache_dir={args.cache_dir}")
    if not rows:
        return

    keys = [k for k in rows[0].keys() if k != "file"]
    for k in keys:
        arr = np.asarray([r[k] for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            print(f"{k}: mean={arr.mean():.4f} p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f}")

    print("\nfirst 10 rows:")
    for r in rows[:10]:
        print(r)

if __name__ == "__main__":
    main()
