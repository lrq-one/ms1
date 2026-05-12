#!/usr/bin/env python3
import argparse, glob, os, pickle
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--bin-width", type=float, default=0.01)
    ap.add_argument("--proton-mass", type=float, default=1.007276466812)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    if args.sample_n > 0:
        paths = paths[:args.sample_n]

    shift_bins = int(round(float(args.proton_mass) / float(args.bin_width)))
    rows = []

    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        feat = obj["features"]

        true_idx = np.asarray(feat.get("true_official_idx", []), dtype=np.int64).reshape(-1)
        true_int = np.asarray(feat.get("true_official_intensity", []), dtype=np.float32).reshape(-1)
        m = (true_idx >= 0) & np.isfinite(true_int) & (true_int > 0)
        true_idx = true_idx[m]
        true_int = true_int[m]
        if true_idx.size <= 0:
            continue

        off_idx = feat.get("formulae_peaks_official_idx_agg", feat.get("formulae_peaks_official_idx"))
        fmask = np.asarray(feat.get("formulae_mask", []), dtype=np.float32).reshape(-1) > 0.5
        off_idx = np.asarray(off_idx, dtype=np.int64)
        use_n = min(off_idx.shape[0], fmask.shape[0])

        support = set()
        for i in range(use_n):
            if not fmask[i]:
                continue
            row = off_idx[i]
            row = row[row >= 0]
            support.update(int(x) for x in row.tolist())

        true_set = set(int(x) for x in true_idx.tolist())
        plus_set = set(int(x + shift_bins) for x in true_idx.tolist())
        minus_set = set(int(x - shift_bins) for x in true_idx.tolist())

        def recall(s):
            return len(s & support) / max(1, len(s))

        def intensity_cov(s):
            keep = np.asarray([int(b) in support for b in s], dtype=bool)
            return float(true_int[keep].sum() / max(1e-12, true_int.sum()))

        rows.append({
            "file": os.path.basename(p),
            "true_recall": recall(true_set),
            "plusH_recall": recall(plus_set),
            "minusH_recall": recall(minus_set),
            "true_int_cov": intensity_cov(true_idx),
            "plusH_int_cov": intensity_cov(true_idx + shift_bins),
            "minusH_int_cov": intensity_cov(true_idx - shift_bins),
            "true_n": int(true_idx.size),
        })

    print("shift_bins=", shift_bins)
    for k in ["true_recall", "plusH_recall", "minusH_recall", "true_int_cov", "plusH_int_cov", "minusH_int_cov", "true_n"]:
        arr = np.asarray([r[k] for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            print(f"{k}: mean={arr.mean():.4f} p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f}")

    print("\nfirst 10 rows:")
    for r in rows[:10]:
        print(r)

if __name__ == "__main__":
    main()
