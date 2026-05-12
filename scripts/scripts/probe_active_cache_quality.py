#!/usr/bin/env python3
import argparse, glob, os, pickle, random
import numpy as np

def sparse_to_dense(idx, val, n):
    out = np.zeros((n,), dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    val = np.asarray(val, dtype=np.float32).reshape(-1)
    use = min(idx.size, val.size)
    idx, val = idx[:use], val[:use]
    m = (idx >= 0) & (idx < n) & np.isfinite(val) & (val > 0)
    if np.any(m):
        np.add.at(out, idx[m], val[m])
    return out

def build_union(off_idx, off_int, mask, nbin):
    off_idx = np.asarray(off_idx, dtype=np.int64)
    off_int = np.asarray(off_int, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    cn = min(off_idx.shape[0], off_int.shape[0], mask.shape[0])
    u = np.zeros((nbin,), dtype=np.float32)
    for i in range(cn):
        if not mask[i]:
            continue
        d = sparse_to_dense(off_idx[i], off_int[i], nbin)
        u = np.maximum(u, d)
    return u

def metrics(union, true):
    us = union > 0
    ts = true > 0
    inter = us & ts
    prec = inter.sum() / max(1, us.sum())
    rec = inter.sum() / max(1, ts.sum())
    cov = true[us].sum() / max(1e-12, true.sum())

    true_idx = np.nonzero(ts)[0]
    if true_idx.size > 0:
        k = min(20, true_idx.size)
        top = np.argsort(-true)[:k]
        top = set(int(x) for x in top if true[int(x)] > 0)
        top20 = len(top.intersection(set(np.nonzero(us)[0].tolist()))) / max(1, len(top))
    else:
        top20 = np.nan
    return prec, rec, cov, top20, int(us.sum()), int(ts.sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--official-max-mz", type=float, default=1005.0)
    ap.add_argument("--official-bin-width", type=float, default=0.01)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    rng = random.Random(args.seed)
    if args.sample_n > 0 and len(paths) > args.sample_n:
        paths = rng.sample(paths, args.sample_n)

    nbin = int(np.floor(args.official_max_mz / args.official_bin_width)) + 1

    rows = []
    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        feat = obj.get("features", {})
        true = sparse_to_dense(feat.get("true_official_idx", []), feat.get("true_official_intensity", []), nbin)

        off_idx = feat.get("formulae_peaks_official_idx_agg", feat.get("formulae_peaks_official_idx"))
        off_int = feat.get("formulae_peaks_official_intensity_agg", feat.get("formulae_peaks_official_intensity"))
        if off_idx is None or off_int is None:
            continue

        fmask = np.asarray(feat.get("formulae_mask", []), dtype=np.float32).reshape(-1) > 0.5
        src = np.asarray(feat.get("formulae_source_flag", np.zeros_like(fmask)), dtype=np.int64).reshape(-1)
        active = np.asarray(feat.get("formulae_active_mask", np.zeros_like(fmask)), dtype=np.float32).reshape(-1) > 0.5

        n = min(fmask.size, src.size, active.size, np.asarray(off_idx).shape[0])
        fmask = fmask[:n]
        src = src[:n]
        active = active[:n]

        masks = {
            "all": fmask,
            "active": fmask & active,
            "source": fmask & (src > 0),
            "struct": fmask & ((src == 1) | (src == 3)),
        }

        row = {"file": os.path.basename(p)}
        for name, m in masks.items():
            u = build_union(off_idx, off_int, m, nbin)
            prec, rec, cov, top20, usn, tsn = metrics(u, true)
            row[f"{name}_precision"] = prec
            row[f"{name}_support_recall"] = rec
            row[f"{name}_intensity_coverage"] = cov
            row[f"{name}_top20_recall"] = top20
            row[f"{name}_support_bins"] = usn
            row[f"{name}_candidate_n"] = int(m.sum())
            row["true_bins"] = tsn
        rows.append(row)

    if not rows:
        print("[active-probe] no rows")
        return

    keys = [k for k in rows[0].keys() if k != "file"]
    print(f"[active-probe] n_samples={len(rows)} cache_dir={args.cache_dir}")
    for k in keys:
        arr = np.asarray([r[k] for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            print(f"{k}: mean={arr.mean():.4f} p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f}")

if __name__ == "__main__":
    main()
