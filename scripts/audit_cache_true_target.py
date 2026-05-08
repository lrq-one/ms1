#!/usr/bin/env python3
import os, glob, pickle, argparse, math
import numpy as np

def official_bin_indices(mz, bin_width, mode="floor"):
    mz = np.asarray(mz, dtype=np.float64)
    bw = float(max(1e-6, bin_width))
    if mode in ("round", "nearest", "nominal"):
        return np.rint(mz / bw).astype(np.int64)
    return np.floor(mz / bw + 1e-8).astype(np.int64)

def to_peak_matrix(x):
    if x is None:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        arr = np.asarray(x, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        arr = arr.reshape(-1, 2)
        valid = np.isfinite(arr[:,0]) & np.isfinite(arr[:,1]) & (arr[:,1] > 0)
        return arr[valid].astype(np.float32)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

def build_true_from_spect(spect, bin_width, max_mz, mode):
    peaks = to_peak_matrix(spect)
    if peaks.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    mz = peaks[:,0].astype(np.float64)
    inten = peaks[:,1].astype(np.float64)
    valid = (
        np.isfinite(mz)
        & np.isfinite(inten)
        & (inten > 0)
        & (mz >= 0)
        & (mz < float(max_mz))
    )
    mz = mz[valid]
    inten = inten[valid]
    if mz.size == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    idx = official_bin_indices(mz, bin_width, mode)
    uniq, inv = np.unique(idx, return_inverse=True)
    val = np.zeros((uniq.shape[0],), dtype=np.float64)
    np.add.at(val, inv, inten)
    order = np.argsort(uniq, kind="stable")
    return uniq[order].astype(np.int64), val[order].astype(np.float32)

def dense(idx, val, n):
    out = np.zeros((n,), dtype=np.float32)
    idx = np.asarray(idx, dtype=np.int64).reshape(-1)
    val = np.asarray(val, dtype=np.float32).reshape(-1)
    use = min(idx.size, val.size)
    idx = idx[:use]
    val = val[:use]
    m = (idx >= 0) & (idx < n) & np.isfinite(val) & (val > 0)
    if np.any(m):
        np.add.at(out, idx[m], val[m])
    return out

def cosine(a, b):
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / (den + 1e-12)) if den > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--sample-n", type=int, default=200)
    ap.add_argument("--official-bin-width", type=float, default=0.01)
    ap.add_argument("--official-max-mz", type=float, default=1005.0)
    ap.add_argument("--official-bin-mode", default=os.environ.get("OFFICIAL_BIN_MODE", "floor"))
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))[:args.sample_n]
    bin_n = int(math.floor(args.official_max_mz / args.official_bin_width)) + 1

    rows = []
    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)

        feat = obj.get("features", {}) if isinstance(obj, dict) else {}

        spect = obj.get("spect", None)
        if spect is None:
            spect = obj.get("spect_raw", None)
        if spect is None:
            spect = feat.get("spect", None)
        if spect is None:
            rows.append((os.path.basename(p), "NO_SPECT", None, None, None))
            continue

        idx_re, val_re = build_true_from_spect(
            spect,
            args.official_bin_width,
            args.official_max_mz,
            args.official_bin_mode,
        )

        idx_cache = feat.get("true_official_idx", None)
        val_cache = feat.get("true_official_intensity", None)

        if idx_cache is None or val_cache is None:
            rows.append((os.path.basename(p), "NO_TRUE_CACHE", None, None, None))
            continue

        d_re = dense(idx_re, val_re, bin_n)
        d_cache = dense(idx_cache, val_cache, bin_n)

        cos = cosine(d_re, d_cache)
        support_same = bool(np.array_equal(np.nonzero(d_re > 0)[0], np.nonzero(d_cache > 0)[0]))

        rows.append((
            os.path.basename(p),
            "OK",
            cos,
            int((d_re > 0).sum()),
            int((d_cache > 0).sum()),
            support_same,
        ))

    ok = [r for r in rows if r[1] == "OK"]
    print("checked:", len(rows))
    print("ok:", len(ok))
    if ok:
        cos_arr = np.asarray([r[2] for r in ok], dtype=float)
        print("cos mean:", float(np.mean(cos_arr)))
        print("cos min:", float(np.min(cos_arr)))
        print("cos p50:", float(np.percentile(cos_arr, 50)))
        print("support_same ratio:", sum(bool(r[5]) for r in ok) / max(1, len(ok)))

    print("\nfirst 20 rows:")
    for r in rows[:20]:
        print(r)

if __name__ == "__main__":
    main()