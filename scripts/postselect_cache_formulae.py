#!/usr/bin/env python3
import os
import glob
import pickle
import argparse
import shutil
import numpy as np

FORMULA_KEYS_PREFIX = (
    "formulae_",
)

def _as1d(x, n, default=0.0, dtype=np.float32):
    if x is None:
        return np.full((n,), default, dtype=dtype)
    arr = np.asarray(x, dtype=dtype).reshape(-1)
    if arr.shape[0] < n:
        pad = np.full((n - arr.shape[0],), default, dtype=dtype)
        arr = np.concatenate([arr, pad], axis=0)
    return arr[:n]

def _get_features(obj):
    if isinstance(obj, dict) and isinstance(obj.get("features", None), dict):
        return obj["features"]
    if isinstance(obj, dict):
        return obj
    return {}

def _select_indices(features, max_keep=4096, low_mz_max=220.0):
    mask = np.asarray(features.get("formulae_mask", []), dtype=np.float32).reshape(-1) > 0.5
    M = int(mask.shape[0])
    if M <= 0:
        return np.zeros((0,), dtype=np.int64)

    valid_ids = np.where(mask)[0]
    if valid_ids.size <= max_keep:
        return valid_ids.astype(np.int64)

    off_idx = np.asarray(features.get("formulae_peaks_official_idx", []), dtype=np.int64)
    off_int = np.asarray(
        features.get(
            "formulae_peaks_official_intensity",
            features.get("formulae_peaks_intensity", []),
        ),
        dtype=np.float32,
    )

    if off_idx.ndim != 2:
        return valid_ids[:max_keep].astype(np.int64)

    if off_int.shape != off_idx.shape:
        off_int = np.ones_like(off_idx, dtype=np.float32)

    n = min(M, off_idx.shape[0], off_int.shape[0])
    mask = mask[:n]
    valid_ids = np.where(mask)[0]
    off_idx = off_idx[:n]
    off_int = off_int[:n]

    valid_peak = (off_idx >= 0) & np.isfinite(off_int) & (off_int > 0)
    peak_n = valid_peak.sum(axis=1).astype(np.float32)

    min_bin = np.full((n,), 10**9, dtype=np.int64)
    max_bin = np.full((n,), -1, dtype=np.int64)

    for i in range(n):
        if mask[i] and np.any(valid_peak[i]):
            vals = off_idx[i][valid_peak[i]]
            min_bin[i] = int(np.min(vals))
            max_bin[i] = int(np.max(vals))

    source = _as1d(features.get("formulae_source_flag", None), n, 0.0)
    active = _as1d(features.get("formulae_active_mask", None), n, 0.0)
    prior = _as1d(features.get("formulae_prior_score", None), n, 0.0)
    break_depth = _as1d(features.get("formulae_break_depth", None), n, 9.0)
    ring_cut = _as1d(features.get("formulae_ring_cut_flag", None), n, 0.0)

    selected = []
    selected_set = set()

    def add_many(cands, limit):
        nonlocal selected, selected_set
        if limit <= 0:
            return
        for ci in cands:
            ci = int(ci)
            if ci < 0 or ci >= n:
                continue
            if not mask[ci]:
                continue
            if ci in selected_set:
                continue
            selected.append(ci)
            selected_set.add(ci)
            if len(selected) >= max_keep:
                return
            limit -= 1
            if limit <= 0:
                return

    source_budget = int(os.environ.get("POSTSELECT_SOURCE_BUDGET", "1024"))
    low_budget = int(os.environ.get("POSTSELECT_LOW_MZ_BUDGET", "1536"))
    diverse_budget = int(os.environ.get("POSTSELECT_DIVERSE_BUDGET", "1024"))

    # 1. source / active / prior-supported candidates
    source_score = (
        8.0 * (source > 0).astype(np.float32)
        + 4.0 * (active > 0).astype(np.float32)
        + prior
        + 0.05 * peak_n
        - 0.10 * np.clip(break_depth, 0, 10)
        - 0.30 * (ring_cut > 0).astype(np.float32)
    )
    source_ids = valid_ids[np.argsort(-source_score[valid_ids], kind="stable")]
    add_many(source_ids, source_budget)

    # 2. low-mz candidates: 修复 72/86/96/110 这种高能 HCD 小离子
    bw = float(os.environ.get("OFFICIAL_BIN_WIDTH", "0.01"))
    low_bin_max = int(float(low_mz_max) / max(1e-6, bw))
    low_ids = valid_ids[min_bin[valid_ids] <= low_bin_max]

    low_score = (
        2.0 * (source[low_ids] > 0).astype(np.float32)
        + 1.0 * (active[low_ids] > 0).astype(np.float32)
        + prior[low_ids]
        + 0.05 * peak_n[low_ids]
        - 0.00002 * min_bin[low_ids].astype(np.float32)
    )
    low_ids = low_ids[np.argsort(-low_score, kind="stable")]
    add_many(low_ids, low_budget)

    # 3. mass-diverse candidates: 避免只保留低 m/z，保留中高 m/z 结构峰
    bucket_da = float(os.environ.get("POSTSELECT_BUCKET_DA", "25.0"))
    bucket_size = max(1, int(bucket_da / max(1e-6, bw)))
    bucket = np.floor_divide(np.maximum(min_bin, 0), bucket_size)

    diverse_order = []
    per_bucket = int(os.environ.get("POSTSELECT_PER_BUCKET", "32"))

    for b in sorted(set(bucket[valid_ids].tolist())):
        ids_b = valid_ids[bucket[valid_ids] == b]
        if ids_b.size <= 0:
            continue

        score_b = (
            4.0 * (source[ids_b] > 0).astype(np.float32)
            + 2.0 * (active[ids_b] > 0).astype(np.float32)
            + prior[ids_b]
            + 0.03 * peak_n[ids_b]
        )
        ids_b = ids_b[np.argsort(-score_b, kind="stable")]
        diverse_order.extend(ids_b[:per_bucket].tolist())

        if len(diverse_order) >= diverse_budget:
            break

    add_many(diverse_order, diverse_budget)

    # 4. fill: 先验 + source + active
    fill_score = (
        4.0 * (source > 0).astype(np.float32)
        + 2.0 * (active > 0).astype(np.float32)
        + prior
        + 0.03 * peak_n
        - 0.05 * np.clip(break_depth, 0, 10)
    )
    fill_ids = valid_ids[np.argsort(-fill_score[valid_ids], kind="stable")]
    add_many(fill_ids, max_keep - len(selected))

    # 5. 如果还没满，按原始顺序补
    if len(selected) < max_keep:
        add_many(valid_ids, max_keep - len(selected))

    return np.asarray(selected[:max_keep], dtype=np.int64)

def _slice_first_axis(arr, take, out_n, pad_value=0):
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr

    if arr.shape[0] == 0:
        shape = (out_n,) + arr.shape[1:]
        return np.full(shape, pad_value, dtype=arr.dtype)

    take = np.asarray(take, dtype=np.int64)
    take = take[(take >= 0) & (take < arr.shape[0])]

    out = arr[take]
    if out.shape[0] < out_n:
        pad_shape = (out_n - out.shape[0],) + out.shape[1:]
        pad = np.full(pad_shape, pad_value, dtype=out.dtype)
        out = np.concatenate([out, pad], axis=0)
    return out[:out_n]

def _pad_value_for_key(key):
    if key.endswith("_idx") or "idx" in key:
        return -1
    return 0

def postselect_one(obj, max_keep=4096, low_mz_max=220.0):
    features = _get_features(obj)
    if not isinstance(features, dict):
        return obj

    mask = np.asarray(features.get("formulae_mask", []), dtype=np.float32).reshape(-1)
    if mask.size <= max_keep:
        return obj

    take = _select_indices(features, max_keep=max_keep, low_mz_max=low_mz_max)

    new_features = {}
    M = int(mask.shape[0])

    for k, v in features.items():
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == M and k.startswith("formulae_"):
            new_features[k] = _slice_first_axis(
                v,
                take,
                out_n=max_keep,
                pad_value=_pad_value_for_key(k),
            )
        else:
            new_features[k] = v

    new_mask = np.zeros((max_keep,), dtype=np.float32)
    valid_n = min(int(take.shape[0]), max_keep)
    new_mask[:valid_n] = 1.0
    new_features["formulae_mask"] = new_mask

    new_features["postselect_original_candidate_n"] = np.asarray([M], dtype=np.int64)
    new_features["postselect_selected_candidate_n"] = np.asarray([valid_n], dtype=np.int64)
    new_features["postselect_mode"] = "balanced_v1"

    if isinstance(obj, dict) and isinstance(obj.get("features", None), dict):
        obj = dict(obj)
        obj["features"] = new_features
        obj["postselect_info"] = {
            "mode": "balanced_v1",
            "original_candidate_n": int(M),
            "selected_candidate_n": int(valid_n),
            "max_keep": int(max_keep),
            "low_mz_max": float(low_mz_max),
        }
        return obj

    return new_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-keep", type=int, default=4096)
    ap.add_argument("--low-mz-max", type=float, default=220.0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if os.path.exists(args.out_dir):
        if args.overwrite:
            shutil.rmtree(args.out_dir)
        else:
            raise SystemExit(f"out-dir exists: {args.out_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    paths = sorted(glob.glob(os.path.join(args.in_dir, "*.pkl")))
    print("input pkl:", len(paths))

    for p in paths:
        obj = pickle.load(open(p, "rb"))
        obj2 = postselect_one(obj, max_keep=args.max_keep, low_mz_max=args.low_mz_max)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        with open(out_p, "wb") as f:
            pickle.dump(obj2, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("written:", args.out_dir)

if __name__ == "__main__":
    main()