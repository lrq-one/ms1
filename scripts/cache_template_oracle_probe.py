#!/usr/bin/env python3
import argparse
import glob
import os
import pickle
import random
from collections import Counter, defaultdict

import numpy as np


def get_nested(m, names):
    for name in names:
        if name in m:
            return m[name], name

    feat = m.get("features", None)
    if isinstance(feat, dict):
        for name in names:
            if name in feat:
                return feat[name], f"features.{name}"

    return None, None


def to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def make_true_dense(m, bin_n):
    # 优先 official sparse true target
    idx_names = [
        "true_all_official_idx",
        "true_official_idx",
        "target_official_idx",
        "spect_official_idx",
    ]
    int_names = [
        "true_all_official_intensity",
        "true_official_intensity",
        "target_official_intensity",
        "spect_official_intensity",
    ]

    idx, idx_src = get_nested(m, idx_names)
    inten, int_src = get_nested(m, int_names)

    idx = to_np(idx)
    inten = to_np(inten)

    if idx is not None and inten is not None:
        y = np.zeros((bin_n,), dtype=np.float32)

        idx = idx.reshape(-1).astype(np.int64, copy=False)
        inten = inten.reshape(-1).astype(np.float32, copy=False)

        ok = (idx >= 0) & (idx < bin_n) & np.isfinite(inten) & (inten > 0)
        if ok.any():
            np.add.at(y, idx[ok], inten[ok])

        s = float(y.sum())
        if s > 0:
            y /= s
            return y, f"{idx_src}+{int_src}"

    # fallback: dense
    dense_names = [
        "spect_dense",
        "true_dense",
        "true_official_dense",
        "target_dense",
    ]
    dense, dense_src = get_nested(m, dense_names)
    dense = to_np(dense)

    if dense is not None:
        dense = dense.reshape(-1).astype(np.float32, copy=False)
        if dense.shape[0] < bin_n:
            dense = np.pad(dense, (0, bin_n - dense.shape[0]))
        dense = dense[:bin_n]
        dense = np.where(np.isfinite(dense), dense, 0.0)
        dense = np.clip(dense, 0.0, None)

        s = float(dense.sum())
        if s > 0:
            dense = dense / s
            return dense, dense_src

    return None, None


def get_candidate_peaks(m):
    idx_names = [
        "formulae_peaks_official_idx",
        "formulae_peaks_official_idx_agg",
        "formulae_peaks_mass_idx",
        "formulae_peaks_idx",
    ]
    int_names = [
        "formulae_peaks_official_intensity",
        "formulae_peaks_official_intensity_agg",
        "formulae_peaks_intensity",
    ]

    idx, idx_src = get_nested(m, idx_names)
    inten, int_src = get_nested(m, int_names)

    idx = to_np(idx)
    inten = to_np(inten)

    if idx is None or inten is None:
        return None, None, None

    if idx.ndim == 1:
        idx = idx[None, :]
    if inten.ndim == 1:
        inten = inten[None, :]

    if idx.ndim > 2:
        idx = idx.reshape(idx.shape[0], -1)
    if inten.ndim > 2:
        inten = inten.reshape(inten.shape[0], -1)

    m_n = min(idx.shape[0], inten.shape[0])
    p_n = min(idx.shape[1], inten.shape[1])

    idx = idx[:m_n, :p_n].astype(np.int64, copy=False)
    inten = inten[:m_n, :p_n].astype(np.float32, copy=False)

    return idx, inten, f"{idx_src}+{int_src}"


def normalize_candidate_row(idx_row, int_row, bin_n):
    ok = (
        (idx_row >= 0)
        & (idx_row < bin_n)
        & np.isfinite(int_row)
        & (int_row > 0)
    )
    if not ok.any():
        return None, None

    idx = idx_row[ok].astype(np.int64, copy=False)
    val = int_row[ok].astype(np.float32, copy=False)

    # 合并重复 bin
    uniq, inv = np.unique(idx, return_inverse=True)
    acc = np.zeros((uniq.shape[0],), dtype=np.float32)
    np.add.at(acc, inv, val)

    s = float(acc.sum())
    if s <= 0:
        return None, None

    acc = acc / s
    return uniq, acc


def sparse_cos_to_true(idx, val, true_dense):
    if idx is None or val is None:
        return 0.0
    dot = float((val * true_dense[idx]).sum())
    n1 = float(np.sqrt((val * val).sum()))
    n2 = float(np.linalg.norm(true_dense))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 0.0
    return dot / (n1 * n2)


def audit_one(m, bin_n, topks):
    true_dense, true_src = make_true_dense(m, bin_n)
    if true_dense is None or float(true_dense.sum()) <= 0:
        return None, {"missing_true": 1}

    cand_idx, cand_int, cand_src = get_candidate_peaks(m)
    if cand_idx is None or cand_int is None:
        return None, {"missing_candidate": 1}

    M = int(cand_idx.shape[0])
    if M <= 0:
        return None, {"empty_candidate": 1}

    true_mask = true_dense > 1e-12
    true_nnz = int(true_mask.sum())

    scores = np.full((M,), -1e9, dtype=np.float32)
    false_mass = np.zeros((M,), dtype=np.float32)
    overlap_mass = np.zeros((M,), dtype=np.float32)
    single_cos = np.zeros((M,), dtype=np.float32)

    valid_rows = 0

    for i in range(M):
        idx, val = normalize_candidate_row(cand_idx[i], cand_int[i], bin_n)
        if idx is None:
            continue

        valid_rows += 1

        hit = true_mask[idx]
        overlap = float(val[hit].sum()) if hit.any() else 0.0
        false = 1.0 - overlap

        # weighted overlap with true intensity
        true_dot = float((val * true_dense[idx]).sum())

        overlap_mass[i] = overlap
        false_mass[i] = false
        single_cos[i] = sparse_cos_to_true(idx, val, true_dense)

        # oracle ranking score: 真峰命中越高越好，假峰略惩罚
        scores[i] = true_dot + 0.10 * overlap - 0.02 * false

    if valid_rows <= 0:
        return None, {"no_valid_candidate_rows": 1}

    order = np.argsort(-scores)

    out = {
        "M": float(M),
        "valid_rows": float(valid_rows),
        "true_nnz": float(true_nnz),
        "best_single_cos": float(np.max(single_cos)),
        "mean_single_cos": float(np.mean(single_cos[scores > -1e8])),
        "candidate_overlap_rate": float((overlap_mass > 0).mean()),
        "mean_candidate_overlap_mass": float(np.mean(overlap_mass)),
        "mean_candidate_false_mass": float(np.mean(false_mass)),
        "true_source": true_src,
        "candidate_source": cand_src,
    }

    for k in topks:
        kk = min(int(k), M)
        if kk <= 0:
            continue

        sel = order[:kk]
        sel_scores = scores[sel]

        # 只用非负 oracle 权重；如果全非正，则 uniform
        w = np.clip(sel_scores, 0.0, None)
        if float(w.sum()) <= 1e-12:
            w = np.ones((kk,), dtype=np.float32) / float(kk)
        else:
            w = w / float(w.sum())

        pred = np.zeros((bin_n,), dtype=np.float32)
        covered = np.zeros((bin_n,), dtype=bool)

        for wi, ci in zip(w, sel):
            idx, val = normalize_candidate_row(cand_idx[ci], cand_int[ci], bin_n)
            if idx is None:
                continue
            pred[idx] += float(wi) * val
            covered[idx] = True

        ps = float(pred.sum())
        if ps > 0:
            pred /= ps

        dot = float(np.dot(pred, true_dense))
        den = float(np.linalg.norm(pred) * np.linalg.norm(true_dense))
        cos = dot / den if den > 1e-12 else 0.0

        out[f"oracle_cos@{k}"] = float(cos)
        out[f"true_coverage@{k}"] = float(true_dense[covered].sum())
        out[f"pred_false_mass@{k}"] = float(pred[~true_mask].sum())

    return out, {}


def summarize(rows, key):
    vals = []
    for r in rows:
        v = r.get(key, None)
        if isinstance(v, (int, float)) and np.isfinite(v):
            vals.append(float(v))

    if not vals:
        return None

    vals = np.asarray(vals, dtype=np.float32)
    return {
        "mean": float(vals.mean()),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bin_n", type=int, default=100501)
    ap.add_argument("--topks", default="64,128,256,512,768,1024,1536,2048,4096")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    if not files:
        raise SystemExit(f"No pkl files found: {args.cache_dir}")

    random.seed(args.seed)
    if args.n > 0 and len(files) > args.n:
        files = random.sample(files, args.n)

    topks = [int(x.strip()) for x in args.topks.split(",") if x.strip()]

    rows = []
    bad_counter = Counter()
    true_src_counter = Counter()
    cand_src_counter = Counter()

    for p in files:
        try:
            with open(p, "rb") as f:
                m = pickle.load(f)

            r, bad = audit_one(m, args.bin_n, topks)

            if bad:
                bad_counter.update(bad)

            if r is not None:
                rows.append(r)
                true_src_counter[str(r.get("true_source"))] += 1
                cand_src_counter[str(r.get("candidate_source"))] += 1

        except Exception as e:
            bad_counter[f"exception:{type(e).__name__}"] += 1
            print("[BAD]", p, type(e).__name__, e)

    print("======== CACHE TEMPLATE ORACLE PROBE ========")
    print("cache_dir:", args.cache_dir)
    print("files_seen:", len(files))
    print("valid_rows:", len(rows))
    print("bad_counter:", dict(bad_counter))
    print("true_source_counter:", dict(true_src_counter))
    print("candidate_source_counter:", dict(cand_src_counter))

    if not rows:
        raise SystemExit("No valid rows. Need to inspect cache keys first.")

    keys = [
        "M",
        "valid_rows",
        "true_nnz",
        "best_single_cos",
        "mean_single_cos",
        "candidate_overlap_rate",
        "mean_candidate_overlap_mass",
        "mean_candidate_false_mass",
    ]

    for k in topks:
        keys.extend([
            f"oracle_cos@{k}",
            f"true_coverage@{k}",
            f"pred_false_mass@{k}",
        ])

    for key in keys:
        s = summarize(rows, key)
        if s is None:
            continue
        print(
            f"{key}: "
            f"mean={s['mean']:.4f} "
            f"median={s['median']:.4f} "
            f"p10={s['p10']:.4f} "
            f"p90={s['p90']:.4f}"
        )


if __name__ == "__main__":
    main()