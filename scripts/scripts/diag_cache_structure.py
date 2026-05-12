#!/usr/bin/env python3
import os
import sys
import glob
import pickle
import json
import numpy as np
from pathlib import Path

def arr(x):
    if x is None:
        return None
    try:
        import torch
        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    try:
        return np.asarray(x)
    except Exception:
        return None

def stat_one(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)

    feat = obj.get("features", obj) if isinstance(obj, dict) else {}

    out = {"file": os.path.basename(path)}

    fm = arr(feat.get("formulae_mask"))
    if fm is not None:
        fm = fm.reshape(-1)
        out["formula_valid_n"] = int((fm > 0.5).sum())
    else:
        out["formula_valid_n"] = None

    active = arr(feat.get("formulae_active_mask"))
    if active is not None:
        active = active.reshape(-1)
        out["active_n"] = int((active > 0.5).sum())
    else:
        out["active_n"] = None

    teacher = arr(feat.get("teacher_formula_probs"))
    if teacher is not None:
        teacher = teacher.reshape(-1)
        out["teacher_pos_n"] = int((teacher > 1e-12).sum())
        out["teacher_entropy_like"] = float(-(teacher[teacher > 0] * np.log(teacher[teacher > 0] + 1e-12)).sum())
    else:
        out["teacher_pos_n"] = None
        out["teacher_entropy_like"] = None

    gid = arr(feat.get("formulae_instance_group_id"))
    if gid is not None:
        gid = gid.reshape(-1)
        if fm is not None and fm.shape[0] == gid.shape[0]:
            gid = gid[fm > 0.5]
        gid = gid[gid >= 0]
        out["instance_n"] = int(gid.shape[0])
        out["unique_group_n"] = int(np.unique(gid).shape[0]) if gid.size else 0
        out["dup_ratio"] = float(1.0 - out["unique_group_n"] / max(1, out["instance_n"]))
    else:
        out["instance_n"] = None
        out["unique_group_n"] = None
        out["dup_ratio"] = None

    fn_mask = arr(feat.get("fragment_node_mask"))
    fn_label = arr(feat.get("fragment_node_label"))
    fn_gid = arr(feat.get("fragment_node_group_formula_id"))
    if fn_mask is not None:
        fn_mask = fn_mask.reshape(-1)
        valid = fn_mask > 0.5
        out["fn_valid_n"] = int(valid.sum())
        if fn_label is not None:
            lab = fn_label.reshape(-1)
            out["fn_pos_n"] = int(((lab > 0.5) & valid).sum())
            out["fn_pos_ratio"] = float(((lab > 0.5) & valid).sum() / max(1, valid.sum()))
        if fn_gid is not None:
            g = fn_gid.reshape(-1)
            g = g[valid & (g >= 0)]
            out["fn_mapped_formula_n"] = int(np.unique(g).shape[0]) if g.size else 0
    else:
        out["fn_valid_n"] = None
        out["fn_pos_n"] = None
        out["fn_pos_ratio"] = None
        out["fn_mapped_formula_n"] = None

    true_idx = arr(feat.get("true_official_idx"))
    out["true_peak_n"] = int((true_idx.reshape(-1) >= 0).sum()) if true_idx is not None else None

    off_idx = arr(feat.get("formulae_peaks_official_idx_agg"))
    if off_idx is not None:
        out["cand_official_peak_mean"] = float(((off_idx >= 0).sum(axis=-1)).mean())
    else:
        out["cand_official_peak_mean"] = None

    return out

def main():
    cache_dir = sys.argv[1]
    sample_n = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pkl")))[:sample_n]
    rows = [stat_one(p) for p in files]

    keys = sorted(set(k for r in rows for k in r))
    print("\t".join(keys))
    for r in rows:
        print("\t".join(str(r.get(k, "")) for k in keys))

    nums = {}
    for k in keys:
        vals = []
        for r in rows:
            v = r.get(k)
            if isinstance(v, (int, float)) and np.isfinite(v):
                vals.append(float(v))
        if vals:
            nums[k] = {
                "mean": float(np.mean(vals)),
                "p50": float(np.percentile(vals, 50)),
                "p90": float(np.percentile(vals, 90)),
            }

    print("\nSUMMARY")
    print(json.dumps(nums, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
    