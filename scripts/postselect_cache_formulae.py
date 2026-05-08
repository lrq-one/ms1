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
    import os
    import heapq
    import numpy as np

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

    def _as1d_local(x, default=0.0, dtype=np.float32):
        if x is None:
            return np.full((n,), default, dtype=dtype)
        arr = np.asarray(x, dtype=dtype).reshape(-1)
        if arr.shape[0] < n:
            arr = np.concatenate(
                [arr, np.full((n - arr.shape[0],), default, dtype=dtype)],
                axis=0,
            )
        return arr[:n]

    source = _as1d_local(features.get("formulae_source_flag", None), 0.0)
    active = _as1d_local(features.get("formulae_active_mask", None), 0.0)
    prior = _as1d_local(features.get("formulae_prior_score", None), 0.0)
    break_depth = _as1d_local(features.get("formulae_break_depth", None), 9.0)
    ring_cut = _as1d_local(features.get("formulae_ring_cut_flag", None), 0.0)

    bw = float(os.environ.get("OFFICIAL_BIN_WIDTH", "0.01"))

    cand_bins = []
    min_bin = np.full((n,), 10**9, dtype=np.int64)

    for i in range(n):
        if not mask[i] or not np.any(valid_peak[i]):
            cand_bins.append(())
            continue

        bins = np.unique(off_idx[i][valid_peak[i]].astype(np.int64))
        bins = bins[bins >= 0]
        tup = tuple(int(x) for x in bins.tolist())
        cand_bins.append(tup)

        if bins.size > 0:
            min_bin[i] = int(np.min(bins))

    selected = []
    selected_set = set()

    def add_one(ci):
        ci = int(ci)
        if ci < 0 or ci >= n:
            return False
        if not mask[ci]:
            return False
        if ci in selected_set:
            return False
        selected.append(ci)
        selected_set.add(ci)
        return True

    def add_many(cands, limit):
        if limit <= 0:
            return
        for ci in cands:
            if len(selected) >= max_keep:
                return
            if add_one(ci):
                limit -= 1
                if limit <= 0:
                    return

    def base_score(ci):
        ci = int(ci)
        return (
            5.0 * float(source[ci] > 0)
            + 2.0 * float(active[ci] > 0)
            + float(prior[ci])
            + 0.03 * float(peak_n[ci])
            - 0.08 * float(min(max(break_depth[ci], 0), 10))
            - 0.20 * float(ring_cut[ci] > 0)
        )

    def region_bins_tuple(ci, lo_bin, hi_bin):
        out = []
        for b in cand_bins[int(ci)]:
            if b >= lo_bin and b < hi_bin:
                out.append(int(b))
        return tuple(out)

    def lazy_greedy_region_setcover(lo_mz, hi_mz, budget, name="region"):
        if budget <= 0 or len(selected) >= max_keep:
            return

        lo_bin = int(float(lo_mz) / max(1e-6, bw))
        hi_bin = int(float(hi_mz) / max(1e-6, bw))

        region_cache = {}
        heap = []

        # 已选候选也算覆盖，避免区域间重复浪费
        covered = set()
        for ci in selected:
            for b in region_bins_tuple(ci, lo_bin, hi_bin):
                covered.add(int(b))

        for ci in valid_ids:
            ci = int(ci)
            if ci in selected_set:
                continue

            rb = region_bins_tuple(ci, lo_bin, hi_bin)
            if len(rb) <= 0:
                continue

            region_cache[ci] = rb
            gain = len(rb)
            tie = base_score(ci)

            # Python heap 是小根堆，所以用负号实现最大堆
            heapq.heappush(heap, (-gain, -tie, int(min_bin[ci]), ci))

        picked = 0

        while heap and picked < budget and len(selected) < max_keep:
            neg_gain_old, neg_tie, mb, ci = heapq.heappop(heap)
            ci = int(ci)

            if ci in selected_set:
                continue

            rb = region_cache.get(ci, ())
            if not rb:
                continue

            true_gain = 0
            for b in rb:
                if b not in covered:
                    true_gain += 1

            if true_gain <= 0:
                continue

            # lazy greedy:
            # 如果重新计算后的 gain 仍然足够，就选它；
            # 否则把更新后的 gain 放回堆，等待下一轮比较。
            old_gain = -int(neg_gain_old)
            if true_gain < old_gain:
                heapq.heappush(heap, (-true_gain, neg_tie, mb, ci))
                continue

            add_one(ci)
            for b in rb:
                covered.add(int(b))
            picked += 1

        # 如果 set cover 没用完预算，用 region 内 base score 补足
        if picked < budget and len(selected) < max_keep:
            rest = []
            for ci, rb in region_cache.items():
                if ci in selected_set:
                    continue
                if not rb:
                    continue
                rest.append(ci)

            if rest:
                rest = np.asarray(rest, dtype=np.int64)
                scores = np.asarray([base_score(int(ci)) for ci in rest], dtype=np.float32)
                rest = rest[np.argsort(-scores, kind="stable")]
                add_many(rest, budget - picked)

    # 1. source / active 只保少量，避免挤掉 coverage
    source_budget = int(os.environ.get("POSTSELECT_SOURCE_BUDGET", "256"))
    source_scores = np.asarray([base_score(int(ci)) for ci in valid_ids], dtype=np.float32)
    source_order = valid_ids[np.argsort(-source_scores, kind="stable")]
    add_many(source_order, source_budget)

    # 2. 分区 lazy set cover
    low_budget = int(os.environ.get("POSTSELECT_LOW_MZ_BUDGET", "1024"))
    mid_budget = int(os.environ.get("POSTSELECT_MID_MZ_BUDGET", "1536"))
    high_budget = int(os.environ.get("POSTSELECT_HIGH_MZ_BUDGET", "768"))

    low_hi = float(os.environ.get("POSTSELECT_LOW_MZ_MAX", str(low_mz_max)))
    mid_hi = float(os.environ.get("POSTSELECT_MID_MZ_MAX", "500.0"))
    high_hi = float(os.environ.get("POSTSELECT_HIGH_MZ_MAX", "1500.0"))

    lazy_greedy_region_setcover(0.0, low_hi, low_budget, name="low")
    lazy_greedy_region_setcover(low_hi, mid_hi, mid_budget, name="mid")
    lazy_greedy_region_setcover(mid_hi, high_hi, high_budget, name="high")

    # 3. fill：补满 4096
    if len(selected) < max_keep:
        rest = np.asarray([ci for ci in valid_ids if int(ci) not in selected_set], dtype=np.int64)
        if rest.size > 0:
            scores = np.asarray([base_score(int(ci)) for ci in rest], dtype=np.float32)
            rest = rest[np.argsort(-scores, kind="stable")]
            add_many(rest, max_keep - len(selected))

    # 4. 兜底
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

    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="postselect", ncols=120)
    except Exception:
        iterator = paths

    for ii, p in enumerate(iterator, 1):
        obj = pickle.load(open(p, "rb"))
        obj2 = postselect_one(obj, max_keep=args.max_keep, low_mz_max=args.low_mz_max)
        out_p = os.path.join(args.out_dir, os.path.basename(p))
        with open(out_p, "wb") as f:
            pickle.dump(obj2, f, protocol=pickle.HIGHEST_PROTOCOL)

        if not hasattr(iterator, "set_description") and ii % 50 == 0:
            print(f"[postselect] processed={ii}/{len(paths)}", flush=True)

    print("written:", args.out_dir)

if __name__ == "__main__":
    main()