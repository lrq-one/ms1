import math

import numpy as np
import torch

from rassp.training.spectrum_targets import (
    _to_sparse_peak_array,
    _official_bin_indices_np,
    _precursor_keep_mask_np,
)


def _bin_peaks_to_official_sparse(
    peaks,
    bin_width=0.01,
    max_mz=1005.0,
    exclude_precursor=False,
    precursor_mz=None,
):
    if peaks is None:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)
    peaks = np.asarray(peaks, dtype=np.float32)
    if peaks.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    if peaks.ndim != 2 or peaks.shape[1] != 2:
        try:
            peaks = peaks.reshape(-1, 2)
        except Exception:
            return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    mz = peaks[:, 0]
    intensity = peaks[:, 1]
    valid = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    mz = mz[valid]
    intensity = intensity[valid]

    max_bin = int(math.floor(float(max_mz) / float(bin_width)))
    idx = _official_bin_indices_np(mz, bin_width).astype(np.int32)
    valid = (idx >= 0) & (idx <= max_bin)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    mz = mz[valid]
    idx = idx[valid]
    intensity = intensity[valid]

    if exclude_precursor:
        keep = _precursor_keep_mask_np(
            mz=mz,
            precursor_mz=precursor_mz,
            bin_width=bin_width,
            exclude_precursor=True,
        )
        idx = idx[keep]
        intensity = intensity[keep]

    if idx.size == 0:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.float32)

    order = np.argsort(idx, kind="mergesort")
    idx_sorted = idx[order]
    int_sorted = intensity[order]
    uniq_idx, start = np.unique(idx_sorted, return_index=True)
    uniq_int = np.add.reduceat(int_sorted, start).astype(np.float32)
    return uniq_idx.astype(np.int32), uniq_int


def _cosine_sparse(idx_a, val_a, idx_b, val_b, sqrt_intensity=False):
    if idx_a.size == 0 or idx_b.size == 0:
        return float(0.0)

    a = np.asarray(val_a, dtype=np.float64)
    b = np.asarray(val_b, dtype=np.float64)
    if sqrt_intensity:
        a = np.sqrt(np.clip(a, 0.0, None))
        b = np.sqrt(np.clip(b, 0.0, None))

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= 1e-12 or nb <= 1e-12:
        return float(0.0)

    i = 0
    j = 0
    dot = 0.0
    while i < idx_a.size and j < idx_b.size:
        ia = int(idx_a[i])
        ib = int(idx_b[j])
        if ia == ib:
            dot += float(a[i] * b[j])
            i += 1
            j += 1
        elif ia < ib:
            i += 1
        else:
            j += 1

    return float(dot / (na * nb + 1e-12))


def _js_similarity_sparse(idx_a, val_a, idx_b, val_b):
    if idx_a.size == 0 and idx_b.size == 0:
        return float(1.0)
    if idx_a.size == 0 or idx_b.size == 0:
        return float(0.0)

    pa = np.asarray(val_a, dtype=np.float64)
    pb = np.asarray(val_b, dtype=np.float64)
    sa = float(pa.sum())
    sb = float(pb.sum())
    if sa <= 1e-12 or sb <= 1e-12:
        return float(0.0)
    pa = pa / sa
    pb = pb / sb

    i = 0
    j = 0
    js = 0.0
    while i < idx_a.size or j < idx_b.size:
        if j >= idx_b.size or (i < idx_a.size and int(idx_a[i]) < int(idx_b[j])):
            a = float(pa[i])
            b = 0.0
            i += 1
        elif i >= idx_a.size or int(idx_b[j]) < int(idx_a[i]):
            a = 0.0
            b = float(pb[j])
            j += 1
        else:
            a = float(pa[i])
            b = float(pb[j])
            i += 1
            j += 1

        m = 0.5 * (a + b)
        if a > 0:
            js += 0.5 * a * math.log(a / m)
        if b > 0:
            js += 0.5 * b * math.log(b / m)

    js = max(0.0, float(js))
    js_sim = 1.0 - (js / math.log(2.0))
    return float(np.clip(js_sim, 0.0, 1.0))


def _matched_intensity_coverage(pred_idx, true_idx, true_val):
    if true_idx.size == 0:
        return float(np.nan)
    total_true = float(np.sum(true_val))
    if total_true <= 1e-12:
        return float(np.nan)
    pred_set = set(int(x) for x in pred_idx.tolist())
    keep = np.array([int(x) in pred_set for x in true_idx.tolist()], dtype=bool)
    covered = float(np.sum(true_val[keep]))
    return float(covered / total_true)


def _topk_peak_recall(pred_idx, true_idx, true_val, k=20):
    if true_idx.size == 0:
        return float(np.nan)
    k = max(1, min(int(k), int(true_idx.size)))
    order = np.argsort(-true_val, kind="mergesort")[:k]
    true_top = set(int(x) for x in true_idx[order].tolist())
    if len(true_top) == 0:
        return float(np.nan)
    pred_set = set(int(x) for x in pred_idx.tolist())
    hit = len(true_top.intersection(pred_set))
    return float(hit / float(len(true_top)))


def compute_batch_official_metrics(
    raw_batch,
    pred_spect,
    spect_bin_centers,
    metric_cfg,
    pred_exact_peaks=None,
    debug_ctx=None,
):
    """Compute official-style and diagnostic metrics for one validation batch."""
    pred_np = pred_spect.detach().cpu().numpy().astype(np.float32)

    true_raw_list = raw_batch.get("spect_raw", None)
    if true_raw_list is None:
        true_raw_list = raw_batch.get("spect", None)

    precursor_list = raw_batch.get("precursor_mz", None)
    if precursor_list is None:
        precursor_list = [None] * pred_np.shape[0]

    out = {
        "official_cos_no_precursor": [],
        "official_js_no_precursor": [],
        "cos_with_precursor": [],
        "sqrt_cos_with_precursor": [],
        "sqrt_cos_no_precursor": [],
        "matched_intensity_coverage": [],
        "topk_peak_recall": [],
        "pred_exact_enabled": [],
        "official_metric_source": [],
        "pred_official_n": [],
        "true_official_n": [],
        "overlap_n": [],
        "false_pred_n": [],
        "pred_intensity_on_true_ratio": [],
        "retrieval_pred_sparse": [],
        "retrieval_true_sparse": [],
    }

    bin_width = float(metric_cfg.get("bin_width", 0.01))
    official_max_mz = float(metric_cfg.get("max_mz", 1005.0))
    official_bin_n = int(math.floor(float(official_max_mz) / float(bin_width))) + 1
    official_bin_centers = ((np.arange(official_bin_n, dtype=np.float64) + 0.5) * float(bin_width)).astype(np.float32)

    def _idx_to_official_mz(idx_arr):
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
        if idx_arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return ((idx_arr.astype(np.float64) + 0.5) * bin_width).astype(np.float32)

    def _fmt_mz_summary(mz_arr):
        mz_arr = np.asarray(mz_arr, dtype=np.float64)
        if mz_arr.size == 0:
            return "min=NA p10=NA p50=NA p90=NA max=NA"
        q = np.percentile(mz_arr, [0, 10, 50, 90, 100])
        return (
            f"min={float(q[0]):.3f} p10={float(q[1]):.3f} "
            f"p50={float(q[2]):.3f} p90={float(q[3]):.3f} max={float(q[4]):.3f}"
        )

    for i in range(pred_np.shape[0]):
        pred_peaks_i = None
        if isinstance(pred_exact_peaks, (list, tuple)) and i < len(pred_exact_peaks):
            pred_peaks_i = pred_exact_peaks[i]
        elif torch.is_tensor(pred_exact_peaks) and pred_exact_peaks.dim() >= 3 and i < int(pred_exact_peaks.shape[0]):
            pred_peaks_i = pred_exact_peaks[i]
        elif isinstance(pred_exact_peaks, np.ndarray) and pred_exact_peaks.ndim >= 3 and i < int(pred_exact_peaks.shape[0]):
            pred_peaks_i = pred_exact_peaks[i]

        metric_source_i = "coarse_fallback"

        if pred_peaks_i is not None:
            pred_peaks = _to_sparse_peak_array(
                pred_peaks_i,
                spect_bin_centers=None,
                min_intensity=metric_cfg.get("pred_min_intensity", 1e-8),
            )
            pred_exact_used = 1
            metric_source_i = "pred_exact_peaks"
        else:
            if pred_np.ndim == 2 and int(pred_np.shape[1]) == int(official_bin_n):
                pred_centers_i = official_bin_centers
                pred_exact_used = 2
                metric_source_i = "official_dense"
            else:
                pred_centers_i = spect_bin_centers
                pred_exact_used = 0
                metric_source_i = "coarse_fallback"

            pred_peaks = _to_sparse_peak_array(
                pred_np[i],
                spect_bin_centers=pred_centers_i,
                min_intensity=metric_cfg.get("pred_min_intensity", 1e-8),
            )

        true_raw = None
        if isinstance(true_raw_list, (list, tuple)) and i < len(true_raw_list):
            true_raw = true_raw_list[i]

        true_peaks = _to_sparse_peak_array(
            true_raw,
            spect_bin_centers=spect_bin_centers,
            min_intensity=0.0,
        )

        precursor = None
        if isinstance(precursor_list, (list, tuple)) and i < len(precursor_list):
            precursor = precursor_list[i]

        pred_idx_with, pred_val_with = _bin_peaks_to_official_sparse(
            pred_peaks,
            bin_width=metric_cfg["bin_width"],
            max_mz=metric_cfg["max_mz"],
            exclude_precursor=False,
            precursor_mz=precursor,
        )
        true_idx_with, true_val_with = _bin_peaks_to_official_sparse(
            true_peaks,
            bin_width=metric_cfg["bin_width"],
            max_mz=metric_cfg["max_mz"],
            exclude_precursor=False,
            precursor_mz=precursor,
        )

        pred_idx_no, pred_val_no = _bin_peaks_to_official_sparse(
            pred_peaks,
            bin_width=metric_cfg["bin_width"],
            max_mz=metric_cfg["max_mz"],
            exclude_precursor=metric_cfg.get("exclude_precursor", True),
            precursor_mz=precursor,
        )
        true_idx_no, true_val_no = _bin_peaks_to_official_sparse(
            true_peaks,
            bin_width=metric_cfg["bin_width"],
            max_mz=metric_cfg["max_mz"],
            exclude_precursor=metric_cfg.get("exclude_precursor", True),
            precursor_mz=precursor,
        )

        out["official_cos_no_precursor"].append(
            _cosine_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no, sqrt_intensity=False)
        )
        out["official_js_no_precursor"].append(
            _js_similarity_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no)
        )
        out["cos_with_precursor"].append(
            _cosine_sparse(pred_idx_with, pred_val_with, true_idx_with, true_val_with, sqrt_intensity=False)
        )
        out["sqrt_cos_with_precursor"].append(
            _cosine_sparse(pred_idx_with, pred_val_with, true_idx_with, true_val_with, sqrt_intensity=True)
        )
        out["sqrt_cos_no_precursor"].append(
            _cosine_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no, sqrt_intensity=True)
        )
        out["matched_intensity_coverage"].append(
            _matched_intensity_coverage(pred_idx_no, true_idx_no, true_val_no)
        )
        out["topk_peak_recall"].append(
            _topk_peak_recall(
                pred_idx_no,
                true_idx_no,
                true_val_no,
                k=metric_cfg.get("topk_peak_recall_k", 20),
            )
        )
        pred_set = set(int(x) for x in pred_idx_no.tolist())
        true_set = set(int(x) for x in true_idx_no.tolist())
        overlap_set = pred_set.intersection(true_set)
        overlap_n = len(overlap_set)
        false_pred_n = max(0, int(pred_idx_no.size) - int(overlap_n))

        if pred_idx_no.size > 0 and pred_val_no.size > 0:
            pred_total_int = float(np.sum(np.clip(pred_val_no, 0.0, None)))
            if pred_total_int > 1e-12:
                keep_pred_true = np.array(
                    [int(x) in true_set for x in pred_idx_no.tolist()],
                    dtype=bool,
                )
                pred_true_int = float(np.sum(np.clip(pred_val_no[keep_pred_true], 0.0, None)))
                pred_intensity_on_true_ratio = pred_true_int / pred_total_int
            else:
                pred_intensity_on_true_ratio = float("nan")
        else:
            pred_intensity_on_true_ratio = float("nan")

        out["pred_exact_enabled"].append(float(pred_exact_used))
        out["official_metric_source"].append(metric_source_i)
        out["pred_official_n"].append(float(pred_idx_no.size))
        out["true_official_n"].append(float(true_idx_no.size))
        out["overlap_n"].append(float(overlap_n))
        out["false_pred_n"].append(float(false_pred_n))
        out["pred_intensity_on_true_ratio"].append(float(pred_intensity_on_true_ratio))

        if debug_ctx is not None and bool(debug_ctx.get("enabled", False)):
            printed = int(debug_ctx.get("printed", 0))
            max_samples = max(0, int(debug_ctx.get("max_samples", 0)))
            if printed < max_samples:
                epoch_tag = debug_ctx.get("epoch", None)
                batch_tag = debug_ctx.get("batch", None)
                print(
                    f"[DEBUG_EVAL_SUPPORT] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"pred_exact_used={pred_exact_used} "
                    f"pred_dense_nnz={int((pred_np[i] > 1e-8).sum())} "
                    f"pred_sparse_n={int(len(pred_peaks))} "
                    f"pred_official_n={int(pred_idx_no.size)} "
                    f"true_official_n={int(true_idx_no.size)} "
                    f"overlap_n={int(overlap_n)}",
                    flush=True,
                )
                pred_top20_mz = _idx_to_official_mz(pred_idx_no[:20])
                true_top20_mz = _idx_to_official_mz(true_idx_no[:20])
                print(
                    f"[DEBUG_EVAL_PRED_TOP20_MZ] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"{pred_top20_mz.tolist()}",
                    flush=True,
                )
                print(
                    f"[DEBUG_EVAL_TRUE_TOP20_MZ] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"{true_top20_mz.tolist()}",
                    flush=True,
                )

                pred_official_mz = _idx_to_official_mz(pred_idx_no)
                true_official_mz = _idx_to_official_mz(true_idx_no)

                pred_top_order = (
                    np.argsort(-pred_val_no, kind="mergesort")[:20]
                    if pred_val_no.size > 0
                    else np.asarray([], dtype=np.int64)
                )
                true_top_order = (
                    np.argsort(-true_val_no, kind="mergesort")[:20]
                    if true_val_no.size > 0
                    else np.asarray([], dtype=np.int64)
                )
                pred_top20_mz = (
                    _idx_to_official_mz(pred_idx_no[pred_top_order])
                    if pred_top_order.size > 0
                    else np.zeros((0,), dtype=np.float32)
                )
                true_top20_mz = (
                    _idx_to_official_mz(true_idx_no[true_top_order])
                    if true_top_order.size > 0
                    else np.zeros((0,), dtype=np.float32)
                )

                print(
                    f"[debug eval mz] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"pred_official_mz={_fmt_mz_summary(pred_official_mz)} "
                    f"true_official_mz={_fmt_mz_summary(true_official_mz)}",
                    flush=True,
                )
                print(
                    f"[debug eval mz] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"pred_top20_mz={pred_top20_mz.tolist()}",
                    flush=True,
                )
                print(
                    f"[debug eval mz] epoch={epoch_tag} batch={batch_tag} sample={i} "
                    f"true_top20_mz={true_top20_mz.tolist()}",
                    flush=True,
                )

                debug_ctx["printed"] = printed + 1

        out["retrieval_pred_sparse"].append((pred_idx_no.astype(np.int32), pred_val_no.astype(np.float32)))
        out["retrieval_true_sparse"].append((true_idx_no.astype(np.int32), true_val_no.astype(np.float32)))

    return out
