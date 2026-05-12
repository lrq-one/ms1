import math

import torch

from rassp.training.formula_targets import build_true_official_dense_from_cached_sparse_batch
from rassp.training.spectrum_targets import build_true_official_dense_from_raw


def build_candidate_peak_targets_from_batch(
    batch,
    official_bin_width=0.01,
    official_max_mz=1005.0,
):
    # peak-level supervision must follow non-aggregated official peak order
    # so that target peaks stay aligned with the peak head inputs.
    off_idx = batch.get('formulae_peaks_official_idx', None)
    off_int = batch.get('formulae_peaks_official_intensity', None)

    true_idx = batch.get('true_official_idx', None)
    true_val = batch.get('true_official_intensity', None)

    if (not torch.is_tensor(off_idx)) or (not torch.is_tensor(off_int)):
        return None, None
    if true_idx is None or true_val is None:
        return None, None

    batch_n = int(off_idx.shape[0])
    formula_n = int(off_idx.shape[1])
    peak_n = int(off_idx.shape[2])

    bin_n = int(math.floor(float(official_max_mz) / float(official_bin_width))) + 1
    device = off_idx.device

    true_dense, used_cache = build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=bin_n,
    )
    if not used_cache:
        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get('spect_raw', None),
            precursor_mz=batch.get('precursor_mz', None),
            official_bin_width=float(official_bin_width),
            official_max_mz=float(official_max_mz),
            exclude_precursor=True,
            batch_n=batch_n,
            device=device,
        )

    safe_idx = off_idx.long().clamp(0, max(0, bin_n - 1))
    peak_valid = (
        (off_idx >= 0)
        & (off_idx < bin_n)
        & torch.isfinite(off_int)
        & (off_int > 0)
    )

    true_at_peak = torch.gather(
        true_dense, 1, safe_idx.reshape(batch_n, -1)
    ).reshape(batch_n, formula_n, peak_n)

    peak_target = true_at_peak * peak_valid.float()
    peak_target_sum = peak_target.sum(dim=-1, keepdim=True)

    peak_target_prob = peak_target / peak_target_sum.clamp_min(1e-8)
    peak_target_valid = (peak_target_sum.squeeze(-1) > 0)

    return peak_target_prob, peak_target_valid
 
# Function overview: compute_candidate_exact_overlap_scores_from_batch builds per-candidate exact-overlap scores for diagnostics.


