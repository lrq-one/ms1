# Clean minimal training entry for current mainline
# Focus: setwise scorer + candidate-local peak head
# Objective: selector + rerank + spectral + peak + oos

import sys
import os
import time
import math
import random
import pickle
import json
import re
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Subset

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from rassp.featurize import msutil
from rassp.msutil import masscompute
from rassp import dataset, netutil
from rassp.model.selector_topk import coverage_aware_topk, group_unique_topk, plain_topk
from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value
from rassp.training.batch_utils import (
    ADDUCT_VOCAB,
    INSTRUMENT_VOCAB,
    MS_LEVEL_VOCAB,
    my_collate_fn,
    prepare_batch_cpu,
    move_batch_to_device,
)
from rassp.training.spectrum_targets import build_true_official_dense_from_raw
from rassp.training.official_metrics import compute_batch_official_metrics
from rassp.training.formula_targets import (
    build_true_official_dense_from_cached_sparse_batch,
    apply_teacher_topk_to_target,
    compute_formula_target_probs_from_batch,
)
from rassp.training.runtime_selector_targets import (
    build_candidate_local_quality_target,
    build_selector_teacher_dist_from_official_overlap,
    build_selector_teacher_dist_setcover,
)
from rassp.training.peak_targets import build_candidate_peak_targets_from_batch
from rassp.training.rerank_targets import build_setcover_rerank_target_from_quality
from rassp.training.loss_utils import (
    masked_prob_kl,
    compute_precursor_loss_from_batch,
)
from rassp.training.train_loss_components import (
    _apply_selector_aux_logit_bias,
    _cosine_loss_dense,
    _dense_kl_loss,
    _false_support_mass_loss_dense,
    _get_active_candidate_mask_from_batch,
    _get_reranker_scores_from_res,
    _get_selector_logits_from_res,
    _group_masked_candidate_kl,
    _masked_candidate_kl,
    _masked_formula_entropy_loss,
    _normalize_dense_prob,
    _renormalize_target_probs,
    _sqrt_cosine_loss_dense,
    compute_official_dense_spectral_loss,
    selector_pairwise_utility_loss,
)
from rassp.training.selector_metrics import (
    build_group_unique_topk_mask_from_scores,
    build_mask_from_topk_indices,
    build_topk_mask_from_scores,
    build_true_official_dense_for_batch,
    mask_precision,
    mask_ratio_in_topk,
    mask_recall,
    build_true_official_dense_from_batch,
    compute_candidate_support_stats,
    compute_selector_eval_pack,
    compute_selector_quality_metrics,
    select_model_topk_indices,
)
from rassp.training.selector_losses import (
    compute_selector_false_support_loss,
    compute_selector_utility_target_loss,
)
from rassp.training.config import (
    get_loss_config,
    get_run_config,
    get_selector_config,
)
from rassp.training.train_epoch import train_one_epoch
from rassp.training.validate_epoch import validate_one_epoch
from rassp.training.logging_utils import format_metric_line
from rassp.training.checkpointing import is_better_metric, save_checkpoint

# ================= 杈呭姪绫讳笌鍑芥暟 =================

# -------------------------------------------------------------------------
# Code-reading map for this training entry:
# 1) Data source and row policies:
#    - rassp/dataset/__init__.py :: ParquetDataset.__getitem__
# 2) Per-molecule feature construction:
#    - rassp/featurize/featurize.py :: MolFeaturizer.__call__
# 3) Batch assembly and device transfer:
#    - prepare_batch_cpu / move_batch_to_device in this file
# 4) Core model scoring path:
#    - rassp/model/formulaenets.py :: GraphVertSpect.forward
#    - rassp/model/formulaenets.py :: MolAttentionGRUNewSparse.forward
# 5) Spectrum projection (candidate probabilities -> m/z bins):
#    - project_formula_probs_to_spectrum_dense
#    - project_formula_probs_to_exact_sparse
# 6) Validation scoring and checkpoint selection:
#    - compute_batch_official_metrics / _compute_retrieval_hits in this file
# -------------------------------------------------------------------------



# Function overview: _parse_csv_env handles a specific workflow step in this module.

def _parse_csv_env(raw):
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(',') if x.strip()]


# Function overview: _resolve_formula_atomicnos handles a specific workflow step in this module.

def _resolve_formula_atomicnos():
    # Primary tier defaults: H, C, N, O, F, P, S, Cl
    base_default = [1, 6, 7, 8, 9, 15, 16, 17]
    # Secondary tier (optional): Na, Br, I
    secondary = [11, 35, 53]

    raw = os.environ.get('FORMULA_ATOMICNOS', '').strip()
    if raw:
        vals = []
        for token in raw.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                an = int(token)
            except Exception:
                continue
            if an > 0 and an not in vals:
                vals.append(an)
        if vals:
            return vals

    vals = list(base_default)
    if os.environ.get('FORMULA_ATOMICNOS_TIER2', '0') == '1':
        for an in secondary:
            if an not in vals:
                vals.append(an)
    return vals


def _parse_optional_positive_int_env(name, default=None):
    """
    Parse optional step-limit env var.

    Semantics:
      unset / empty / <=0  -> default, usually None means no limit
      positive integer     -> that many steps
    """
    raw = os.environ.get(name, None)
    if raw is None:
        return default

    raw = str(raw).strip()
    if raw == "":
        return default

    try:
        val = int(raw)
    except Exception:
        return default

    if val <= 0:
        return default

    return int(val)

def _finite_mean(values):
    if values is None:
        return float('nan')
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float('nan')
    return float(arr.mean())


def _parse_formula_oh_sizes(raw_value, element_n):
    if not raw_value:
        return None
    try:
        vals = [int(x.strip()) for x in raw_value.split(',') if x.strip()]
    except Exception:
        return None
    if len(vals) != element_n:
        return None
    if any(v <= 1 for v in vals):
        return None
    return vals


def _resolve_formula_oh_sizes(element_n):
    override = _parse_formula_oh_sizes(os.environ.get('FORMULA_OH_SIZES', '').strip(), element_n)
    if override is not None:
        return override, 'env'
    return [50] * int(element_n), 'fixed50'


def _build_parquet_dataset_with_optional_cache(parquet_path, cache_dir, spect_bin, featurizer_config):
    prev_cache = os.environ.get('FEAT_CACHE_DIR')
    try:
        if cache_dir:
            os.environ['FEAT_CACHE_DIR'] = cache_dir
        elif 'FEAT_CACHE_DIR' in os.environ:
            del os.environ['FEAT_CACHE_DIR']
        return dataset.ParquetDataset(parquet_path, spect_bin, featurizer_config, {})
    finally:
        if prev_cache is None:
            if 'FEAT_CACHE_DIR' in os.environ:
                del os.environ['FEAT_CACHE_DIR']
        else:
            os.environ['FEAT_CACHE_DIR'] = prev_cache


def _masked_selector_bce(selector_logits, target_mask, formulae_mask=None, pos_weight=4.0):
    if (not torch.is_tensor(selector_logits)) or (not torch.is_tensor(target_mask)):
        return None

    logits = selector_logits.float()
    target = target_mask.float()

    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    elif logits.dim() > 2:
        logits = logits.reshape(logits.shape[0], -1)

    if target.dim() == 1:
        target = target.unsqueeze(0)
    elif target.dim() > 2:
        target = target.reshape(target.shape[0], -1)

    use_b = min(int(logits.shape[0]), int(target.shape[0]))
    use_m = min(int(logits.shape[1]), int(target.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None

    logits = logits[:use_b, :use_m]
    target = target[:use_b, :use_m].to(device=logits.device, dtype=logits.dtype)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:use_b, :use_m].to(device=logits.device)
        valid = fm > 0.5
    else:
        valid = torch.ones_like(target, dtype=torch.bool)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none'
    )

    pw = float(pos_weight)
    if np.isfinite(pw) and pw > 1.0:
        weight = torch.where(
            target > 0.5,
            torch.full_like(target, pw),
            torch.ones_like(target),
        )
        loss = loss * weight

    if not bool(valid.any().item()):
        return logits.sum() * 0.0

    use_balanced = os.environ.get("SELECTOR_BALANCED_BCE", "1") == "1"
    if not use_balanced:
        return loss[valid].mean()

    pos = valid & (target > 0.5)
    neg = valid & (target <= 0.5)

    if bool(pos.any().item()):
        pos_loss = loss[pos].mean()
    else:
        pos_loss = logits.sum() * 0.0

    if bool(neg.any().item()):
        neg_loss = loss[neg].mean()
    else:
        neg_loss = logits.sum() * 0.0

    try:
        pos_part_w = float(os.environ.get("SELECTOR_BALANCED_POS_PART", "0.7"))
    except Exception:
        pos_part_w = 0.7
    pos_part_w = float(np.clip(pos_part_w, 0.0, 1.0))

    return pos_part_w * pos_loss + (1.0 - pos_part_w) * neg_loss


def _rerank_teacher_ratio_for_epoch(
    epoch,
    selector_only_warmup_epochs,
    mix_stage1_epochs,
    mix_stage2_epochs,
    mix_teacher_ratio_stage1,
    mix_teacher_ratio_stage2,
    mix_teacher_ratio_stage3,
):
    if int(epoch) < int(selector_only_warmup_epochs):
        return 1.0

    stage1 = max(0, int(mix_stage1_epochs))
    stage2 = max(0, int(mix_stage2_epochs))
    p = int(epoch) - int(selector_only_warmup_epochs)

    if p < stage1:
        ratio = float(mix_teacher_ratio_stage1)
    elif p < (stage1 + stage2):
        ratio = float(mix_teacher_ratio_stage2)
    else:
        ratio = float(mix_teacher_ratio_stage3)

    return float(np.clip(ratio, 0.0, 1.0))


def _mix_teacher_model_masks(teacher_mask, model_mask, teacher_ratio):
    teacher_ok = torch.is_tensor(teacher_mask)
    model_ok = torch.is_tensor(model_mask)

    if (not teacher_ok) and (not model_ok):
        return None, float('nan')
    if teacher_ok and (not model_ok):
        return (teacher_mask > 0.5).float(), 1.0
    if model_ok and (not teacher_ok):
        return (model_mask > 0.5).float(), 0.0

    tm = teacher_mask.float()
    mm = model_mask.float()

    if tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)
    if mm.dim() > 2:
        mm = mm.reshape(mm.shape[0], -1)

    use_b = min(int(tm.shape[0]), int(mm.shape[0]))
    use_m = min(int(tm.shape[1]), int(mm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None, float('nan')

    tm = (tm[:use_b, :use_m] > 0.5).float()
    mm = (mm[:use_b, :use_m] > 0.5).float()

    ratio = float(np.clip(float(teacher_ratio), 0.0, 1.0))
    gate = (torch.rand((use_b, 1), device=tm.device) < ratio).float()
    mixed = gate * tm + (1.0 - gate) * mm
    mixed = (mixed > 0.5).float()

    fallback = torch.where(tm.sum(dim=-1, keepdim=True) > 0, tm, mm)
    row_ok = mixed.sum(dim=-1, keepdim=True) > 0
    mixed = torch.where(row_ok, mixed, fallback)

    return mixed, float(gate.mean().detach().cpu().item())


def _compute_peak_aux_loss_from_batch(
    batch,
    res,
    official_bin_width=0.01,
    official_max_mz=1005.0,
):
    if not isinstance(res, dict):
        return None

    peak_logits = res.get('peak_reweight_logits', None)
    if not torch.is_tensor(peak_logits):
        return None

    peak_target_prob, peak_target_valid = build_candidate_peak_targets_from_batch(
        batch,
        official_bin_width=official_bin_width,
        official_max_mz=official_max_mz,
    )

    if (not torch.is_tensor(peak_target_prob)) or (not torch.is_tensor(peak_target_valid)):
        return None

    use_b = min(int(peak_logits.shape[0]), int(peak_target_prob.shape[0]), int(peak_target_valid.shape[0]))
    use_m = min(int(peak_logits.shape[1]), int(peak_target_prob.shape[1]), int(peak_target_valid.shape[1]))
    use_k = min(int(peak_logits.shape[2]), int(peak_target_prob.shape[2]))
    if use_b <= 0 or use_m <= 0 or use_k <= 0:
        return None

    logits = peak_logits[:use_b, :use_m, :use_k]
    target = peak_target_prob[:use_b, :use_m, :use_k].to(device=logits.device, dtype=logits.dtype)
    valid_formula = peak_target_valid[:use_b, :use_m].to(device=logits.device)

    peak_idx = batch.get('formulae_peaks_official_idx', None)
    peak_int = batch.get('formulae_peaks_official_intensity', None)
    if not torch.is_tensor(peak_idx):
        peak_idx = batch.get('formulae_peaks_mass_idx', None)
    if not torch.is_tensor(peak_int):
        peak_int = batch.get('formulae_peaks_intensity', None)

    if torch.is_tensor(peak_idx) and torch.is_tensor(peak_int):
        pidx = peak_idx[:use_b, :use_m, :use_k].to(device=logits.device)
        pint = peak_int[:use_b, :use_m, :use_k].to(device=logits.device, dtype=logits.dtype)
        peak_valid = (pidx >= 0) & torch.isfinite(pint) & (pint > 0)
    else:
        peak_valid = torch.ones((use_b, use_m, use_k), dtype=torch.bool, device=logits.device)

    logits = logits.masked_fill(~peak_valid, _neg_mask_fill_value(logits))
    target = target * peak_valid.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    per_formula = F.kl_div(
        F.log_softmax(logits, dim=-1),
        target,
        reduction='none',
    ).sum(dim=-1)

    formula_mask = batch.get('formulae_mask', None)
    if torch.is_tensor(formula_mask):
        fm = formula_mask[:use_b, :use_m].to(device=logits.device) > 0.5
        valid_formula = valid_formula & fm

    valid_formula = valid_formula & (peak_valid.sum(dim=-1) > 0)
    if not bool(valid_formula.any().item()):
        return logits.sum() * 0.0

    return per_formula[valid_formula].mean()


def _build_oos_target_from_batch(batch, official_bin_width=0.01, official_max_mz=1005.0):
    if not isinstance(batch, dict):
        return None, None

    off_idx = batch.get('formulae_peaks_official_idx_agg', None)
    off_int = batch.get('formulae_peaks_official_intensity_agg', None)
    if not torch.is_tensor(off_idx):
        off_idx = batch.get('formulae_peaks_official_idx', None)
    if not torch.is_tensor(off_int):
        off_int = batch.get('formulae_peaks_official_intensity', None)

    if (not torch.is_tensor(off_idx)) or (not torch.is_tensor(off_int)):
        return None, None

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)
    if off_idx.dim() != 3 or off_int.dim() != 3:
        return None, None

    batch_n = min(int(off_idx.shape[0]), int(off_int.shape[0]))
    formula_n = min(int(off_idx.shape[1]), int(off_int.shape[1]))
    peak_n = min(int(off_idx.shape[2]), int(off_int.shape[2]))
    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None, None

    device = off_idx.device
    off_idx = off_idx[:batch_n, :formula_n, :peak_n].long()
    off_int = off_int[:batch_n, :formula_n, :peak_n].float()

    formulae_mask = batch.get('formulae_mask', None)
    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(batch_n, int(fm.shape[0]))
        use_m = min(formula_n, int(fm.shape[1]))
        off_idx = off_idx[:use_b, :use_m, :]
        off_int = off_int[:use_b, :use_m, :]
        fm = fm[:use_b, :use_m]
        batch_n = use_b
        formula_n = use_m
    else:
        fm = torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)

    if batch_n <= 0 or formula_n <= 0:
        return None, None

    bwidth = float(max(1e-6, official_bin_width))
    max_mz = float(max(bwidth, official_max_mz))
    official_bin_n = int(math.floor(max_mz / bwidth)) + 1

    true_dense, used_cached_true = build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )
    if not used_cached_true:
        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get('spect_raw', None),
            precursor_mz=batch.get('precursor_mz', None),
            official_bin_width=bwidth,
            official_max_mz=max_mz,
            exclude_precursor=True,
            batch_n=batch_n,
            device=device,
        )

    safe_idx = off_idx.clamp(0, max(0, official_bin_n - 1))
    valid = (
        (off_idx >= 0)
        & (off_idx < official_bin_n)
        & torch.isfinite(off_int)
        & (off_int > 0)
        & (fm > 0.5).unsqueeze(-1)
    )

    support_dense = torch.zeros((batch_n, official_bin_n), dtype=torch.float32, device=device)
    support_dense.scatter_add_(
        1,
        safe_idx.reshape(batch_n, -1),
        valid.float().reshape(batch_n, -1),
    )
    support_mask = support_dense > 0

    true_total = true_dense.sum(dim=-1)
    in_support = (true_dense * support_mask.float()).sum(dim=-1)

    oos_ratio = 1.0 - (in_support / true_total.clamp_min(1e-12))
    oos_ratio = oos_ratio.clamp(0.0, 1.0)
    oos_valid = true_total > 1e-12

    return oos_ratio, oos_valid


def _compute_oos_loss_from_batch(batch, res, official_bin_width=0.01, official_max_mz=1005.0):
    if not isinstance(res, dict):
        return None

    oos_logit = res.get('oos_logit', None)
    if not torch.is_tensor(oos_logit):
        return None

    oos_target, oos_valid = _build_oos_target_from_batch(
        batch,
        official_bin_width=official_bin_width,
        official_max_mz=official_max_mz,
    )
    if (not torch.is_tensor(oos_target)) or (not torch.is_tensor(oos_valid)):
        return None

    use_b = min(int(oos_logit.shape[0]), int(oos_target.shape[0]), int(oos_valid.shape[0]))
    if use_b <= 0:
        return None

    logits = oos_logit[:use_b].float()
    target = oos_target[:use_b].to(device=logits.device, dtype=logits.dtype)
    valid = oos_valid[:use_b].to(device=logits.device)

    if not bool(valid.any().item()):
        return logits.sum() * 0.0

    return F.binary_cross_entropy_with_logits(logits[valid], target[valid], reduction='mean')

def _prune_batch_by_candidate_mask(
    batch,
    cand_mask,
    keep_topk=64,
    fill_scores=None,
    group_id=None,
    group_unique=False,
):
    """
    Physically prune candidate-axis tensors in batch for second-pass rerank.
    cand_mask: [B, M] binary-like mask, usually teacher_topk_mask or model_topk_mask.
    """
    if (not torch.is_tensor(cand_mask)) or cand_mask.dim() != 2:
        return dict(batch)

    out = dict(batch)
    mask = cand_mask.float()
    bsz, cand_n = mask.shape
    kk = max(1, min(int(keep_topk), int(cand_n)))

    if torch.is_tensor(fill_scores):
        fs = fill_scores.float()
        if fs.dim() > 2:
            fs = fs.reshape(fs.shape[0], -1)
        use_b = min(int(fs.shape[0]), int(mask.shape[0]))
        use_m = min(int(fs.shape[1]), int(mask.shape[1]))
        fs = fs[:use_b, :use_m]
        mask_use = mask[:use_b, :use_m]

        if use_b < bsz or use_m < cand_n:
            score = torch.zeros_like(mask)
            score[:use_b, :use_m] = fs
            mask_aligned = torch.zeros_like(mask)
            mask_aligned[:use_b, :use_m] = mask_use
            mask = mask_aligned
        else:
            score = fs
    else:
        score = torch.zeros_like(mask)

    # selected mask always dominates fill scores;
    # fill_scores only chooses padding candidates when positives < keep_topk.
    if bool(group_unique):
        gid_src = group_id
        if gid_src is None:
            gid_src = batch.get("formulae_instance_group_id", None)

        if torch.is_tensor(gid_src):
            gid = gid_src.to(device=mask.device, dtype=torch.long)
            if gid.dim() == 1:
                gid = gid.unsqueeze(0)
            elif gid.dim() > 2:
                gid = gid.reshape(gid.shape[0], -1)

            use_b = min(int(gid.shape[0]), int(mask.shape[0]))
            use_m = min(int(gid.shape[1]), int(mask.shape[1]))

            gid_full = torch.arange(
                cand_n,
                device=mask.device,
                dtype=torch.long,
            ).view(1, -1).expand(bsz, -1).clone()

            gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
            gid = gid_full
        else:
            gid = torch.arange(
                cand_n,
                device=mask.device,
                dtype=torch.long,
            ).view(1, -1).expand(bsz, -1)

        # valid candidates for filler. Prefer formulae_mask if available.
        fm_for_prune = batch.get("formulae_mask", None)
        if torch.is_tensor(fm_for_prune) and fm_for_prune.dim() >= 2:
            fm = fm_for_prune.to(device=mask.device).float()
            if fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)

            valid = torch.zeros_like(mask, dtype=torch.bool)
            use_b = min(int(fm.shape[0]), bsz)
            use_m = min(int(fm.shape[1]), cand_n)
            valid[:use_b, :use_m] = fm[:use_b, :use_m] > 0.5
        else:
            valid = torch.ones_like(mask, dtype=torch.bool)

        top_idx_rows = []

        for bi in range(bsz):
            selected_idx = torch.nonzero(mask[bi] > 0.5, as_tuple=False).reshape(-1)

            # First: among selected mask, keep only best candidate per group.
            chosen = []
            seen = set()

            if selected_idx.numel() > 0:
                selected_scores = score[bi, selected_idx]
                order = torch.argsort(selected_scores, descending=True)

                for oi in order.detach().cpu().tolist():
                    idx = int(selected_idx[oi].detach().cpu().item())
                    g = int(gid[bi, idx].detach().cpu().item())
                    if g in seen:
                        continue
                    seen.add(g)
                    chosen.append(idx)
                    if len(chosen) >= kk:
                        break

            # Second: fill remaining slots using fill_scores/score, still one per group.
            if len(chosen) < kk:
                valid_idx = torch.nonzero(valid[bi], as_tuple=False).reshape(-1)
                if valid_idx.numel() > 0:
                    filler_scores = score[bi, valid_idx]
                    order = torch.argsort(filler_scores, descending=True)

                    chosen_set = set(chosen)
                    for oi in order.detach().cpu().tolist():
                        idx = int(valid_idx[oi].detach().cpu().item())
                        if idx in chosen_set:
                            continue

                        g = int(gid[bi, idx].detach().cpu().item())
                        if g in seen:
                            continue

                        seen.add(g)
                        chosen.append(idx)
                        chosen_set.add(idx)

                        if len(chosen) >= kk:
                            break

            # Last fallback: if unique groups are fewer than kk, allow any valid filler.
            # This should be rare, but prevents gather shape failure.
            if len(chosen) < kk:
                valid_idx = torch.nonzero(valid[bi], as_tuple=False).reshape(-1)
                chosen_set = set(chosen)

                for idx_t in valid_idx.detach().cpu().tolist():
                    idx = int(idx_t)
                    if idx in chosen_set:
                        continue
                    chosen.append(idx)
                    chosen_set.add(idx)
                    if len(chosen) >= kk:
                        break

            if len(chosen) <= 0:
                chosen = [0]

            while len(chosen) < kk:
                chosen.append(chosen[-1])

            top_idx_rows.append(
                torch.as_tensor(chosen[:kk], dtype=torch.long, device=mask.device)
            )

        top_idx = torch.stack(top_idx_rows, dim=0)

    else:
        score = score + (mask > 0.5).float() * 1e6
        top_idx = torch.topk(score, k=kk, dim=-1).indices

    out['formula_topk_orig_idx'] = top_idx

    candidate_keys = [
        'formulae_features',
        'formulae_peaks',
        'formulae_peaks_mass_idx',
        'formulae_peaks_intensity',
        'formulae_peaks_official_idx',
        'formulae_peaks_official_intensity',
        'formulae_peaks_official_idx_agg',
        'formulae_peaks_official_intensity_agg',
        'formulae_aux_feat',
        'formulae_frag_aux_feat',   # 鍏抽敭锛歠ull-train rerank 瑁佸壀鏃跺繀椤讳竴璧疯
        'formulae_active_mask',
        'formulae_prior_score',
        'formulae_source_flag',
        'formulae_break_depth',
        'formulae_ring_cut_flag',
        'formulae_mask',
        'teacher_formula_probs',
        'formulae_instance_is_source',
        'formulae_instance_group_id',
        'formulae_instance_depth',
        'formulae_instance_h_shift',
    ]

    for k in candidate_keys:
        v = out.get(k, None)
        if (not torch.is_tensor(v)) or v.dim() < 2:
            continue
        if int(v.shape[0]) != bsz or int(v.shape[1]) != cand_n:
            continue

        gather_idx = top_idx
        while gather_idx.dim() < v.dim():
            gather_idx = gather_idx.unsqueeze(-1)
        gather_idx = gather_idx.expand(-1, -1, *v.shape[2:])
        out[k] = torch.gather(v, 1, gather_idx)

    if torch.is_tensor(out.get('formulae_n_kept', None)):
        # After gather, formulae_mask already marks which selected/padded candidates are valid.
        # Do not blindly set n_kept to kk when some filler slots came from invalid candidates.
        if torch.is_tensor(out.get('formulae_mask', None)):
            out['formulae_n_kept'] = out['formulae_mask'].float().sum(dim=-1).long().to(
                dtype=out['formulae_n_kept'].dtype,
                device=out['formulae_n_kept'].device,
            )
        else:
            out['formulae_n_kept'] = torch.full(
                (bsz,),
                kk,
                dtype=out['formulae_n_kept'].dtype,
                device=out['formulae_n_kept'].device,
            )

    if torch.is_tensor(out.get('formulae_mask', None)):
        # Keep the gathered original formulae_mask. Only guard against all-zero rows.
        fm = out['formulae_mask'].float()
        row_has_valid = fm.sum(dim=-1, keepdim=True) > 0
        out['formulae_mask'] = torch.where(row_has_valid, fm, torch.ones_like(fm))

    return out

def train_mssubsetnet():
    def log(msg):
        print(msg, flush=True)

    run_cfg = get_run_config()
    selector_cfg = get_selector_config()
    loss_cfg = get_loss_config()

    use_group_unique_teacher = str(os.environ.get("USE_GROUP_UNIQUE_TEACHER", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    use_group_unique_model = str(os.environ.get("USE_GROUP_UNIQUE_MODEL", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    use_group_unique_prune = str(os.environ.get("USE_GROUP_UNIQUE_PRUNE", "0")).strip().lower() in {
        "1", "true", "yes", "y", "on"
    }
    print(
        "group_unique_env:",
        "teacher=", use_group_unique_teacher,
        "model=", use_group_unique_model,
        "prune=", use_group_unique_prune,
        flush=True,
    )

    seed = int(os.environ.get('SEED', '1024'))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if not getattr(masscompute, 'HAS_MASSEVAL', False):
        raise RuntimeError('Compiled masseval backend is required for train_ms_subsetnet.')

    batch_size = int(os.environ.get('BATCH_SIZE', '16'))
    epochs = int(os.environ.get('EPOCHS', '5'))
    lr = float(os.environ.get('LR', '3e-4'))
    weight_decay = float(os.environ.get('WEIGHT_DECAY', '1e-5'))
    grad_clip = float(os.environ.get('GRAD_CLIP', '1.0'))
    loader_workers = max(0, int(os.environ.get('NUM_WORKERS', '4')))
    loader_prefetch = max(1, int(os.environ.get('PREFETCH_FACTOR', '2')))
    amp_enabled = os.environ.get('AMP', '0') == '1'
    amp_dtype_name = os.environ.get('AMP_DTYPE', 'fp16').strip().lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ('bf16', 'bfloat16') else torch.float16
    if amp_enabled and amp_dtype == torch.bfloat16 and torch.cuda.is_available() and (not torch.cuda.is_bf16_supported()):
        amp_dtype = torch.float16
    scaler = GradScaler(enabled=(amp_enabled and amp_dtype == torch.float16))

    formula_only = os.environ.get('FORMULA_ONLY', '1') == '1'
    use_msms_conditioning = os.environ.get('USE_MSMS_CONDITIONING', '1') == '1'
    prob_softmax = os.environ.get('FORMULA_PROB_SOFTMAX', '1') == '1'
    normalize_output = os.environ.get('FORMULA_NORMALIZE_OUTPUT', '1') == '1'

    rerank_loss_weight = max(
        0.0,
        float(
            os.environ.get(
                'RERANK_LOSS_WEIGHT',
                os.environ.get('MAIN_CANDIDATE_KL_WEIGHT', '1.0'),
            )
        ),
    )
    main_candidate_kl_weight = rerank_loss_weight
    rerank_kl_weight = max(0.0, float(os.environ.get('RERANK_KL_WEIGHT', '0.7')))
    rerank_bce_weight = max(0.0, float(os.environ.get('RERANK_BCE_WEIGHT', '0.3')))
    official_spectral_loss_weight = max(
        0.0,
        float(os.environ.get('OFFICIAL_SPECTRAL_LOSS_WEIGHT', os.environ.get('SPECTRAL_LOSS_WEIGHT', '1.0'))),
    )
    formula_entropy_loss_weight = max(
        0.0,
        float(os.environ.get('FORMULA_ENTROPY_LOSS_WEIGHT', '0.0')),
    )
    false_support_loss_weight = max(
        0.0,
        float(os.environ.get('FALSE_SUPPORT_LOSS_WEIGHT', '0.0')),
    )
    selector_utility_loss_weight = max(
        0.0,
        float(os.environ.get('SELECTOR_UTILITY_LOSS_WEIGHT', '0.0')),
    )
    # coarse_spectral_aux_weight = max(0.0, float(os.environ.get('COARSE_SPECTRAL_AUX_WEIGHT', '0.1')))
    coarse_spectral_aux_weight = 0.0
    official_spectral_kl_weight = max(0.0, float(os.environ.get('OFFICIAL_SPECTRAL_KL_WEIGHT', '0.2')))
    peak_aux_loss_weight = max(0.0, float(os.environ.get('PEAK_AUX_LOSS_WEIGHT', '0.05')))
    oos_loss_weight = max(0.0, float(os.environ.get('OOS_LOSS_WEIGHT', '0.05')))
    precursor_loss_weight = max(0.0, float(os.environ.get('PRECURSOR_LOSS_WEIGHT', '0.05')))

    precursor_loss_start_epoch = max(0, int(os.environ.get('PRECURSOR_LOSS_START_EPOCH', '0')))
    main_target_bin_width = float(os.environ.get('MAIN_TARGET_BIN_WIDTH', os.environ.get('OFFICIAL_BIN_WIDTH', '0.01')))
    main_target_max_mz = float(os.environ.get('MAIN_TARGET_MAX_MZ', os.environ.get('OFFICIAL_MAX_MZ', '1005.0')))
    target_support_temperature = float(os.environ.get('TARGET_SUPPORT_TEMPERATURE', '1.0'))
    target_support_topk = max(0, int(os.environ.get('TARGET_SUPPORT_TOPK', '16')))
    formula_target_mode = 'quality_hybrid_official_cached'

    selector_topk = max(1, int(os.environ.get('SELECTOR_TOPK', '64')))
    teacher_topk_train = max(0, int(os.environ.get('TEACHER_TOPK_TRAIN', str(selector_topk))))
    teacher_topk_eval = max(0, int(os.environ.get('TEACHER_TOPK_EVAL', str(selector_topk))))
    model_topk_eval = max(1, int(os.environ.get('MODEL_TOPK_EVAL', str(selector_topk))))

    selector_loss_weight = max(0.0, float(os.environ.get('SELECTOR_LOSS_WEIGHT', '1.0')))
    selector_pos_weight = max(1.0, float(os.environ.get('SELECTOR_POS_WEIGHT', '4.0')))

    # selector should learn both:
    # 1) binary inclusion into teacher topK
    # 2) listwise ranking distribution over candidates
    selector_bce_weight = max(0.0, float(os.environ.get('SELECTOR_BCE_WEIGHT', '0.2')))
    selector_kl_weight = max(0.0, float(os.environ.get('SELECTOR_KL_WEIGHT', '0.2')))
    selector_pairwise_weight = max(0.0, float(os.environ.get('SELECTOR_PAIRWISE_WEIGHT', '0.4')))
    selector_utility_kl_weight = max(0.0, float(os.environ.get('SELECTOR_UTILITY_KL_WEIGHT', '0.2')))

    train_selector_only_stage = os.environ.get("TRAIN_SELECTOR_ONLY_STAGE", "0") == "1"

    # If we are continuing from a selector checkpoint into reranker/full training,
    # do not silently skip main_candidate_kl for the first 3 epochs.
    # The old default is still kept for true from-scratch full training.
    _default_selector_warmup = "0" if (
        os.environ.get("LOAD_MODEL_PATH", "").strip()
        and not train_selector_only_stage
    ) else "3"

    selector_only_warmup_epochs = max(
        0,
        int(os.environ.get("SELECTOR_ONLY_WARMUP_EPOCHS", _default_selector_warmup)),
    )
    rerank_mix_stage1_epochs = max(0, int(os.environ.get('RERANK_MIX_STAGE1_EPOCHS', '2')))
    rerank_mix_stage2_epochs = max(0, int(os.environ.get('RERANK_MIX_STAGE2_EPOCHS', '2')))
    rerank_mix_teacher_ratio_stage1 = float(os.environ.get('RERANK_MIX_TEACHER_RATIO_STAGE1', '0.7'))
    rerank_mix_teacher_ratio_stage2 = float(os.environ.get('RERANK_MIX_TEACHER_RATIO_STAGE2', '0.5'))
    rerank_mix_teacher_ratio_stage3 = float(os.environ.get('RERANK_MIX_TEACHER_RATIO_STAGE3', '0.2'))

    spectral_loss_start_epoch = max(
        0,
        int(
            os.environ.get(
                'SPECTRAL_LOSS_START_EPOCH',
                str(selector_only_warmup_epochs + rerank_mix_stage1_epochs),
            )
        ),
    )
    peak_aux_start_epoch = max(
        0,
        int(os.environ.get('PEAK_AUX_START_EPOCH', str(spectral_loss_start_epoch + 1))),
    )
    oos_loss_start_epoch = max(
        0,
        int(os.environ.get('OOS_LOSS_START_EPOCH', str(peak_aux_start_epoch + 1))),
    )

    official_metric_cfg = {
        'bin_width': float(max(1e-6, float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01')))),
        'max_mz': float(max(1.0, float(os.environ.get('OFFICIAL_MAX_MZ', '1005.0')))),
        'exclude_precursor': os.environ.get('OFFICIAL_EXCLUDE_PRECURSOR', '1') == '1',
        'pred_min_intensity': float(max(0.0, float(os.environ.get('OFFICIAL_PRED_MIN_INTENSITY', '1e-8')))),
        'topk_peak_recall_k': int(max(1, int(os.environ.get('OFFICIAL_TOPK_PEAK_K', '20')))),
    }
    official_bin_n = int(math.floor(official_metric_cfg['max_mz'] / official_metric_cfg['bin_width'])) + 1
    early_stop_patience = max(0, int(os.environ.get('EARLY_STOP_PATIENCE', '0')))
    early_stop_min_delta = max(0.0, float(os.environ.get('EARLY_STOP_MIN_DELTA', '0.0')))
    early_stop_warmup_epochs = max(0, int(os.environ.get('EARLY_STOP_WARMUP_EPOCHS', '0')))
    model_select_metric_name = os.environ.get("MODEL_SELECT_METRIC", "official_cos").strip().lower()
    # Selector-only stage should not select by official_cos because reranker/render is untrained.
    if train_selector_only_stage and model_select_metric_name in ("official_cos", "cos", "official"):
        selector_stage_metric = os.environ.get(
            "SELECTOR_STAGE_MODEL_SELECT_METRIC",
            "model_topk_oracle_cos_256",
        ).strip().lower()
        model_select_metric_name = selector_stage_metric or "model_topk_oracle_cos_256"
    # 0 / negative / unset means no limit.
    # This avoids the bug where MAX_TRAIN_STEPS=0 makes every epoch run 0 batches.
    max_train_steps = _parse_optional_positive_int_env('MAX_TRAIN_STEPS', default=None)
    max_val_steps = _parse_optional_positive_int_env('MAX_VAL_STEPS', default=None)
    spect_bin_config = {'first_bin_center': 1.0, 'bin_width': 1.0, 'bin_number': 1024}
    spect_bin = msutil.binutils.create_spectrum_bins(**spect_bin_config)

    formula_atomicnos = _resolve_formula_atomicnos()
    feat_vert_args = netutil.dict_combine(netutil.default_feat_vert_args, {'feat_atomicno_onehot': formula_atomicnos})
    mol_global_features = _parse_csv_env(os.environ.get('MOL_GLOBAL_FEATURES', 'exact_mol_wt,mol_logp,tpsa,num_hbd,num_hba,num_rot_bonds'))
    use_mol_global_feat = os.environ.get('USE_MOL_GLOBAL_FEAT', '0') == '1'

    featurizer_config = {
        'MAX_N': 128,
        'feat_vert_args': feat_vert_args,
        'adj_args': netutil.default_adj_args,
        'mol_args': {'global_features': mol_global_features if use_mol_global_feat else []},
        'vert_subset_samples_n': 0 if formula_only else int(os.environ.get('VERT_SUBSET_SAMPLES_N', '128')),
        'subset_gen_config': {'name': 'break_and_rearrange', 'num_breaks': 3},
        'element_oh': feat_vert_args['feat_atomicno_onehot'],
        'explicit_formulae_config': {
            'formula_possible_atomicno': feat_vert_args['feat_atomicno_onehot'],
            'clip_mass': 1023,
            'use_highres': os.environ.get('FORMULA_USE_HIGHRES', '0') == '1',
            'max_formulae': int(os.environ.get('MAX_FORMULAE', '4096')),
            'overflow_mode': os.environ.get('FORMULAE_OVERFLOW_MODE', 'truncate').strip().lower(),
            'overflow_sample_seed': int(os.environ.get('FORMULAE_OVERFLOW_SAMPLE_SEED', '0')),
        },
    }

    script_dir = os.path.abspath(os.path.dirname(__file__))
    train_parquet = os.environ.get('TRAIN_PARQUET', os.path.join(script_dir, 'data/massspecgym/massspecgym_train.parquet'))
    val_parquet = os.environ.get('VAL_PARQUET', os.path.join(script_dir, 'data/massspecgym/massspecgym_val.parquet'))
    train_cache_dir = os.environ.get('FEAT_CACHE_DIR_TRAIN', '').strip()
    val_cache_dir = os.environ.get('FEAT_CACHE_DIR_VAL', '').strip()

    log(f'鈿欙笍 mode=formula_only:{int(formula_only)} batch_size={batch_size} epochs={epochs}')
    log(f'馃И formula_elements: atomicnos={formula_atomicnos}')
    log(f'馃З mainline_signature=setwise_peak_clean_min')
    log(f'馃З objective: selector + rerank + official_dense_spectral + peak + oos')
    log(f'馃З formula_target_mode={formula_target_mode} support_temp={target_support_temperature:.4f} support_topk={target_support_topk}')
    log(
        f'馃З loss_weights: selector={selector_loss_weight:.3f} rerank={main_candidate_kl_weight:.3f} '
        f'official_spectral={official_spectral_loss_weight:.3f} '
        f'official_kl={official_spectral_kl_weight:.3f} peak={peak_aux_loss_weight:.3f} '
        f'oos={oos_loss_weight:.3f} precursor={precursor_loss_weight:.3f} '
        f'entropy={formula_entropy_loss_weight:.3f} '
        f'false_support={false_support_loss_weight:.3f} '
    )
    log(f'馃З teacher_topk_train={teacher_topk_train}')
    log(f'馃З teacher_topk_eval={teacher_topk_eval}')
    log(f'馃З selector_topk={selector_topk} model_topk_eval={model_topk_eval}')
    log(
        f'馃З step_limits: '
        f'train={max_train_steps if max_train_steps is not None else "<full>"} '
        f'val={max_val_steps if max_val_steps is not None else "<full>"}'
    )
    log(f'馃З selector_pos_weight={selector_pos_weight:.3f}')
    log(
        f'馃З selector_components: bce_weight={selector_bce_weight:.3f} '
        f'kl_weight={selector_kl_weight:.3f} '
        f'pairwise_weight={selector_pairwise_weight:.3f} '
        f'utility_kl_weight={selector_utility_kl_weight:.3f}'
    )
    log(f'馃З train_selector_only_stage={int(train_selector_only_stage)}')
    log(f'馃З selector_only_warmup_epochs={selector_only_warmup_epochs}')
    log(
        f'馃З rerank_mix_teacher_ratio: stage1={rerank_mix_teacher_ratio_stage1:.2f} '
        f'stage2={rerank_mix_teacher_ratio_stage2:.2f} stage3={rerank_mix_teacher_ratio_stage3:.2f} '
        f'epochs=({rerank_mix_stage1_epochs},{rerank_mix_stage2_epochs})'
    )
    log(
        f'馃З loss_start_epoch: spectral={spectral_loss_start_epoch} '
        f'peak={peak_aux_start_epoch} oos={oos_loss_start_epoch}'
    )
    log(
        f'precursor_loss_start_epoch={precursor_loss_start_epoch}'
    )
    log(f'馃З model_select_metric={model_select_metric_name}')
    train_ds = _build_parquet_dataset_with_optional_cache(train_parquet, train_cache_dir, spect_bin, featurizer_config)
    val_ds = _build_parquet_dataset_with_optional_cache(val_parquet, val_cache_dir, spect_bin, featurizer_config)

    train_max_samples = os.environ.get('TRAIN_MAX_SAMPLES')
    val_max_samples = os.environ.get('VAL_MAX_SAMPLES')
    if train_max_samples:
        n = min(len(train_ds), max(1, int(train_max_samples)))
        train_ds = Subset(train_ds, random.sample(range(len(train_ds)), n))
    if val_max_samples:
        n = min(len(val_ds), max(1, int(val_max_samples)))
        val_ds = Subset(val_ds, random.sample(range(len(val_ds)), n))

    first_sample = train_ds[0]
    if 'formulae_features' in first_sample:
        ff_shape = np.asarray(first_sample['formulae_features']).shape
        if len(ff_shape) > 0 and int(ff_shape[-1]) != len(formula_atomicnos):
            raise RuntimeError(
                '缂撳瓨鐨?formula 缁村害涓庡綋鍓嶅厓绱犻泦鍚堜笉涓€鑷? '
                f'cache_dim={int(ff_shape[-1])}, configured_dim={len(formula_atomicnos)}, atomicnos={formula_atomicnos}'
            )

    formula_oh_sizes, formula_oh_source = _resolve_formula_oh_sizes(len(formula_atomicnos))
    log(f'馃М formula_oh_sizes={formula_oh_sizes} source={formula_oh_source}')

    def worker_init_fn(_worker_id):
        os.environ['OMP_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        torch.set_num_threads(1)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': loader_workers,
        'pin_memory': True,
        'worker_init_fn': worker_init_fn,
        'collate_fn': my_collate_fn,
    }
    if loader_workers > 0:
        loader_kwargs.update({'prefetch_factor': loader_prefetch, 'persistent_workers': True})

    train_dl = torch.utils.data.DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_dl = torch.utils.data.DataLoader(val_ds, shuffle=False, **loader_kwargs)

    sample0 = next(iter(train_dl))
    formulae_aux_dim = 0
    try:
        aux0 = np.asarray(sample0['formulae_aux_feat'])
        if aux0.ndim >= 1:
            formulae_aux_dim = int(aux0.shape[-1])
    except Exception:
        formulae_aux_dim = 0

    fragment_local_aux_dim = 0
    use_fragment_local_aux = os.environ.get("USE_FRAGMENT_LOCAL_AUX", "0").strip() == "1"

    if use_fragment_local_aux:
        try:
            frag0 = np.asarray(sample0.get("formulae_frag_aux_feat", None))
            if frag0 is not None and frag0.ndim >= 1:
                fragment_local_aux_dim = int(frag0.shape[-1])
        except Exception:
            fragment_local_aux_dim = 0

        try:
            fmask0 = np.asarray(sample0.get("formulae_mask", None), dtype=np.float32)
            if fmask0.ndim >= 2:
                # sample0 鏄?raw collate 鍚庣殑 batch锛岄€氬父 shape 鏄?[B, M]
                valid0 = fmask0 > 0.5
            else:
                valid0 = None

            if "frag0" in locals() and frag0 is not None:
                frag_arr = np.asarray(frag0, dtype=np.float32)
                if frag_arr.ndim >= 3:
                    row_norm = np.linalg.norm(frag_arr, axis=-1)
                    if valid0 is not None and valid0.shape == row_norm.shape:
                        nonzero_ratio = float(((row_norm > 1e-8) & valid0).sum() / max(1, valid0.sum()))
                    else:
                        nonzero_ratio = float((row_norm > 1e-8).mean())
                elif frag_arr.ndim == 2:
                    row_norm = np.linalg.norm(frag_arr, axis=-1)
                    nonzero_ratio = float((row_norm > 1e-8).mean())
                else:
                    nonzero_ratio = float("nan")
            else:
                nonzero_ratio = float("nan")
        except Exception:
            nonzero_ratio = float("nan")
    else:
        nonzero_ratio = float("nan")

    log(
        f"馃З fragment_local_aux: use={int(use_fragment_local_aux)} "
        f"dim={int(fragment_local_aux_dim)} "
        f"nonzero_ratio_sample0={nonzero_ratio:.4f}"
    )

    from rassp.model import formulaenets
    spect_out_config = {
        'embedding_key_size': int(os.environ.get('FORMULA_EMBEDDING_KEY_SIZE', '64')),
        'internal_d': int(os.environ.get('INTERNAL_D', '512')),
        'formula_oh_sizes': formula_oh_sizes,
        'formula_oh_accum': True,
        'formula_oh_normalize': False,
        'ce_emb_dim': 32,
        'adduct_emb_dim': 16,
        'adduct_vocab_size': int(os.environ.get('NUM_ADDUCTS', str(max(ADDUCT_VOCAB.values(), default=0) + 1))),
        'instrument_emb_dim': int(os.environ.get('INSTRUMENT_EMB_DIM', '16')),
        'instrument_vocab_size': int(os.environ.get('NUM_INSTRUMENTS', str(max(INSTRUMENT_VOCAB.values(), default=7) + 1))),
        'ms_level_emb_dim': int(os.environ.get('MS_LEVEL_EMB_DIM', '8')),
        'ms_level_vocab_size': int(os.environ.get('NUM_MS_LEVELS', str(max(MS_LEVEL_VOCAB.values(), default=4) + 1))),
        'formulae_aux_dim': int(formulae_aux_dim),
        'fragment_local_aux_dim': int(fragment_local_aux_dim),
        'use_msms_conditioning': bool(use_msms_conditioning),
        'score_cond_concat': os.environ.get('SCORE_COND_CONCAT', '1') == '1',
        'prob_softmax': prob_softmax,
        'normalize_1_output': normalize_output,
        'pred_exact_topk': max(0, int(os.environ.get('PRED_EXACT_TOPK_FORMULA', '0'))),
        'pred_exact_min_prob': max(0.0, float(os.environ.get('PRED_EXACT_MIN_FORMULA_PROB', '0.0'))),
    }

    model = formulaenets.GraphVertSpect(
        g_feature_n=first_sample['vect_feat'].shape[-1],
        spect_bin=spect_bin,
        int_d=spect_out_config['internal_d'],
        layer_n=4,
        gml_config={'layer_config': {'dropout': float(os.environ.get('DROPOUT', '0.1'))}},
        spect_out_class='MolAttentionGRUNewSparse',
        spect_out_config=spect_out_config,
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    load_model_path = os.environ.get("LOAD_MODEL_PATH", "").strip()
    if load_model_path:
        if not os.path.exists(load_model_path):
            raise FileNotFoundError(f"LOAD_MODEL_PATH not found: {load_model_path}")
        obj = torch.load(load_model_path, map_location=device)
        if isinstance(obj, dict):
            missing, unexpected = model.load_state_dict(obj, strict=False)
        else:
            missing, unexpected = model.load_state_dict(obj.state_dict(), strict=False)
        log(
            f"馃攣 loaded model from {load_model_path} | "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    assert hasattr(model.spect_out, 'set_score_norm')
    assert hasattr(model.spect_out, 'selector_head')
    assert hasattr(model.spect_out, 'peak_score_mlp')
    assert hasattr(model.spect_out, 'oos_head')

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


    os.makedirs('checkpoints', exist_ok=True)
    best_val_official_cos = -1.0
    early_stop_wait = 0
    history = {
        'train_loss': [],
        'train_main_candidate_kl': [],
        'train_official_spectral_loss': [],
        'train_peak_aux': [],
        'train_oos_loss': [],
        'train_rerank_teacher_ratio': [],
        'train_rerank_kl': [],
        'train_rerank_bce': [],
        'train_rerank_loss': [],
        'train_formula_entropy': [],
        'val_formula_entropy': [],
        'val_loss': [],
        'val_main_candidate_kl': [],
        'val_rerank_kl': [],
        'val_rerank_bce': [],
        'val_rerank_loss': [],
        'val_official_spectral_loss': [],
        'val_peak_aux': [],
        'val_oos_loss': [],
        'val_official_cos_no_precursor': [],
        'val_official_js_no_precursor': [],
        'val_topk_peak_recall@20': [],
        'train_false_support': [],
        'val_false_support': [],
        'val_matched_intensity_coverage': [],
        'train_selector_loss': [],
        'train_selector_bce': [],
        'train_selector_kl': [],
        'train_selector_quality_mean': [],
        'train_selector_pos_rate': [],
        'train_target_pos_false_mass': [],
        'train_target_pos_overlap_exact': [],
        'train_target_pos_exact_support_mass': [],
        'train_target_strict_keep_rate': [],
        'train_target_fallback_rate': [],
        'train_target_clean_pos_rate': [],
        'train_target_pool_pos_rate': [],
        'train_target_pool_pos_false_mass': [],
        'train_target_pool_pos_overlap_tol': [],
        'train_target_teacher_pos_rate': [],
        'train_target_teacher_dist_n': [],
        'train_target_teacher_added_rate': [],
        'train_selector_dyn_pos_weight': [],
        'train_use_rerank_delta': [],
        'val_selector_loss': [],
        'val_selector_bce': [],
        'val_selector_kl': [],
        'val_selector_quality_mean': [],
        'val_selector_pos_rate': [],
        'val_target_pos_false_mass': [],
        'val_target_pos_overlap_exact': [],
        'val_target_pos_exact_support_mass': [],
        'val_target_strict_keep_rate': [],
        'val_target_fallback_rate': [],
        'val_target_clean_pos_rate': [],
        'val_target_pool_pos_rate': [],
        'val_target_pool_pos_false_mass': [],
        'val_target_pool_pos_overlap_tol': [],
        'val_target_teacher_pos_rate': [],
        'val_target_teacher_dist_n': [],
        'val_target_teacher_added_rate': [],
        'val_selector_dyn_pos_weight': [],
        'val_use_rerank_delta': [],
        'val_model_topk_teacher_recall': [],
        'train_precursor_loss': [],
        'val_precursor_loss': [],
        'train_fn_loss': [],
        'val_fn_loss': [],
        'val_active_teacher_recall': [],
        'val_fragaux_teacher_recall': [],
        'val_fragaux_model_topk_ratio@32': [],
        'val_fragaux_model_topk_ratio@64': [],
        'val_fragaux_model_topk_ratio@128': [],
        'val_fragaux_model_topk_ratio@256': [],
        'val_selector_recall@32': [],
        'val_selector_recall@64': [],
        'val_selector_recall@128': [],
        'val_selector_recall@256': [],
        'val_selector_precision@32': [],
        'val_selector_precision@64': [],
        'val_selector_precision@128': [],
        'val_selector_precision@256': [],
        'val_selector_quality_mean@32': [],
        'val_selector_quality_mean@64': [],
        'val_selector_quality_mean@128': [],
        'val_selector_quality_mean@256': [],
        'val_selected_true_hit_mass@32': [],
        'val_selected_true_hit_mass@64': [],
        'val_selected_true_hit_mass@128': [],
        'val_selected_true_hit_mass@256': [],
        'val_selected_false_mass@32': [],
        'val_selected_false_mass@64': [],
        'val_selected_false_mass@128': [],
        'val_selected_false_mass@256': [],
        'val_teacher_oracle_cos': [],
        'val_teacher_oracle_false_support': [],
        'val_teacher_oracle_pred_int_on_true': [],
        'val_teacher_oracle_pred_n': [],
        'val_model_topk_oracle_cos@256': [],
        'val_model_topk_oracle_false_support@256': [],
        'val_utility_top64_oracle_cos': [],
        'val_utility_top64_false_support': [],
        'val_utility_top64_true_hit_mass': [],
        'val_utility_top64_false_mass': [],
    }

    for epoch in range(epochs):
        model.train()
        train_losses = []
        train_main_kl_vals = []
        train_official_spectral_vals = []
        train_peak_aux_vals = []
        train_oos_vals = []
        train_formula_entropy_vals = []
        train_false_support_vals = []
        train_precursor_loss_vals = []
        train_fn_loss_vals = []
        train_selector_loss_vals = []
        train_selector_bce_vals = []
        train_selector_kl_vals = []
        train_selector_quality_mean_vals = []
        train_selector_pos_rate_vals = []
        train_target_pos_false_mass_vals = []
        train_target_pos_overlap_exact_vals = []
        train_target_pos_exact_support_mass_vals = []
        train_target_strict_keep_rate_vals = []
        train_target_fallback_rate_vals = []
        train_target_clean_pos_rate_vals = []
        train_target_pool_pos_rate_vals = []
        train_target_pool_pos_false_mass_vals = []
        train_target_pool_pos_overlap_tol_vals = []
        train_target_teacher_pos_rate_vals = []
        train_target_teacher_dist_n_vals = []
        train_target_teacher_added_rate_vals = []
        train_selector_dyn_pos_weight_vals = []
        train_use_rerank_delta_vals = []
        train_rerank_kl_vals = []
        train_rerank_bce_vals = []
        train_rerank_loss_vals = []
        train_rerank_teacher_ratio_vals = []
        train_update_n = 0
        for step, raw_batch in enumerate(tqdm(train_dl, desc=f'Epoch {epoch+1}/{epochs} [Train]'), start=1):
            if max_train_steps is not None and step > max_train_steps:
                break
            processed = prepare_batch_cpu(raw_batch, spect_bin)
            batch = move_batch_to_device(processed, device)
            optimizer.zero_grad(set_to_none=True)

            formulae_mask = batch.get('formulae_mask', None)
            if torch.is_tensor(formulae_mask):
                formulae_mask = formulae_mask.float()
                if formulae_mask.dim() > 2:
                    formulae_mask = formulae_mask.reshape(formulae_mask.shape[0], -1)
            else:
                formulae_mask = None

            selector_quality = None
            selector_pos_label = None
            selector_valid_mask = None
            selector_extra = {}
            if torch.is_tensor(formulae_mask):
                if os.environ.get("BUILD_RUNTIME_SELECTOR_TEACHER", "1") == "1":
                    teacher_mode = os.environ.get("RUNTIME_SELECTOR_TEACHER_MODE", "setcover").strip().lower()

                    if teacher_mode in ("setcover", "greedy", "set_cover"):
                        batch["selector_teacher_dist"] = build_selector_teacher_dist_setcover(
                            batch=batch,
                            formulae_mask=formulae_mask,
                            official_bin_n=official_bin_n,
                        )
                    else:
                        batch["selector_teacher_dist"] = build_selector_teacher_dist_from_official_overlap(
                            batch=batch,
                            formulae_mask=formulae_mask,
                            official_bin_n=official_bin_n,
                        )
                selector_quality, selector_pos_label, selector_valid_mask, selector_extra = (
                    build_candidate_local_quality_target(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )
                )

            # ---------- teacher top64 target ----------
            teacher_target_full = compute_formula_target_probs_from_batch(
                batch,
                bin_width=main_target_bin_width,
                max_mz=main_target_max_mz,
                target_mode=formula_target_mode,
                support_temperature=target_support_temperature,
                support_topk=target_support_topk,
            )
            teacher_target_full = _renormalize_target_probs(
                teacher_target_full,
                batch.get('formulae_mask', None),
            )

            teacher_formula_mask = batch.get('formulae_mask', None)
            teacher_topk_for_train = int(teacher_topk_train) if int(teacher_topk_train) > 0 else int(selector_topk)
            if torch.is_tensor(teacher_target_full):
                if use_group_unique_teacher:
                    teacher_positive_mask = (teacher_target_full > 0)

                    teacher_topk_mask = build_group_unique_topk_mask_from_scores(
                        teacher_target_full,
                        formulae_mask=teacher_formula_mask,
                        group_id=batch.get("formulae_instance_group_id", None),
                        topk=teacher_topk_for_train,
                        candidate_mask=teacher_positive_mask,
                    )

                    if torch.is_tensor(teacher_topk_mask):
                        teacher_topk_probs = teacher_target_full * teacher_topk_mask.to(
                            device=teacher_target_full.device,
                            dtype=teacher_target_full.dtype,
                        )
                        teacher_topk_probs = _renormalize_target_probs(
                            teacher_topk_probs,
                            teacher_formula_mask,
                        )
                    else:
                        teacher_topk_probs, teacher_topk_mask = apply_teacher_topk_to_target(
                            teacher_target_full,
                            teacher_formula_mask,
                            topk=teacher_topk_for_train,
                        )
                else:
                    teacher_topk_probs, teacher_topk_mask = apply_teacher_topk_to_target(
                        teacher_target_full,
                        teacher_formula_mask,
                        topk=teacher_topk_for_train,
                    )
            else:
                teacher_topk_probs = None
                teacher_topk_mask = None

            if torch.is_tensor(selector_pos_label):
                if use_group_unique_teacher:
                    selector_positive_mask = selector_pos_label > 0
                    unique_selector_pos = build_group_unique_topk_mask_from_scores(
                        selector_pos_label.float(),
                        formulae_mask=teacher_formula_mask,
                        group_id=batch.get("formulae_instance_group_id", None),
                        topk=teacher_topk_for_train,
                        candidate_mask=selector_positive_mask,
                    )
                    teacher_topk_mask = (
                        unique_selector_pos
                        if torch.is_tensor(unique_selector_pos)
                        else selector_pos_label
                    )
                else:
                    teacher_topk_mask = selector_pos_label
                teacher_topk_probs = None

            true_official_dense = build_true_official_dense_for_batch(
                batch,
                official_metric_cfg,
                device,
            )

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # ============================================================
                # PASS 1: full-candidate selector
                # ============================================================
                res_full = model(**batch, selector_only_forward=True)

                precursor_loss = compute_precursor_loss_from_batch(batch, res_full)
                if not torch.is_tensor(precursor_loss):
                    precursor_loss = res_full['spect'].sum() * 0.0

                fn_loss = res_full['spect'].sum() * 0.0
                if os.environ.get("TRAIN_FRAGMENT_NODE_SELECTOR", "0") == "1":
                    fn_logits = res_full.get('fragment_node_logits', None)
                    fn_label = batch.get('fragment_node_label', None)
                    fn_mask = batch.get('fragment_node_mask', None)
                    if fn_logits is not None and fn_label is not None and fn_mask is not None:
                        if torch.is_tensor(fn_logits) and torch.is_tensor(fn_label) and torch.is_tensor(fn_mask):
                            fn_valid = fn_mask > 0.5
                            if fn_valid.any():
                                try:
                                    pos_weight = float(os.environ.get("FN_POS_WEIGHT", "8.0"))
                                except Exception:
                                    pos_weight = 8.0
                                bce_loss = F.binary_cross_entropy_with_logits(
                                    fn_logits[fn_valid],
                                    fn_label[fn_valid].float(),
                                    pos_weight=fn_logits.new_tensor([pos_weight]),
                                    reduction='mean',
                                )
                                fn_loss = bce_loss

                selector_logits = _get_selector_logits_from_res(res_full)

                # Raw logits: only used for trainable selector losses.
                # Do NOT add rule/prior bias into BCE/KL loss.
                selector_logits_for_loss = selector_logits

                # Biased logits: only used for topK candidate selection / rerank pruning.
                selector_logits_for_topk = _apply_selector_aux_logit_bias(selector_logits, batch)

                selector_target_mask = selector_pos_label
                selector_mask_full = formulae_mask

                selector_bce_loss = res_full['spect'].sum() * 0.0
                selector_kl_loss = res_full['spect'].sum() * 0.0
                selector_pairwise_loss = res_full['spect'].sum() * 0.0
                selector_utility_kl_loss = res_full['spect'].sum() * 0.0
                selector_pos_rate = res_full['spect'].sum() * 0.0
                selector_quality_mean = res_full['spect'].sum() * 0.0
                selector_dyn_pos_weight = res_full['spect'].sum() * 0.0

                if (
                    torch.is_tensor(selector_quality)
                    and torch.is_tensor(selector_pos_label)
                    and torch.is_tensor(selector_valid_mask)
                ):
                    if os.environ.get("SELECTOR_DYNAMIC_POS_WEIGHT", "1") == "1":
                        pos_n = (selector_pos_label * selector_valid_mask).sum()
                        neg_n = ((1.0 - selector_pos_label) * selector_valid_mask).sum()
                        try:
                            dyn_pos_weight_min = float(os.environ.get("SELECTOR_DYN_POS_WEIGHT_MIN", "3.0"))
                        except Exception:
                            dyn_pos_weight_min = 3.0
                        try:
                            dyn_pos_weight_max = float(os.environ.get("SELECTOR_DYN_POS_WEIGHT_MAX", "50.0"))
                        except Exception:
                            dyn_pos_weight_max = 50.0
                        dyn_pos_weight = (neg_n / pos_n.clamp_min(1.0)).clamp(
                            dyn_pos_weight_min,
                            dyn_pos_weight_max,
                        )
                    else:
                        dyn_pos_weight = torch.tensor(
                            float(selector_pos_weight),
                            device=selector_logits.device,
                            dtype=selector_logits.dtype,
                        )

                    selector_logits_masked = selector_logits_for_loss.masked_fill(
                        selector_valid_mask <= 0.5, 0.0
                    )
                    bce_raw = F.binary_cross_entropy_with_logits(
                        selector_logits_masked,
                        selector_pos_label,
                        reduction='none',
                        pos_weight=dyn_pos_weight,
                    )
                    selector_bce_loss = (
                        bce_raw * selector_valid_mask
                    ).sum() / selector_valid_mask.sum().clamp_min(1.0)

                    try:
                        gamma = float(os.environ.get("QUALITY_TARGET_GAMMA", "2.0"))
                    except Exception:
                        gamma = 2.0

                    use_teacher_kl = os.environ.get("SELECTOR_USE_TEACHER_KL", "1") == "1"

                    if (
                        use_teacher_kl
                        and isinstance(selector_extra, dict)
                        and torch.is_tensor(selector_extra.get("teacher_dist", None))
                    ):
                        teacher_dist_t = selector_extra["teacher_dist"].to(
                            device=selector_logits_for_loss.device,
                            dtype=selector_logits_for_loss.dtype,
                        )
                        target_dist = teacher_dist_t * selector_valid_mask.float()
                        target_sum = target_dist.sum(dim=1, keepdim=True)
                        target_dist = target_dist / target_sum.clamp_min(1e-8)

                        clean_pos_for_kl = selector_pos_label.float()
                        if torch.is_tensor(selector_extra.get("clean_pos_label", None)):
                            clean_pos_for_kl = selector_extra["clean_pos_label"].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )

                        fallback_dist = clean_pos_for_kl.float() * selector_valid_mask.float()
                        fallback_dist = fallback_dist / fallback_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

                        target_dist = torch.where(
                            target_sum > 1e-8,
                            target_dist,
                            fallback_dist,
                        )
                    else:
                        clean_pos_for_kl = selector_pos_label.float()
                        if isinstance(selector_extra, dict) and torch.is_tensor(selector_extra.get('clean_pos_label', None)):
                            clean_pos_for_kl = selector_extra['clean_pos_label'].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )

                        target_dist = selector_quality.clamp_min(0.0) ** gamma
                        target_dist = target_dist * clean_pos_for_kl.float() * selector_valid_mask.float()

                        target_sum = target_dist.sum(dim=1, keepdim=True)

                        uniform_pos = clean_pos_for_kl.float() * selector_valid_mask.float()
                        uniform_pos = uniform_pos / uniform_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)

                        uniform_all_pos = selector_pos_label.float() * selector_valid_mask.float()
                        uniform_all_pos = uniform_all_pos / uniform_all_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)

                        uniform_pos = torch.where(
                            uniform_pos.sum(dim=1, keepdim=True) > 1e-8,
                            uniform_pos,
                            uniform_all_pos,
                        )

                        target_dist = torch.where(
                            target_sum > 1e-8,
                            target_dist / target_sum.clamp_min(1e-8),
                            uniform_pos,
                        )

                    log_probs = F.log_softmax(
                        selector_logits_for_loss.masked_fill(
                            selector_valid_mask <= 0.5,
                            _neg_mask_fill_value(selector_logits_for_loss),
                        ),
                        dim=1,
                    )

                    selector_kl_loss = F.kl_div(
                        log_probs,
                        target_dist,
                        reduction='none',
                    ).sum(dim=1)

                    valid_rows = (selector_valid_mask.sum(dim=1) > 0).float()
                    selector_kl_loss = (
                        selector_kl_loss * valid_rows
                    ).sum() / valid_rows.sum().clamp_min(1.0)

                    selector_pos_rate = (
                        selector_pos_label.sum() / selector_valid_mask.sum().clamp_min(1.0)
                    )
                    selector_quality_mean = selector_quality.mean()
                    selector_dyn_pos_weight = dyn_pos_weight.detach()

                    utility_t = None
                    utility_dist_t = None
                    if isinstance(selector_extra, dict):
                        if torch.is_tensor(selector_extra.get("utility", None)):
                            utility_t = selector_extra["utility"].to(
                                device=selector_logits_for_loss.device,
                                dtype=selector_logits_for_loss.dtype,
                            )
                        if torch.is_tensor(selector_extra.get("utility_dist", None)):
                            utility_dist_t = selector_extra["utility_dist"].to(
                                device=selector_logits_for_loss.device,
                                dtype=selector_logits_for_loss.dtype,
                            )

                    if torch.is_tensor(utility_t):
                        selector_pairwise_loss = selector_pairwise_utility_loss(
                            selector_logits=selector_logits_for_loss,
                            utility=utility_t,
                            valid_mask=selector_valid_mask,
                            high_q=float(os.environ.get("SELECTOR_PAIRWISE_HIGH_Q", "0.80")),
                            low_q=float(os.environ.get("SELECTOR_PAIRWISE_LOW_Q", "0.40")),
                            margin=float(os.environ.get("SELECTOR_PAIRWISE_MARGIN", "0.2")),
                            max_pairs=int(os.environ.get("SELECTOR_PAIRWISE_MAX_PAIRS", "2048")),
                        )
                        if (not torch.is_tensor(selector_pairwise_loss)) or (not torch.isfinite(selector_pairwise_loss)):
                            selector_pairwise_loss = res_full['spect'].sum() * 0.0

                    if torch.is_tensor(utility_dist_t):
                        utility_log_probs = F.log_softmax(
                            selector_logits_for_loss.masked_fill(
                                selector_valid_mask <= 0.5,
                                _neg_mask_fill_value(selector_logits_for_loss),
                            ),
                            dim=1,
                        )
                        selector_utility_kl_loss = F.kl_div(
                            utility_log_probs,
                            utility_dist_t,
                            reduction='none',
                        ).sum(dim=1)
                        valid_rows = (selector_valid_mask.sum(dim=1) > 0).float()
                        selector_utility_kl_loss = (
                            selector_utility_kl_loss * valid_rows
                        ).sum() / valid_rows.sum().clamp_min(1.0)
                        if (not torch.is_tensor(selector_utility_kl_loss)) or (not torch.isfinite(selector_utility_kl_loss)):
                            selector_utility_kl_loss = res_full['spect'].sum() * 0.0

                target_pos_false_mass = res_full['spect'].sum() * 0.0
                target_pos_overlap_exact = res_full['spect'].sum() * 0.0
                target_clean_pos_rate = res_full['spect'].sum() * 0.0
                target_pool_pos_rate = res_full['spect'].sum() * 0.0
                target_pool_pos_false_mass = res_full['spect'].sum() * 0.0
                target_pool_pos_overlap_tol = res_full['spect'].sum() * 0.0
                target_teacher_pos_rate = res_full['spect'].sum() * 0.0
                target_teacher_dist_n = res_full['spect'].sum() * 0.0
                target_teacher_added_rate = res_full['spect'].sum() * 0.0
                if (
                    isinstance(selector_extra, dict)
                    and torch.is_tensor(selector_pos_label)
                    and torch.is_tensor(selector_extra.get('false_support_mass_exact', None))
                    and torch.is_tensor(selector_extra.get('overlap_intensity_exact', None))
                ):
                    pos = selector_pos_label.float()
                    pos_den = pos.sum().clamp_min(1.0)
                    target_pos_false_mass = (
                        selector_extra['false_support_mass_exact'].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        * pos
                    ).sum() / pos_den
                    target_pos_overlap_exact = (
                        selector_extra['overlap_intensity_exact'].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        * pos
                    ).sum() / pos_den
                    if torch.is_tensor(selector_extra.get('clean_pos_label', None)):
                        target_clean_pos_rate = (
                            selector_extra['clean_pos_label'].to(device=selector_pos_label.device).float()
                            * formulae_mask.float()
                        ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                    if torch.is_tensor(selector_extra.get('pool_pos_label', None)):
                        target_pool_pos_rate = (
                            selector_extra['pool_pos_label'].to(device=selector_pos_label.device).float()
                            * formulae_mask.float()
                        ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                        pool_pos = selector_extra['pool_pos_label'].to(
                            device=selector_pos_label.device,
                            dtype=selector_pos_label.dtype,
                        )
                        pool_den = pool_pos.sum().clamp_min(1.0)
                        if torch.is_tensor(selector_extra.get('false_support_mass_exact', None)):
                            target_pool_pos_false_mass = (
                                selector_extra['false_support_mass_exact'].to(
                                    device=pool_pos.device,
                                    dtype=pool_pos.dtype,
                                ) * pool_pos
                            ).sum() / pool_den
                        if torch.is_tensor(selector_extra.get('overlap_intensity_tol', None)):
                            target_pool_pos_overlap_tol = (
                                selector_extra['overlap_intensity_tol'].to(
                                    device=pool_pos.device,
                                    dtype=pool_pos.dtype,
                                ) * pool_pos
                            ).sum() / pool_den
                    if torch.is_tensor(selector_extra.get("teacher_pos_label", None)):
                        teacher_pos = selector_extra["teacher_pos_label"].to(
                            device=selector_pos_label.device,
                            dtype=selector_pos_label.dtype,
                        )
                        target_teacher_pos_rate = (
                            teacher_pos * formulae_mask.float()
                        ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                    if torch.is_tensor(selector_extra.get("teacher_dist", None)):
                        teacher_dist_t = selector_extra["teacher_dist"].to(
                            device=selector_pos_label.device,
                            dtype=selector_pos_label.dtype,
                        )
                        target_teacher_dist_n = (
                            teacher_dist_t > 0
                        ).float().sum(dim=1).mean()
                    if torch.is_tensor(selector_extra.get("teacher_added_label", None)):
                        teacher_added = selector_extra["teacher_added_label"].to(
                            device=selector_pos_label.device,
                            dtype=selector_pos_label.dtype,
                        )
                        target_teacher_added_rate = (
                            teacher_added * formulae_mask.float()
                        ).sum() / formulae_mask.float().sum().clamp_min(1.0)

                selector_loss = (
                    float(selector_bce_weight) * selector_bce_loss
                    + float(selector_kl_weight) * selector_kl_loss
                    + float(selector_pairwise_weight) * selector_pairwise_loss
                    + float(selector_utility_kl_weight) * selector_utility_kl_loss
                )

                target_pos_exact_support_mass = res_full['spect'].sum() * 0.0
                target_strict_keep_rate = res_full['spect'].sum() * 0.0
                target_fallback_rate = res_full['spect'].sum() * 0.0

                if (
                    isinstance(selector_extra, dict)
                    and torch.is_tensor(selector_pos_label)
                ):
                    pos = selector_pos_label.float()
                    pos_den = pos.sum().clamp_min(1.0)

                    if torch.is_tensor(selector_extra.get('exact_support_mass', None)):
                        exact_support_mass_t = selector_extra['exact_support_mass'].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        target_pos_exact_support_mass = (
                            exact_support_mass_t * pos
                        ).sum() / pos_den

                    if torch.is_tensor(selector_extra.get('strict_keep', None)):
                        strict_keep_t = selector_extra['strict_keep'].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        )
                        target_strict_keep_rate = (
                            strict_keep_t * formulae_mask.float()
                        ).sum() / formulae_mask.float().sum().clamp_min(1.0)

                    if torch.is_tensor(selector_extra.get('fallback_used', None)):
                        target_fallback_rate = selector_extra['fallback_used'].to(
                            device=pos.device,
                            dtype=pos.dtype,
                        ).float().mean()

                pair_loss_total = selector_logits_for_loss.new_tensor(0.0)
                pair_count = 0

                # Optional: selector pairwise ranking loss (hard-negative sampling)
                if os.environ.get('ENABLE_SELECTOR_PAIRWISE', '0') == '1':
                    try:
                        pair_margin = float(os.environ.get('SELECTOR_PAIRWISE_MARGIN', '0.1'))
                    except Exception:
                        pair_margin = 0.1
                    try:
                        pair_weight = float(os.environ.get('SELECTOR_PAIRWISE_WEIGHT', '1.0'))
                    except Exception:
                        pair_weight = 1.0
                    try:
                        num_neg = int(os.environ.get('SELECTOR_PAIRWISE_NEG', '8'))
                    except Exception:
                        num_neg = 8

                    pair_loss_total = selector_logits_for_loss.new_tensor(0.0)
                    pair_count = 0
                    # selector_logits_for_loss: [B, M]
                    scores = selector_logits_for_loss
                    if torch.is_tensor(scores) and torch.is_tensor(selector_target_mask):
                        tgt_mask = selector_target_mask.float()
                        fm = selector_mask_full.float() if torch.is_tensor(selector_mask_full) else torch.ones_like(tgt_mask)
                        B = min(int(scores.shape[0]), int(tgt_mask.shape[0]))
                        M = min(int(scores.shape[1]), int(tgt_mask.shape[1]))
                        sc = scores[:B, :M]
                        tg = tgt_mask[:B, :M]
                        fm = fm[:B, :M]
                        for bi in range(B):
                            pos_idx = torch.where((tg[bi] > 0.5) & (fm[bi] > 0.5))[0]
                            if pos_idx.numel() == 0:
                                continue
                            # negatives: top-k by model that are not pos and valid
                            valid_idx = torch.where(fm[bi] > 0.5)[0]
                            if valid_idx.numel() == 0:
                                continue
                            scores_row = sc[bi, valid_idx]
                            # sort desc
                            _, order = torch.sort(scores_row, descending=True)
                            # select negatives not in pos
                            negs = []
                            for o in order.tolist():
                                cand = int(valid_idx[o])
                                if (tg[bi, cand] > 0.5):
                                    continue
                                negs.append(cand)
                                if len(negs) >= num_neg:
                                    break
                            if len(negs) == 0:
                                continue
                            # use worst positive (lowest score among positives) and sampled negatives
                            pos_scores = sc[bi, pos_idx]
                            pos_score = pos_scores.min() if pos_scores.numel() > 0 else pos_scores.mean()
                            neg_scores = sc[bi, negs]
                            # pairwise hinge
                            diff = pair_margin - (pos_score.unsqueeze(0) - neg_scores)
                            hinge = torch.clamp(diff, min=0.0)
                            pair_loss_total = pair_loss_total + hinge.mean()
                            pair_count += 1
                if pair_count > 0:
                    pair_loss_avg = pair_loss_total / float(max(1, pair_count))
                    selector_loss = selector_loss + float(pair_weight) * pair_loss_avg
                # ============================================================
                # Selector false-support loss.
                # This must be computed before selector-only/full-stage branching,
                # so that TRAIN_SELECTOR_ONLY_STAGE=1 can still use it.
                # ============================================================
                selector_false_support_loss = selector_loss.new_zeros(())
                selector_utility_loss = selector_loss.new_zeros(())

                if selector_utility_loss_weight > 0.0:
                    selector_utility_loss = compute_selector_utility_target_loss(
                        selector_logits=selector_logits_for_topk,
                        batch=batch,
                    )

                    if (not torch.is_tensor(selector_utility_loss)) or (not torch.isfinite(selector_utility_loss)):
                        selector_utility_loss = selector_loss.new_zeros(())

                if float(false_support_loss_weight) > 0.0 and os.environ.get("USE_SELECTOR_FALSE_SUPPORT_LOSS", "1") == "1":
                    selector_false_support_loss = compute_selector_false_support_loss(
                        selector_logits=selector_logits_for_topk,
                        batch=batch,
                        topk=int(os.environ.get("MODEL_TOPK_EVAL", os.environ.get("SELECTOR_TOPK", "64"))),
                    )

                    if (not torch.is_tensor(selector_false_support_loss)) or (not torch.isfinite(selector_false_support_loss)):
                        selector_false_support_loss = selector_loss.new_zeros(())
                # ============================================================
                # Optional Stage 1: selector-only training.
                # In this stage, do NOT run rerank / official projection / peak / OOS.
                # ============================================================
                if bool(train_selector_only_stage):
                    rerank_teacher_ratio = 1.0

                    main_candidate_kl = selector_loss.new_zeros(())
                    rerank_kl = selector_loss.new_zeros(())
                    rerank_bce = selector_loss.new_zeros(())
                    official_spectral_loss = selector_loss.new_zeros(())
                    peak_aux_loss = selector_loss.new_zeros(())
                    oos_loss = selector_loss.new_zeros(())
                    formula_entropy_loss = selector_loss.new_zeros(())

                    # Important:
                    # In selector-only stage, false_support_loss should mean selector-level
                    # false-support loss, not dense spectral false-support loss.
                    false_support_loss = selector_false_support_loss

                    loss = float(selector_loss_weight) * selector_loss

                    if float(selector_utility_loss_weight) > 0.0:
                        loss = loss + float(selector_utility_loss_weight) * selector_utility_loss

                    if float(false_support_loss_weight) > 0.0:
                        loss = loss + float(false_support_loss_weight) * false_support_loss

                    if epoch >= precursor_loss_start_epoch and precursor_loss_weight > 0:
                        loss = loss + float(precursor_loss_weight) * precursor_loss

                else:
                    # ============================================================
                    # PASS 2: teacher/model mixed-topK reranker
                    # ============================================================
                    topk_candidate_mask_train = None
                    if os.environ.get("MODEL_TOPK_USE_ACTIVE_MASK", "0") == "1":
                        topk_candidate_mask_train = _get_active_candidate_mask_from_batch(
                            batch,
                            formulae_mask=batch.get('formulae_mask', None),
                        )

                    if use_group_unique_model:
                        model_topk_mask_train = build_group_unique_topk_mask_from_scores(
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            group_id=batch.get("formulae_instance_group_id", None),
                            topk=selector_topk,
                            candidate_mask=topk_candidate_mask_train,
                        )
                    else:
                        model_topk_mask_train = build_topk_mask_from_scores(
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            topk=selector_topk,
                            candidate_mask=topk_candidate_mask_train,
                        )

                    rerank_mask = model_topk_mask_train
                    rerank_teacher_ratio = 0.0

                    batch_with_teacher = dict(batch)
                    if torch.is_tensor(teacher_target_full):
                        batch_with_teacher['teacher_formula_probs'] = teacher_target_full

                    batch_rerank = _prune_batch_by_candidate_mask(
                        batch_with_teacher,
                        rerank_mask,
                        keep_topk=selector_topk,
                        fill_scores=selector_logits_for_topk.detach(),
                        group_id=batch.get("formulae_instance_group_id", None),
                        group_unique=(use_group_unique_prune),
                    )

                    if torch.is_tensor(batch_rerank.get('teacher_formula_probs', None)):
                        batch_rerank['teacher_formula_probs'] = _renormalize_target_probs(
                            batch_rerank.get('teacher_formula_probs', None),
                            batch_rerank.get('formulae_mask', None),
                        )

                    res = model(**batch_rerank)
                    pred_spect_coarse = res['spect'] if isinstance(res, dict) else res
                    pred_spect_official = res.get('spect_out_official', None) if isinstance(res, dict) else None

                    formulae_scores = _get_reranker_scores_from_res(res)

                    formula_entropy_loss = _masked_formula_entropy_loss(
                        formulae_scores,
                        batch_rerank.get('formulae_mask', None),
                    )
                    if not torch.is_tensor(formula_entropy_loss):
                        formula_entropy_loss = pred_spect_coarse.sum() * 0.0

                    rerank_logits_pool = res.get('rerank_logits_pool', None)
                    pool_idx = res.get('selector_pool_idx', None)
                    rerank_loss = pred_spect_coarse.sum() * 0.0
                    rerank_kl = pred_spect_coarse.sum() * 0.0
                    rerank_bce = pred_spect_coarse.sum() * 0.0

                    if torch.is_tensor(rerank_logits_pool) and torch.is_tensor(pool_idx):
                        formulae_mask_rerank = batch_rerank.get('formulae_mask', None)
                        if torch.is_tensor(formulae_mask_rerank):
                            formulae_mask_rerank = formulae_mask_rerank.float()
                            if formulae_mask_rerank.dim() > 2:
                                formulae_mask_rerank = formulae_mask_rerank.reshape(
                                    formulae_mask_rerank.shape[0], -1
                                )
                        else:
                            formulae_mask_rerank = torch.ones(
                                (int(rerank_logits_pool.shape[0]), int(rerank_logits_pool.shape[1])),
                                dtype=torch.float32,
                                device=rerank_logits_pool.device,
                            )

                        selector_quality_rerank, _, _, _ = build_candidate_local_quality_target(
                            batch=batch_rerank,
                            formulae_mask=formulae_mask_rerank,
                            official_bin_n=official_bin_n,
                        )

                        rerank_target_dist, rerank_pos, rerank_pool_mask = (
                            build_setcover_rerank_target_from_quality(
                                selector_quality=selector_quality_rerank.detach(),
                                pool_idx=pool_idx,
                                formulae_mask=formulae_mask_rerank,
                            )
                        )

                        logp_pool = F.log_softmax(
                            rerank_logits_pool.masked_fill(
                                rerank_pool_mask <= 0.5,
                                _neg_mask_fill_value(rerank_logits_pool),
                            ),
                            dim=1,
                        )

                        rerank_kl = F.kl_div(
                            logp_pool,
                            rerank_target_dist,
                            reduction='none',
                        ).sum(dim=1)

                        valid_rows = (rerank_pool_mask.sum(dim=1) > 0).float()
                        rerank_kl = (
                            rerank_kl * valid_rows
                        ).sum() / valid_rows.sum().clamp_min(1.0)

                        rerank_bce_raw = F.binary_cross_entropy_with_logits(
                            rerank_logits_pool,
                            rerank_pos,
                            reduction='none',
                        )
                        rerank_bce = (
                            rerank_bce_raw * rerank_pool_mask
                        ).sum() / rerank_pool_mask.sum().clamp_min(1.0)

                        rerank_loss = float(rerank_kl_weight) * rerank_kl + float(rerank_bce_weight) * rerank_bce

                    main_candidate_kl = rerank_loss

                    if torch.is_tensor(pred_spect_official) and torch.is_tensor(true_official_dense):
                        official_spectral_loss = compute_official_dense_spectral_loss(
                            pred_spect_official,
                            true_official_dense,
                            kl_weight=official_spectral_kl_weight,
                        )
                        false_support_loss = _false_support_mass_loss_dense(
                            pred_spect_official,
                            true_official_dense,
                        )
                        if not torch.is_tensor(false_support_loss):
                            false_support_loss = pred_spect_coarse.sum() * 0.0
                    else:
                        official_spectral_loss = pred_spect_coarse.sum() * 0.0
                        false_support_loss = pred_spect_coarse.sum() * 0.0

                    peak_aux_loss = _compute_peak_aux_loss_from_batch(
                        batch_rerank,
                        res if isinstance(res, dict) else {},
                        official_bin_width=official_metric_cfg['bin_width'],
                        official_max_mz=official_metric_cfg['max_mz'],
                    )
                    if (not torch.is_tensor(peak_aux_loss)) or (not torch.isfinite(peak_aux_loss)):
                        peak_aux_loss = pred_spect_coarse.sum() * 0.0

                    oos_loss = _compute_oos_loss_from_batch(
                        batch_rerank,
                        res if isinstance(res, dict) else {},
                        official_bin_width=official_metric_cfg['bin_width'],
                        official_max_mz=official_metric_cfg['max_mz'],
                    )
                    if not torch.is_tensor(oos_loss):
                        oos_loss = pred_spect_coarse.sum() * 0.0

                    loss = float(selector_loss_weight) * selector_loss

                    if epoch >= precursor_loss_start_epoch and precursor_loss_weight > 0:
                        loss = loss + float(precursor_loss_weight) * precursor_loss

                    if epoch >= selector_only_warmup_epochs and main_candidate_kl_weight > 0:
                        loss = loss + float(main_candidate_kl_weight) * main_candidate_kl

                    if epoch >= selector_only_warmup_epochs and formula_entropy_loss_weight > 0:
                        loss = loss + float(formula_entropy_loss_weight) * formula_entropy_loss
                    if epoch >= spectral_loss_start_epoch and false_support_loss_weight > 0:
                        loss = loss + float(false_support_loss_weight) * false_support_loss

                    if epoch >= spectral_loss_start_epoch and official_spectral_loss_weight > 0:
                        loss = loss + float(official_spectral_loss_weight) * official_spectral_loss

                    if epoch >= peak_aux_start_epoch and peak_aux_loss_weight > 0:
                        loss = loss + float(peak_aux_loss_weight) * peak_aux_loss

                    if epoch >= oos_loss_start_epoch and oos_loss_weight > 0:
                        loss = loss + float(oos_loss_weight) * oos_loss

            if os.environ.get("TRAIN_FRAGMENT_NODE_SELECTOR", "0") == "1":
                loss = loss + float(os.environ.get("FN_LOSS_WEIGHT", "1.0")) * fn_loss

            if not torch.isfinite(loss):
                continue

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
                optimizer.step()
            train_update_n += 1
            train_losses.append(float(loss.detach().item()))
            train_main_kl_vals.append(float(main_candidate_kl.detach().item()))
            train_official_spectral_vals.append(float(official_spectral_loss.detach().item()))
            train_peak_aux_vals.append(float(peak_aux_loss.detach().item()))
            train_oos_vals.append(float(oos_loss.detach().item()))
            train_formula_entropy_vals.append(float(formula_entropy_loss.detach().item()))
            train_precursor_loss_vals.append(float(precursor_loss.detach().item()))
            train_fn_loss_vals.append(float(fn_loss.detach().item()))
            train_selector_loss_vals.append(float(selector_loss.detach().item()))
            train_selector_bce_vals.append(float(selector_bce_loss.detach().item()))
            train_selector_kl_vals.append(float(selector_kl_loss.detach().item()))
            train_selector_quality_mean_vals.append(float(selector_quality_mean.detach().item()))
            train_selector_pos_rate_vals.append(float(selector_pos_rate.detach().item()))
            train_target_pos_false_mass_vals.append(float(target_pos_false_mass.detach().item()))
            train_target_pos_overlap_exact_vals.append(float(target_pos_overlap_exact.detach().item()))
            train_target_pos_exact_support_mass_vals.append(
                float(target_pos_exact_support_mass.detach().item())
            )
            train_target_strict_keep_rate_vals.append(
                float(target_strict_keep_rate.detach().item())
            )
            train_target_fallback_rate_vals.append(
                float(target_fallback_rate.detach().item())
            )
            train_target_clean_pos_rate_vals.append(
                float(target_clean_pos_rate.detach().item())
            )
            train_target_pool_pos_rate_vals.append(
                float(target_pool_pos_rate.detach().item())
            )
            train_target_pool_pos_false_mass_vals.append(
                float(target_pool_pos_false_mass.detach().item())
            )
            train_target_pool_pos_overlap_tol_vals.append(
                float(target_pool_pos_overlap_tol.detach().item())
            )
            train_target_teacher_pos_rate_vals.append(
                float(target_teacher_pos_rate.detach().item())
            )
            train_target_teacher_dist_n_vals.append(
                float(target_teacher_dist_n.detach().item())
            )
            train_target_teacher_added_rate_vals.append(
                float(target_teacher_added_rate.detach().item())
            )
            train_selector_dyn_pos_weight_vals.append(
                float(selector_dyn_pos_weight.detach().item())
            )
            use_rerank_delta_val = 0.0
            if isinstance(res_full, dict):
                v = res_full.get('use_rerank_delta', 0.0)
                if torch.is_tensor(v):
                    try:
                        use_rerank_delta_val = float(v.detach().reshape(-1)[0].item())
                    except Exception:
                        use_rerank_delta_val = 0.0
                else:
                    try:
                        use_rerank_delta_val = float(v)
                    except Exception:
                        use_rerank_delta_val = 0.0

            train_use_rerank_delta_vals.append(use_rerank_delta_val)
            train_rerank_kl_vals.append(float(rerank_kl.detach().item()))
            train_rerank_bce_vals.append(float(rerank_bce.detach().item()))
            train_rerank_loss_vals.append(float(main_candidate_kl.detach().item()))
            train_rerank_teacher_ratio_vals.append(float(rerank_teacher_ratio))
            train_false_support_vals.append(float(false_support_loss.detach().item()))
        model.eval()    
        val_losses = []
        val_main_kl_vals = []
        val_official_spectral_vals = []
        val_peak_aux_vals = []
        val_oos_vals = []
        val_formula_entropy_vals = []
        val_false_support_vals = []
        val_precursor_loss_vals = []
        val_fn_loss_vals = []
        official_cos_vals = []
        official_js_vals = []
        official_recall_vals = []
        official_cov_vals = []
        official_pred_n_vals = []
        official_true_n_vals = []
        official_overlap_n_vals = []
        official_false_pred_n_vals = []
        official_pred_int_on_true_vals = []
        val_selector_loss_vals = []
        val_selector_bce_vals = []
        val_selector_kl_vals = []
        val_selector_quality_mean_vals = []
        val_selector_pos_rate_vals = []
        val_target_pos_false_mass_vals = []
        val_target_pos_overlap_exact_vals = []
        val_target_pos_exact_support_mass_vals = []
        val_target_strict_keep_rate_vals = []
        val_target_fallback_rate_vals = []
        val_target_clean_pos_rate_vals = []
        val_target_pool_pos_rate_vals = []
        val_target_pool_pos_false_mass_vals = []
        val_target_pool_pos_overlap_tol_vals = []
        val_target_teacher_pos_rate_vals = []
        val_target_teacher_dist_n_vals = []
        val_target_teacher_added_rate_vals = []
        val_selector_dyn_pos_weight_vals = []
        val_use_rerank_delta_vals = []
        val_rerank_kl_vals = []
        val_rerank_bce_vals = []
        val_rerank_loss_vals = []
        val_model_topk_teacher_recall_vals = []     
        val_active_teacher_recall_vals = []
        val_fragaux_teacher_recall_vals = []
        val_fragaux_model_topk_ratio_32_vals = []
        val_fragaux_model_topk_ratio_64_vals = []
        val_fragaux_model_topk_ratio_128_vals = []
        val_fragaux_model_topk_ratio_256_vals = []
        val_selector_recall_32_vals = []
        val_selector_recall_64_vals = []
        val_selector_recall_128_vals = []
        val_selector_recall_256_vals = []
        val_selector_precision_32_vals = []
        val_selector_precision_64_vals = []
        val_selector_precision_128_vals = []
        val_selector_precision_256_vals = []
        val_selector_quality_mean_32_vals = []
        val_selector_quality_mean_64_vals = []
        val_selector_quality_mean_128_vals = []
        val_selector_quality_mean_256_vals = []
        val_selected_true_hit_mass_32_vals = []
        val_selected_true_hit_mass_64_vals = []
        val_selected_true_hit_mass_128_vals = []
        val_selected_true_hit_mass_256_vals = []
        val_selected_false_mass_32_vals = []
        val_selected_false_mass_64_vals = []
        val_selected_false_mass_128_vals = []
        val_selected_false_mass_256_vals = []
        val_teacher_oracle_cos_vals = []
        val_teacher_oracle_false_support_vals = []
        val_teacher_oracle_pred_int_on_true_vals = []
        val_teacher_oracle_pred_n_vals = []
        val_model_topk_oracle_cos_256_vals = []
        val_model_topk_oracle_false_support_256_vals = []
        val_utility_top64_oracle_cos_vals = []
        val_utility_top64_false_support_vals = []
        val_utility_top64_true_hit_mass_vals = []
        val_utility_top64_false_mass_vals = []
        epoch_best_sparse = None
        epoch_worst_sparse = None
        epoch_best_cos = -1.0
        epoch_worst_cos = 1e9
        epoch_worst_nonempty_sparse = None
        epoch_worst_nonempty_cos = 1e9

        epoch_worst_overlap_sparse = None
        epoch_worst_overlap_cos = 1e9
        with torch.no_grad():
            for step, raw_batch in enumerate(tqdm(val_dl, desc=f'Epoch {epoch+1}/{epochs} [Val]', leave=False), start=1):
                if max_val_steps is not None and step > max_val_steps:
                    break
                processed = prepare_batch_cpu(raw_batch, spect_bin)
                batch = move_batch_to_device(processed, device)

                formulae_mask = batch.get('formulae_mask', None)
                if torch.is_tensor(formulae_mask):
                    formulae_mask = formulae_mask.float()
                    if formulae_mask.dim() > 2:
                        formulae_mask = formulae_mask.reshape(formulae_mask.shape[0], -1)
                else:
                    formulae_mask = None

                selector_quality = None
                selector_pos_label = None
                selector_valid_mask = None
                selector_extra = {}
                if torch.is_tensor(formulae_mask):
                    if os.environ.get("BUILD_RUNTIME_SELECTOR_TEACHER", "1") == "1":
                        teacher_mode = os.environ.get("RUNTIME_SELECTOR_TEACHER_MODE", "setcover").strip().lower()

                        if teacher_mode in ("setcover", "greedy", "set_cover"):
                            batch["selector_teacher_dist"] = build_selector_teacher_dist_setcover(
                                batch=batch,
                                formulae_mask=formulae_mask,
                                official_bin_n=official_bin_n,
                            )
                        else:
                            batch["selector_teacher_dist"] = build_selector_teacher_dist_from_official_overlap(
                                batch=batch,
                                formulae_mask=formulae_mask,
                                official_bin_n=official_bin_n,
                            )
                    selector_quality, selector_pos_label, selector_valid_mask, selector_extra = (
                        build_candidate_local_quality_target(
                            batch=batch,
                            formulae_mask=formulae_mask,
                            official_bin_n=official_bin_n,
                        )
                    )

                # teacher target 鍙敤浜庤瘖鏂?KL锛屼笉鐢ㄤ簬楠岃瘉 forward
                teacher_target_full = compute_formula_target_probs_from_batch(
                    batch,
                    bin_width=main_target_bin_width,
                    max_mz=main_target_max_mz,
                    target_mode=formula_target_mode,
                    support_temperature=target_support_temperature,
                    support_topk=target_support_topk,
                )

                teacher_formula_mask = batch.get('formulae_mask', None)
                teacher_topk_for_eval = int(teacher_topk_eval) if int(teacher_topk_eval) > 0 else int(model_topk_eval)

                if torch.is_tensor(teacher_target_full):
                    if use_group_unique_teacher:
                        teacher_positive_mask = (teacher_target_full > 0)

                        teacher_topk_mask = build_group_unique_topk_mask_from_scores(
                            teacher_target_full,
                            formulae_mask=teacher_formula_mask,
                            group_id=batch.get("formulae_instance_group_id", None),
                            topk=teacher_topk_for_eval,
                            candidate_mask=teacher_positive_mask,
                        )

                        if torch.is_tensor(teacher_topk_mask):
                            teacher_topk_probs = teacher_target_full * teacher_topk_mask.to(
                                device=teacher_target_full.device,
                                dtype=teacher_target_full.dtype,
                            )
                            teacher_topk_probs = _renormalize_target_probs(
                                teacher_topk_probs,
                                teacher_formula_mask,
                            )
                        else:
                            teacher_topk_probs, teacher_topk_mask = apply_teacher_topk_to_target(
                                teacher_target_full,
                                teacher_formula_mask,
                                topk=teacher_topk_for_eval,
                            )
                    else:
                        teacher_topk_probs, teacher_topk_mask = apply_teacher_topk_to_target(
                            teacher_target_full,
                            teacher_formula_mask,
                            topk=teacher_topk_for_eval,
                        )
                else:
                    teacher_topk_probs = None
                    teacher_topk_mask = None

                if torch.is_tensor(selector_pos_label):
                    if use_group_unique_teacher:
                        selector_positive_mask = selector_pos_label > 0
                        unique_selector_pos = build_group_unique_topk_mask_from_scores(
                            selector_pos_label.float(),
                            formulae_mask=teacher_formula_mask,
                            group_id=batch.get("formulae_instance_group_id", None),
                            topk=teacher_topk_for_eval,
                            candidate_mask=selector_positive_mask,
                        )
                        teacher_topk_mask = (
                            unique_selector_pos
                            if torch.is_tensor(unique_selector_pos)
                            else selector_pos_label
                        )
                    else:
                        teacher_topk_mask = selector_pos_label
                    teacher_topk_probs = None

                teacher_target_full = _renormalize_target_probs(
                    teacher_target_full,
                    teacher_formula_mask,
                )

                active_teacher_recall = float("nan")
                active_mask_diag = _get_active_candidate_mask_from_batch(
                    batch,
                    formulae_mask=batch.get('formulae_mask', None),
                )
                if torch.is_tensor(active_mask_diag) and torch.is_tensor(teacher_topk_mask):
                    active_teacher_recall = mask_recall(active_mask_diag, teacher_topk_mask)


                fragaux_teacher_recall = float("nan")
                frag_source_mask = None

                frag_aux = batch.get("formulae_frag_aux_feat", None)
                if torch.is_tensor(frag_aux):
                    fa = frag_aux.float()
                    if fa.dim() == 2:
                        fa = fa.unsqueeze(0)
                    elif fa.dim() > 3:
                        fa = fa.reshape(fa.shape[0], fa.shape[1], -1)

                    if fa.dim() == 3:
                        frag_source_mask = (torch.linalg.norm(fa, dim=-1) > 1e-8).float()

                        fm = batch.get("formulae_mask", None)
                        if torch.is_tensor(fm):
                            fm = fm.float()
                            if fm.dim() > 2:
                                fm = fm.reshape(fm.shape[0], -1)

                            use_b = min(int(frag_source_mask.shape[0]), int(fm.shape[0]))
                            use_m = min(int(frag_source_mask.shape[1]), int(fm.shape[1]))
                            frag_source_mask = frag_source_mask[:use_b, :use_m]
                            fm = fm[:use_b, :use_m]
                            frag_source_mask = frag_source_mask * (fm > 0.5).float()

                if torch.is_tensor(frag_source_mask) and torch.is_tensor(teacher_topk_mask):
                    fragaux_teacher_recall = mask_recall(frag_source_mask, teacher_topk_mask)

                true_official_dense = build_true_official_dense_for_batch(
                    batch,
                    official_metric_cfg,
                    device,
                )

                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    # ============================================================
                    # PASS 1: full-candidate selector
                    # ============================================================
                    res_full = model(**batch, selector_only_forward=True)

                    val_precursor_loss = compute_precursor_loss_from_batch(batch, res_full)
                    if not torch.is_tensor(val_precursor_loss):
                        val_precursor_loss = res_full['spect'].sum() * 0.0

                    val_fn_loss = res_full['spect'].sum() * 0.0
                    if os.environ.get("TRAIN_FRAGMENT_NODE_SELECTOR", "0") == "1":
                        fn_logits = res_full.get('fragment_node_logits', None)
                        fn_label = batch.get('fragment_node_label', None)
                        fn_mask = batch.get('fragment_node_mask', None)
                        if fn_logits is not None and fn_label is not None and fn_mask is not None:
                            if torch.is_tensor(fn_logits) and torch.is_tensor(fn_label) and torch.is_tensor(fn_mask):
                                fn_valid = fn_mask > 0.5
                                if fn_valid.any():
                                    try:
                                        pos_weight = float(os.environ.get("FN_POS_WEIGHT", "8.0"))
                                    except Exception:
                                        pos_weight = 8.0
                                    bce_loss = F.binary_cross_entropy_with_logits(
                                        fn_logits[fn_valid],
                                        fn_label[fn_valid].float(),
                                        pos_weight=fn_logits.new_tensor([pos_weight]),
                                        reduction='mean',
                                    )
                                    val_fn_loss = bce_loss

                    selector_logits = _get_selector_logits_from_res(res_full)

                    # Raw logits for diagnostic selector losses.
                    selector_logits_for_loss = selector_logits

                    # Biased logits only for topK selection.
                    selector_logits_for_topk = _apply_selector_aux_logit_bias(selector_logits, batch)

                    val_selector_bce = res_full['spect'].sum() * 0.0
                    val_selector_kl = res_full['spect'].sum() * 0.0
                    val_selector_pairwise = res_full['spect'].sum() * 0.0
                    val_selector_utility_kl = res_full['spect'].sum() * 0.0
                    val_selector_pos_rate = res_full['spect'].sum() * 0.0
                    val_selector_quality_mean = res_full['spect'].sum() * 0.0
                    val_selector_dyn_pos_weight = res_full['spect'].sum() * 0.0

                    if (
                        torch.is_tensor(selector_quality)
                        and torch.is_tensor(selector_pos_label)
                        and torch.is_tensor(selector_valid_mask)
                    ):
                        if os.environ.get("SELECTOR_DYNAMIC_POS_WEIGHT", "1") == "1":
                            pos_n = (selector_pos_label * selector_valid_mask).sum()
                            neg_n = ((1.0 - selector_pos_label) * selector_valid_mask).sum()
                            try:
                                dyn_pos_weight_min = float(os.environ.get("SELECTOR_DYN_POS_WEIGHT_MIN", "3.0"))
                            except Exception:
                                dyn_pos_weight_min = 3.0
                            try:
                                dyn_pos_weight_max = float(os.environ.get("SELECTOR_DYN_POS_WEIGHT_MAX", "50.0"))
                            except Exception:
                                dyn_pos_weight_max = 50.0
                            dyn_pos_weight = (neg_n / pos_n.clamp_min(1.0)).clamp(
                                dyn_pos_weight_min,
                                dyn_pos_weight_max,
                            )
                        else:
                            dyn_pos_weight = torch.tensor(
                                float(selector_pos_weight),
                                device=selector_logits.device,
                                dtype=selector_logits.dtype,
                            )

                        selector_logits_masked = selector_logits_for_loss.masked_fill(
                            selector_valid_mask <= 0.5, 0.0
                        )
                        bce_raw = F.binary_cross_entropy_with_logits(
                            selector_logits_masked,
                            selector_pos_label,
                            reduction='none',
                            pos_weight=dyn_pos_weight,
                        )
                        val_selector_bce = (
                            bce_raw * selector_valid_mask
                        ).sum() / selector_valid_mask.sum().clamp_min(1.0)

                        try:
                            gamma = float(os.environ.get("QUALITY_TARGET_GAMMA", "2.0"))
                        except Exception:
                            gamma = 2.0

                        use_teacher_kl = os.environ.get("SELECTOR_USE_TEACHER_KL", "1") == "1"

                        if (
                            use_teacher_kl
                            and isinstance(selector_extra, dict)
                            and torch.is_tensor(selector_extra.get("teacher_dist", None))
                        ):
                            teacher_dist_t = selector_extra["teacher_dist"].to(
                                device=selector_logits_for_loss.device,
                                dtype=selector_logits_for_loss.dtype,
                            )
                            target_dist = teacher_dist_t * selector_valid_mask.float()
                            target_sum = target_dist.sum(dim=1, keepdim=True)

                            clean_pos_for_kl = selector_pos_label.float()
                            if torch.is_tensor(selector_extra.get("clean_pos_label", None)):
                                clean_pos_for_kl = selector_extra["clean_pos_label"].to(
                                    device=selector_pos_label.device,
                                    dtype=selector_pos_label.dtype,
                                )

                            fallback_dist = clean_pos_for_kl.float() * selector_valid_mask.float()
                            fallback_sum = fallback_dist.sum(dim=1, keepdim=True)
                            fallback_dist = fallback_dist / fallback_sum.clamp_min(1e-8)

                            all_pos_dist = selector_pos_label.float() * selector_valid_mask.float()
                            all_pos_sum = all_pos_dist.sum(dim=1, keepdim=True)
                            all_pos_dist = all_pos_dist / all_pos_sum.clamp_min(1e-8)

                            fallback_dist = torch.where(
                                fallback_sum > 1e-8,
                                fallback_dist,
                                all_pos_dist,
                            )

                            target_dist = torch.where(
                                target_sum > 1e-8,
                                target_dist / target_sum.clamp_min(1e-8),
                                fallback_dist,
                            )
                        else:
                            clean_pos_for_kl = selector_pos_label.float()
                            if isinstance(selector_extra, dict) and torch.is_tensor(selector_extra.get('clean_pos_label', None)):
                                clean_pos_for_kl = selector_extra['clean_pos_label'].to(
                                    device=selector_pos_label.device,
                                    dtype=selector_pos_label.dtype,
                                )

                            target_dist = selector_quality.clamp_min(0.0) ** gamma
                            target_dist = target_dist * clean_pos_for_kl.float() * selector_valid_mask.float()

                            target_sum = target_dist.sum(dim=1, keepdim=True)

                            uniform_pos = clean_pos_for_kl.float() * selector_valid_mask.float()
                            uniform_pos = uniform_pos / uniform_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)

                            uniform_all_pos = selector_pos_label.float() * selector_valid_mask.float()
                            uniform_all_pos = uniform_all_pos / uniform_all_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)

                            uniform_pos = torch.where(
                                uniform_pos.sum(dim=1, keepdim=True) > 1e-8,
                                uniform_pos,
                                uniform_all_pos,
                            )

                            target_dist = torch.where(
                                target_sum > 1e-8,
                                target_dist / target_sum.clamp_min(1e-8),
                                uniform_pos,
                            )

                        log_probs = F.log_softmax(
                            selector_logits_for_loss.masked_fill(
                                selector_valid_mask <= 0.5,
                                _neg_mask_fill_value(selector_logits_for_loss),
                            ),
                            dim=1,
                        )

                        val_selector_kl = F.kl_div(
                            log_probs,
                            target_dist,
                            reduction='none',
                        ).sum(dim=1)

                        valid_rows = (selector_valid_mask.sum(dim=1) > 0).float()
                        val_selector_kl = (
                            val_selector_kl * valid_rows
                        ).sum() / valid_rows.sum().clamp_min(1.0)

                        val_selector_pos_rate = (
                            selector_pos_label.sum() / selector_valid_mask.sum().clamp_min(1.0)
                        )
                        val_selector_quality_mean = selector_quality.mean()
                        val_selector_dyn_pos_weight = dyn_pos_weight.detach()

                        utility_t = None
                        utility_dist_t = None
                        if isinstance(selector_extra, dict):
                            if torch.is_tensor(selector_extra.get("utility", None)):
                                utility_t = selector_extra["utility"].to(
                                    device=selector_logits_for_loss.device,
                                    dtype=selector_logits_for_loss.dtype,
                                )
                            if torch.is_tensor(selector_extra.get("utility_dist", None)):
                                utility_dist_t = selector_extra["utility_dist"].to(
                                    device=selector_logits_for_loss.device,
                                    dtype=selector_logits_for_loss.dtype,
                                )

                        if torch.is_tensor(utility_t):
                            val_selector_pairwise = selector_pairwise_utility_loss(
                                selector_logits=selector_logits_for_loss,
                                utility=utility_t,
                                valid_mask=selector_valid_mask,
                                high_q=float(os.environ.get("SELECTOR_PAIRWISE_HIGH_Q", "0.80")),
                                low_q=float(os.environ.get("SELECTOR_PAIRWISE_LOW_Q", "0.40")),
                                margin=float(os.environ.get("SELECTOR_PAIRWISE_MARGIN", "0.2")),
                                max_pairs=int(os.environ.get("SELECTOR_PAIRWISE_MAX_PAIRS", "2048")),
                            )
                            if (not torch.is_tensor(val_selector_pairwise)) or (not torch.isfinite(val_selector_pairwise)):
                                val_selector_pairwise = res_full['spect'].sum() * 0.0

                        if torch.is_tensor(utility_dist_t):
                            utility_log_probs = F.log_softmax(
                                selector_logits_for_loss.masked_fill(
                                    selector_valid_mask <= 0.5,
                                    _neg_mask_fill_value(selector_logits_for_loss),
                                ),
                                dim=1,
                            )
                            val_selector_utility_kl = F.kl_div(
                                utility_log_probs,
                                utility_dist_t,
                                reduction='none',
                            ).sum(dim=1)
                            valid_rows = (selector_valid_mask.sum(dim=1) > 0).float()
                            val_selector_utility_kl = (
                                val_selector_utility_kl * valid_rows
                            ).sum() / valid_rows.sum().clamp_min(1.0)
                            if (not torch.is_tensor(val_selector_utility_kl)) or (not torch.isfinite(val_selector_utility_kl)):
                                val_selector_utility_kl = res_full['spect'].sum() * 0.0

                    val_target_pos_false_mass = res_full['spect'].sum() * 0.0
                    val_target_pos_overlap_exact = res_full['spect'].sum() * 0.0
                    val_target_pos_exact_support_mass = res_full['spect'].sum() * 0.0
                    val_target_strict_keep_rate = res_full['spect'].sum() * 0.0
                    val_target_fallback_rate = res_full['spect'].sum() * 0.0
                    val_target_clean_pos_rate = res_full['spect'].sum() * 0.0
                    val_target_pool_pos_rate = res_full['spect'].sum() * 0.0
                    val_target_pool_pos_false_mass = res_full['spect'].sum() * 0.0
                    val_target_pool_pos_overlap_tol = res_full['spect'].sum() * 0.0
                    val_target_teacher_pos_rate = res_full['spect'].sum() * 0.0
                    val_target_teacher_dist_n = res_full['spect'].sum() * 0.0
                    val_target_teacher_added_rate = res_full['spect'].sum() * 0.0
                    if (
                        isinstance(selector_extra, dict)
                        and torch.is_tensor(selector_pos_label)
                    ):
                        pos = selector_pos_label.float()
                        pos_den = pos.sum().clamp_min(1.0)
                        if torch.is_tensor(selector_extra.get('false_support_mass_exact', None)):
                            val_target_pos_false_mass = (
                                selector_extra['false_support_mass_exact'].to(
                                    device=pos.device,
                                    dtype=pos.dtype,
                                )
                                * pos
                            ).sum() / pos_den
                        if torch.is_tensor(selector_extra.get('overlap_intensity_exact', None)):
                            val_target_pos_overlap_exact = (
                                selector_extra['overlap_intensity_exact'].to(
                                    device=pos.device,
                                    dtype=pos.dtype,
                                )
                                * pos
                            ).sum() / pos_den
                        if torch.is_tensor(selector_extra.get('exact_support_mass', None)):
                            exact_support_mass_t = selector_extra['exact_support_mass'].to(
                                device=pos.device,
                                dtype=pos.dtype,
                            )
                            val_target_pos_exact_support_mass = (
                                exact_support_mass_t * pos
                            ).sum() / pos_den
                        if torch.is_tensor(selector_extra.get('strict_keep', None)):
                            strict_keep_t = selector_extra['strict_keep'].to(
                                device=pos.device,
                                dtype=pos.dtype,
                            )
                            val_target_strict_keep_rate = (
                                strict_keep_t * formulae_mask.float()
                            ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                        if torch.is_tensor(selector_extra.get('fallback_used', None)):
                            val_target_fallback_rate = selector_extra['fallback_used'].to(
                                device=pos.device,
                                dtype=pos.dtype,
                            ).float().mean()
                        if torch.is_tensor(selector_extra.get('clean_pos_label', None)):
                            val_target_clean_pos_rate = (
                                selector_extra['clean_pos_label'].to(device=selector_pos_label.device).float()
                                * formulae_mask.float()
                            ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                        if torch.is_tensor(selector_extra.get('pool_pos_label', None)):
                            val_target_pool_pos_rate = (
                                selector_extra['pool_pos_label'].to(device=selector_pos_label.device).float()
                                * formulae_mask.float()
                            ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                            pool_pos = selector_extra['pool_pos_label'].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )
                            pool_den = pool_pos.sum().clamp_min(1.0)
                            if torch.is_tensor(selector_extra.get('false_support_mass_exact', None)):
                                val_target_pool_pos_false_mass = (
                                    selector_extra['false_support_mass_exact'].to(
                                        device=pool_pos.device,
                                        dtype=pool_pos.dtype,
                                    ) * pool_pos
                                ).sum() / pool_den
                            if torch.is_tensor(selector_extra.get('overlap_intensity_tol', None)):
                                val_target_pool_pos_overlap_tol = (
                                    selector_extra['overlap_intensity_tol'].to(
                                        device=pool_pos.device,
                                        dtype=pool_pos.dtype,
                                    ) * pool_pos
                                ).sum() / pool_den
                        if torch.is_tensor(selector_extra.get("teacher_pos_label", None)):
                            teacher_pos = selector_extra["teacher_pos_label"].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )
                            val_target_teacher_pos_rate = (
                                teacher_pos * formulae_mask.float()
                            ).sum() / formulae_mask.float().sum().clamp_min(1.0)
                        if torch.is_tensor(selector_extra.get("teacher_dist", None)):
                            teacher_dist_t = selector_extra["teacher_dist"].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )
                            val_target_teacher_dist_n = (
                                teacher_dist_t > 0
                            ).float().sum(dim=1).mean()
                        if torch.is_tensor(selector_extra.get("teacher_added_label", None)):
                            teacher_added = selector_extra["teacher_added_label"].to(
                                device=selector_pos_label.device,
                                dtype=selector_pos_label.dtype,
                            )
                            val_target_teacher_added_rate = (
                                teacher_added * formulae_mask.float()
                            ).sum() / formulae_mask.float().sum().clamp_min(1.0)

                    val_selector_loss = (
                        float(selector_bce_weight) * val_selector_bce
                        + float(selector_kl_weight) * val_selector_kl
                        + float(selector_pairwise_weight) * val_selector_pairwise
                        + float(selector_utility_kl_weight) * val_selector_utility_kl
                    )
                    topk_candidate_mask_val = None
                    if os.environ.get("MODEL_TOPK_USE_ACTIVE_MASK", "0") == "1":
                        topk_candidate_mask_val = _get_active_candidate_mask_from_batch(
                            batch,
                            formulae_mask=batch.get('formulae_mask', None),
                        )

                    use_coverage_topk = os.environ.get("USE_COVERAGE_AWARE_TOPK", "0") == "1"
                    model_topk_idx = select_model_topk_indices(
                        selector_logits=selector_logits_for_topk,
                        batch=batch,
                        k=model_topk_eval,
                        use_coverage=use_coverage_topk,
                        use_group_unique=use_group_unique_model,
                        candidate_mask=topk_candidate_mask_val,
                    )
                    model_topk_mask = build_mask_from_topk_indices(
                        model_topk_idx,
                        selector_logits_for_topk,
                        formulae_mask=batch.get('formulae_mask', None),
                        candidate_mask=topk_candidate_mask_val,
                    )

                    model_topk_teacher_recall = mask_recall(model_topk_mask, teacher_topk_mask)

                    k_list = [32, 64, 128, 256]
                    selector_masks = {}
                    quality_metrics = {}
                    if torch.is_tensor(selector_quality) and torch.is_tensor(formulae_mask):
                        quality_metrics = compute_selector_quality_metrics(
                            selector_logits_for_topk,
                            selector_quality,
                            formulae_mask,
                            ks=k_list,
                        )
                    if step == 1:
                        if torch.is_tensor(topk_candidate_mask_val):
                            print(
                                "[ACTIVE_DEBUG]",
                                "active_mean=",
                                float(topk_candidate_mask_val.float().mean().detach().cpu().item()),
                                "active_n=",
                                float(topk_candidate_mask_val.float().sum(dim=1).mean().detach().cpu().item()),
                                flush=True,
                            )
                        else:
                            print("[ACTIVE_DEBUG] active_mask=None", flush=True)

                    selector_eval = compute_selector_eval_pack(
                        selector_logits=selector_logits_for_topk,
                        batch=batch,
                        formulae_mask=formulae_mask,
                        teacher_mask=teacher_topk_mask,
                        active_mask=topk_candidate_mask_val,
                        topk_list=tuple(k_list),
                        use_group_unique=use_group_unique_model,
                        use_coverage=use_coverage_topk,
                    )
                    for k in k_list:
                        topk_idx = select_model_topk_indices(
                            selector_logits=selector_logits_for_topk,
                            batch=batch,
                            k=k,
                            use_coverage=use_coverage_topk,
                            use_group_unique=use_group_unique_model,
                            candidate_mask=topk_candidate_mask_val,
                        )
                        selector_masks[k] = build_mask_from_topk_indices(
                            topk_idx,
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            candidate_mask=topk_candidate_mask_val,
                        )

                    fragaux_model_topk_ratio_32 = float("nan")
                    fragaux_model_topk_ratio_64 = float("nan")
                    fragaux_model_topk_ratio_128 = float("nan")
                    fragaux_model_topk_ratio_256 = float("nan")

                    if torch.is_tensor(frag_source_mask):
                        fragaux_model_topk_ratio_32 = mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(32, None),
                        )
                        fragaux_model_topk_ratio_64 = mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(64, None),
                        )
                        fragaux_model_topk_ratio_128 = mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(128, None),
                        )
                        fragaux_model_topk_ratio_256 = mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(256, None),
                        )

                    teacher_stats = {}
                    if torch.is_tensor(teacher_target_full):
                        teacher_stats = compute_candidate_support_stats(
                            batch,
                            teacher_target_full,
                            official_bin_width=official_metric_cfg['bin_width'],
                            official_max_mz=official_metric_cfg['max_mz'],
                        )

                    for k in k_list:
                        mask_k = selector_masks.get(k, None)
                        rec_k = selector_eval.get(
                            f'selector_recall@{k}',
                            mask_recall(mask_k, teacher_topk_mask),
                        )
                        prec_k = quality_metrics.get(
                            f'selector_precision_at_{k}',
                            selector_eval.get(
                                f'selector_precision@{k}',
                                mask_precision(mask_k, teacher_topk_mask),
                            ),
                        )
                        q_mean_k = quality_metrics.get(
                            f'selector_quality_mean_at_{k}',
                            float('nan'),
                        )
                        if torch.is_tensor(prec_k):
                            prec_k = float(prec_k.detach().cpu().item())
                        if torch.is_tensor(q_mean_k):
                            q_mean_k = float(q_mean_k.detach().cpu().item())
                        stats_k = compute_candidate_support_stats(
                            batch,
                            mask_k,
                            official_bin_width=official_metric_cfg['bin_width'],
                            official_max_mz=official_metric_cfg['max_mz'],
                        )
                        selected_true_k = selector_eval.get(
                            f'selected_true_hit_mass@{k}',
                            stats_k.get('pred_int_on_true', float('nan')) if stats_k else float('nan'),
                        )
                        selected_false_k = selector_eval.get(
                            f'selected_false_mass@{k}',
                            stats_k.get('false_support', float('nan')) if stats_k else float('nan'),
                        )

                        if k == 32:
                            val_selector_recall_32_vals.append(rec_k)
                            val_selector_precision_32_vals.append(prec_k)
                            val_selector_quality_mean_32_vals.append(q_mean_k)
                            val_selected_true_hit_mass_32_vals.append(selected_true_k)
                            val_selected_false_mass_32_vals.append(selected_false_k)
                        elif k == 64:
                            val_selector_recall_64_vals.append(rec_k)
                            val_selector_precision_64_vals.append(prec_k)
                            val_selector_quality_mean_64_vals.append(q_mean_k)
                            val_selected_true_hit_mass_64_vals.append(selected_true_k)
                            val_selected_false_mass_64_vals.append(selected_false_k)
                        elif k == 128:
                            val_selector_recall_128_vals.append(rec_k)
                            val_selector_precision_128_vals.append(prec_k)
                            val_selector_quality_mean_128_vals.append(q_mean_k)
                            val_selected_true_hit_mass_128_vals.append(selected_true_k)
                            val_selected_false_mass_128_vals.append(selected_false_k)
                        elif k == 256:
                            val_selector_recall_256_vals.append(rec_k)
                            val_selector_precision_256_vals.append(prec_k)
                            val_selector_quality_mean_256_vals.append(q_mean_k)
                            val_selected_true_hit_mass_256_vals.append(selected_true_k)
                            val_selected_false_mass_256_vals.append(selected_false_k)

                    if teacher_stats:
                        val_teacher_oracle_cos_vals.append(teacher_stats.get('official_cos', float('nan')))
                        val_teacher_oracle_false_support_vals.append(teacher_stats.get('false_support', float('nan')))
                        val_teacher_oracle_pred_int_on_true_vals.append(teacher_stats.get('pred_int_on_true', float('nan')))
                        val_teacher_oracle_pred_n_vals.append(teacher_stats.get('pred_n', float('nan')))

                    utility_top64_stats = {}
                    if isinstance(selector_extra, dict) and torch.is_tensor(selector_extra.get("utility", None)):
                        utility_scores = selector_extra["utility"].to(
                            device=selector_logits_for_topk.device,
                            dtype=selector_logits_for_topk.dtype,
                        )
                        utility_topk_idx = select_model_topk_indices(
                            selector_logits=utility_scores,
                            batch=batch,
                            k=64,
                            use_coverage=False,
                            use_group_unique=use_group_unique_model,
                            candidate_mask=None,
                        )
                        utility_topk_mask = build_mask_from_topk_indices(
                            utility_topk_idx,
                            utility_scores,
                            formulae_mask=batch.get('formulae_mask', None),
                            candidate_mask=None,
                        )
                        utility_top64_stats = compute_candidate_support_stats(
                            batch,
                            utility_topk_mask,
                            official_bin_width=official_metric_cfg['bin_width'],
                            official_max_mz=official_metric_cfg['max_mz'],
                        )
                    if utility_top64_stats:
                        val_utility_top64_oracle_cos_vals.append(utility_top64_stats.get('official_cos', float('nan')))
                        val_utility_top64_false_support_vals.append(utility_top64_stats.get('false_support', float('nan')))
                        val_utility_top64_true_hit_mass_vals.append(utility_top64_stats.get('pred_int_on_true', float('nan')))
                        val_utility_top64_false_mass_vals.append(utility_top64_stats.get('false_support', float('nan')))

                    model_topk_stats_256 = compute_candidate_support_stats(
                        batch,
                        selector_masks.get(256, model_topk_mask),
                        official_bin_width=official_metric_cfg['bin_width'],
                        official_max_mz=official_metric_cfg['max_mz'],
                    )
                    if model_topk_stats_256:
                        val_model_topk_oracle_cos_256_vals.append(model_topk_stats_256.get('official_cos', float('nan')))
                        val_model_topk_oracle_false_support_256_vals.append(model_topk_stats_256.get('false_support', float('nan')))

                    model_topk_stats_eval = compute_candidate_support_stats(
                        batch,
                        model_topk_mask,
                        official_bin_width=official_metric_cfg['bin_width'],
                        official_max_mz=official_metric_cfg['max_mz'],
                    )

                    if model_topk_stats_eval:
                        # Reuse the existing 256 lists only if model_topk_eval == 256.
                        # Otherwise this is printed directly for debugging.
                        if not hasattr(train_mssubsetnet, "_model_topk_eval_debug_vals"):
                            train_mssubsetnet._model_topk_eval_debug_vals = []
                        train_mssubsetnet._model_topk_eval_debug_vals.append((
                            float(model_topk_eval),
                            float(model_topk_stats_eval.get('official_cos', float('nan'))),
                            float(model_topk_stats_eval.get('false_support', float('nan'))),
                        ))

                    if epoch == 0 and step == 1:
                        teacher_pos_n = float("nan")
                        teacher_prob_n = float("nan")

                        if torch.is_tensor(teacher_topk_mask):
                            teacher_pos_n = float(
                                teacher_topk_mask.float().sum(dim=-1).mean().detach().cpu().item()
                            )

                        if torch.is_tensor(teacher_topk_probs):
                            teacher_prob_n = float(
                                (teacher_topk_probs > 1e-12).float().sum(dim=-1).mean().detach().cpu().item()
                            )

                        model_topk_n = float("nan")
                        teacher_topk_n = float("nan")
                        if torch.is_tensor(model_topk_mask):
                            model_topk_n = float(model_topk_mask.float().sum(dim=-1).mean().detach().cpu().item())
                        if torch.is_tensor(teacher_topk_mask):
                            teacher_topk_n = float(teacher_topk_mask.float().sum(dim=-1).mean().detach().cpu().item())

                        print(
                            "[TOPK_DEBUG]",
                            "model_topk_teacher_recall=", model_topk_teacher_recall,
                            "teacher_pos_n=", teacher_pos_n,
                            "teacher_prob_n=", teacher_prob_n,
                            "teacher_topk_n=", teacher_topk_n,
                            "model_topk_n=", model_topk_n,
                            "use_group_unique_teacher=", str(int(bool(use_group_unique_teacher))),
                            "use_group_unique_model=", str(int(bool(use_group_unique_model))),
                            "use_group_unique_prune=", str(int(bool(use_group_unique_prune))),
                            flush=True,
                        )
                    # ========= ===================================================
                    # PASS 2: model-top64 reranker
                    # ============================================================
                    batch_with_teacher = dict(batch)
                    if torch.is_tensor(teacher_target_full):
                        batch_with_teacher['teacher_formula_probs'] = teacher_target_full

                    batch_rerank = _prune_batch_by_candidate_mask(
                        batch_with_teacher,
                        model_topk_mask,
                        keep_topk=model_topk_eval,
                        fill_scores=selector_logits_for_topk.detach(),
                        group_id=batch.get("formulae_instance_group_id", None),
                        group_unique=(use_group_unique_prune),
                    )
                    if torch.is_tensor(batch_rerank.get('teacher_formula_probs', None)):
                        batch_rerank['teacher_formula_probs'] = _renormalize_target_probs(
                            batch_rerank.get('teacher_formula_probs', None),
                            batch_rerank.get('formulae_mask', None),
                        )

                    res = model(**batch_rerank)
                    pred_spect_coarse = res['spect'] if isinstance(res, dict) else res
                    pred_spect_official = res.get('spect_out_official', None) if isinstance(res, dict) else None

                    formulae_scores = _get_reranker_scores_from_res(res)
                    val_formula_entropy = _masked_formula_entropy_loss(
                        formulae_scores,
                        batch_rerank.get('formulae_mask', None),
                    )
                    if not torch.is_tensor(val_formula_entropy):
                        val_formula_entropy = pred_spect_coarse.sum() * 0.0

                    if torch.is_tensor(pred_spect_official) and torch.is_tensor(true_official_dense):
                        val_official_spectral = compute_official_dense_spectral_loss(
                            pred_spect_official,
                            true_official_dense,
                            kl_weight=official_spectral_kl_weight,
                        )
                        val_false_support = _false_support_mass_loss_dense(
                            pred_spect_official,
                            true_official_dense,
                        )
                        if not torch.is_tensor(val_false_support):
                            val_false_support = pred_spect_coarse.sum() * 0.0
                    else:
                        val_official_spectral = pred_spect_coarse.sum() * 0.0
                        val_false_support = pred_spect_coarse.sum() * 0.0


                rerank_logits_pool = res.get('rerank_logits_pool', None)
                pool_idx = res.get('selector_pool_idx', None)
                val_rerank_kl = pred_spect_coarse.sum() * 0.0
                val_rerank_bce = pred_spect_coarse.sum() * 0.0
                val_rerank_loss = pred_spect_coarse.sum() * 0.0

                if torch.is_tensor(rerank_logits_pool) and torch.is_tensor(pool_idx):
                    formulae_mask_rerank = batch_rerank.get('formulae_mask', None)
                    if torch.is_tensor(formulae_mask_rerank):
                        formulae_mask_rerank = formulae_mask_rerank.float()
                        if formulae_mask_rerank.dim() > 2:
                            formulae_mask_rerank = formulae_mask_rerank.reshape(
                                formulae_mask_rerank.shape[0], -1
                            )
                    else:
                        formulae_mask_rerank = torch.ones(
                            (int(rerank_logits_pool.shape[0]), int(rerank_logits_pool.shape[1])),
                            dtype=torch.float32,
                            device=rerank_logits_pool.device,
                        )

                    selector_quality_rerank, _, _, _ = build_candidate_local_quality_target(
                        batch=batch_rerank,
                        formulae_mask=formulae_mask_rerank,
                        official_bin_n=official_bin_n,
                    )

                    rerank_target_dist, rerank_pos, rerank_pool_mask = (
                        build_setcover_rerank_target_from_quality(
                            selector_quality=selector_quality_rerank.detach(),
                            pool_idx=pool_idx,
                            formulae_mask=formulae_mask_rerank,
                        )
                    )

                    logp_pool = F.log_softmax(
                        rerank_logits_pool.masked_fill(
                            rerank_pool_mask <= 0.5,
                            _neg_mask_fill_value(rerank_logits_pool),
                        ),
                        dim=1,
                    )

                    val_rerank_kl = F.kl_div(
                        logp_pool,
                        rerank_target_dist,
                        reduction='none',
                    ).sum(dim=1)

                    valid_rows = (rerank_pool_mask.sum(dim=1) > 0).float()
                    val_rerank_kl = (
                        val_rerank_kl * valid_rows
                    ).sum() / valid_rows.sum().clamp_min(1.0)

                    rerank_bce_raw = F.binary_cross_entropy_with_logits(
                        rerank_logits_pool,
                        rerank_pos,
                        reduction='none',
                    )
                    val_rerank_bce = (
                        rerank_bce_raw * rerank_pool_mask
                    ).sum() / rerank_pool_mask.sum().clamp_min(1.0)

                    val_rerank_loss = float(rerank_kl_weight) * val_rerank_kl + float(rerank_bce_weight) * val_rerank_bce

                val_main_kl = val_rerank_loss

                val_peak_aux = _compute_peak_aux_loss_from_batch(
                    batch_rerank,
                    res if isinstance(res, dict) else {},
                    official_bin_width=official_metric_cfg['bin_width'],
                    official_max_mz=official_metric_cfg['max_mz'],
                )
                if (not torch.is_tensor(val_peak_aux)) or (not torch.isfinite(val_peak_aux)):
                    val_peak_aux = pred_spect_coarse.sum() * 0.0

                val_oos = _compute_oos_loss_from_batch(
                    batch_rerank,
                    res if isinstance(res, dict) else {},
                    official_bin_width=official_metric_cfg['bin_width'],
                    official_max_mz=official_metric_cfg['max_mz'],
                )
                if not torch.is_tensor(val_oos):
                    val_oos = pred_spect_coarse.sum() * 0.0

                val_loss = float(selector_loss_weight) * val_selector_loss

                if epoch >= precursor_loss_start_epoch and precursor_loss_weight > 0:
                    val_loss = val_loss + float(precursor_loss_weight) * val_precursor_loss

                if os.environ.get("TRAIN_FRAGMENT_NODE_SELECTOR", "0") == "1":
                    val_loss = val_loss + float(os.environ.get("FN_LOSS_WEIGHT", "1.0")) * val_fn_loss

                if epoch >= selector_only_warmup_epochs:
                    val_loss = val_loss + float(main_candidate_kl_weight) * val_main_kl

                if epoch >= selector_only_warmup_epochs and formula_entropy_loss_weight > 0:
                    val_loss = val_loss + float(formula_entropy_loss_weight) * val_formula_entropy
                if epoch >= spectral_loss_start_epoch and false_support_loss_weight > 0:
                    val_loss = val_loss + float(false_support_loss_weight) * val_false_support
                if epoch >= spectral_loss_start_epoch and official_spectral_loss_weight > 0:
                    val_loss = val_loss + float(official_spectral_loss_weight) * val_official_spectral

                if epoch >= peak_aux_start_epoch and peak_aux_loss_weight > 0:
                    val_loss = val_loss + float(peak_aux_loss_weight) * val_peak_aux

                if epoch >= oos_loss_start_epoch and oos_loss_weight > 0:
                    val_loss = val_loss + float(oos_loss_weight) * val_oos

                pred_spect_for_metric = pred_spect_official if torch.is_tensor(pred_spect_official) else pred_spect_coarse

                metrics = compute_batch_official_metrics(
                    raw_batch,
                    pred_spect_for_metric,
                    spect_bin.get_bin_centers().astype(np.float32),
                    official_metric_cfg,
                    pred_exact_peaks=None,
                )
                official_cos_vals.extend(metrics['official_cos_no_precursor'])
                official_js_vals.extend(metrics['official_js_no_precursor'])
                official_recall_vals.extend(metrics['topk_peak_recall'])
                official_cov_vals.extend(metrics['matched_intensity_coverage'])
                official_pred_n_vals.extend(metrics.get('pred_official_n', []))
                official_true_n_vals.extend(metrics.get('true_official_n', []))
                official_overlap_n_vals.extend(metrics.get('overlap_n', []))
                official_false_pred_n_vals.extend(metrics.get('false_pred_n', []))
                official_pred_int_on_true_vals.extend(metrics.get('pred_intensity_on_true_ratio', []))
                val_losses.append(float(val_loss.detach().item()))
                val_main_kl_vals.append(float(val_main_kl.detach().item()))
                val_official_spectral_vals.append(float(val_official_spectral.detach().item()))
                val_peak_aux_vals.append(float(val_peak_aux.detach().item()))
                val_oos_vals.append(float(val_oos.detach().item()))
                val_formula_entropy_vals.append(float(val_formula_entropy.detach().item()))
                val_precursor_loss_vals.append(float(val_precursor_loss.detach().item()))
                val_fn_loss_vals.append(float(val_fn_loss.detach().item()))
                val_selector_loss_vals.append(float(val_selector_loss.detach().item()))
                val_selector_bce_vals.append(float(val_selector_bce.detach().item()))
                val_selector_kl_vals.append(float(val_selector_kl.detach().item()))
                val_selector_quality_mean_vals.append(float(val_selector_quality_mean.detach().item()))
                val_selector_pos_rate_vals.append(float(val_selector_pos_rate.detach().item()))
                val_target_pos_false_mass_vals.append(float(val_target_pos_false_mass.detach().item()))
                val_target_pos_overlap_exact_vals.append(float(val_target_pos_overlap_exact.detach().item()))
                val_target_pos_exact_support_mass_vals.append(
                    float(val_target_pos_exact_support_mass.detach().item())
                )
                val_target_strict_keep_rate_vals.append(
                    float(val_target_strict_keep_rate.detach().item())
                )
                val_target_fallback_rate_vals.append(
                    float(val_target_fallback_rate.detach().item())
                )
                val_target_clean_pos_rate_vals.append(
                    float(val_target_clean_pos_rate.detach().item())
                )
                val_target_pool_pos_rate_vals.append(
                    float(val_target_pool_pos_rate.detach().item())
                )
                val_target_pool_pos_false_mass_vals.append(
                    float(val_target_pool_pos_false_mass.detach().item())
                )
                val_target_pool_pos_overlap_tol_vals.append(
                    float(val_target_pool_pos_overlap_tol.detach().item())
                )
                val_target_teacher_pos_rate_vals.append(
                    float(val_target_teacher_pos_rate.detach().item())
                )
                val_target_teacher_dist_n_vals.append(
                    float(val_target_teacher_dist_n.detach().item())
                )
                val_target_teacher_added_rate_vals.append(
                    float(val_target_teacher_added_rate.detach().item())
                )
                val_selector_dyn_pos_weight_vals.append(
                    float(val_selector_dyn_pos_weight.detach().item())
                )
                use_rerank_delta_val = 0.0
                if isinstance(res_full, dict):
                    v = res_full.get('use_rerank_delta', 0.0)
                    if torch.is_tensor(v):
                        try:
                            use_rerank_delta_val = float(v.detach().reshape(-1)[0].item())
                        except Exception:
                            use_rerank_delta_val = 0.0
                    else:
                        try:
                            use_rerank_delta_val = float(v)
                        except Exception:
                            use_rerank_delta_val = 0.0

                val_use_rerank_delta_vals.append(use_rerank_delta_val)
                val_rerank_kl_vals.append(float(val_rerank_kl.detach().item()))
                val_rerank_bce_vals.append(float(val_rerank_bce.detach().item()))
                val_rerank_loss_vals.append(float(val_main_kl.detach().item()))
                val_model_topk_teacher_recall_vals.append(float(model_topk_teacher_recall))
                val_active_teacher_recall_vals.append(float(active_teacher_recall))
                val_fragaux_teacher_recall_vals.append(float(fragaux_teacher_recall))
                val_fragaux_model_topk_ratio_32_vals.append(float(fragaux_model_topk_ratio_32))
                val_fragaux_model_topk_ratio_64_vals.append(float(fragaux_model_topk_ratio_64))
                val_fragaux_model_topk_ratio_128_vals.append(float(fragaux_model_topk_ratio_128))
                val_fragaux_model_topk_ratio_256_vals.append(float(fragaux_model_topk_ratio_256))
                val_false_support_vals.append(float(val_false_support.detach().item()))
                batch_cos = metrics['official_cos_no_precursor']
                batch_pred_sparse = metrics['retrieval_pred_sparse']
                batch_true_sparse = metrics['retrieval_true_sparse']
                batch_pred_n = metrics['pred_official_n']
                batch_overlap_n = metrics['overlap_n']
                for bi, cos_i in enumerate(batch_cos):
                    if not np.isfinite(cos_i):
                        continue

                    pred_n_i = int(batch_pred_n[bi]) if np.isfinite(batch_pred_n[bi]) else 0
                    overlap_n_i = int(batch_overlap_n[bi]) if np.isfinite(batch_overlap_n[bi]) else 0

                    cur_sparse = (batch_pred_sparse[bi], batch_true_sparse[bi])

                    # best overall
                    if cos_i > epoch_best_cos:
                        epoch_best_cos = float(cos_i)
                        epoch_best_sparse = cur_sparse

                    # worst overall (can be empty prediction)
                    if cos_i < epoch_worst_cos:
                        epoch_worst_cos = float(cos_i)
                        epoch_worst_sparse = cur_sparse

                    # worst but still has some predicted official peaks
                    if pred_n_i > 0 and cos_i < epoch_worst_nonempty_cos:
                        epoch_worst_nonempty_cos = float(cos_i)
                        epoch_worst_nonempty_sparse = cur_sparse

                    # worst but still has overlap
                    if overlap_n_i > 0 and cos_i < epoch_worst_overlap_cos:
                        epoch_worst_overlap_cos = float(cos_i)
                        epoch_worst_overlap_sparse = cur_sparse
        avg_train_loss = _finite_mean(train_losses)
        avg_train_selector_loss = _finite_mean(train_selector_loss_vals)
        avg_train_selector_bce = _finite_mean(train_selector_bce_vals)
        avg_train_selector_kl = _finite_mean(train_selector_kl_vals)
        avg_train_selector_quality_mean = _finite_mean(train_selector_quality_mean_vals)
        avg_train_selector_pos_rate = _finite_mean(train_selector_pos_rate_vals)
        avg_train_target_pos_false_mass = _finite_mean(train_target_pos_false_mass_vals)
        avg_train_target_pos_overlap_exact = _finite_mean(train_target_pos_overlap_exact_vals)
        avg_train_target_pos_exact_support_mass = _finite_mean(train_target_pos_exact_support_mass_vals)
        avg_train_target_strict_keep_rate = _finite_mean(train_target_strict_keep_rate_vals)
        avg_train_target_fallback_rate = _finite_mean(train_target_fallback_rate_vals)
        avg_train_target_clean_pos_rate = _finite_mean(train_target_clean_pos_rate_vals)
        avg_train_target_pool_pos_rate = _finite_mean(train_target_pool_pos_rate_vals)
        avg_train_target_pool_pos_false_mass = _finite_mean(train_target_pool_pos_false_mass_vals)
        avg_train_target_pool_pos_overlap_tol = _finite_mean(train_target_pool_pos_overlap_tol_vals)
        avg_train_target_teacher_pos_rate = _finite_mean(train_target_teacher_pos_rate_vals)
        avg_train_target_teacher_dist_n = _finite_mean(train_target_teacher_dist_n_vals)
        avg_train_target_teacher_added_rate = _finite_mean(train_target_teacher_added_rate_vals)
        avg_train_selector_dyn_pos_weight = _finite_mean(train_selector_dyn_pos_weight_vals)
        avg_train_use_rerank_delta = _finite_mean(train_use_rerank_delta_vals)
        avg_val_selector_loss = _finite_mean(val_selector_loss_vals)
        avg_val_selector_bce = _finite_mean(val_selector_bce_vals)
        avg_val_selector_kl = _finite_mean(val_selector_kl_vals)
        avg_val_selector_quality_mean = _finite_mean(val_selector_quality_mean_vals)
        avg_val_selector_pos_rate = _finite_mean(val_selector_pos_rate_vals)
        avg_val_target_pos_false_mass = _finite_mean(val_target_pos_false_mass_vals)
        avg_val_target_pos_overlap_exact = _finite_mean(val_target_pos_overlap_exact_vals)
        avg_val_target_pos_exact_support_mass = _finite_mean(val_target_pos_exact_support_mass_vals)
        avg_val_target_strict_keep_rate = _finite_mean(val_target_strict_keep_rate_vals)
        avg_val_target_fallback_rate = _finite_mean(val_target_fallback_rate_vals)
        avg_val_target_clean_pos_rate = _finite_mean(val_target_clean_pos_rate_vals)
        avg_val_target_pool_pos_rate = _finite_mean(val_target_pool_pos_rate_vals)
        avg_val_target_pool_pos_false_mass = _finite_mean(val_target_pool_pos_false_mass_vals)
        avg_val_target_pool_pos_overlap_tol = _finite_mean(val_target_pool_pos_overlap_tol_vals)
        avg_val_target_teacher_pos_rate = _finite_mean(val_target_teacher_pos_rate_vals)
        avg_val_target_teacher_dist_n = _finite_mean(val_target_teacher_dist_n_vals)
        avg_val_target_teacher_added_rate = _finite_mean(val_target_teacher_added_rate_vals)
        avg_val_selector_dyn_pos_weight = _finite_mean(val_selector_dyn_pos_weight_vals)
        avg_val_use_rerank_delta = _finite_mean(val_use_rerank_delta_vals)
        avg_val_model_topk_teacher_recall = _finite_mean(val_model_topk_teacher_recall_vals)
        avg_val_active_teacher_recall = _finite_mean(val_active_teacher_recall_vals)
        avg_val_fragaux_teacher_recall = _finite_mean(val_fragaux_teacher_recall_vals)
        avg_val_fragaux_model_topk_ratio_32 = _finite_mean(val_fragaux_model_topk_ratio_32_vals)
        avg_val_fragaux_model_topk_ratio_64 = _finite_mean(val_fragaux_model_topk_ratio_64_vals)
        avg_val_fragaux_model_topk_ratio_128 = _finite_mean(val_fragaux_model_topk_ratio_128_vals)
        avg_val_fragaux_model_topk_ratio_256 = _finite_mean(val_fragaux_model_topk_ratio_256_vals)
        avg_train_rerank_teacher_ratio = _finite_mean(train_rerank_teacher_ratio_vals)
        avg_train_rerank_kl = _finite_mean(train_rerank_kl_vals)
        avg_train_rerank_bce = _finite_mean(train_rerank_bce_vals)
        avg_train_rerank_loss = _finite_mean(train_rerank_loss_vals)
        avg_train_main_kl = _finite_mean(train_main_kl_vals)
        avg_train_official_spectral = _finite_mean(train_official_spectral_vals)
        avg_train_peak_aux = _finite_mean(train_peak_aux_vals)
        avg_train_oos = _finite_mean(train_oos_vals)
        avg_train_formula_entropy = _finite_mean(train_formula_entropy_vals)
        avg_train_false_support = _finite_mean(train_false_support_vals)
        avg_train_precursor_loss = _finite_mean(train_precursor_loss_vals)
        avg_train_fn_loss = _finite_mean(train_fn_loss_vals)
        avg_val_loss = _finite_mean(val_losses)
        avg_val_formula_entropy = _finite_mean(val_formula_entropy_vals)
        avg_val_main_kl = _finite_mean(val_main_kl_vals)
        avg_val_rerank_kl = _finite_mean(val_rerank_kl_vals)
        avg_val_rerank_bce = _finite_mean(val_rerank_bce_vals)
        avg_val_rerank_loss = _finite_mean(val_rerank_loss_vals)
        avg_val_official_spectral = _finite_mean(val_official_spectral_vals)
        avg_val_peak_aux = _finite_mean(val_peak_aux_vals)
        avg_val_oos = _finite_mean(val_oos_vals)
        avg_val_precursor_loss = _finite_mean(val_precursor_loss_vals)
        avg_val_fn_loss = _finite_mean(val_fn_loss_vals)
        avg_val_cos = _finite_mean(official_cos_vals)
        avg_val_js = _finite_mean(official_js_vals)
        avg_val_recall = _finite_mean(official_recall_vals)
        avg_val_cov = _finite_mean(official_cov_vals)
        avg_val_pred_n = _finite_mean(official_pred_n_vals)
        avg_val_true_n = _finite_mean(official_true_n_vals)
        avg_val_overlap_n = _finite_mean(official_overlap_n_vals)
        avg_val_false_pred_n = _finite_mean(official_false_pred_n_vals)
        avg_val_pred_int_on_true = _finite_mean(official_pred_int_on_true_vals)
        avg_val_false_support = _finite_mean(val_false_support_vals)
        avg_val_selector_recall_32 = _finite_mean(val_selector_recall_32_vals)
        avg_val_selector_recall_64 = _finite_mean(val_selector_recall_64_vals)
        avg_val_selector_recall_128 = _finite_mean(val_selector_recall_128_vals)
        avg_val_selector_recall_256 = _finite_mean(val_selector_recall_256_vals)
        avg_val_selector_precision_32 = _finite_mean(val_selector_precision_32_vals)
        avg_val_selector_precision_64 = _finite_mean(val_selector_precision_64_vals)
        avg_val_selector_precision_128 = _finite_mean(val_selector_precision_128_vals)
        avg_val_selector_precision_256 = _finite_mean(val_selector_precision_256_vals)
        avg_val_selector_quality_mean_32 = _finite_mean(val_selector_quality_mean_32_vals)
        avg_val_selector_quality_mean_64 = _finite_mean(val_selector_quality_mean_64_vals)
        avg_val_selector_quality_mean_128 = _finite_mean(val_selector_quality_mean_128_vals)
        avg_val_selector_quality_mean_256 = _finite_mean(val_selector_quality_mean_256_vals)
        avg_val_selected_true_hit_mass_32 = _finite_mean(val_selected_true_hit_mass_32_vals)
        avg_val_selected_true_hit_mass_64 = _finite_mean(val_selected_true_hit_mass_64_vals)
        avg_val_selected_true_hit_mass_128 = _finite_mean(val_selected_true_hit_mass_128_vals)
        avg_val_selected_true_hit_mass_256 = _finite_mean(val_selected_true_hit_mass_256_vals)
        avg_val_selected_false_mass_32 = _finite_mean(val_selected_false_mass_32_vals)
        avg_val_selected_false_mass_64 = _finite_mean(val_selected_false_mass_64_vals)
        avg_val_selected_false_mass_128 = _finite_mean(val_selected_false_mass_128_vals)
        avg_val_selected_false_mass_256 = _finite_mean(val_selected_false_mass_256_vals)
        avg_val_teacher_oracle_cos = _finite_mean(val_teacher_oracle_cos_vals)
        avg_val_teacher_oracle_false_support = _finite_mean(val_teacher_oracle_false_support_vals)
        avg_val_teacher_oracle_pred_int_on_true = _finite_mean(val_teacher_oracle_pred_int_on_true_vals)
        avg_val_teacher_oracle_pred_n = _finite_mean(val_teacher_oracle_pred_n_vals)
        avg_val_model_topk_oracle_cos_256 = _finite_mean(val_model_topk_oracle_cos_256_vals)
        avg_val_model_topk_oracle_false_support_256 = _finite_mean(val_model_topk_oracle_false_support_256_vals)
        avg_val_utility_top64_oracle_cos = _finite_mean(val_utility_top64_oracle_cos_vals)
        avg_val_utility_top64_false_support = _finite_mean(val_utility_top64_false_support_vals)
        avg_val_utility_top64_true_hit_mass = _finite_mean(val_utility_top64_true_hit_mass_vals)
        avg_val_utility_top64_false_mass = _finite_mean(val_utility_top64_false_mass_vals)
        eval_debug_vals = getattr(train_mssubsetnet, "_model_topk_eval_debug_vals", [])
        if len(eval_debug_vals) > 0:
            arr = np.asarray(eval_debug_vals, dtype=np.float64)
            eval_k_used = int(arr[-1, 0])
            avg_val_model_topk_oracle_cos_eval = float(np.nanmean(arr[:, 1]))
            avg_val_model_topk_oracle_false_support_eval = float(np.nanmean(arr[:, 2]))
            train_mssubsetnet._model_topk_eval_debug_vals = []
        else:
            eval_k_used = int(model_topk_eval)
            avg_val_model_topk_oracle_cos_eval = float("nan")
            avg_val_model_topk_oracle_false_support_eval = float("nan")
        avg_val_overlap_ratio = (
            float(avg_val_overlap_n / max(avg_val_pred_n, 1e-8))
            if np.isfinite(avg_val_pred_n) and avg_val_pred_n > 0
            else float("nan")
        )


        if model_select_metric_name in ("selector_recall", "topk_recall", "selector"):
            model_select_metric = (
                avg_val_model_topk_teacher_recall
                if np.isfinite(avg_val_model_topk_teacher_recall)
                else -1e9
            )

        elif model_select_metric_name in ("val_loss", "loss"):
            # 娉ㄦ剰锛氬灞?is_best 鐢ㄧ殑鏄秺澶ц秺濂斤紝鎵€浠?loss 瑕佸彇璐熸暟
            model_select_metric = (
                -avg_val_loss
                if np.isfinite(avg_val_loss)
                else -1e9
            )

        elif model_select_metric_name in ("selector_loss",):
            model_select_metric = (
                -avg_val_selector_loss
                if np.isfinite(avg_val_selector_loss)
                else -1e9
            )

        elif model_select_metric_name in (
            "selector_precision_at_256",
            "selector_precision_256",
            "selector_precision",
        ):
            model_select_metric = (
                avg_val_selector_precision_256
                if np.isfinite(avg_val_selector_precision_256)
                else -1e9
            )

        elif model_select_metric_name in ("matched_cov", "coverage"):
            model_select_metric = avg_val_cov if np.isfinite(avg_val_cov) else -1e9

        elif model_select_metric_name in (
            "model_topk_oracle_cos_256",
            "topk_oracle_cos_256",
            "oracle_topk_cos",
        ):
            model_select_metric = (
                avg_val_model_topk_oracle_cos_256
                if np.isfinite(avg_val_model_topk_oracle_cos_256)
                else -1e9
            )

        elif model_select_metric_name in (
            "selected_true_mass_32",
            "true_mass_32",
            "selected_true32",
        ):
            model_select_metric = (
                avg_val_selected_true_hit_mass_32
                if np.isfinite(avg_val_selected_true_hit_mass_32)
                else -1e9
            )

        elif model_select_metric_name in (
            "selected_true_mass_64",
            "true_mass_64",
            "selected_true64",
        ):
            model_select_metric = (
                avg_val_selected_true_hit_mass_64
                if np.isfinite(avg_val_selected_true_hit_mass_64)
                else -1e9
            )

        elif model_select_metric_name in (
            "selected_false_mass_32",
            "false_mass_32",
            "selected_false32",
        ):
            model_select_metric = (
                -avg_val_selected_false_mass_32
                if np.isfinite(avg_val_selected_false_mass_32)
                else -1e9
            )

        elif model_select_metric_name in (
            "selected_false_mass_64",
            "false_mass_64",
            "selected_false64",
        ):
            model_select_metric = (
                -avg_val_selected_false_mass_64
                if np.isfinite(avg_val_selected_false_mass_64)
                else -1e9
            )

        elif model_select_metric_name in ("official_cos", "cos", "official"):
            model_select_metric = avg_val_cos if np.isfinite(avg_val_cos) else -1e9

        else:
            # 榛樿浠嶇劧鐢?official cos
            model_select_metric = avg_val_cos if np.isfinite(avg_val_cos) else -1e9
        is_best = model_select_metric > (best_val_official_cos + early_stop_min_delta)
        if is_best:
            best_val_official_cos = model_select_metric
            early_stop_wait = 0
            torch.save(model.state_dict(), 'checkpoints/best_model.pt')
            torch.save(model, 'checkpoints/best_model.model')
            with open('checkpoints/best_model.meta', 'wb') as f:
                pickle.dump({
                    'spectrum_bin_config': spect_bin_config,
                    'featurize_config': featurizer_config,
                    'model_config': spect_out_config,
                    'formula_atomicnos': formula_atomicnos,
                    'loss': 'selector + precursor + rerank + official_dense_spectral + peak + oos',
                }, f)
        else:
            if np.isfinite(model_select_metric):
                early_stop_wait += 1

        log(
            f'Epoch {epoch+1}/{epochs} | '
            f'train_loss={avg_train_loss:.4f} | '
            f'train_rerank_teacher_ratio={avg_train_rerank_teacher_ratio:.4f} | '
            f'train_selector_loss={avg_train_selector_loss:.4f} | '
            f'train_selector_bce={avg_train_selector_bce:.4f} | '
            f'train_selector_kl={avg_train_selector_kl:.4f} | '
            f'train_selector_quality_mean={avg_train_selector_quality_mean:.4f} | '
            f'train_selector_pos_rate={avg_train_selector_pos_rate:.4f} | '
            f'train_target_pos_false_mass={avg_train_target_pos_false_mass:.4f} | '
            f'train_target_pos_overlap_exact={avg_train_target_pos_overlap_exact:.4f} | '
            f'train_target_pos_exact_support_mass={avg_train_target_pos_exact_support_mass:.4f} | '
            f'train_target_strict_keep_rate={avg_train_target_strict_keep_rate:.4f} | '
            f'train_target_fallback_rate={avg_train_target_fallback_rate:.4f} | '
            f'train_target_clean_pos_rate={avg_train_target_clean_pos_rate:.4f} | '
            f'train_target_pool_pos_rate={avg_train_target_pool_pos_rate:.4f} | '
            f'train_target_pool_pos_false_mass={avg_train_target_pool_pos_false_mass:.4f} | '
            f'train_target_pool_pos_overlap_tol={avg_train_target_pool_pos_overlap_tol:.4f} | '
            f'train_target_teacher_pos_rate={avg_train_target_teacher_pos_rate:.4f} | '
            f'train_target_teacher_dist_n={avg_train_target_teacher_dist_n:.4f} | '
            f'train_target_teacher_added_rate={avg_train_target_teacher_added_rate:.4f} | '
            f'train_selector_dyn_pos_weight={avg_train_selector_dyn_pos_weight:.4f} | '
            f'train_use_rerank_delta={avg_train_use_rerank_delta:.1f} | '
            f'train_main_candidate_kl={avg_train_main_kl:.4f} | '
            f'train_rerank_kl={avg_train_rerank_kl:.4f} | '
            f'train_rerank_bce={avg_train_rerank_bce:.4f} | '
            f'train_rerank_loss={avg_train_rerank_loss:.4f} | '
            f'train_official_spectral={avg_train_official_spectral:.4f} | '
            f'train_peak_aux={avg_train_peak_aux:.4f} | '
            f'train_oos={avg_train_oos:.4f} | '
            f'train_formula_entropy={avg_train_formula_entropy:.4f} | '
            f'train_false_support={avg_train_false_support:.4f} | '
            f'train_precursor={avg_train_precursor_loss:.4f} | '
            f'train_fn_loss={avg_train_fn_loss:.4f} | '
            f'val_loss={avg_val_loss:.4f} | '
            f'val_selector_loss={avg_val_selector_loss:.4f} | '
            f'val_selector_bce={avg_val_selector_bce:.4f} | '
            f'val_selector_kl={avg_val_selector_kl:.4f} | '
            f'val_selector_quality_mean={avg_val_selector_quality_mean:.4f} | '
            f'val_selector_pos_rate={avg_val_selector_pos_rate:.4f} | '
            f'val_target_pos_false_mass={avg_val_target_pos_false_mass:.4f} | '
            f'val_target_pos_overlap_exact={avg_val_target_pos_overlap_exact:.4f} | '
            f'val_target_pos_exact_support_mass={avg_val_target_pos_exact_support_mass:.4f} | '
            f'val_target_strict_keep_rate={avg_val_target_strict_keep_rate:.4f} | '
            f'val_target_fallback_rate={avg_val_target_fallback_rate:.4f} | '
            f'val_target_clean_pos_rate={avg_val_target_clean_pos_rate:.4f} | '
            f'val_target_pool_pos_rate={avg_val_target_pool_pos_rate:.4f} | '
            f'val_target_pool_pos_false_mass={avg_val_target_pool_pos_false_mass:.4f} | '
            f'val_target_pool_pos_overlap_tol={avg_val_target_pool_pos_overlap_tol:.4f} | '
            f'val_target_teacher_pos_rate={avg_val_target_teacher_pos_rate:.4f} | '
            f'val_target_teacher_dist_n={avg_val_target_teacher_dist_n:.4f} | '
            f'val_target_teacher_added_rate={avg_val_target_teacher_added_rate:.4f} | '
            f'val_selector_dyn_pos_weight={avg_val_selector_dyn_pos_weight:.4f} | '
            f'val_use_rerank_delta={avg_val_use_rerank_delta:.1f} | '
            f'val_model_topk_teacher_recall@{model_topk_eval}={avg_val_model_topk_teacher_recall:.4f} | '
            f'val_active_teacher_recall={avg_val_active_teacher_recall:.4f} | '
            f'val_fragaux_teacher_recall={avg_val_fragaux_teacher_recall:.4f} | '
            f'val_fragaux_model_topk_ratio@32={avg_val_fragaux_model_topk_ratio_32:.4f} | '
            f'val_fragaux_model_topk_ratio@64={avg_val_fragaux_model_topk_ratio_64:.4f} | '
            f'val_fragaux_model_topk_ratio@128={avg_val_fragaux_model_topk_ratio_128:.4f} | '
            f'val_fragaux_model_topk_ratio@256={avg_val_fragaux_model_topk_ratio_256:.4f} | '
            f'val_selector_recall@32={avg_val_selector_recall_32:.4f} | '
            f'val_selector_recall@64={avg_val_selector_recall_64:.4f} | '
            f'val_selector_recall@128={avg_val_selector_recall_128:.4f} | '
            f'val_selector_recall@256={avg_val_selector_recall_256:.4f} | '
            f'val_selector_precision@32={avg_val_selector_precision_32:.4f} | '
            f'val_selector_precision@64={avg_val_selector_precision_64:.4f} | '
            f'val_selector_precision@128={avg_val_selector_precision_128:.4f} | '
            f'val_selector_precision@256={avg_val_selector_precision_256:.4f} | '
            f'val_selector_quality_mean@32={avg_val_selector_quality_mean_32:.4f} | '
            f'val_selector_quality_mean@64={avg_val_selector_quality_mean_64:.4f} | '
            f'val_selector_quality_mean@128={avg_val_selector_quality_mean_128:.4f} | '
            f'val_selector_quality_mean@256={avg_val_selector_quality_mean_256:.4f} | '
            f'val_main_candidate_kl={avg_val_main_kl:.4f} | '
            f'val_rerank_kl={avg_val_rerank_kl:.4f} | '
            f'val_rerank_bce={avg_val_rerank_bce:.4f} | '
            f'val_rerank_loss={avg_val_rerank_loss:.4f} | '
            f'val_official_spectral={avg_val_official_spectral:.4f} | '
            f'val_peak_aux={avg_val_peak_aux:.4f} | '
            f'val_oos={avg_val_oos:.4f} | '
            f'val_formula_entropy={avg_val_formula_entropy:.4f} | '
            f'val_pred_n={avg_val_pred_n:.1f} | '
            f'val_true_n={avg_val_true_n:.1f} | '
            f'val_overlap_n={avg_val_overlap_n:.1f} | '
            f'val_false_pred_n={avg_val_false_pred_n:.1f} | '
            f'val_overlap_ratio={avg_val_overlap_ratio:.4f} | '
            f'val_pred_int_on_true={avg_val_pred_int_on_true:.4f} | '
            f'val_precursor={avg_val_precursor_loss:.4f} | '
            f'val_fn_loss={avg_val_fn_loss:.4f} | '
            f'val_official_cos_no_precursor={avg_val_cos:.4f} | '
            f'val_official_js_no_precursor={avg_val_js:.4f} | '
            f'val_topk_peak_recall@20={avg_val_recall:.4f} | '
            f'val_false_support={avg_val_false_support:.4f} | '
            f'val_matched_intensity_coverage={avg_val_cov:.4f} | '
            f'val_selected_true_hit_mass@32={avg_val_selected_true_hit_mass_32:.4f} | '
            f'val_selected_true_hit_mass@64={avg_val_selected_true_hit_mass_64:.4f} | '
            f'val_selected_true_hit_mass@128={avg_val_selected_true_hit_mass_128:.4f} | '
            f'val_selected_true_hit_mass@256={avg_val_selected_true_hit_mass_256:.4f} | '
            f'val_selected_false_mass@32={avg_val_selected_false_mass_32:.4f} | '
            f'val_selected_false_mass@64={avg_val_selected_false_mass_64:.4f} | '
            f'val_selected_false_mass@128={avg_val_selected_false_mass_128:.4f} | '
            f'val_selected_false_mass@256={avg_val_selected_false_mass_256:.4f} | '
            f'val_teacher_oracle_cos={avg_val_teacher_oracle_cos:.4f} | '
            f'val_teacher_oracle_false_support={avg_val_teacher_oracle_false_support:.4f} | '
            f'val_teacher_oracle_pred_int_on_true={avg_val_teacher_oracle_pred_int_on_true:.4f} | '
            f'val_teacher_oracle_pred_n={avg_val_teacher_oracle_pred_n:.2f} | '
            f'val_model_topk_oracle_cos@256={avg_val_model_topk_oracle_cos_256:.4f} | '
            f'val_model_topk_oracle_false_support@256={avg_val_model_topk_oracle_false_support_256:.4f} | '
            f'val_utility_top64_oracle_cos={avg_val_utility_top64_oracle_cos:.4f} | '
            f'val_utility_top64_false_support={avg_val_utility_top64_false_support:.4f} | '
            f'val_utility_top64_true_hit_mass={avg_val_utility_top64_true_hit_mass:.4f} | '
            f'val_utility_top64_false_mass={avg_val_utility_top64_false_mass:.4f} | '
            f'val_model_topk_oracle_cos@{eval_k_used}={avg_val_model_topk_oracle_cos_eval:.4f} | '
            f'val_model_topk_oracle_false_support@{eval_k_used}={avg_val_model_topk_oracle_false_support_eval:.4f}'
            + (' | BEST' if is_best else '')
        )

        history['train_loss'].append(avg_train_loss)
        history['train_main_candidate_kl'].append(avg_train_main_kl)
        history['train_official_spectral_loss'].append(avg_train_official_spectral)
        history['train_peak_aux'].append(avg_train_peak_aux)
        history['train_oos_loss'].append(avg_train_oos)
        history['train_formula_entropy'].append(avg_train_formula_entropy)
        history['train_precursor_loss'].append(avg_train_precursor_loss)
        history['val_precursor_loss'].append(avg_val_precursor_loss)
        history['train_fn_loss'].append(avg_train_fn_loss)
        history['val_fn_loss'].append(avg_val_fn_loss)
        history['train_rerank_teacher_ratio'].append(avg_train_rerank_teacher_ratio)
        history['train_rerank_kl'].append(avg_train_rerank_kl)
        history['train_rerank_bce'].append(avg_train_rerank_bce)
        history['train_rerank_loss'].append(avg_train_rerank_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_main_candidate_kl'].append(avg_val_main_kl)
        history['val_rerank_kl'].append(avg_val_rerank_kl)
        history['val_rerank_bce'].append(avg_val_rerank_bce)
        history['val_rerank_loss'].append(avg_val_rerank_loss)
        history['val_official_spectral_loss'].append(avg_val_official_spectral)
        history['val_peak_aux'].append(avg_val_peak_aux)
        history['val_oos_loss'].append(avg_val_oos)
        history['val_formula_entropy'].append(avg_val_formula_entropy)
        history['val_official_cos_no_precursor'].append(avg_val_cos)
        history['val_official_js_no_precursor'].append(avg_val_js)
        history['val_topk_peak_recall@20'].append(avg_val_recall)
        history['val_matched_intensity_coverage'].append(avg_val_cov)
        history['train_selector_loss'].append(avg_train_selector_loss)
        history['train_selector_bce'].append(avg_train_selector_bce)
        history['train_selector_kl'].append(avg_train_selector_kl)
        history['train_selector_quality_mean'].append(avg_train_selector_quality_mean)
        history['train_selector_pos_rate'].append(avg_train_selector_pos_rate)
        history['train_target_pos_false_mass'].append(avg_train_target_pos_false_mass)
        history['train_target_pos_overlap_exact'].append(avg_train_target_pos_overlap_exact)
        history['train_target_pos_exact_support_mass'].append(avg_train_target_pos_exact_support_mass)
        history['train_target_strict_keep_rate'].append(avg_train_target_strict_keep_rate)
        history['train_target_fallback_rate'].append(avg_train_target_fallback_rate)
        history['train_target_clean_pos_rate'].append(avg_train_target_clean_pos_rate)
        history['train_target_pool_pos_rate'].append(avg_train_target_pool_pos_rate)
        history['train_target_pool_pos_false_mass'].append(avg_train_target_pool_pos_false_mass)
        history['train_target_pool_pos_overlap_tol'].append(avg_train_target_pool_pos_overlap_tol)
        history['train_target_teacher_pos_rate'].append(avg_train_target_teacher_pos_rate)
        history['train_target_teacher_dist_n'].append(avg_train_target_teacher_dist_n)
        history['train_target_teacher_added_rate'].append(avg_train_target_teacher_added_rate)
        history['train_selector_dyn_pos_weight'].append(avg_train_selector_dyn_pos_weight)
        history['train_use_rerank_delta'].append(avg_train_use_rerank_delta)
        history['val_selector_bce'].append(avg_val_selector_bce)
        history['val_selector_kl'].append(avg_val_selector_kl)
        history['val_selector_quality_mean'].append(avg_val_selector_quality_mean)
        history['val_selector_pos_rate'].append(avg_val_selector_pos_rate)
        history['val_target_pos_false_mass'].append(avg_val_target_pos_false_mass)
        history['val_target_pos_overlap_exact'].append(avg_val_target_pos_overlap_exact)
        history['val_target_pos_exact_support_mass'].append(avg_val_target_pos_exact_support_mass)
        history['val_target_strict_keep_rate'].append(avg_val_target_strict_keep_rate)
        history['val_target_fallback_rate'].append(avg_val_target_fallback_rate)
        history['val_target_clean_pos_rate'].append(avg_val_target_clean_pos_rate)
        history['val_target_pool_pos_rate'].append(avg_val_target_pool_pos_rate)
        history['val_target_pool_pos_false_mass'].append(avg_val_target_pool_pos_false_mass)
        history['val_target_pool_pos_overlap_tol'].append(avg_val_target_pool_pos_overlap_tol)
        history['val_target_teacher_pos_rate'].append(avg_val_target_teacher_pos_rate)
        history['val_target_teacher_dist_n'].append(avg_val_target_teacher_dist_n)
        history['val_target_teacher_added_rate'].append(avg_val_target_teacher_added_rate)
        history['val_selector_dyn_pos_weight'].append(avg_val_selector_dyn_pos_weight)
        history['val_use_rerank_delta'].append(avg_val_use_rerank_delta)
        history['val_selector_loss'].append(avg_val_selector_loss)
        history['val_model_topk_teacher_recall'].append(avg_val_model_topk_teacher_recall)
        history['val_active_teacher_recall'].append(avg_val_active_teacher_recall)
        history['val_fragaux_teacher_recall'].append(avg_val_fragaux_teacher_recall)
        history['val_fragaux_model_topk_ratio@32'].append(avg_val_fragaux_model_topk_ratio_32)
        history['val_fragaux_model_topk_ratio@64'].append(avg_val_fragaux_model_topk_ratio_64)
        history['val_fragaux_model_topk_ratio@128'].append(avg_val_fragaux_model_topk_ratio_128)
        history['val_fragaux_model_topk_ratio@256'].append(avg_val_fragaux_model_topk_ratio_256)
        history['val_selector_recall@32'].append(avg_val_selector_recall_32)
        history['val_selector_recall@64'].append(avg_val_selector_recall_64)
        history['val_selector_recall@128'].append(avg_val_selector_recall_128)
        history['val_selector_recall@256'].append(avg_val_selector_recall_256)
        history['val_selector_precision@32'].append(avg_val_selector_precision_32)
        history['val_selector_precision@64'].append(avg_val_selector_precision_64)
        history['val_selector_precision@128'].append(avg_val_selector_precision_128)
        history['val_selector_precision@256'].append(avg_val_selector_precision_256)
        history['val_selector_quality_mean@32'].append(avg_val_selector_quality_mean_32)
        history['val_selector_quality_mean@64'].append(avg_val_selector_quality_mean_64)
        history['val_selector_quality_mean@128'].append(avg_val_selector_quality_mean_128)
        history['val_selector_quality_mean@256'].append(avg_val_selector_quality_mean_256)
        history['val_selected_true_hit_mass@32'].append(avg_val_selected_true_hit_mass_32)
        history['val_selected_true_hit_mass@64'].append(avg_val_selected_true_hit_mass_64)
        history['val_selected_true_hit_mass@128'].append(avg_val_selected_true_hit_mass_128)
        history['val_selected_true_hit_mass@256'].append(avg_val_selected_true_hit_mass_256)
        history['val_selected_false_mass@32'].append(avg_val_selected_false_mass_32)
        history['val_selected_false_mass@64'].append(avg_val_selected_false_mass_64)
        history['val_selected_false_mass@128'].append(avg_val_selected_false_mass_128)
        history['val_selected_false_mass@256'].append(avg_val_selected_false_mass_256)
        history['val_teacher_oracle_cos'].append(avg_val_teacher_oracle_cos)
        history['val_teacher_oracle_false_support'].append(avg_val_teacher_oracle_false_support)
        history['val_teacher_oracle_pred_int_on_true'].append(avg_val_teacher_oracle_pred_int_on_true)
        history['val_teacher_oracle_pred_n'].append(avg_val_teacher_oracle_pred_n)
        history['val_model_topk_oracle_cos@256'].append(avg_val_model_topk_oracle_cos_256)
        history['val_model_topk_oracle_false_support@256'].append(avg_val_model_topk_oracle_false_support_256)
        history['val_utility_top64_oracle_cos'].append(avg_val_utility_top64_oracle_cos)
        history['val_utility_top64_false_support'].append(avg_val_utility_top64_false_support)
        history['val_utility_top64_true_hit_mass'].append(avg_val_utility_top64_true_hit_mass)
        history['val_utility_top64_false_mass'].append(avg_val_utility_top64_false_mass)
        history['train_false_support'].append(avg_train_false_support)
        history['val_false_support'].append(avg_val_false_support)
        if epoch_best_sparse is not None:
            _save_epoch_official_sparse_plot(
                pred_sparse=epoch_best_sparse[0],
                true_sparse=epoch_best_sparse[1],
                epoch=epoch + 1,
                out_dir='outputs',
                tag='best',
                topk=30,
                bin_width=official_metric_cfg['bin_width'],
            )

        if epoch_worst_sparse is not None:
            _save_epoch_official_sparse_plot(
                pred_sparse=epoch_worst_sparse[0],
                true_sparse=epoch_worst_sparse[1],
                epoch=epoch + 1,
                out_dir='outputs',
                tag='worst_any',
                topk=30,
                bin_width=official_metric_cfg['bin_width'],
            )

        if epoch_worst_nonempty_sparse is not None:
            _save_epoch_official_sparse_plot(
                pred_sparse=epoch_worst_nonempty_sparse[0],
                true_sparse=epoch_worst_nonempty_sparse[1],
                epoch=epoch + 1,
                out_dir='outputs',
                tag='worst_nonempty',
                topk=30,
                bin_width=official_metric_cfg['bin_width'],
            )

        if epoch_worst_overlap_sparse is not None:
            _save_epoch_official_sparse_plot(
                pred_sparse=epoch_worst_overlap_sparse[0],
                true_sparse=epoch_worst_overlap_sparse[1],
                epoch=epoch + 1,
                out_dir='outputs',
                tag='worst_overlap_sparse',
                topk=30,
                bin_width=official_metric_cfg['bin_width'],
            )

            _save_epoch_official_overlap_plot(
                pred_sparse=epoch_worst_overlap_sparse[0],
                true_sparse=epoch_worst_overlap_sparse[1],
                epoch=epoch + 1,
                out_dir='outputs',
                tag='worst_overlap',
                topk=20,
                bin_width=official_metric_cfg['bin_width'],
            )

        
        _save_training_curves(history, out_dir='outputs')

        if train_update_n > 0:
            scheduler.step()
        else:
            log(
                f'鈿狅笍 epoch={epoch+1}: no optimizer update was performed; '
                f'skip scheduler.step(). Check MAX_TRAIN_STEPS or non-finite loss.'
            )
        if (
            early_stop_patience > 0
            and (epoch + 1) >= early_stop_warmup_epochs
            and early_stop_wait >= early_stop_patience
        ):
            log(f'鈴癸笍 early_stop: epoch={epoch+1} best_val_official_cos={best_val_official_cos:.6f}')
            break



def _save_epoch_official_sparse_plot(
    pred_sparse,
    true_sparse,
    epoch,
    out_dir='outputs',
    tag='worst',
    topk=30,
    bin_width=0.01,
):
    if pred_sparse is None or true_sparse is None:
        return

    os.makedirs(out_dir, exist_ok=True)

    pred_idx, pred_val = pred_sparse
    true_idx, true_val = true_sparse

    pred_idx = np.asarray(pred_idx, dtype=np.int64).reshape(-1)
    pred_val = np.asarray(pred_val, dtype=np.float32).reshape(-1)
    true_idx = np.asarray(true_idx, dtype=np.int64).reshape(-1)
    true_val = np.asarray(true_val, dtype=np.float32).reshape(-1)

    if pred_idx.size == 0 and true_idx.size == 0:
        return

    # ---------- normalize each spectrum independently ----------
    if pred_val.size > 0 and np.max(pred_val) > 0:
        pred_val_plot = pred_val / np.max(pred_val)
    else:
        pred_val_plot = pred_val.copy()

    if true_val.size > 0 and np.max(true_val) > 0:
        true_val_plot = true_val / np.max(true_val)
    else:
        true_val_plot = true_val.copy()

    union_idx = np.union1d(pred_idx, true_idx)
    if union_idx.size == 0:
        return

    pred_map = {int(i): float(v) for i, v in zip(pred_idx.tolist(), pred_val_plot.tolist())}
    true_map = {int(i): float(v) for i, v in zip(true_idx.tolist(), true_val_plot.tolist())}

    # ---------- choose top-k after normalization ----------
    score = np.array(
        [max(pred_map.get(int(i), 0.0), true_map.get(int(i), 0.0)) for i in union_idx],
        dtype=np.float32,
    )
    order = np.argsort(-score, kind='stable')
    sel = union_idx[order[: min(topk, len(order))]]
    sel = np.sort(sel)

    pred_sel = np.array([pred_map.get(int(i), 0.0) for i in sel], dtype=np.float32)
    true_sel = np.array([true_map.get(int(i), 0.0) for i in sel], dtype=np.float32)

    mz = (sel.astype(np.float64) + 0.5) * float(bin_width)

    plt.figure(figsize=(14, 5))
    plt.stem(mz - 0.01, true_sel, linefmt='C0-', markerfmt=' ', basefmt=' ')
    plt.stem(mz + 0.01, pred_sel, linefmt='C1-', markerfmt=' ', basefmt=' ')

    plt.xlabel('m/z')
    plt.ylabel('relative intensity (max-normalized)')
    plt.title(f'Epoch {epoch} official sparse comparison ({tag}, top-{len(sel)})')
    plt.legend(['true', 'pred'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_official_{tag}.png'), dpi=180)
    plt.close()

def _save_epoch_official_overlap_plot(
    pred_sparse,
    true_sparse,
    epoch,
    out_dir='outputs',
    tag='worst_overlap',
    topk=20,
    bin_width=0.01,
):
    if pred_sparse is None or true_sparse is None:
        return

    os.makedirs(out_dir, exist_ok=True)

    pred_idx, pred_val = pred_sparse
    true_idx, true_val = true_sparse

    pred_idx = np.asarray(pred_idx, dtype=np.int64).reshape(-1)
    pred_val = np.asarray(pred_val, dtype=np.float32).reshape(-1)
    true_idx = np.asarray(true_idx, dtype=np.int64).reshape(-1)
    true_val = np.asarray(true_val, dtype=np.float32).reshape(-1)

    pred_map_raw = {int(i): float(v) for i, v in zip(pred_idx.tolist(), pred_val.tolist())}
    true_map_raw = {int(i): float(v) for i, v in zip(true_idx.tolist(), true_val.tolist())}

    overlap = np.intersect1d(pred_idx, true_idx)
    if overlap.size == 0:
        return

    pred_max = max(float(np.max(pred_val)) if pred_val.size > 0 else 0.0, 1e-12)
    true_max = max(float(np.max(true_val)) if true_val.size > 0 else 0.0, 1e-12)

    score = np.array(
        [max(pred_map_raw[int(i)] / pred_max, true_map_raw[int(i)] / true_max) for i in overlap],
        dtype=np.float32,
    )
    order = np.argsort(-score, kind='stable')
    sel = overlap[order[: min(topk, len(order))]]
    sel = np.sort(sel)

    pred_sel = np.array([pred_map_raw[int(i)] / pred_max for i in sel], dtype=np.float32)
    true_sel = np.array([true_map_raw[int(i)] / true_max for i in sel], dtype=np.float32)

    mz = (sel.astype(np.float64) + 0.5) * float(bin_width)

    plt.figure(figsize=(14, 5))
    plt.stem(mz - 0.01, true_sel, linefmt='C0-', markerfmt=' ', basefmt=' ')
    plt.stem(mz + 0.01, pred_sel, linefmt='C1-', markerfmt=' ', basefmt=' ')
    plt.xlabel('m/z')
    plt.ylabel('relative intensity (overlap peaks)')
    plt.title(f'Epoch {epoch} official overlap comparison ({tag}, top-{len(sel)})')
    plt.legend(['true', 'pred'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'epoch_{epoch}_official_{tag}.png'), dpi=180)
    plt.close()

def _save_training_curves(history, out_dir='outputs'):
    if not isinstance(history, dict) or len(history) == 0:
        return
    os.makedirs(out_dir, exist_ok=True)
    epochs = np.arange(1, len(history.get('train_loss', [])) + 1, dtype=np.int32)
    if epochs.size <= 0:
        return

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    if len(history.get('train_loss', [])) == epochs.size:
        plt.plot(epochs, history['train_loss'], label='train_loss')
    if len(history.get('val_loss', [])) == epochs.size:
        plt.plot(epochs, history['val_loss'], label='val_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Total loss')
    plt.legend()

    plt.subplot(2, 2, 2)
    if len(history.get('train_main_candidate_kl', [])) == epochs.size:
        plt.plot(epochs, history['train_main_candidate_kl'], label='train_main_candidate_kl')
    if len(history.get('val_main_candidate_kl', [])) == epochs.size:
        plt.plot(epochs, history['val_main_candidate_kl'], label='val_main_candidate_kl')
    if len(history.get('train_official_spectral_loss', [])) == epochs.size:
        plt.plot(epochs, history['train_official_spectral_loss'], label='train_official_spectral_loss')
    if len(history.get('val_official_spectral_loss', [])) == epochs.size:
        plt.plot(epochs, history['val_official_spectral_loss'], label='val_official_spectral_loss')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Rerank/Official spectral loss')
    plt.legend()

    plt.subplot(2, 2, 3)
    if len(history.get('train_peak_aux', [])) == epochs.size:
        plt.plot(epochs, history['train_peak_aux'], label='train_peak_aux')
    if len(history.get('val_peak_aux', [])) == epochs.size:
        plt.plot(epochs, history['val_peak_aux'], label='val_peak_aux')
    if len(history.get('train_oos_loss', [])) == epochs.size:
        plt.plot(epochs, history['train_oos_loss'], label='train_oos_loss')
    if len(history.get('val_oos_loss', [])) == epochs.size:
        plt.plot(epochs, history['val_oos_loss'], label='val_oos_loss')
    if len(history.get('train_precursor_loss', [])) == epochs.size:
        plt.plot(epochs, history['train_precursor_loss'], label='train_precursor_loss')
    if len(history.get('val_precursor_loss', [])) == epochs.size:
        plt.plot(epochs, history['val_precursor_loss'], label='val_precursor_loss')
    if len(history.get('train_fn_loss', [])) == epochs.size:
        plt.plot(epochs, history['train_fn_loss'], label='train_fn_loss')
    if len(history.get('val_fn_loss', [])) == epochs.size:
        plt.plot(epochs, history['val_fn_loss'], label='val_fn_loss')
    plt.xlabel('epoch')
    plt.ylabel('aux loss')
    plt.title('Peak/OOS/Precursor auxiliary')
    plt.legend()

    plt.subplot(2, 2, 4)
    if len(history.get('val_official_cos_no_precursor', [])) == epochs.size:
        plt.plot(epochs, history['val_official_cos_no_precursor'], label='val_official_cos_no_precursor')
    if len(history.get('val_topk_peak_recall@20', [])) == epochs.size:
        plt.plot(epochs, history['val_topk_peak_recall@20'], label='val_topk_peak_recall@20')
    if len(history.get('val_model_topk_teacher_recall', [])) == epochs.size:
        plt.plot(epochs, history['val_model_topk_teacher_recall'], label='val_model_topk_teacher_recall')
    if len(history.get('train_rerank_teacher_ratio', [])) == epochs.size:
        plt.plot(epochs, history['train_rerank_teacher_ratio'], label='train_rerank_teacher_ratio')
    if len(history.get('val_active_teacher_recall', [])) == epochs.size:
        plt.plot(epochs, history['val_active_teacher_recall'], label='val_active_teacher_recall')

    plt.xlabel('epoch')
    plt.ylabel('metric')
    plt.title('Validation metrics')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'training_results.png'), dpi=160)
    plt.close()


if __name__ == '__main__':
    train_mssubsetnet()



