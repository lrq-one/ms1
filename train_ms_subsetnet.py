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

def _neg_mask_fill_value(x):
    if torch.is_tensor(x) and torch.is_floating_point(x):
        if x.dtype in (torch.float16, torch.bfloat16):
            # fp16/bf16 cannot represent -1e9; keep a safe finite floor for softmax masking.
            return -1e4
        try:
            return float(torch.finfo(x.dtype).min)
        except Exception:
            return -1e9
    return -1e9

# ================= 辅助类与函数 =================

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


# Function overview: project_formula_probs_to_official_dense handles a specific workflow step in this module.

def _extract_precursor_mz_at(precursor_mz, i):
    if precursor_mz is None:
        return None
    try:
        if torch.is_tensor(precursor_mz):
            flat = precursor_mz.detach().reshape(-1)
            if i < int(flat.shape[0]):
                v = float(flat[i].cpu().item())
                return v if np.isfinite(v) else None
            return None
        if isinstance(precursor_mz, np.ndarray):
            flat = precursor_mz.reshape(-1)
            if i < int(flat.shape[0]):
                v = float(flat[i])
                return v if np.isfinite(v) else None
            return None
        if isinstance(precursor_mz, (list, tuple)):
            if i < len(precursor_mz):
                v = float(precursor_mz[i])
                return v if np.isfinite(v) else None
            return None
        v = float(precursor_mz)
        return v if np.isfinite(v) else None
    except Exception:
        return None


# Function overview: build_true_official_dense_from_raw handles a specific workflow step in this module.

def build_true_official_dense_from_raw(
    spect_raw_list,
    precursor_mz,
    official_bin_width,
    official_max_mz,
    exclude_precursor,
    batch_n,
    device,
):
    """Build dense official-bin target spectra from raw sparse peaks (target-only, non-differentiable path)."""
    bin_width = float(max(1e-6, official_bin_width))
    max_mz = float(max(bin_width, official_max_mz))
    official_bin_n = int(math.floor(max_mz / bin_width)) + 1

    out = torch.zeros((int(batch_n), official_bin_n), dtype=torch.float32, device=device)
    if not isinstance(spect_raw_list, (list, tuple)):
        return out

    use_n = min(int(batch_n), len(spect_raw_list))
    for i in range(use_n):
        peaks = _to_sparse_peak_array(spect_raw_list[i], spect_bin_centers=None, min_intensity=0.0)
        if peaks.size == 0:
            continue

        mz = peaks[:, 0].astype(np.float64)
        intensity = peaks[:, 1].astype(np.float64)

        valid = (
            np.isfinite(mz)
            & np.isfinite(intensity)
            & (intensity > 0)
            & (mz >= 0.0)
            & (mz < max_mz)
        )
        mz = mz[valid]
        intensity = intensity[valid]
        if mz.size == 0:
            continue

        idx = _official_bin_indices_np(mz, bin_width).astype(np.int64)
        if exclude_precursor:
            pmz = _extract_precursor_mz_at(precursor_mz, i)
            keep = _precursor_keep_mask_np(
                mz=mz,
                precursor_mz=pmz,
                bin_width=bin_width,
                exclude_precursor=True,
            )
            idx = idx[keep]
            intensity = intensity[keep]
        if idx.size == 0:
            continue

        idx_t = torch.as_tensor(idx, dtype=torch.long, device=device).clamp(0, max(0, official_bin_n - 1))
        val_t = torch.as_tensor(intensity.astype(np.float32), dtype=torch.float32, device=device)
        out[i].scatter_add_(0, idx_t, val_t)

    return out


# Function overview: cosine_loss_dense handles a specific workflow step in this module.

def my_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [d[key] for d in batch]
    return batch_dict


# =========================================================================
# 🎯 [数据处理] 稀疏谱转密集谱
# 原始的质谱数据是稀疏的 [(m/z=100, intensity=0.5), (m/z=200, intensity=0.8)]
# 这个函数的作用是把它们“装进” 1024 个箱子(Bin)里，变成一个长度为 1024 的数组。
# 模型最终预测的也就是这 1024 个箱子的高度。
# =========================================================================
# Function overview: to_dense_binned_spectrum handles a specific workflow step in this module.

def to_dense_binned_spectrum(s, spect_bin_obj):
    """
    Convert a sparse spectrum to a dense binned 1024 vector.

    Accepts either:
    - already dense vector (len == 1024)
    - sparse peaks as iterable of (mz, intensity)
    """
    if isinstance(s, torch.Tensor):
        s = s.detach().cpu().numpy()

    if isinstance(s, np.ndarray) and s.ndim == 1 and s.shape[0] == 1024:
        return torch.as_tensor(s, dtype=torch.float32)

    if isinstance(s, (list, tuple)) and len(s) == 1024 and not isinstance(s[0], (list, tuple, np.ndarray, torch.Tensor)):
        return torch.as_tensor(s, dtype=torch.float32)

    # Common dataset format: object array with each element like [mz, intensity]
    if isinstance(s, np.ndarray) and s.dtype == object and s.ndim == 1:
        try:
            peaks = np.stack([np.asarray(x, dtype=np.float32) for x in s], axis=0)
        except Exception:
            peaks = np.asarray(s, dtype=np.float32)
    else:
        peaks = np.asarray(s, dtype=np.float32)
    if peaks.size == 0:
        return torch.zeros(1024, dtype=torch.float32)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        # Fallback: try flattening odd inputs; if still invalid, return zeros.
        try:
            peaks = peaks.reshape(-1, 2)
        except Exception:
            return torch.zeros(1024, dtype=torch.float32)

    _, _, spect_out = spect_bin_obj.histogram(peaks[:, 0], peaks[:, 1])
    return torch.as_tensor(spect_out, dtype=torch.float32)



# =========================================================================
# 🎯 [特征定义] MS/MS 加合物词表
# 为了让串联质谱模型知道前体离子(Precursor)是怎么带电的（挂了氢H+还是钠Na+）。
# 我们给常见的加合物进行编号，相当于 NLP 里的单词分词，送进网络做 Embedding。
# =========================================================================
ADDUCT_VOCAB = {
    '[M+H]+': 1,
    '[M+Na]+': 2,
    '[M+K]+': 3,
    '[M-H]-': 4,
    '[M+NH4]+': 5,
}

MISSING_TOKEN = '__MISSING__'
UNKNOWN_TOKEN = '__UNK__'

INSTRUMENT_VOCAB = {
    'orbitrap': 1,
    'qtof': 2,
    'tof': 3,
    'iontrap': 4,
    'fticr': 5,
    'triplequad': 6,
}


# Function overview: _load_vocab_from_json handles a specific workflow step in this module.

def _load_vocab_from_json(env_var_name, fallback_vocab, field_name):
    path = os.environ.get(env_var_name, '').strip()
    if not path:
        return dict(fallback_vocab)
    if not os.path.exists(path):
        return dict(fallback_vocab)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, dict) and field_name in obj and isinstance(obj[field_name], dict):
            raw = obj[field_name]
        elif isinstance(obj, dict):
            raw = obj
        else:
            return dict(fallback_vocab)

        out = {}
        for k, v in raw.items():
            try:
                idx = int(v)
            except Exception:
                continue
            if idx >= 0:
                out[str(k)] = idx
        return out if len(out) > 0 else dict(fallback_vocab)
    except Exception:
        return dict(fallback_vocab)


# Function overview: _vocab_uses_separate_unknown handles a specific workflow step in this module.

def _vocab_uses_separate_unknown(vocab):
    return isinstance(vocab, dict) and MISSING_TOKEN in vocab and UNKNOWN_TOKEN in vocab


# Function overview: _vocab_missing_index handles a specific workflow step in this module.

def _vocab_missing_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(MISSING_TOKEN, 0))
        except Exception:
            return 0
    return 0


# Function overview: _vocab_unknown_index handles a specific workflow step in this module.

def _vocab_unknown_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(UNKNOWN_TOKEN, 1))
        except Exception:
            return 1
    return 0


# Function overview: _instrument_norm_key handles a specific workflow step in this module.

def _instrument_norm_key(val):
    return str(val).strip().lower().replace(' ', '').replace('-', '').replace('_', '')


# Function overview: _build_normalized_instrument_vocab handles a specific workflow step in this module.

def _build_normalized_instrument_vocab(vocab):
    out = {}
    if not isinstance(vocab, dict):
        return out
    for k, v in vocab.items():
        if str(k) in (MISSING_TOKEN, UNKNOWN_TOKEN):
            continue
        try:
            idx = int(v)
        except Exception:
            continue
        if idx <= 0:
            continue
        nk = _instrument_norm_key(k)
        if nk and nk not in out:
            out[nk] = idx
    return out




# Function overview: encode_adduct_batch handles a specific workflow step in this module.

ADDUCT_VOCAB = _load_vocab_from_json('ADDUCT_VOCAB_PATH', ADDUCT_VOCAB, 'adduct_vocab')
INSTRUMENT_VOCAB = _load_vocab_from_json('INSTRUMENT_VOCAB_PATH', INSTRUMENT_VOCAB, 'instrument_vocab')
MS_LEVEL_VOCAB = _load_vocab_from_json('MS_LEVEL_VOCAB_PATH', {}, 'ms_level_vocab')
INSTRUMENT_VOCAB_NORM = _build_normalized_instrument_vocab(INSTRUMENT_VOCAB)

def encode_adduct_batch(values):
    missing_idx = _vocab_missing_index(ADDUCT_VOCAB)
    unknown_idx = _vocab_unknown_index(ADDUCT_VOCAB)
    out = []
    for val in values:
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            out.append(int(val))
            continue
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
            else:
                out.append(int(val))
            continue
        sval = str(val).strip()
        out.append(ADDUCT_VOCAB.get(sval, unknown_idx))
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: _normalize_instrument_key handles a specific workflow step in this module.

def _normalize_instrument_key(val):
    return _instrument_norm_key(val)


# Function overview: encode_instrument_batch handles a specific workflow step in this module.

def encode_instrument_batch(values):
    missing_idx = _vocab_missing_index(INSTRUMENT_VOCAB)
    unknown_idx = _vocab_unknown_index(INSTRUMENT_VOCAB)
    out = []
    for val in values:
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            out.append(int(val))
            continue
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
            else:
                out.append(int(val))
            continue
        key = _normalize_instrument_key(val)
        idx = INSTRUMENT_VOCAB_NORM.get(key, unknown_idx)
        if idx == unknown_idx:
            if 'orbitrap' in key:
                idx = INSTRUMENT_VOCAB_NORM.get('orbitrap', unknown_idx)
            elif 'qtof' in key:
                idx = INSTRUMENT_VOCAB_NORM.get('qtof', unknown_idx)
            elif 'tof' == key:
                idx = INSTRUMENT_VOCAB_NORM.get('tof', unknown_idx)
            elif 'iontrap' in key:
                idx = INSTRUMENT_VOCAB_NORM.get('iontrap', unknown_idx)
            elif 'fticr' in key:
                idx = INSTRUMENT_VOCAB_NORM.get('fticr', unknown_idx)
            elif 'triplequad' in key or 'triplequadrupole' in key:
                idx = INSTRUMENT_VOCAB_NORM.get('triplequad', unknown_idx)
        out.append(idx)
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: encode_ms_level_batch handles a specific workflow step in this module.

def encode_ms_level_batch(values):
    out = []
    has_vocab = len(MS_LEVEL_VOCAB) > 0
    missing_idx = _vocab_missing_index(MS_LEVEL_VOCAB)
    unknown_idx = _vocab_unknown_index(MS_LEVEL_VOCAB)
    for val in values:
        if val is None or (isinstance(val, str) and not str(val).strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            raw = int(val)
        elif isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
                continue
            raw = int(val)
        else:
            sval = str(val).strip().lower()
            if not sval:
                out.append(missing_idx)
                continue
            else:
                try:
                    raw = int(float(sval))
                except Exception:
                    m = re.search(r"\d+", sval)
                    if m is None:
                        out.append(unknown_idx if has_vocab else 0)
                        continue
                    raw = int(m.group(0))
        if has_vocab:
            out.append(int(MS_LEVEL_VOCAB.get(str(raw), unknown_idx)))
        else:
            out.append(raw)
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: _stack_list_to_tensor handles a specific workflow step in this module.

def _stack_list_to_tensor(values, target_dtype=None):
    if values is None or len(values) == 0:
        return values

    first = values[0]
    try:
        if torch.is_tensor(first):
            t = torch.stack([v if torch.is_tensor(v) else torch.as_tensor(v) for v in values])
            return t.to(dtype=target_dtype) if target_dtype is not None else t

        arr = np.asarray(values)
        if arr.dtype != object:
            t = torch.as_tensor(arr)
            return t.to(dtype=target_dtype) if target_dtype is not None else t
    except Exception:
        pass

    try:
        elems = [torch.as_tensor(v) for v in values]
        t = torch.stack(elems)
        return t.to(dtype=target_dtype) if target_dtype is not None else t
    except Exception:
        return values


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

def get_formulae_official_intensity_from_batch(batch):
    """Prefer official-bin intensity tensor when present; fallback to legacy intensity."""
    if not isinstance(batch, dict):
        return None
    off_int = batch.get('formulae_peaks_official_intensity', None)
    if torch.is_tensor(off_int) and off_int.dim() == 3:
        return off_int
    return batch.get('formulae_peaks_intensity', None)


# Function overview: prepare_batch_cpu handles a specific workflow step in this module.

def prepare_batch_cpu(raw_batch, spect_bin):
    try:
        prefix_topk = int(os.environ.get('CANDIDATE_PREFIX_TOPK', '0'))
    except Exception:
        prefix_topk = 0

    processed_batch = {}
    for k, v in raw_batch.items():
        if k == 'spect':
            dense = _stack_list_to_tensor(v, target_dtype=torch.float32)
            if torch.is_tensor(dense) and dense.dim() == 2 and dense.shape[1] == 1024:
                processed_batch[k] = dense
            else:
                spect_list = [to_dense_binned_spectrum(s, spect_bin) for s in v]
                processed_batch[k] = torch.stack(spect_list)

        elif k == 'spect_raw':
            # Keep sparse/raw spectra on CPU for official validation metrics.
            processed_batch[k] = v

        elif k in ['formulae_feats', 'formula_feats']:
            processed_batch[k] = _stack_list_to_tensor(v, target_dtype=torch.long)

        elif k in [
            'true_official_idx',
            'true_official_intensity',
            'true_top20_official_idx',
            'true_top20_official_intensity',
            'true_all_official_idx',
            'true_all_official_intensity',
        ]:
            # 这些是变长 sparse target，先保留 list。
            processed_batch[k] = v

        elif k in ['vect_feat', 'adj', 'input_mask', 'mol_feat', 'ce', 'precursor_mz',
            'ce_missing', 'adduct_missing', 'instrument_missing', 'ms_level_missing',
            'precursor_mz_missing', 'formulae_mask', 'teacher_formula_probs']:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            'true_precursor_intensity',
            'true_precursor_prob_in_all',
            'true_precursor_present',
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            'true_precursor_bin',
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in ['formulae_n_raw', 'formulae_n_kept', 'formulae_truncated']:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v
        elif k in ['formulae_active_mask', 'formulae_prior_score']:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in ['formulae_source_flag', 'formulae_break_depth', 'formulae_ring_cut_flag']:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k == 'adduct':
            processed_batch[k] = encode_adduct_batch(v)

        elif k == 'instrument_type':
            processed_batch[k] = encode_instrument_batch(v)

        elif k == 'ms_level':
            processed_batch[k] = encode_ms_level_batch(v)

        elif k in [
            'formulae_features', 'formulae_peaks', 'formulae_peaks_mass_idx', 'formulae_peaks_intensity',
            'formulae_peaks_official_idx', 'formulae_peaks_official_intensity',
            'formulae_peaks_official_idx_agg', 'formulae_peaks_official_intensity_agg',
            'formulae_aux_feat', 'formulae_frag_aux_feat', 'formulae_instance_is_source',
            'formulae_instance_group_id', 'formulae_instance_depth', 'formulae_instance_h_shift',
            'vert_element_oh',
            'fragment_node_mz', 'fragment_node_intensity', 'fragment_node_local_feat',
            'fragment_node_depth', 'fragment_node_h_shift', 'fragment_node_is_brics',
            'fragment_node_ring_cut', 'fragment_node_atom_count', 'fragment_node_cut_count',
            'fragment_node_source_type', 'fragment_node_mask', 'fragment_node_label',
            'fragment_node_true_intensity', 'fragment_node_true_intensity_share',
            'fragment_node_bin_dup_count', 'fragment_node_label_top20'
        ]:
            if k in ['formulae_features']:
                stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            elif ('idx' in k) or ('mass_idx' in k):
                stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            else:
                stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            'fragment_node_formula', 'fragment_node_official_idx', 'fragment_node_group_formula_id', 'fragment_node_n_valid'
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        else:
            processed_batch[k] = v

    # Optional controlled ablation: keep only the first-K candidates from a shared cache.
    # This enables a clean max_formulae prefix sweep on identical samples/candidate ordering.
    if prefix_topk > 0 and torch.is_tensor(processed_batch.get('formulae_features', None)):
        cand_n = int(processed_batch['formulae_features'].shape[1])
        keep_n = max(1, min(int(prefix_topk), cand_n))

        for kk in [
            'formulae_features',
            'formulae_peaks',
            'formulae_peaks_mass_idx',
            'formulae_peaks_intensity',
            'formulae_peaks_official_idx',
            'formulae_peaks_official_intensity',
            'formulae_peaks_official_idx_agg',
            'formulae_peaks_official_intensity_agg',
            'formulae_aux_feat',
            'formulae_frag_aux_feat',
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
        ]:
            vv = processed_batch.get(kk, None)
            if torch.is_tensor(vv) and vv.dim() >= 2 and int(vv.shape[1]) >= keep_n:
                processed_batch[kk] = vv[:, :keep_n, ...]

        n_kept = processed_batch.get('formulae_n_kept', None)
        if torch.is_tensor(n_kept):
            processed_batch['formulae_n_kept'] = torch.clamp(n_kept.long(), max=int(keep_n))

        n_raw = processed_batch.get('formulae_n_raw', None)
        if torch.is_tensor(n_raw):
            trunc = (n_raw.long() > int(keep_n)).long()
            if torch.is_tensor(processed_batch.get('formulae_truncated', None)):
                processed_batch['formulae_truncated'] = torch.maximum(
                    processed_batch['formulae_truncated'].long(),
                    trunc,
                )
            else:
                processed_batch['formulae_truncated'] = trunc
        # Robust fallback: some batches may still leave peak tensors as python lists.
    # Force-convert them here if shapes are regular.
    for kk in [
        'formulae_peaks_mass_idx',
        'formulae_peaks_intensity',
        'formulae_peaks_official_idx',
        'formulae_peaks_official_intensity',
        'formulae_peaks_official_idx_agg',
        'formulae_peaks_official_intensity_agg',

        
        'true_precursor_bin',
        'true_precursor_intensity',
        'true_precursor_prob_in_all',
        'true_precursor_present',
    ]:
        vv = processed_batch.get(kk, None)
        if isinstance(vv, list):
            try:
                arr = np.asarray(vv)
                if (
                    ('idx' in kk)
                    or ('mass_idx' in kk)
                    or kk in [
                        'true_precursor_bin',
                    ]
                ):
                    processed_batch[kk] = torch.as_tensor(arr, dtype=torch.long)
                else:
                    processed_batch[kk] = torch.as_tensor(arr, dtype=torch.float32)
            except Exception:
                pass
    return processed_batch


# Function overview: move_batch_to_device handles a specific workflow step in this module.

def move_batch_to_device(processed_batch, device):
    batch = {}
    for k, v in processed_batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
            batch[k] = [i.to(device, non_blocking=True) for i in v]
        else:
            batch[k] = v
    return batch


# Function overview: compute_formula_distribution_metrics summarizes score/probability behavior over candidates.

def _to_float_scalar(v, default=np.nan):
    if v is None:
        return float(default)
    if torch.is_tensor(v):
        try:
            return float(v.detach().reshape(-1)[0].cpu().item())
        except Exception:
            return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


# Function overview: _to_sparse_peak_array handles a specific workflow step in this module.

def _to_sparse_peak_array(x, spect_bin_centers=None, min_intensity=0.0):
    """Convert mixed spectrum representations to sparse (mz, intensity) peaks."""
    if x is None:
        return np.zeros((0, 2), dtype=np.float32)

    if torch.is_tensor(x):
        arr = x.detach().cpu().numpy()
    else:
        arr = np.asarray(x)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(e, dtype=np.float32) for e in arr], axis=0)
        except Exception:
            arr = np.asarray(arr, dtype=np.float32)

    if arr.ndim == 1:
        if spect_bin_centers is None:
            return np.zeros((0, 2), dtype=np.float32)
        vec = np.asarray(arr, dtype=np.float32)
        n = min(vec.shape[0], spect_bin_centers.shape[0])
        vec = vec[:n]
        idx = np.nonzero(vec > float(min_intensity))[0]
        if idx.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return np.stack([
            spect_bin_centers[idx].astype(np.float32),
            vec[idx].astype(np.float32),
        ], axis=-1)

    if arr.ndim == 2 and arr.shape[1] == 2:
        out = np.asarray(arr, dtype=np.float32)
        valid = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & (out[:, 1] > float(min_intensity))
        out = out[valid]
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return out

    try:
        out = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
        valid = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & (out[:, 1] > float(min_intensity))
        out = out[valid]
        if out.size == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return out
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)



def _official_bin_indices_np(mz, bin_width):
    bw = float(max(1e-6, float(bin_width)))
    mode = str(os.environ.get("OFFICIAL_BIN_MODE", "floor")).strip().lower()
    arr = np.asarray(mz, dtype=np.float64)
    if mode in ("round", "nearest", "nominal"):
        return np.rint(arr / bw).astype(np.int64)
    return np.floor(arr / bw + 1e-8).astype(np.int64)

def _precursor_keep_mask_np(mz, precursor_mz, bin_width, exclude_precursor):
    mz = np.asarray(mz, dtype=np.float64)
    keep = np.ones((mz.shape[0],), dtype=bool)
    if not bool(exclude_precursor):
        return keep

    try:
        pmz = float(precursor_mz)
    except Exception:
        return keep

    if not np.isfinite(pmz) or pmz <= 0:
        return keep

    try:
        tol_da = float(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.0"))
    except Exception:
        tol_da = 0.0
    try:
        isotope_n = int(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "0"))
    except Exception:
        isotope_n = 0

    if tol_da > 0.0:
        for iso_k in range(max(0, isotope_n) + 1):
            keep &= (np.abs(mz - (pmz + float(iso_k))) > float(tol_da))
    else:
        idx = _official_bin_indices_np(mz, bin_width)
        p_idx = int(_official_bin_indices_np(np.asarray([pmz], dtype=np.float64), bin_width)[0])
        keep &= (idx != p_idx)

    return keep


# Function overview: _bin_peaks_to_official_sparse handles a specific workflow step in this module.

def _bin_peaks_to_official_sparse(peaks, bin_width=0.01, max_mz=1005.0,
                                  exclude_precursor=False, precursor_mz=None):
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

    order = np.argsort(idx, kind='mergesort')
    idx_sorted = idx[order]
    int_sorted = intensity[order]
    uniq_idx, start = np.unique(idx_sorted, return_index=True)
    uniq_int = np.add.reduceat(int_sorted, start).astype(np.float32)
    return uniq_idx.astype(np.int32), uniq_int


# Function overview: _cosine_sparse handles a specific workflow step in this module.

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


# Function overview: _js_similarity_sparse handles a specific workflow step in this module.

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


# Function overview: _matched_intensity_coverage handles a specific workflow step in this module.

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


# Function overview: _topk_peak_recall handles a specific workflow step in this module.

def _topk_peak_recall(pred_idx, true_idx, true_val, k=20):
    if true_idx.size == 0:
        return float(np.nan)
    k = max(1, min(int(k), int(true_idx.size)))
    order = np.argsort(-true_val, kind='mergesort')[:k]
    true_top = set(int(x) for x in true_idx[order].tolist())
    if len(true_top) == 0:
        return float(np.nan)
    pred_set = set(int(x) for x in pred_idx.tolist())
    hit = len(true_top.intersection(pred_set))
    return float(hit / float(len(true_top)))


# Function overview: compute_batch_official_metrics handles a specific workflow step in this module.

def compute_batch_official_metrics(raw_batch, pred_spect, spect_bin_centers, metric_cfg, pred_exact_peaks=None, debug_ctx=None):
    """Compute official-style and diagnostic metrics for one validation batch."""
    pred_np = pred_spect.detach().cpu().numpy().astype(np.float32)

    true_raw_list = raw_batch.get('spect_raw', None)
    if true_raw_list is None:
        true_raw_list = raw_batch.get('spect', None)

    precursor_list = raw_batch.get('precursor_mz', None)
    if precursor_list is None:
        precursor_list = [None] * pred_np.shape[0]

    # Metric semantics used by this training script:
    # - official_cos_no_precursor: cosine on official bins after removing precursor bin.
    # - official_js_no_precursor: Jensen-Shannon similarity on official bins.
    # - cos_with_precursor / sqrt_cos_*: diagnostic cosine variants.
    # - matched_intensity_coverage: fraction of true intensity explained by predicted support.
    # - topk_peak_recall: recall over top-K true-intensity peaks.
    # - retrieval_*: sparse vectors used to compute hit@K retrieval metrics.
    out = {
        'official_cos_no_precursor': [],
        'official_js_no_precursor': [],
        'cos_with_precursor': [],
        'sqrt_cos_with_precursor': [],
        'sqrt_cos_no_precursor': [],
        'matched_intensity_coverage': [],
        'topk_peak_recall': [],
        'pred_exact_enabled': [],
        'official_metric_source': [],
        'pred_official_n': [],
        'true_official_n': [],
        'overlap_n': [],
        'false_pred_n': [],
        'pred_intensity_on_true_ratio': [],
        'retrieval_pred_sparse': [],
        'retrieval_true_sparse': [],
    }

    bin_width = float(metric_cfg.get('bin_width', 0.01))
    official_max_mz = float(metric_cfg.get('max_mz', 1005.0))
    official_bin_n = int(math.floor(float(official_max_mz) / float(bin_width))) + 1
    official_bin_centers = ((np.arange(official_bin_n, dtype=np.float64) + 0.5) * float(bin_width)).astype(np.float32)
    # Function overview: _idx_to_official_mz converts official bin index to bin-center m/z.
    def _idx_to_official_mz(idx_arr):
        idx_arr = np.asarray(idx_arr, dtype=np.int64)
        if idx_arr.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return ((idx_arr.astype(np.float64) + 0.5) * bin_width).astype(np.float32)

    # Function overview: _fmt_mz_summary formats min/p10/p50/p90/max for compact debug logs.
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

        metric_source_i = 'coarse_fallback'

        if pred_peaks_i is not None:
            pred_peaks = _to_sparse_peak_array(
                pred_peaks_i,
                spect_bin_centers=None,
                min_intensity=metric_cfg.get('pred_min_intensity', 1e-8),
            )
            pred_exact_used = 1
            metric_source_i = 'pred_exact_peaks'
        else:
            # Critical fix:
            # If pred_spect is official dense [B, official_bin_n],
            # convert dense index -> official m/z centers using 0.01 Da centers,
            # not the old 1Da/1024 coarse spect_bin centers.
            if pred_np.ndim == 2 and int(pred_np.shape[1]) == int(official_bin_n):
                pred_centers_i = official_bin_centers
                pred_exact_used = 2
                metric_source_i = 'official_dense'
            else:
                pred_centers_i = spect_bin_centers
                pred_exact_used = 0
                metric_source_i = 'coarse_fallback'

            pred_peaks = _to_sparse_peak_array(
                pred_np[i],
                spect_bin_centers=pred_centers_i,
                min_intensity=metric_cfg.get('pred_min_intensity', 1e-8),
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
            bin_width=metric_cfg['bin_width'],
            max_mz=metric_cfg['max_mz'],
            exclude_precursor=False,
            precursor_mz=precursor,
        )
        true_idx_with, true_val_with = _bin_peaks_to_official_sparse(
            true_peaks,
            bin_width=metric_cfg['bin_width'],
            max_mz=metric_cfg['max_mz'],
            exclude_precursor=False,
            precursor_mz=precursor,
        )

        pred_idx_no, pred_val_no = _bin_peaks_to_official_sparse(
            pred_peaks,
            bin_width=metric_cfg['bin_width'],
            max_mz=metric_cfg['max_mz'],
            exclude_precursor=metric_cfg.get('exclude_precursor', True),
            precursor_mz=precursor,
        )
        true_idx_no, true_val_no = _bin_peaks_to_official_sparse(
            true_peaks,
            bin_width=metric_cfg['bin_width'],
            max_mz=metric_cfg['max_mz'],
            exclude_precursor=metric_cfg.get('exclude_precursor', True),
            precursor_mz=precursor,
        )

        out['official_cos_no_precursor'].append(_cosine_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no, sqrt_intensity=False))
        out['official_js_no_precursor'].append(_js_similarity_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no))
        out['cos_with_precursor'].append(_cosine_sparse(pred_idx_with, pred_val_with, true_idx_with, true_val_with, sqrt_intensity=False))
        out['sqrt_cos_with_precursor'].append(_cosine_sparse(pred_idx_with, pred_val_with, true_idx_with, true_val_with, sqrt_intensity=True))
        out['sqrt_cos_no_precursor'].append(_cosine_sparse(pred_idx_no, pred_val_no, true_idx_no, true_val_no, sqrt_intensity=True))
        out['matched_intensity_coverage'].append(_matched_intensity_coverage(pred_idx_no, true_idx_no, true_val_no))
        out['topk_peak_recall'].append(_topk_peak_recall(pred_idx_no, true_idx_no, true_val_no, k=metric_cfg.get('topk_peak_recall_k', 20)))
        pred_set = set(int(x) for x in pred_idx_no.tolist())
        true_set = set(int(x) for x in true_idx_no.tolist())
        overlap_set = pred_set.intersection(true_set)
        overlap_n = len(overlap_set)
        false_pred_n = max(0, int(pred_idx_no.size) - int(overlap_n))

        # predicted intensity precision:
        # how much predicted intensity is assigned to true-support bins.
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
                pred_intensity_on_true_ratio = float('nan')
        else:
            pred_intensity_on_true_ratio = float('nan')

        out['pred_exact_enabled'].append(float(pred_exact_used))
        out['official_metric_source'].append(metric_source_i)
        out['pred_official_n'].append(float(pred_idx_no.size))
        out['true_official_n'].append(float(true_idx_no.size))
        out['overlap_n'].append(float(overlap_n))
        out['false_pred_n'].append(float(false_pred_n))
        out['pred_intensity_on_true_ratio'].append(float(pred_intensity_on_true_ratio))

        if debug_ctx is not None and bool(debug_ctx.get('enabled', False)):
            printed = int(debug_ctx.get('printed', 0))
            max_samples = max(0, int(debug_ctx.get('max_samples', 0)))
            if printed < max_samples:
                epoch_tag = debug_ctx.get('epoch', None)
                batch_tag = debug_ctx.get('batch', None)
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

                pred_top_order = np.argsort(-pred_val_no, kind='mergesort')[:20] if pred_val_no.size > 0 else np.asarray([], dtype=np.int64)
                true_top_order = np.argsort(-true_val_no, kind='mergesort')[:20] if true_val_no.size > 0 else np.asarray([], dtype=np.int64)
                pred_top20_mz = _idx_to_official_mz(pred_idx_no[pred_top_order]) if pred_top_order.size > 0 else np.zeros((0,), dtype=np.float32)
                true_top20_mz = _idx_to_official_mz(true_idx_no[true_top_order]) if true_top_order.size > 0 else np.zeros((0,), dtype=np.float32)

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
                debug_ctx['printed'] = printed + 1

        out['retrieval_pred_sparse'].append((pred_idx_no.astype(np.int32), pred_val_no.astype(np.float32)))
        out['retrieval_true_sparse'].append((true_idx_no.astype(np.int32), true_val_no.astype(np.float32)))

    return out


# Function overview: _compute_retrieval_hits handles a specific workflow step in this module.

def _build_true_official_dense_from_cached_sparse_batch(batch, batch_n, device, official_bin_n):
    out = torch.zeros((int(batch_n), int(official_bin_n)), dtype=torch.float32, device=device)
    idx_src = batch.get('true_official_idx', None)
    val_src = batch.get('true_official_intensity', None)
    used_cache = False

    if torch.is_tensor(idx_src) and torch.is_tensor(val_src):
        idx_t = idx_src.long()
        val_t = val_src.float()
        if idx_t.dim() == 1:
            idx_t = idx_t.unsqueeze(0)
        if val_t.dim() == 1:
            val_t = val_t.unsqueeze(0)
        if idx_t.dim() == 2 and val_t.dim() == 2:
            use_b = min(int(batch_n), int(idx_t.shape[0]), int(val_t.shape[0]))
            use_k = min(int(idx_t.shape[1]), int(val_t.shape[1]))
            if use_b > 0 and use_k > 0:
                idx_t = idx_t[:use_b, :use_k].to(device=device)
                val_t = val_t[:use_b, :use_k].to(device=device)
                valid = (
                    (idx_t >= 0)
                    & (idx_t < int(official_bin_n))
                    & torch.isfinite(val_t)
                    & (val_t > 0)
                )
                if bool(valid.any().item()):
                    idx_safe = idx_t.clamp(0, max(0, int(official_bin_n) - 1))
                    out[:use_b].scatter_add_(1, idx_safe, val_t * valid.float())
                    used_cache = True
        return out, used_cache

    if isinstance(idx_src, (list, tuple)) and isinstance(val_src, (list, tuple)):
        use_b = min(int(batch_n), len(idx_src), len(val_src))
        for bi in range(use_b):
            try:
                idx_i = np.asarray(idx_src[bi], dtype=np.int64).reshape(-1)
                val_i = np.asarray(val_src[bi], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if idx_i.size <= 0 or val_i.size <= 0:
                continue
            use_k = min(int(idx_i.shape[0]), int(val_i.shape[0]))
            idx_i = idx_i[:use_k]
            val_i = val_i[:use_k]
            valid_i = (
                (idx_i >= 0)
                & (idx_i < int(official_bin_n))
                & np.isfinite(val_i)
                & (val_i > 0)
            )
            if not np.any(valid_i):
                continue
            idx_t = torch.as_tensor(idx_i[valid_i], dtype=torch.long, device=device)
            val_t = torch.as_tensor(val_i[valid_i], dtype=torch.float32, device=device)
            out[bi].scatter_add_(0, idx_t, val_t)
            used_cache = True
    return out, used_cache

def _build_cached_true_top20_tensors(batch, batch_n, official_bin_n, device, default_k=20):
    k = max(1, int(default_k))
    out_idx = torch.full((int(batch_n), k), -1, dtype=torch.long, device=device)
    out_val = torch.zeros((int(batch_n), k), dtype=torch.float32, device=device)
    out_valid = torch.zeros((int(batch_n), k), dtype=torch.bool, device=device)
    used_cache = False

    idx_src = batch.get('true_top20_official_idx', None)
    val_src = batch.get('true_top20_official_intensity', None)
    if idx_src is None or val_src is None:
        return out_idx, out_val, out_valid, used_cache

    if torch.is_tensor(idx_src) and torch.is_tensor(val_src):
        idx_t = idx_src.long()
        val_t = val_src.float()
        if idx_t.dim() == 1:
            idx_t = idx_t.unsqueeze(0)
        if val_t.dim() == 1:
            val_t = val_t.unsqueeze(0)
        if idx_t.dim() != 2 or val_t.dim() != 2:
            return out_idx, out_val, out_valid, used_cache

        use_b = min(int(batch_n), int(idx_t.shape[0]), int(val_t.shape[0]))
        use_k = min(int(idx_t.shape[1]), int(val_t.shape[1]))
        if use_b <= 0 or use_k <= 0:
            return out_idx, out_val, out_valid, used_cache

        idx_t = idx_t[:use_b, :use_k].to(device=device)
        val_t = val_t[:use_b, :use_k].to(device=device)

        valid = (
            (idx_t >= 0)
            & (idx_t < int(official_bin_n))
            & torch.isfinite(val_t)
            & (val_t > 0)
        )
        if bool(valid.any().item()):
            order = torch.argsort(val_t, dim=-1, descending=True)
            idx_sorted = torch.gather(idx_t, 1, order)
            val_sorted = torch.gather(val_t, 1, order)
            valid_sorted = torch.gather(valid, 1, order)
            keep = min(k, int(idx_sorted.shape[1]))
            out_idx[:use_b, :keep] = idx_sorted[:, :keep]
            out_val[:use_b, :keep] = val_sorted[:, :keep]
            out_valid[:use_b, :keep] = valid_sorted[:, :keep]
            used_cache = True
        return out_idx, out_val, out_valid, used_cache

    if isinstance(idx_src, (list, tuple)) and isinstance(val_src, (list, tuple)):
        use_b = min(int(batch_n), len(idx_src), len(val_src))
        for bi in range(use_b):
            try:
                idx_i = np.asarray(idx_src[bi], dtype=np.int64).reshape(-1)
                val_i = np.asarray(val_src[bi], dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if idx_i.size <= 0 or val_i.size <= 0:
                continue
            use_n = min(int(idx_i.shape[0]), int(val_i.shape[0]))
            idx_i = idx_i[:use_n]
            val_i = val_i[:use_n]
            valid_i = (
                (idx_i >= 0)
                & (idx_i < int(official_bin_n))
                & np.isfinite(val_i)
                & (val_i > 0)
            )
            if not np.any(valid_i):
                continue
            idx_v = idx_i[valid_i]
            val_v = val_i[valid_i]
            order = np.argsort(-val_v, kind='stable')
            take = min(k, int(order.shape[0]))
            if take <= 0:
                continue
            sel = order[:take]
            out_idx[bi, :take] = torch.as_tensor(idx_v[sel], dtype=torch.long, device=device)
            out_val[bi, :take] = torch.as_tensor(val_v[sel], dtype=torch.float32, device=device)
            out_valid[bi, :take] = True
            used_cache = True
    return out_idx, out_val, out_valid, used_cache


def _get_teacher_formula_target_from_batch(batch):
    tq = batch.get('teacher_formula_probs', None)
    if not torch.is_tensor(tq):
        return None

    tq = tq.float()

    if tq.dim() == 1:
        tq = tq.unsqueeze(0)
    elif tq.dim() > 2:
        tq = tq.reshape(tq.shape[0], -1)

    # 优先用 formulae_mask 对齐 candidate 维度
    fm = batch.get('formulae_mask', None)
    if torch.is_tensor(fm):
        fm = fm.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(tq.shape[0]), int(fm.shape[0]))
        use_m = min(int(tq.shape[1]), int(fm.shape[1]))
        if use_b <= 0 or use_m <= 0:
            return None

        tq = tq[:use_b, :use_m]
        fm = fm[:use_b, :use_m]

        tq = torch.where(torch.isfinite(tq), tq, torch.zeros_like(tq))
        tq = tq.clamp_min(0.0)
        tq = tq * (fm > 0.5).float()

        row_sum = tq.sum(dim=-1, keepdim=True)
        bad = row_sum <= 1e-12

        if bool(bad.any().item()):
            # 极少数 teacher 全 0 的行，用 masked-uniform 保底，避免 KL 崩掉
            fallback = (fm > 0.5).float()
            fallback = fallback / fallback.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            tq = torch.where(
                bad.expand_as(tq),
                fallback,
                tq / row_sum.clamp_min(1e-12),
            )
        else:
            tq = tq / row_sum.clamp_min(1e-12)

        return tq

    # 没 mask 时也保证是合法分布
    tq = torch.where(torch.isfinite(tq), tq, torch.zeros_like(tq))
    tq = tq.clamp_min(0.0)
    row_sum = tq.sum(dim=-1, keepdim=True)
    valid = row_sum > 1e-12

    if bool(valid.any().item()):
        tq = torch.where(
            valid.expand_as(tq),
            tq / row_sum.clamp_min(1e-12),
            torch.full_like(tq, 1.0 / max(1, int(tq.shape[1]))),
        )
        return tq

    return None

def masked_prob_kl(pred_prob, target_q, mask, eps=1e-8):
    """
    KL(target_q || pred_prob), only for rows where target_q has nonzero mass.
    """
    pred_prob = pred_prob.clamp_min(eps)
    target_q = target_q.float().clamp_min(0.0)
    mask = (mask > 0.5).float()

    target_q = target_q * mask
    target_sum = target_q.sum(dim=-1, keepdim=True)

    valid = target_sum.squeeze(-1) > eps
    target_q = target_q / target_sum.clamp_min(eps)

    kl = target_q * (target_q.clamp_min(eps).log() - pred_prob.log())
    kl = kl.sum(dim=-1)

    if valid.any():
        return kl[valid].mean()

    return pred_prob.new_tensor(0.0)
def _compute_precursor_loss_from_batch(batch, res):
    """
    Precursor stability loss.
    Uses:
      res['precursor_logit']                 [B]
      batch['true_precursor_prob_in_all']    [B, 1] or [B]
    """
    if not isinstance(res, dict):
        return None

    precursor_logit = res.get("precursor_logit", None)
    precursor_target = batch.get("true_precursor_prob_in_all", None)

    if not (torch.is_tensor(precursor_logit) and torch.is_tensor(precursor_target)):
        return None

    logit = precursor_logit.float().reshape(-1)
    target = precursor_target.float().reshape(-1).to(device=logit.device)

    use_n = min(int(logit.shape[0]), int(target.shape[0]))
    if use_n <= 0:
        return None

    logit = logit[:use_n]
    target = target[:use_n].clamp(0.0, 1.0)

    valid = torch.isfinite(target) & torch.isfinite(logit)
    if not bool(valid.any().item()):
        return logit.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        logit[valid],
        target[valid],
        reduction="mean",
    )
def apply_teacher_topk_to_target(target_probs, formulae_mask=None, topk=0):
    if (not torch.is_tensor(target_probs)) or int(topk) <= 0:
        return target_probs, formulae_mask

    tp = target_probs.float()
    if tp.dim() == 1:
        tp = tp.unsqueeze(0)
    elif tp.dim() > 2:
        tp = tp.reshape(tp.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(tp.shape[0]), int(fm.shape[0]))
        use_m = min(int(tp.shape[1]), int(fm.shape[1]))
        tp = tp[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        fm = (fm > 0.5).float()
    else:
        fm = torch.ones_like(tp)

    kk = max(1, min(int(topk), int(tp.shape[1])))

    # topK by teacher probability
    score = tp.masked_fill(fm <= 0.5, -1e9)
    top_idx = torch.topk(score, k=kk, dim=-1).indices

    keep = torch.zeros_like(tp)
    keep.scatter_(1, top_idx, 1.0)
    keep = keep * fm

    # Critical fix:
    # teacher_topk_mask should not mark zero-probability candidates as positives.
    # Otherwise topK=128 with TARGET_SUPPORT_TOPK=16 creates many fake positive labels.
    try:
        pos_eps = float(os.environ.get("TEACHER_TOPK_POS_EPS", "1e-12"))
    except Exception:
        pos_eps = 1e-12

    positive = (tp > float(pos_eps)).float() * fm

    if os.environ.get("TEACHER_TOPK_MASK_POSITIVE_ONLY", "1") == "1":
        eff_mask = keep * positive

        # If a row somehow has no positive candidate after masking, fall back to keep.
        row_has_pos = eff_mask.sum(dim=-1, keepdim=True) > 0
        eff_mask = torch.where(row_has_pos, eff_mask, keep)
    else:
        eff_mask = keep

    tp_masked = tp * eff_mask

    row_sum = tp_masked.sum(dim=-1, keepdim=True)

    # fallback distribution over effective mask if probabilities are zero
    fallback = eff_mask / eff_mask.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    tp_out = torch.where(
        row_sum > 1e-12,
        tp_masked / row_sum.clamp_min(1e-12),
        fallback,
    )

    return tp_out, eff_mask
# Function overview: compute_formula_target_probs_from_batch builds candidate-level target probabilities over formula support.

def compute_formula_target_probs_from_batch(
    batch,
    bin_width=0.1,
    max_mz=1005.0,
    target_mode='exact_overlap',
    support_temperature=1.0,
    support_topk=0,
):
    use_teacher = str(os.environ.get('USE_TEACHER_FORMULA_TARGET', '1')).strip().lower() not in ('0', 'false', 'no', 'off')

    if use_teacher:
        teacher_target = _get_teacher_formula_target_from_batch(batch)
        if torch.is_tensor(teacher_target):
            return teacher_target
    # target_mode is kept in signature for compatibility, but target generation is now single-path:
    # official cached fields + quality_hybrid scoring.
    del target_mode

    off_idx = batch.get('formulae_peaks_official_idx_agg', None)
    off_int = batch.get('formulae_peaks_official_intensity_agg', None)
    if off_idx is None:
        off_idx = batch.get('formulae_peaks_official_idx', None)
    if off_int is None:
        off_int = get_formulae_official_intensity_from_batch(batch)

    if not (torch.is_tensor(off_idx) and torch.is_tensor(off_int)):
        return None

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)
    if off_idx.dim() != 3 or off_int.dim() != 3:
        return None

    batch_n = min(int(off_idx.shape[0]), int(off_int.shape[0]))
    formula_n = min(int(off_idx.shape[1]), int(off_int.shape[1]))
    peak_n = min(int(off_idx.shape[2]), int(off_int.shape[2]))
    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None

    device = off_idx.device
    off_idx = off_idx[:batch_n, :formula_n, :peak_n].long()
    off_int = off_int[:batch_n, :formula_n, :peak_n].float()

    formulae_mask = batch.get('formulae_mask', None)
    if torch.is_tensor(formulae_mask):
        mask = formulae_mask.float()
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], -1)
        use_b = min(batch_n, int(mask.shape[0]))
        use_m = min(formula_n, int(mask.shape[1]))
        off_idx = off_idx[:use_b, :use_m, :]
        off_int = off_int[:use_b, :use_m, :]
        mask = mask[:use_b, :use_m]
        batch_n = use_b
        formula_n = use_m
        peak_n = min(peak_n, int(off_idx.shape[2]), int(off_int.shape[2]))
    else:
        mask = torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)

    if batch_n <= 0 or formula_n <= 0 or peak_n <= 0:
        return None

    try:
        temp = float(support_temperature)
    except Exception:
        temp = 1.0
    if (not np.isfinite(temp)) or temp <= 0:
        temp = 1.0

    try:
        topk = int(support_topk)
    except Exception:
        topk = 0
    topk = max(0, topk)

    bwidth = float(max(1e-6, bin_width))
    max_mz = float(max(bwidth, max_mz))
    official_bin_n = int(math.floor(max_mz / bwidth)) + 1

    true_dense, used_cached_true = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )

    if not used_cached_true:
        raw_batch = batch.get('spect_raw', None)
        if not getattr(compute_formula_target_probs_from_batch, '_warned_raw_fallback', False):
            print(
                '[target] missing cached true_official_idx/intensity, fallback to spect_raw target build; '
                'rebuild cache via cache_featurizer_condv2 for official cache mode.',
                flush=True,
            )
            compute_formula_target_probs_from_batch._warned_raw_fallback = True
        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=raw_batch,
            precursor_mz=batch.get('precursor_mz', None),
            official_bin_width=bwidth,
            official_max_mz=max_mz,
            exclude_precursor=True,
            batch_n=batch_n,
            device=device,
        )

    true_norm = torch.sqrt((true_dense ** 2).sum(dim=-1).clamp_min(1e-12))
    true_support = true_dense > 0

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )
    safe_idx = off_idx.clamp(0, max(0, int(official_bin_n) - 1))

    true_at_peak = torch.gather(
        true_dense,
        1,
        safe_idx.reshape(batch_n, -1),
    ).reshape(batch_n, formula_n, peak_n)

    candidate_dot = (true_at_peak * off_int * valid_peak.float()).sum(dim=-1)
    cand_norm = torch.sqrt(((off_int * valid_peak.float()) ** 2).sum(dim=-1).clamp_min(1e-12))
    candidate_overlap = candidate_dot / (cand_norm * true_norm.unsqueeze(-1) + 1e-12)

    support_at_peak = torch.gather(
        true_support.float(),
        1,
        safe_idx.reshape(batch_n, -1),
    ).reshape(batch_n, formula_n, peak_n)
    overlap_support_score = (support_at_peak * valid_peak.float()).sum(dim=-1) / valid_peak.float().sum(dim=-1).clamp_min(1.0)

    # Candidate intensity precision:
    # Among this candidate's own rendered intensity, how much is assigned
    # to bins that actually exist in the true spectrum.
    # This penalizes candidates that touch a few true peaks but spray most
    # intensity onto false-support bins.
    candidate_int_precision_score = (
        support_at_peak * off_int * valid_peak.float()
    ).sum(dim=-1) / (
        off_int * valid_peak.float()
    ).sum(dim=-1).clamp_min(1e-8)
    top20_idx, top20_val, top20_valid, used_cached_top20 = _build_cached_true_top20_tensors(
        batch=batch,
        batch_n=batch_n,
        official_bin_n=official_bin_n,
        device=device,
        default_k=20,
    )
    if not used_cached_top20:
        if not getattr(compute_formula_target_probs_from_batch, '_warned_top20_fallback', False):
            print(
                '[target] missing cached true_top20_official_idx/intensity, fallback to dense topk(true_official).',
                flush=True,
            )
            compute_formula_target_probs_from_batch._warned_top20_fallback = True
        k20_dense = min(20, int(true_dense.shape[-1]))
        if k20_dense > 0:
            dense_top_val, dense_top_idx = torch.topk(true_dense, k=k20_dense, dim=-1)
            top20_idx[:, :k20_dense] = dense_top_idx
            top20_val[:, :k20_dense] = dense_top_val
            top20_valid[:, :k20_dense] = dense_top_val > 0

    try:
        weak_thr = float(os.environ.get('QUALITY_HYBRID_WEAK_INTENSITY_MAX', '0.05'))
    except Exception:
        weak_thr = 0.05
    if (not np.isfinite(weak_thr)) or weak_thr <= 0:
        weak_thr = 0.05

    k20 = int(top20_idx.shape[1])
    hit_top20_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    weak_hit_top20_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    hit_top20_intensity_score = torch.zeros((batch_n, formula_n), dtype=torch.float32, device=device)
    weak_top20_valid = top20_valid & (top20_val <= float(weak_thr))

    if k20 > 0:
        for kk in range(k20):
            tk = top20_idx[:, kk].view(batch_n, 1, 1)
            tk_valid = top20_valid[:, kk].view(batch_n, 1)
            hit_k = ((safe_idx == tk) & valid_peak).any(dim=-1).float()
            hit_top20_score += hit_k * tk_valid.float()
            hit_top20_intensity_score += hit_k * tk_valid.float() * top20_val[:, kk].view(batch_n, 1)
            tk_weak = weak_top20_valid[:, kk].view(batch_n, 1)
            weak_hit_top20_score += hit_k * tk_weak.float()
        hit_top20_score = hit_top20_score / float(max(1, k20))
        top20_int_den = (top20_val * top20_valid.float()).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        hit_top20_intensity_score = hit_top20_intensity_score / top20_int_den
        weak_den = weak_top20_valid.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        weak_hit_top20_score = weak_hit_top20_score / weak_den
        no_weak = weak_top20_valid.float().sum(dim=-1, keepdim=True) <= 0
        weak_hit_top20_score = torch.where(no_weak, hit_top20_score, weak_hit_top20_score)

    q1 = candidate_overlap * mask
    q2 = hit_top20_score * mask
    q3 = overlap_support_score * mask
    q4 = weak_hit_top20_score * mask
    q5 = hit_top20_intensity_score * mask
    q6 = candidate_int_precision_score * mask

    if os.environ.get('QUALITY_HYBRID_NORMALIZE', '1') == '1':
        def _row_minmax(x):
            x_safe = torch.where(mask > 0, x, torch.full_like(x, float('inf')))
            row_min = torch.amin(x_safe, dim=-1, keepdim=True)
            row_min = torch.where(torch.isfinite(row_min), row_min, torch.zeros_like(row_min))

            x_safe_max = torch.where(mask > 0, x, torch.full_like(x, float('-inf')))
            row_max = torch.amax(x_safe_max, dim=-1, keepdim=True)
            row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

            denom = (row_max - row_min).clamp_min(1e-8)
            out = (x - row_min) / denom
            return torch.where(mask > 0, out, torch.zeros_like(out))

        q1 = _row_minmax(q1)
        q2 = _row_minmax(q2)
        q3 = _row_minmax(q3)
        q4 = _row_minmax(q4)
        q5 = _row_minmax(q5)
        q6 = _row_minmax(q6)

    w1 = float(os.environ.get('QUALITY_HYBRID_W_COS', '0.7'))
    w2 = float(os.environ.get('QUALITY_HYBRID_W_HIT20', '0.2'))
    w3 = float(os.environ.get('QUALITY_HYBRID_W_OVERLAP', '0.1'))
    w4 = float(os.environ.get('QUALITY_HYBRID_W_WEAK20', '0.0'))
    w5 = float(os.environ.get('QUALITY_HYBRID_W_HIT20_INT', '0.0'))
    w6 = float(os.environ.get('QUALITY_HYBRID_W_PREC_INT', '0.0'))

    candidate_overlap = (
        w1 * q1
        + w2 * q2
        + w3 * q3
        + w4 * q4
        + w5 * q5
        + w6 * q6
    ) * mask
    # Optional tiny bonus for source-instance candidates.
    # This prevents set-cover from always selecting formula-only fallback rows
    # when source instance and fallback have identical peak templates.
    try:
        source_instance_teacher_bonus = float(os.environ.get("SOURCE_INSTANCE_TEACHER_BONUS", "0.0"))
    except Exception:
        source_instance_teacher_bonus = 0.0

    if source_instance_teacher_bonus > 0.0:
        inst_src = batch.get("formulae_instance_is_source", None)
        if torch.is_tensor(inst_src):
            inst_src = inst_src.to(device=candidate_overlap.device, dtype=candidate_overlap.dtype)
            if inst_src.dim() == 1:
                inst_src = inst_src.unsqueeze(0)
            if inst_src.dim() > 2:
                inst_src = inst_src.reshape(inst_src.shape[0], -1)

            use_b = min(int(candidate_overlap.shape[0]), int(inst_src.shape[0]))
            use_m = min(int(candidate_overlap.shape[1]), int(inst_src.shape[1]))

            bonus = torch.zeros_like(candidate_overlap)
            bonus[:use_b, :use_m] = inst_src[:use_b, :use_m] * float(source_instance_teacher_bonus)
            candidate_overlap = candidate_overlap + bonus * mask.float()
        # Keep the independent quality teacher before set-cover rewrites it.
    # This is the "q6 / quality-hybrid" teacher used for full-spectrum shape.
    independent_teacher_scores = candidate_overlap.float().clamp_min(0.0) * mask.float()
    independent_sum = independent_teacher_scores.sum(dim=-1, keepdim=True)

    independent_fallback = mask.float() / mask.float().sum(dim=-1, keepdim=True).clamp_min(1e-8)

    independent_teacher_probs = torch.where(
        independent_sum > 1e-12,
        independent_teacher_scores / independent_sum.clamp_min(1e-8),
        independent_fallback,
    )
        # Optional: set-cover residual teacher (fast vectorized pool version)
    # Greedy residual coverage over a prefiltered candidate pool.
    if str(os.environ.get('QUALITY_USE_SETCOVER_TEACHER', '0')).strip() == '1':
        with torch.no_grad():
            try:
                sc_topk = int(os.environ.get('QUALITY_SETCOVER_TOPK', '16'))
            except Exception:
                sc_topk = 16

            try:
                pool_k = int(os.environ.get('QUALITY_SETCOVER_POOL_TOPK', '1024'))
            except Exception:
                pool_k = 1024

            try:
                lambda_false = float(os.environ.get('QUALITY_SETCOVER_LAMBDA_FALSE', '0.5'))
            except Exception:
                lambda_false = 0.5

            try:
                lambda_redun = float(os.environ.get('QUALITY_SETCOVER_LAMBDA_REDUN', '0.2'))
            except Exception:
                lambda_redun = 0.2

            try:
                min_gain = float(os.environ.get('QUALITY_SETCOVER_MIN_GAIN', '1e-8'))
            except Exception:
                min_gain = 1e-8

            pool_k = max(1, min(int(pool_k), int(formula_n)))
            sc_topk = max(1, min(int(sc_topk), int(pool_k)))

            neg_val = _neg_mask_fill_value(candidate_overlap)
            rank_score = candidate_overlap.masked_fill(mask <= 0, neg_val)

            pool_idx_all = torch.topk(rank_score, k=pool_k, dim=-1).indices
            sel_probs = torch.zeros_like(candidate_overlap)

            for bi in range(int(batch_n)):
                pool_idx = pool_idx_all[bi]  # [P]
                pool_valid = mask[bi, pool_idx] > 0

                idx_pool = off_idx[bi, pool_idx].long()            # [P, K]
                int_pool = off_int[bi, pool_idx].float()           # [P, K]
                valid_pool = valid_peak[bi, pool_idx].bool()       # [P, K]

                valid_pool = (
                    valid_pool
                    & (idx_pool >= 0)
                    & (idx_pool < int(official_bin_n))
                    & torch.isfinite(int_pool)
                    & (int_pool > 0)
                    & pool_valid.unsqueeze(-1)
                )

                idx_safe = idx_pool.clamp(0, max(0, int(official_bin_n) - 1))
                int_pool = int_pool * valid_pool.float()

                cand_tot = int_pool.sum(dim=-1)
                true_hit = (true_at_peak[bi, pool_idx].float() * int_pool).sum(dim=-1)
                cand_false_mass = (cand_tot - true_hit).clamp_min(0.0)

                residual = true_dense[bi].float().clone()
                selected_bins_dense = torch.zeros_like(residual)

                selected_local = []
                selected_gain_vals = []
                selected_mask = torch.zeros((pool_k,), dtype=torch.bool, device=device)

                for step in range(sc_topk):
                    # residual gain for every candidate in the pool
                    res_vals = residual[idx_safe]  # [P, K]
                    gain = torch.minimum(res_vals, int_pool).sum(dim=-1)

                    # redundancy with already selected bins
                    sel_vals = selected_bins_dense[idx_safe]
                    redun = (sel_vals * int_pool).sum(dim=-1)

                    score = (
                        gain
                        - float(lambda_false) * cand_false_mass
                        - float(lambda_redun) * redun
                        + 1e-4 * rank_score[bi, pool_idx].clamp_min(0.0)
                    )

                    score = score.masked_fill(~pool_valid, neg_val)
                    score = score.masked_fill(selected_mask, neg_val)
                    score = score.masked_fill(cand_tot <= 0, neg_val)

                    best_score, best_local_t = torch.max(score, dim=0)

                    if not bool(torch.isfinite(best_score).item()):
                        break

                    # 如果第一步之后 residual gain 已经没了，就停止。
                    # 第一步允许选一个最优候选，避免空 teacher。
                    best_local = int(best_local_t.detach().item())
                    best_gain = float(gain[best_local].detach().item())
                    if step > 0 and best_gain <= float(min_gain):
                        break

                    selected_mask[best_local] = True
                    selected_local.append(best_local)
                    selected_gain_vals.append(gain[best_local].clamp_min(0.0))

                    sel_valid = valid_pool[best_local]
                    if bool(sel_valid.any().item()):
                        sel_idxs = idx_safe[best_local, sel_valid]
                        sel_ints = int_pool[best_local, sel_valid]

                        delta = torch.zeros_like(residual)
                        delta.scatter_add_(0, sel_idxs, sel_ints)
                        residual = torch.clamp(residual - delta, min=0.0)

                        selected_bins_dense.scatter_add_(0, sel_idxs, sel_ints)

                if len(selected_local) > 0:
                    selected_orig_idx = pool_idx[
                        torch.as_tensor(selected_local, dtype=torch.long, device=device)
                    ]

                    gains = torch.stack(selected_gain_vals).float()
                    if float(gains.sum().detach().item()) <= 1e-12:
                        gains = candidate_overlap[bi, selected_orig_idx].float().clamp_min(1e-8)

                    weights = gains / gains.sum().clamp_min(1e-8)
                    sel_probs[bi, selected_orig_idx] = weights

            # Fallback: if a row selected nothing, keep original independent teacher.
                        # Fallback: if a row selected nothing, keep original independent teacher.
            sel_sum = sel_probs.sum(dim=-1, keepdim=True)

            if os.environ.get("QUALITY_SETCOVER_HYBRID", "0").strip() == "1":
                try:
                    independent_w = float(os.environ.get("QUALITY_SETCOVER_HYBRID_Q6_WEIGHT", "0.7"))
                except Exception:
                    independent_w = 0.7
                independent_w = float(np.clip(independent_w, 0.0, 1.0))

                setcover_probs = torch.where(
                    sel_sum > 0,
                    sel_probs / sel_sum.clamp_min(1e-8),
                    independent_teacher_probs,
                )

                hybrid_probs = (
                    independent_w * independent_teacher_probs
                    + (1.0 - independent_w) * setcover_probs
                )
                hybrid_probs = hybrid_probs * mask.float()
                hybrid_probs = hybrid_probs / hybrid_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

                candidate_overlap = hybrid_probs
            else:
                candidate_overlap = torch.where(sel_sum > 0, sel_probs, candidate_overlap)
    # new: explicit target sharpening
    try:
        target_gamma = float(os.environ.get('QUALITY_TARGET_GAMMA', '2.0'))
    except Exception:
        target_gamma = 2.0
    if (not np.isfinite(target_gamma)) or target_gamma <= 0:
        target_gamma = 1.5

    # 保留原来的 support_temperature 语义，再叠加一个显式 gamma
    eff_gamma = float(target_gamma) * float(1.0 / temp)

    if abs(eff_gamma - 1.0) > 1e-8:
        positive = candidate_overlap > 0
        candidate_overlap = torch.where(
            positive,
            torch.pow(candidate_overlap.clamp_min(1e-12), eff_gamma),
            candidate_overlap * 0.0,
        )
    if topk > 0 and int(candidate_overlap.shape[1]) > topk:
        k = int(min(topk, int(candidate_overlap.shape[1])))
        rank_score = candidate_overlap.masked_fill(mask <= 0, _neg_mask_fill_value(candidate_overlap))
        top_idx = torch.topk(rank_score, k=k, dim=-1).indices
        keep = torch.zeros_like(candidate_overlap)
        keep.scatter_(1, top_idx, 1.0)
        candidate_overlap = candidate_overlap * keep

    overlap_sum = candidate_overlap.sum(dim=-1, keepdim=True)
    valid_sum = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
    target_probs = torch.where(
        overlap_sum > 0,
        candidate_overlap / overlap_sum.clamp_min(1e-8),
        mask / valid_sum,
    )
    return target_probs


def build_candidate_local_quality_target(
    batch,
    formulae_mask,
    official_bin_n,
    true_key_idx='true_all_official_idx',
    true_key_int='true_all_official_intensity',
    top20_key_idx='true_top20_official_idx',
    eps=1e-8,
):
    """
    Returns:
      quality: [B, M], float in [0, 1]
      pos_label: [B, M], 0/1
      valid_mask: [B, M], 0/1
    """
    device = formulae_mask.device
    B, M = formulae_mask.shape

    off_idx = batch.get('formulae_peaks_official_idx', None)
    off_int = batch.get('formulae_peaks_official_intensity', None)

    if off_idx is None:
        off_idx = batch.get('formulae_peaks_mass_idx', None)
    if off_int is None:
        off_int = batch.get('formulae_peaks_intensity', None)

    if off_idx is None or off_int is None:
        quality = torch.zeros((B, M), dtype=torch.float32, device=device)
        return quality, quality, formulae_mask.float(), {}

    off_idx = off_idx.to(device=device, dtype=torch.long)
    off_int = off_int.to(device=device, dtype=torch.float32)

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    K = min(off_idx.shape[-1], off_int.shape[-1])
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    valid_peak = (
        (off_idx >= 0)
        & (off_idx < int(official_bin_n))
        & torch.isfinite(off_int)
        & (off_int > 0)
    )

    off_int = torch.where(valid_peak, off_int.clamp_min(0.0), torch.zeros_like(off_int))

    off_int_norm = off_int / off_int.sum(dim=-1, keepdim=True).clamp_min(eps)

    true_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    true_idx_list = batch.get(true_key_idx, None)
    true_int_list = batch.get(true_key_int, None)

    if isinstance(true_idx_list, (list, tuple)) and isinstance(true_int_list, (list, tuple)):
        for b in range(min(B, len(true_idx_list), len(true_int_list))):
            ti = true_idx_list[b]
            tv = true_int_list[b]
            if ti is None or tv is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            if not torch.is_tensor(tv):
                tv = torch.as_tensor(tv)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            tv = tv.to(device=device, dtype=torch.float32).reshape(-1)
            n = min(ti.numel(), tv.numel())
            if n <= 0:
                continue
            ti = ti[:n]
            tv = tv[:n]
            keep = (ti >= 0) & (ti < official_bin_n) & torch.isfinite(tv) & (tv > 0)
            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, ti[keep], tv[keep].clamp_min(0.0))
    elif torch.is_tensor(true_idx_list) and torch.is_tensor(true_int_list):
        ti = true_idx_list.to(device=device, dtype=torch.long)
        tv = true_int_list.to(device=device, dtype=torch.float32)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        if tv.dim() == 1:
            tv = tv.unsqueeze(0)
        for b in range(min(B, ti.shape[0], tv.shape[0])):
            idx = ti[b].reshape(-1)
            val = tv[b].reshape(-1)
            n = min(idx.numel(), val.numel())
            idx = idx[:n]
            val = val[:n]
            keep = (idx >= 0) & (idx < official_bin_n) & torch.isfinite(val) & (val > 0)
            if bool(keep.any().item()):
                true_dense[b].scatter_add_(0, idx[keep], val[keep].clamp_min(0.0))

    true_dense = true_dense / true_dense.sum(dim=-1, keepdim=True).clamp_min(eps)
    true_support = (true_dense > 0).float()

    # Allow small official-bin mismatch when building selector target.
    # Exact 0.01-bin matching is too strict for candidate-local supervision.
    try:
        selector_target_bin_tol = int(os.environ.get("SELECTOR_TARGET_BIN_TOL", "1"))
    except Exception:
        selector_target_bin_tol = 1

    selector_target_bin_tol = max(0, int(selector_target_bin_tol))

    if selector_target_bin_tol > 0:
        ksz = 2 * selector_target_bin_tol + 1

        # For intensity overlap, max-pool is intentionally used:
        # a candidate peak gets credit if it falls near a true bin.
        true_dense_for_match = F.max_pool1d(
            true_dense.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)

        true_support_for_match = F.max_pool1d(
            true_support.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)
    else:
        true_dense_for_match = true_dense
        true_support_for_match = true_support

    idx_safe = off_idx.clamp(0, official_bin_n - 1)
    true_at_candidate_bins_exact = torch.gather(
        true_dense.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    support_at_candidate_bins_exact = torch.gather(
        true_support.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    true_at_candidate_bins_tol = torch.gather(
        true_dense_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    support_at_candidate_bins_tol = torch.gather(
        true_support_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )

    overlap_intensity_exact = (
        off_int_norm * true_at_candidate_bins_exact * valid_peak.float()
    ).sum(dim=-1)
    overlap_intensity_tol = (
        off_int_norm * true_at_candidate_bins_tol * valid_peak.float()
    ).sum(dim=-1)
    hit_support_mass_tol = (
        off_int_norm * support_at_candidate_bins_tol * valid_peak.float()
    ).sum(dim=-1)
    false_support_mass_exact = (
        off_int_norm * (1.0 - support_at_candidate_bins_exact) * valid_peak.float()
    ).sum(dim=-1)

    top20_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    top20_idx_list = batch.get(top20_key_idx, None)
    if isinstance(top20_idx_list, (list, tuple)):
        for b in range(min(B, len(top20_idx_list))):
            ti = top20_idx_list[b]
            if ti is None:
                continue
            if not torch.is_tensor(ti):
                ti = torch.as_tensor(ti)
            ti = ti.to(device=device, dtype=torch.long).reshape(-1)
            keep = (ti >= 0) & (ti < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, ti[keep]] = 1.0
    elif torch.is_tensor(top20_idx_list):
        ti = top20_idx_list.to(device=device, dtype=torch.long)
        if ti.dim() == 1:
            ti = ti.unsqueeze(0)
        for b in range(min(B, ti.shape[0])):
            idx = ti[b].reshape(-1)
            keep = (idx >= 0) & (idx < official_bin_n)
            if bool(keep.any().item()):
                top20_dense[b, idx[keep]] = 1.0

    if selector_target_bin_tol > 0:
        ksz = 2 * selector_target_bin_tol + 1
        top20_dense_for_match = F.max_pool1d(
            top20_dense.unsqueeze(1),
            kernel_size=ksz,
            stride=1,
            padding=selector_target_bin_tol,
        ).squeeze(1)
    else:
        top20_dense_for_match = top20_dense
    top20_at_candidate_bins = torch.gather(
        top20_dense_for_match.unsqueeze(1).expand(B, M, official_bin_n),
        2,
        idx_safe,
    )
    hit_top20_mass = (off_int_norm * top20_at_candidate_bins * valid_peak.float()).sum(dim=-1)

    try:
        w_overlap = float(os.environ.get("SELECTOR_QUALITY_W_OVERLAP", "1.50"))
    except Exception:
        w_overlap = 1.50

    try:
        w_support = float(os.environ.get("SELECTOR_QUALITY_W_SUPPORT", "0.35"))
    except Exception:
        w_support = 0.35

    try:
        w_top20 = float(os.environ.get("SELECTOR_QUALITY_W_TOP20", "0.75"))
    except Exception:
        w_top20 = 0.75

    try:
        w_false = float(os.environ.get("SELECTOR_QUALITY_W_FALSE", "1.20"))
    except Exception:
        w_false = 1.20

    # ------------------------------------------------------------------
    # Absolute clean selector target.
    # Do NOT row-minmax bad candidates into positives.
    # ------------------------------------------------------------------

    exact_support_mass = (1.0 - false_support_mass_exact).clamp(0.0, 1.0)

    quality_raw = (
        w_overlap * overlap_intensity_exact
        + 0.40 * overlap_intensity_tol
        + w_support * hit_support_mass_tol
        + w_top20 * hit_top20_mass
        - w_false * false_support_mass_exact
    )

    # A multiplicative clean gate is much stronger than only subtracting false mass.
    try:
        clean_gamma = float(os.environ.get("SELECTOR_CLEAN_GAMMA", "2.0"))
    except Exception:
        clean_gamma = 2.0

    clean_gate = exact_support_mass.clamp(0.0, 1.0) ** clean_gamma

    # Final score: must be positive and clean.
    quality_score = quality_raw.clamp_min(0.0) * clean_gate

    # Absolute positive filters.
    try:
        min_exact_support = float(os.environ.get("SELECTOR_POS_MIN_EXACT_SUPPORT", "0.20"))
    except Exception:
        min_exact_support = 0.20

    try:
        min_tol_support = float(os.environ.get("SELECTOR_POS_MIN_TOL_SUPPORT", "0.25"))
    except Exception:
        min_tol_support = 0.25

    try:
        max_false_support = float(os.environ.get("SELECTOR_POS_MAX_FALSE_SUPPORT", "0.80"))
    except Exception:
        max_false_support = 0.80

    strict_keep = (
        (formulae_mask > 0.5)
        & (quality_score > 0.0)
        & (exact_support_mass >= min_exact_support)
        & (hit_support_mass_tol >= min_tol_support)
        & (false_support_mass_exact <= max_false_support)
    )

    quality_score = torch.where(
        strict_keep,
        quality_score,
        torch.zeros_like(quality_score),
    )

    # Normalize by row max only. No row-min subtraction.
    row_max = quality_score.max(dim=1, keepdim=True).values
    quality = quality_score / row_max.clamp_min(eps)
    quality = torch.where(
        row_max > eps,
        quality,
        torch.zeros_like(quality),
    )
    quality = quality * formulae_mask.float()

    try:
        target_support_topk = int(os.environ.get("TARGET_SUPPORT_TOPK", "64"))
    except Exception:
        target_support_topk = 64

    try:
        target_min_pos = int(os.environ.get("TARGET_MIN_POS", "8"))
    except Exception:
        target_min_pos = 8

    k = max(1, min(target_support_topk, M))

    # First choose topK among clean candidates.
    masked_quality = quality.masked_fill(formulae_mask <= 0.5, -1e9)
    top_idx = torch.topk(masked_quality, k=k, dim=1).indices

    pos_label = torch.zeros_like(quality)
    pos_label.scatter_(1, top_idx, 1.0)
    pos_label = pos_label * strict_keep.float()

    # Fallback: if a row has too few strict positives, add the best fallback candidates.
    # This avoids empty KL rows, but still prevents top64 garbage positives.
    row_pos_n = pos_label.sum(dim=1, keepdim=True)

    fallback_score = (
        0.50 * exact_support_mass
        + 0.35 * hit_support_mass_tol
        + 0.15 * hit_top20_mass
    ) * formulae_mask.float()

    fallback_score = fallback_score.masked_fill(formulae_mask <= 0.5, -1e9)

    fb_k = max(1, min(target_min_pos, M))
    fb_idx = torch.topk(fallback_score, k=fb_k, dim=1).indices
    fb_label = torch.zeros_like(pos_label)
    fb_label.scatter_(1, fb_idx, 1.0)
    fb_label = fb_label * formulae_mask.float()

    need_fb = (row_pos_n < float(target_min_pos)).float()
    pos_label = torch.where(
        need_fb > 0.5,
        torch.maximum(pos_label, fb_label),
        pos_label,
    )

    # But final pos should never exceed target_support_topk.
    if target_support_topk < M:
        pos_quality = quality.masked_fill(pos_label <= 0.5, -1e9)
        keep_idx = torch.topk(pos_quality, k=k, dim=1).indices
        keep_label = torch.zeros_like(pos_label)
        keep_label.scatter_(1, keep_idx, 1.0)
        pos_label = pos_label * keep_label

    has_signal = (pos_label.sum(dim=1, keepdim=True) > 0).float()
    valid_mask = formulae_mask.float() * has_signal

    return quality, pos_label, valid_mask, {
        'overlap_intensity_exact': overlap_intensity_exact.detach(),
        'overlap_intensity_tol': overlap_intensity_tol.detach(),
        'hit_support_mass_tol': hit_support_mass_tol.detach(),
        'hit_top20_mass': hit_top20_mass.detach(),
        'false_support_mass_exact': false_support_mass_exact.detach(),
        'exact_support_mass': exact_support_mass.detach(),
        'strict_keep': strict_keep.float().detach(),
        'quality_score': quality_score.detach(),
    }


def build_setcover_rerank_target_from_quality(
    selector_quality,
    pool_idx,
    formulae_mask,
    eps=1e-8,
):
    """
    Simplified setcover target: normalize candidate-local quality within pool.
    """
    B, K = pool_idx.shape
    pool_quality = torch.gather(selector_quality, 1, pool_idx)
    pool_mask = torch.gather(formulae_mask.float(), 1, pool_idx)

    target = pool_quality.clamp_min(0.0) * pool_mask
    target = target / target.sum(dim=1, keepdim=True).clamp_min(eps)

    pos = torch.zeros_like(target)
    try:
        rerank_pos_k = int(os.environ.get("QUALITY_SETCOVER_TOPK", "24"))
    except Exception:
        rerank_pos_k = 24
    kk = max(1, min(rerank_pos_k, K))

    top_idx = torch.topk(target, k=kk, dim=1).indices
    pos.scatter_(1, top_idx, 1.0)
    pos = pos * pool_mask

    return target, pos, pool_mask


def compute_selector_quality_metrics(selector_logits, selector_quality, formulae_mask, ks=(32, 64, 128, 256)):
    out = {}
    valid_quality = selector_quality * formulae_mask.float()

    for k in ks:
        kk = min(k, selector_logits.shape[1])
        idx = torch.topk(
            selector_logits.masked_fill(formulae_mask <= 0.5, -1e9),
            k=kk,
            dim=1,
        ).indices

        q_top = torch.gather(valid_quality, 1, idx)
        pos_top = (q_top > 0.3).float()

        out[f'selector_quality_mean_at_{k}'] = q_top.mean().detach()
        out[f'selector_precision_at_{k}'] = pos_top.mean().detach()

    return out

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

    true_dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
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


def _masked_candidate_kl(scores_tensor, target_probs, formulae_mask=None):
    if (not torch.is_tensor(scores_tensor)) or (not torch.is_tensor(target_probs)):
        return None

    use_b = min(int(scores_tensor.shape[0]), int(target_probs.shape[0]))
    use_m = min(int(scores_tensor.shape[1]), int(target_probs.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None

    scores_use = scores_tensor[:use_b, :use_m]
    target_use = target_probs[:use_b, :use_m].to(device=scores_use.device, dtype=scores_use.dtype)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(use_b, int(fm.shape[0]))
        use_m = min(use_m, int(fm.shape[1]))
        scores_use = scores_use[:use_b, :use_m]
        target_use = target_use[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        scores_use = scores_use.masked_fill(fm <= 0, _neg_mask_fill_value(scores_use))

    valid_rows = torch.isfinite(target_use).all(dim=-1) & (target_use.sum(dim=-1) > 0)
    if not bool(valid_rows.any().item()):
        return scores_use.sum() * 0.0

    per_row_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(scores_use, dim=-1),
        target_use,
        reduction='none',
    ).sum(dim=-1)
    return per_row_kl[valid_rows].mean()

def _group_masked_candidate_kl(scores_tensor, target_probs, formulae_mask=None, group_id=None, eps=1e-8):
    """
    Group-aware KL for V2 source-instance candidates.

    Instead of:
      KL(target_instance || softmax(instance_scores))

    Use:
      group_score[g] = max score over instances with the same formula group
      target_group[g] = sum target mass over instances with the same formula group
      KL(target_group || softmax(group_score))

    This removes duplicate-source multiplicity pressure from rerank training.
    """
    if (
        not torch.is_tensor(scores_tensor)
        or not torch.is_tensor(target_probs)
        or not torch.is_tensor(group_id)
    ):
        return None

    scores = scores_tensor.float()
    target = target_probs.float()
    gid = group_id.long().to(device=scores.device)

    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if target.dim() == 1:
        target = target.unsqueeze(0)
    elif target.dim() > 2:
        target = target.reshape(target.shape[0], -1)

    if gid.dim() == 1:
        gid = gid.unsqueeze(0)
    elif gid.dim() > 2:
        gid = gid.reshape(gid.shape[0], -1)

    B = min(int(scores.shape[0]), int(target.shape[0]), int(gid.shape[0]))
    M = min(int(scores.shape[1]), int(target.shape[1]), int(gid.shape[1]))
    if B <= 0 or M <= 0:
        return None

    scores = scores[:B, :M]
    target = target[:B, :M].to(device=scores.device, dtype=scores.dtype)
    gid = gid[:B, :M].to(device=scores.device, dtype=torch.long)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(B, int(fm.shape[0]))
        use_m = min(M, int(fm.shape[1]))

        valid = torch.zeros((B, M), dtype=torch.bool, device=scores.device)
        valid[:use_b, :use_m] = fm[:use_b, :use_m].to(device=scores.device) > 0.5
    else:
        valid = torch.ones((B, M), dtype=torch.bool, device=scores.device)

    valid = valid & torch.isfinite(scores) & torch.isfinite(target)
    target = torch.where(torch.isfinite(target), target, torch.zeros_like(target)).clamp_min(0.0)

    losses = []
    neg = _neg_mask_fill_value(scores)

    for bi in range(B):
        valid_b = valid[bi]
        if not bool(valid_b.any().item()):
            continue

        idx_valid = torch.nonzero(valid_b, as_tuple=False).reshape(-1)
        s_b = scores[bi, idx_valid]
        t_b = target[bi, idx_valid].clamp_min(0.0)
        gid_b = gid[bi, idx_valid]

        if float(t_b.sum().detach().item()) <= float(eps):
            continue

        # Critical: group ids are original ids and may be > M after rerank pruning.
        # Do NOT clamp. Remap per row.
        uniq_gid, inv = torch.unique(gid_b, sorted=False, return_inverse=True)
        group_n = int(uniq_gid.shape[0])
        if group_n <= 0:
            continue

        group_scores = s_b.new_full((group_n,), neg)
        group_scores.scatter_reduce_(
            dim=0,
            index=inv,
            src=s_b,
            reduce="amax",
            include_self=True,
        )

        group_target = s_b.new_zeros((group_n,))
        group_target.scatter_add_(0, inv, t_b)
        group_target = group_target / group_target.sum().clamp_min(float(eps))

        logp = torch.log_softmax(group_scores, dim=0)
        kl = group_target * (group_target.clamp_min(float(eps)).log() - logp)
        losses.append(kl.sum())

    if len(losses) <= 0:
        return scores.sum() * 0.0

    return torch.stack(losses).mean()

def _masked_formula_entropy_loss(scores_tensor, formulae_mask=None):
    """
    Encourage formula probability distribution to be less diffuse.

    This is softer than hard FORMULA_RENDER_TOPK.
    It penalizes high entropy among valid candidates.
    """
    if not torch.is_tensor(scores_tensor):
        return None

    scores = scores_tensor.float()
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(int(scores.shape[0]), int(fm.shape[0]))
        use_m = min(int(scores.shape[1]), int(fm.shape[1]))
        scores = scores[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        valid = fm > 0.5
        scores = scores.masked_fill(~valid, _neg_mask_fill_value(scores))
        valid_n = valid.float().sum(dim=-1).clamp_min(2.0)
    else:
        valid_n = torch.full(
            (scores.shape[0],),
            float(max(2, int(scores.shape[1]))),
            dtype=scores.dtype,
            device=scores.device,
        )

    prob = torch.softmax(scores, dim=-1)
    log_prob = torch.log(prob.clamp_min(1e-12))
    ent = -(prob * log_prob).sum(dim=-1)

    # normalize to [0,1] roughly
    ent = ent / torch.log(valid_n).clamp_min(1e-12)

    valid_rows = torch.isfinite(ent)
    if not bool(valid_rows.any().item()):
        return scores.sum() * 0.0

    return ent[valid_rows].mean()
def _renormalize_target_probs(target_probs, formulae_mask=None):
    if not torch.is_tensor(target_probs):
        return None

    tp = target_probs.float()
    if tp.dim() == 1:
        tp = tp.unsqueeze(0)
    elif tp.dim() > 2:
        tp = tp.reshape(tp.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(int(tp.shape[0]), int(fm.shape[0]))
        use_m = min(int(tp.shape[1]), int(fm.shape[1]))
        tp = tp[:use_b, :use_m]
        fm = (fm[:use_b, :use_m] > 0.5).float()
    else:
        fm = torch.ones_like(tp)

    tp = torch.where(torch.isfinite(tp), tp, torch.zeros_like(tp))
    tp = tp.clamp_min(0.0) * fm
    row_sum = tp.sum(dim=-1, keepdim=True)

    fallback = fm / fm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    tp = torch.where(row_sum > 1e-12, tp / row_sum.clamp_min(1e-12), fallback)
    return tp


def _zero_precursor_bin_dense_batch(dense_spect, precursor_mz, bin_width):
    if (not torch.is_tensor(dense_spect)) or precursor_mz is None:
        return dense_spect

    out = dense_spect.clone()
    if not torch.is_tensor(precursor_mz):
        precursor_mz = torch.as_tensor(precursor_mz)

    pmz = precursor_mz.to(device=out.device, dtype=torch.float32).reshape(-1)
    if pmz.shape[0] < int(out.shape[0]):
        pad = torch.zeros((int(out.shape[0]) - int(pmz.shape[0]),), dtype=torch.float32, device=out.device)
        pmz = torch.cat([pmz, pad], dim=0)
    pmz = pmz[: int(out.shape[0])]

    bin_idx = torch.floor(pmz / float(bin_width) + 1e-8).long()
    valid = torch.isfinite(pmz) & (bin_idx >= 0) & (bin_idx < int(out.shape[1]))
    if bool(valid.any().item()):
        row_idx = torch.arange(int(out.shape[0]), device=out.device)
        out[row_idx[valid], bin_idx[valid]] = 0.0
    return out


def _normalize_dense_prob(x):
    x = x.float().clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _sqrt_cosine_loss_dense(pred, true):
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)
    pred_s = torch.sqrt(pred.clamp_min(1e-12))
    true_s = torch.sqrt(true.clamp_min(1e-12))
    num = (pred_s * true_s).sum(dim=-1)
    den = torch.norm(pred_s, dim=-1) * torch.norm(true_s, dim=-1)
    cos = num / den.clamp_min(1e-12)
    return (1.0 - cos).mean()


def _cosine_loss_dense(pred, true):
    """
    Dense cosine loss aligned with official_cos_no_precursor.

    Note:
    cosine is scale-invariant, so per-row normalization does not change
    vector direction; it only stabilizes numerical scale.
    """
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)

    num = (pred * true).sum(dim=-1)
    den = torch.norm(pred, dim=-1) * torch.norm(true, dim=-1)
    cos = num / den.clamp_min(1e-12)
    return (1.0 - cos).mean()

def _dense_kl_loss(pred, true):
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)
    return F.kl_div(
        torch.log(pred.clamp_min(1e-12)),
        true,
        reduction='batchmean',
    )


def _false_support_mass_loss_dense(pred_official, true_official, true_eps=1e-12):
    """
    Penalize predicted intensity mass assigned to bins outside true support.

    This directly targets the current failure:
      val_pred_int_on_true is low,
      false predicted support is huge.
    """
    if (not torch.is_tensor(pred_official)) or (not torch.is_tensor(true_official)):
        return None

    use_b = min(int(pred_official.shape[0]), int(true_official.shape[0]))
    use_m = min(int(pred_official.shape[1]), int(true_official.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None

    pred = pred_official[:use_b, :use_m].float().clamp_min(0.0)
    true = true_official[:use_b, :use_m].float().clamp_min(0.0)

    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    true_support = (true > float(true_eps)).float()

    false_mass = (pred * (1.0 - true_support)).sum(dim=-1)

    valid_rows = torch.isfinite(false_mass)
    if not bool(valid_rows.any().item()):
        return pred.sum() * 0.0

    return false_mass[valid_rows].mean()


def compute_official_dense_spectral_loss(pred_official, true_official, kl_weight=0.2):
    """
    Official dense spectral loss.

    OFFICIAL_SPECTRAL_LOSS_MODE:
      - cos: optimize standard cosine, aligned with official_cos_no_precursor
      - sqrt: old behavior, sqrt-cosine
      - mix: weighted mix of standard cosine and sqrt-cosine
    """
    mode = os.environ.get("OFFICIAL_SPECTRAL_LOSS_MODE", "cos").strip().lower()

    loss_std_cos = _cosine_loss_dense(pred_official, true_official)
    loss_sqrt_cos = _sqrt_cosine_loss_dense(pred_official, true_official)

    if mode in ("sqrt", "sqrt_cos", "sqrtcos"):
        loss_base = loss_sqrt_cos
    elif mode in ("mix", "mixed"):
        try:
            std_w = float(os.environ.get("OFFICIAL_SPECTRAL_STD_COS_WEIGHT", "0.8"))
        except Exception:
            std_w = 0.8
        std_w = float(np.clip(std_w, 0.0, 1.0))
        loss_base = std_w * loss_std_cos + (1.0 - std_w) * loss_sqrt_cos
    else:
        loss_base = loss_std_cos

    if float(kl_weight) > 0:
        loss_kl = _dense_kl_loss(pred_official, true_official)
        return loss_base + float(kl_weight) * loss_kl

    return loss_base


def _build_true_official_dense_for_batch(batch, official_metric_cfg, device):
    if not isinstance(batch, dict):
        return None
    if not torch.is_tensor(batch.get('vect_feat', None)):
        return None

    batch_n = int(batch['vect_feat'].shape[0])
    official_bin_width = float(official_metric_cfg.get('bin_width', 0.01))
    official_max_mz = float(official_metric_cfg.get('max_mz', 1005.0))
    official_bin_n = int(math.floor(float(official_max_mz) / float(official_bin_width))) + 1

    true_official_dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )

    if not used_cache:
        true_official_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get('spect_raw', None),
            precursor_mz=batch.get('precursor_mz', None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=official_metric_cfg.get('exclude_precursor', True),
            batch_n=batch_n,
            device=device,
        )

    if official_metric_cfg.get('exclude_precursor', True):
        true_official_dense = _zero_precursor_bin_dense_batch(
            true_official_dense,
            batch.get('precursor_mz', None),
            official_bin_width,
        )

    return true_official_dense


def build_true_official_dense_from_batch(
    batch,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    exclude_precursor=True,
    device=None,
):
    """Build dense official-bin targets from either cached sparse targets or raw spectra."""
    if not isinstance(batch, dict):
        return None
    if device is None:
        device = torch.device('cpu')

    if torch.is_tensor(batch.get('vect_feat', None)):
        batch_n = int(batch['vect_feat'].shape[0])
    else:
        idx_src = batch.get('true_official_idx', None)
        if torch.is_tensor(idx_src):
            batch_n = int(idx_src.shape[0]) if idx_src.dim() > 0 else 1
        elif isinstance(idx_src, (list, tuple)):
            batch_n = len(idx_src)
        else:
            return None

    official_bin_width = float(max(1e-6, float(official_bin_width)))
    official_max_mz = float(max(official_bin_width, float(official_max_mz)))
    official_bin_n = int(math.floor(float(official_max_mz) / official_bin_width)) + 1

    dense, used_cache = _build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=official_bin_n,
    )
    if not used_cache:
        dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get('spect_raw', None),
            precursor_mz=batch.get('precursor_mz', None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=exclude_precursor,
            batch_n=batch_n,
            device=device,
        )

    if exclude_precursor:
        dense = _zero_precursor_bin_dense_batch(
            dense,
            batch.get('precursor_mz', None),
            official_bin_width,
        )

    return dense


def compute_candidate_support_stats(
    batch,
    cand_probs_or_mask,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    eps=1e-8,
):
    """Project candidate probabilities or masks to official bins and summarize support quality."""
    off_idx = batch.get('formulae_peaks_official_idx_agg', None)
    off_int = batch.get('formulae_peaks_official_intensity_agg', None)
    if off_idx is None:
        off_idx = batch.get('formulae_peaks_official_idx', None)
    if off_int is None:
        off_int = get_formulae_official_intensity_from_batch(batch)

    if not (torch.is_tensor(off_idx) and torch.is_tensor(off_int) and torch.is_tensor(cand_probs_or_mask)):
        return {}

    if off_idx.dim() == 2:
        off_idx = off_idx.unsqueeze(0)
    if off_int.dim() == 2:
        off_int = off_int.unsqueeze(0)

    device = cand_probs_or_mask.device
    off_idx = off_idx.to(device=device).long()
    off_int = off_int.to(device=device).float()
    probs = cand_probs_or_mask.to(device=device).float()
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)
    elif probs.dim() > 2:
        probs = probs.reshape(probs.shape[0], -1)

    B = min(int(probs.shape[0]), int(off_idx.shape[0]), int(off_int.shape[0]))
    M = min(int(probs.shape[1]), int(off_idx.shape[1]), int(off_int.shape[1]))
    K = min(int(off_idx.shape[2]), int(off_int.shape[2]))
    if B <= 0 or M <= 0 or K <= 0:
        return {}

    probs = probs[:B, :M]
    off_idx = off_idx[:B, :M, :K]
    off_int = off_int[:B, :M, :K]

    try:
        official_bin_n = int(np.floor(float(official_max_mz) / float(official_bin_width))) + 1
    except Exception:
        official_bin_n = 1
    official_bin_n = max(1, int(official_bin_n))

    valid = (off_idx >= 0) & (off_idx < official_bin_n) & torch.isfinite(off_int) & (off_int > 0)
    probs_eff = probs
    if probs_eff.dtype != off_int.dtype:
        probs_eff = probs_eff.to(dtype=off_int.dtype)
    contrib = probs_eff.unsqueeze(-1) * off_int * valid.float()

    pred_dense = torch.zeros((B, official_bin_n), dtype=torch.float32, device=device)
    flat_idx = off_idx.clamp(0, max(0, official_bin_n - 1)).reshape(B, -1)
    flat_val = contrib.reshape(B, -1)
    flat_val = flat_val * valid.reshape(B, -1).float()
    pred_dense.scatter_add_(1, flat_idx, flat_val)

    true_dense = build_true_official_dense_from_batch(
        batch,
        official_bin_width=official_bin_width,
        official_max_mz=official_max_mz,
        exclude_precursor=True,
        device=device,
    )
    if true_dense is None:
        return {}
    true_dense = true_dense[:B].to(device=device)

    pred_support = pred_dense > float(eps)
    true_support = true_dense > float(eps)
    overlap = pred_support & true_support
    false = pred_support & (~true_support)

    pred_int_sum = pred_dense.sum(dim=-1).clamp_min(float(eps))
    pred_int_on_true = (pred_dense * true_support.float()).sum(dim=-1) / pred_int_sum
    false_support = (pred_dense * (~true_support).float()).sum(dim=-1) / pred_int_sum
    cos = F.cosine_similarity(pred_dense, true_dense, dim=-1, eps=float(eps))

    return {
        'pred_n': float(pred_support.float().sum(dim=-1).mean().detach().cpu().item()),
        'false_pred_n': float(false.float().sum(dim=-1).mean().detach().cpu().item()),
        'overlap_n': float(overlap.float().sum(dim=-1).mean().detach().cpu().item()),
        'pred_int_on_true': float(pred_int_on_true.mean().detach().cpu().item()),
        'false_support': float(false_support.mean().detach().cpu().item()),
        'official_cos': float(cos.mean().detach().cpu().item()),
    }


def _get_selector_logits_from_res(res_dict):
    """
    Return formula selector logits used by selector losses and topK selection.

    Default behavior:
      use ordinary formula-level selector logits.

    Optional behavior:
      if USE_FN_FORMULA_LOGITS_AS_SELECTOR=1, blend or replace with
      fn_based_formula_logits.  This must be explicit because fragment-node
      mapping can cover only a subset of formula candidates.
    """
    if not isinstance(res_dict, dict):
        return None

    selector_logits = res_dict.get('selector_logits', None)
    if not torch.is_tensor(selector_logits):
        selector_logits = res_dict.get('formulae_scores_raw', None)
    if not torch.is_tensor(selector_logits):
        selector_logits = res_dict.get('formulae_scores_train', None)

    fn_mapped_logits = res_dict.get('fn_based_formula_logits', None)

    use_fn = os.environ.get("USE_FN_FORMULA_LOGITS_AS_SELECTOR", "0") == "1"
    if use_fn and torch.is_tensor(fn_mapped_logits):
        if not torch.is_tensor(selector_logits):
            return fn_mapped_logits

        sel = selector_logits
        fn = fn_mapped_logits.to(device=sel.device, dtype=sel.dtype)

        if sel.dim() == 1:
            sel = sel.unsqueeze(0)
        elif sel.dim() > 2:
            sel = sel.reshape(sel.shape[0], -1)

        if fn.dim() == 1:
            fn = fn.unsqueeze(0)
        elif fn.dim() > 2:
            fn = fn.reshape(fn.shape[0], -1)

        use_b = min(int(sel.shape[0]), int(fn.shape[0]))
        use_m = min(int(sel.shape[1]), int(fn.shape[1]))
        if use_b <= 0 or use_m <= 0:
            return selector_logits

        sel_use = sel[:use_b, :use_m]
        fn_use = fn[:use_b, :use_m]

        # fn_based_formula_logits uses a very negative fill value for
        # formulae without fragment-node mapping.  Those positions must
        # fall back to ordinary selector logits; otherwise valid formulae
        # are silently killed.
        neg = float(_neg_mask_fill_value(fn_use))
        if abs(neg) < 1e20:
            mapped = torch.isfinite(fn_use) & (fn_use > neg * 0.5)
        else:
            mapped = torch.isfinite(fn_use) & (fn_use > -1e20)

        fn_safe = torch.where(mapped, fn_use, sel_use)

        mode = os.environ.get("FN_SELECTOR_MODE", "blend").strip().lower()
        if mode == "replace":
            fused = fn_safe
        else:
            try:
                alpha = float(os.environ.get("FN_SELECTOR_BLEND_ALPHA", "0.2"))
            except Exception:
                alpha = 0.2
            alpha = float(np.clip(alpha, 0.0, 1.0))
            fused = (1.0 - alpha) * sel_use + alpha * fn_safe

        out = sel.clone()
        out[:use_b, :use_m] = fused
        return out

    return selector_logits if torch.is_tensor(selector_logits) else None


def _get_reranker_scores_from_res(res_dict):
    if not isinstance(res_dict, dict):
        return None
    formulae_scores = res_dict.get('formulae_logits', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    formulae_scores = res_dict.get('formulae_scores_train', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    formulae_scores = res_dict.get('formulae_scores_raw', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    return res_dict.get('formulae_scores', None)


def _get_active_candidate_mask_from_batch(batch, formulae_mask=None):
    """
    Read active candidate mask from batch.

    Preferred:
      batch['formulae_active_mask'] if available.

    Fallback:
      formulae_aux_feat struct-aux active column.

    Important:
      formulae_aux_feat layout is usually:
        first 15 dims = build_formulae_aux_feat(...)
        last 7 dims  = [
            any_source,
            struct_source,
            common_loss_source,
            break_depth_norm,
            ring_cut,
            prior_score_z,
            active_mask,
        ]

      Therefore active_mask is column 21 when aux dim is 22,
      not column 6.
    """
    active = batch.get('formulae_active_mask', None)

    if not torch.is_tensor(active):
        aux = batch.get('formulae_aux_feat', None)
        if torch.is_tensor(aux) and aux.dim() >= 3:
            raw_col = os.environ.get("FORMULAE_AUX_ACTIVE_COL", "").strip()

            if raw_col:
                try:
                    active_col = int(raw_col)
                except Exception:
                    active_col = None
            elif int(aux.shape[-1]) >= 22:
                # 15-D formula aux + 7-D struct aux; active is the last struct field.
                active_col = 21
            elif int(aux.shape[-1]) == 7:
                # Struct aux only.
                active_col = 6
            else:
                active_col = None

            if active_col is not None and int(aux.shape[-1]) > active_col:
                active = aux[..., active_col]

    if not torch.is_tensor(active):
        return None

    active = active.float()
    if active.dim() > 2:
        active = active.reshape(active.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(active.shape[0]), int(fm.shape[0]))
        use_m = min(int(active.shape[1]), int(fm.shape[1]))

        active = active[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        active = active * (fm > 0.5).float()

    return (active > 0.5).float()

def _apply_selector_aux_logit_bias(selector_logits, batch):
    """
    Add label-free cache prior bias to selector logits.

    Important:
      formulae_aux_feat layout is usually:
        first 15 dims = formula aux
        last 7 dims  = struct aux:
          0 any_source
          1 struct_source
          2 common_loss_source
          3 break_depth_norm
          4 ring_cut
          5 prior_score_z
          6 active_mask

    So for 22-D aux, struct aux starts at -7, not column 0.
    """
    if os.environ.get("USE_SELECTOR_AUX_LOGIT_BIAS", "0") != "1":
        return selector_logits

    if not torch.is_tensor(selector_logits):
        return selector_logits

    aux = batch.get('formulae_aux_feat', None)
    if not torch.is_tensor(aux) or aux.dim() < 3:
        return selector_logits

    logits = selector_logits
    aux = aux.to(device=logits.device, dtype=logits.dtype)

    use_b = min(int(logits.shape[0]), int(aux.shape[0]))
    use_m = min(int(logits.shape[1]), int(aux.shape[1]))
    logits = logits[:use_b, :use_m]
    aux = aux[:use_b, :use_m]

    d = int(aux.shape[-1])

    # Correct struct-aux extraction.
    if d >= 22:
        saux = aux[..., -7:]
    elif d == 7:
        saux = aux
    else:
        return selector_logits

    def _w(name, default):
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    any_source = saux[..., 0]
    struct_source = saux[..., 1]
    common_loss_source = saux[..., 2]
    break_depth_norm = saux[..., 3]
    ring_cut = saux[..., 4]
    prior_score_z = saux[..., 5].clamp(-3.0, 3.0)
    active_mask = saux[..., 6]

    bias = torch.zeros_like(logits)

    # Soft prior only. Do not hard-mask with active.
    bias = bias + _w("SELECTOR_BIAS_ANY_SOURCE", 0.05) * any_source
    bias = bias + _w("SELECTOR_BIAS_STRUCT_SOURCE", 0.10) * struct_source
    bias = bias + _w("SELECTOR_BIAS_COMMON_LOSS", 0.05) * common_loss_source
    bias = bias + _w("SELECTOR_BIAS_PRIOR_Z", 0.20) * prior_score_z
    bias = bias + _w("SELECTOR_BIAS_ACTIVE", 0.35) * active_mask

    # ring cut / depth are not always reliable; keep weak.
    bias = bias + _w("SELECTOR_BIAS_RING_CUT", -0.03) * ring_cut
    bias = bias + _w("SELECTOR_BIAS_BREAK_DEPTH", -0.02) * break_depth_norm

    out = logits + bias

    fm = batch.get('formulae_mask', None)
    if torch.is_tensor(fm):
        fm = fm.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:use_b, :use_m].to(device=out.device)
        out = out.masked_fill(fm <= 0.5, _neg_mask_fill_value(out))

    return out

def _build_topk_mask_from_scores(scores_tensor, formulae_mask=None, topk=64, candidate_mask=None):
    if not torch.is_tensor(scores_tensor):
        return None

    sc = scores_tensor.float()
    if sc.dim() == 1:
        sc = sc.unsqueeze(0)
    elif sc.dim() > 2:
        sc = sc.reshape(sc.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(sc.shape[0]), int(fm.shape[0]))
        use_m = min(int(sc.shape[1]), int(fm.shape[1]))
        sc = sc[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        fm = (fm > 0.5).float()
    else:
        fm = torch.ones_like(sc)

    # Optional extra candidate mask, e.g. active candidate mask from cache.
    # If a row has no valid active candidates, fall back to the original formulae_mask.
    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.float()
        if cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)

        use_b2 = min(int(sc.shape[0]), int(cm.shape[0]))
        use_m2 = min(int(sc.shape[1]), int(cm.shape[1]))
        sc = sc[:use_b2, :use_m2]
        fm = fm[:use_b2, :use_m2]
        cm = cm[:use_b2, :use_m2]

        fm_full = fm
        fm_active = fm * (cm > 0.5).float()

        row_has_active = fm_active.sum(dim=-1, keepdim=True) > 0
        fm = torch.where(row_has_active, fm_active, fm_full)

    if int(sc.shape[1]) <= 0:
        return None

    kk = max(1, min(int(topk), int(sc.shape[1])))

    masked_scores = sc.masked_fill(fm <= 0, _neg_mask_fill_value(sc))
    top_idx = torch.topk(masked_scores, k=kk, dim=-1).indices

    keep = torch.zeros_like(sc, dtype=torch.float32)
    keep.scatter_(1, top_idx, 1.0)
    keep = keep * fm
    return keep


def _build_group_unique_topk_mask_from_scores(
    scores,
    formulae_mask=None,
    group_id=None,
    topk=256,
    candidate_mask=None,
):
    """
    Build topK mask with at most one candidate per formula group.

    This is required for V2 source-instance candidates:
    multiple source instances of the same formula should not occupy many topK slots.
    """
    if not torch.is_tensor(scores):
        return None

    s = scores.detach()
    if s.dim() == 1:
        s = s.unsqueeze(0)
    elif s.dim() > 2:
        s = s.reshape(s.shape[0], -1)

    B, M = int(s.shape[0]), int(s.shape[1])
    device = s.device

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.to(device=device)
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        valid = torch.zeros((B, M), dtype=torch.bool, device=device)
        use_b = min(B, int(fm.shape[0]))
        use_m = min(M, int(fm.shape[1]))
        valid[:use_b, :use_m] = fm[:use_b, :use_m] > 0.5
    else:
        valid = torch.ones((B, M), dtype=torch.bool, device=device)

    if torch.is_tensor(candidate_mask):
        cm = candidate_mask.to(device=device)
        if cm.dim() == 1:
            cm = cm.unsqueeze(0)
        elif cm.dim() > 2:
            cm = cm.reshape(cm.shape[0], -1)
        use_b = min(B, int(cm.shape[0]))
        use_m = min(M, int(cm.shape[1]))
        cm_full = torch.zeros((B, M), dtype=torch.bool, device=device)
        cm_full[:use_b, :use_m] = cm[:use_b, :use_m] > 0.5
        valid = valid & cm_full

    if torch.is_tensor(group_id):
        gid = group_id.to(device=device, dtype=torch.long)
        if gid.dim() == 1:
            gid = gid.unsqueeze(0)
        elif gid.dim() > 2:
            gid = gid.reshape(gid.shape[0], -1)
        use_b = min(B, int(gid.shape[0]))
        use_m = min(M, int(gid.shape[1]))
        gid_full = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1).clone()
        gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
        gid = gid_full
    else:
        gid = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1)

    out = torch.zeros((B, M), dtype=torch.float32, device=device)
    kk = max(1, min(int(topk), M))

    for b in range(B):
        valid_idx = torch.nonzero(valid[b], as_tuple=False).reshape(-1)
        if valid_idx.numel() <= 0:
            continue

        valid_scores = s[b, valid_idx]
        order = torch.argsort(valid_scores, descending=True)

        seen = set()
        chosen = []

        for oi in order.detach().cpu().tolist():
            idx = int(valid_idx[oi].detach().cpu().item())
            g = int(gid[b, idx].detach().cpu().item())

            if g in seen:
                continue

            seen.add(g)
            chosen.append(idx)

            if len(chosen) >= kk:
                break

        if len(chosen) > 0:
            chosen_t = torch.as_tensor(chosen, dtype=torch.long, device=device)
            out[b, chosen_t] = 1.0

    return out

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


def _mask_recall(pred_mask, true_mask):
    if (not torch.is_tensor(pred_mask)) or (not torch.is_tensor(true_mask)):
        return float('nan')

    pm = pred_mask.float()
    tm = true_mask.float()

    if pm.dim() == 1:
        pm = pm.unsqueeze(0)
    elif pm.dim() > 2:
        pm = pm.reshape(pm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(pm.shape[0]), int(tm.shape[0]))
    use_m = min(int(pm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float('nan')

    pm = pm[:use_b, :use_m] > 0.5
    tm = tm[:use_b, :use_m] > 0.5

    denom = tm.sum(dim=-1).float().clamp_min(1.0)
    hit = (pm & tm).sum(dim=-1).float()
    recall = hit / denom
    return float(recall.mean().detach().cpu().item())


def _mask_precision(pred_mask, true_mask):
    if (not torch.is_tensor(pred_mask)) or (not torch.is_tensor(true_mask)):
        return float('nan')

    pm = pred_mask.float()
    tm = true_mask.float()

    if pm.dim() == 1:
        pm = pm.unsqueeze(0)
    elif pm.dim() > 2:
        pm = pm.reshape(pm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(pm.shape[0]), int(tm.shape[0]))
    use_m = min(int(pm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float('nan')

    pm = pm[:use_b, :use_m] > 0.5
    tm = tm[:use_b, :use_m] > 0.5

    denom = pm.sum(dim=-1).float().clamp_min(1.0)
    hit = (pm & tm).sum(dim=-1).float()
    precision = hit / denom
    return float(precision.mean().detach().cpu().item())

def _mask_ratio_in_topk(source_mask, topk_mask):
    """
    Among selected topK candidates, compute how many have source_mask=1.

    Used for:
      fragaux_model_topk_ratio@K:
      selector topK 里面有多少比例候选带 fragment-local source.
    """
    if (not torch.is_tensor(source_mask)) or (not torch.is_tensor(topk_mask)):
        return float("nan")

    sm = source_mask.float()
    tm = topk_mask.float()

    if sm.dim() == 1:
        sm = sm.unsqueeze(0)
    elif sm.dim() > 2:
        sm = sm.reshape(sm.shape[0], -1)

    if tm.dim() == 1:
        tm = tm.unsqueeze(0)
    elif tm.dim() > 2:
        tm = tm.reshape(tm.shape[0], -1)

    use_b = min(int(sm.shape[0]), int(tm.shape[0]))
    use_m = min(int(sm.shape[1]), int(tm.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return float("nan")

    sm = sm[:use_b, :use_m]
    tm = tm[:use_b, :use_m]

    denom = tm.sum(dim=-1).clamp_min(1.0)
    ratio = ((sm > 0.5).float() * (tm > 0.5).float()).sum(dim=-1) / denom

    return float(ratio.mean().detach().cpu().item())

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

    true_dense, used_cached_true = _build_true_official_dense_from_cached_sparse_batch(
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
        'formulae_frag_aux_feat',   # 关键：full-train rerank 裁剪时必须一起裁
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
    selector_kl_weight = max(0.0, float(os.environ.get('SELECTOR_KL_WEIGHT', '1.0')))

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

    log(f'⚙️ mode=formula_only:{int(formula_only)} batch_size={batch_size} epochs={epochs}')
    log(f'🧪 formula_elements: atomicnos={formula_atomicnos}')
    log(f'🧩 mainline_signature=setwise_peak_clean_min')
    log(f'🧩 objective: selector + rerank + official_dense_spectral + peak + oos')
    log(f'🧩 formula_target_mode={formula_target_mode} support_temp={target_support_temperature:.4f} support_topk={target_support_topk}')
    log(
        f'🧩 loss_weights: selector={selector_loss_weight:.3f} rerank={main_candidate_kl_weight:.3f} '
        f'official_spectral={official_spectral_loss_weight:.3f} '
        f'official_kl={official_spectral_kl_weight:.3f} peak={peak_aux_loss_weight:.3f} '
        f'oos={oos_loss_weight:.3f} precursor={precursor_loss_weight:.3f} '
        f'entropy={formula_entropy_loss_weight:.3f} '
        f'false_support={false_support_loss_weight:.3f} '
    )
    log(f'🧩 teacher_topk_train={teacher_topk_train}')
    log(f'🧩 teacher_topk_eval={teacher_topk_eval}')
    log(f'🧩 selector_topk={selector_topk} model_topk_eval={model_topk_eval}')
    log(
        f'🧩 step_limits: '
        f'train={max_train_steps if max_train_steps is not None else "<full>"} '
        f'val={max_val_steps if max_val_steps is not None else "<full>"}'
    )
    log(f'🧩 selector_pos_weight={selector_pos_weight:.3f}')
    log(
        f'🧩 selector_components: bce_weight={selector_bce_weight:.3f} '
        f'kl_weight={selector_kl_weight:.3f}'
    )
    log(f'🧩 train_selector_only_stage={int(train_selector_only_stage)}')
    log(f'🧩 selector_only_warmup_epochs={selector_only_warmup_epochs}')
    log(
        f'🧩 rerank_mix_teacher_ratio: stage1={rerank_mix_teacher_ratio_stage1:.2f} '
        f'stage2={rerank_mix_teacher_ratio_stage2:.2f} stage3={rerank_mix_teacher_ratio_stage3:.2f} '
        f'epochs=({rerank_mix_stage1_epochs},{rerank_mix_stage2_epochs})'
    )
    log(
        f'🧩 loss_start_epoch: spectral={spectral_loss_start_epoch} '
        f'peak={peak_aux_start_epoch} oos={oos_loss_start_epoch}'
    )
    log(
        f'precursor_loss_start_epoch={precursor_loss_start_epoch}'
    )
    log(f'🧩 model_select_metric={model_select_metric_name}')
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
                '缓存的 formula 维度与当前元素集合不一致: '
                f'cache_dim={int(ff_shape[-1])}, configured_dim={len(formula_atomicnos)}, atomicnos={formula_atomicnos}'
            )

    formula_oh_sizes, formula_oh_source = _resolve_formula_oh_sizes(len(formula_atomicnos))
    log(f'🧮 formula_oh_sizes={formula_oh_sizes} source={formula_oh_source}')

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
                # sample0 是 raw collate 后的 batch，通常 shape 是 [B, M]
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
        f"🧩 fragment_local_aux: use={int(use_fragment_local_aux)} "
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
            f"🔁 loaded model from {load_model_path} | "
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
                if os.environ.get("USE_GROUP_UNIQUE_TEACHER_TOPK", "0") == "1":
                    teacher_positive_mask = (teacher_target_full > 0)

                    teacher_topk_mask = _build_group_unique_topk_mask_from_scores(
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
                teacher_topk_mask = selector_pos_label
                teacher_topk_probs = None

            true_official_dense = _build_true_official_dense_for_batch(
                batch,
                official_metric_cfg,
                device,
            )

            with autocast(enabled=amp_enabled, dtype=amp_dtype):
                # ============================================================
                # PASS 1: full-candidate selector
                # ============================================================
                res_full = model(**batch, selector_only_forward=True)

                precursor_loss = _compute_precursor_loss_from_batch(batch, res_full)
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
                selector_pos_rate = res_full['spect'].sum() * 0.0
                selector_quality_mean = res_full['spect'].sum() * 0.0

                if (
                    torch.is_tensor(selector_quality)
                    and torch.is_tensor(selector_pos_label)
                    and torch.is_tensor(selector_valid_mask)
                ):
                    selector_logits_masked = selector_logits_for_loss.masked_fill(
                        selector_valid_mask <= 0.5, 0.0
                    )
                    bce_raw = F.binary_cross_entropy_with_logits(
                        selector_logits_masked,
                        selector_pos_label,
                        reduction='none',
                        pos_weight=selector_logits_for_loss.new_tensor([float(selector_pos_weight)]),
                    )
                    selector_bce_loss = (
                        bce_raw * selector_valid_mask
                    ).sum() / selector_valid_mask.sum().clamp_min(1.0)

                    try:
                        gamma = float(os.environ.get("QUALITY_TARGET_GAMMA", "2.0"))
                    except Exception:
                        gamma = 2.0

                    target_dist = selector_quality.clamp_min(0.0) ** gamma

                    # Critical: KL should only distribute mass over positive clean candidates.
                    target_dist = target_dist * selector_pos_label.float() * selector_valid_mask.float()

                    target_sum = target_dist.sum(dim=1, keepdim=True)

                    # Fallback if no target mass: use pos_label uniformly.
                    uniform_pos = selector_pos_label.float() * selector_valid_mask.float()
                    uniform_pos = uniform_pos / uniform_pos.sum(dim=1, keepdim=True).clamp_min(1e-8)

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

                target_pos_false_mass = res_full['spect'].sum() * 0.0
                target_pos_overlap_exact = res_full['spect'].sum() * 0.0
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

                selector_loss = (
                    float(selector_bce_weight) * selector_bce_loss
                    + float(selector_kl_weight) * selector_kl_loss
                )

                target_pos_exact_support_mass = res_full['spect'].sum() * 0.0
                target_strict_keep_rate = res_full['spect'].sum() * 0.0

                if (
                    isinstance(selector_extra, dict)
                    and torch.is_tensor(selector_pos_label)
                    and torch.is_tensor(selector_extra.get('exact_support_mass', None))
                    and torch.is_tensor(selector_extra.get('strict_keep', None))
                ):
                    pos = selector_pos_label.float()
                    pos_den = pos.sum().clamp_min(1.0)

                    target_pos_exact_support_mass = (
                        selector_extra['exact_support_mass'].to(device=pos.device, dtype=pos.dtype) * pos
                    ).sum() / pos_den

                    target_strict_keep_rate = (
                        selector_extra['strict_keep'].to(device=pos.device, dtype=pos.dtype) * formulae_mask.float()
                    ).sum() / formulae_mask.float().sum().clamp_min(1.0)

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
                    false_support_loss = selector_loss.new_zeros(())
                    loss = float(selector_loss_weight) * selector_loss

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

                    if os.environ.get("USE_GROUP_UNIQUE_MODEL_TOPK", "0") == "1":
                        model_topk_mask_train = _build_group_unique_topk_mask_from_scores(
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            group_id=batch.get("formulae_instance_group_id", None),
                            topk=selector_topk,
                            candidate_mask=topk_candidate_mask_train,
                        )
                    else:
                        model_topk_mask_train = _build_topk_mask_from_scores(
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
                        group_unique=(os.environ.get("USE_GROUP_UNIQUE_PRUNE", "0") == "1"),
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
                    selector_quality, selector_pos_label, selector_valid_mask, selector_extra = (
                        build_candidate_local_quality_target(
                            batch=batch,
                            formulae_mask=formulae_mask,
                            official_bin_n=official_bin_n,
                        )
                    )

                # teacher target 只用于诊断 KL，不用于验证 forward
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
                    if os.environ.get("USE_GROUP_UNIQUE_TEACHER_TOPK", "0") == "1":
                        teacher_positive_mask = (teacher_target_full > 0)

                        teacher_topk_mask = _build_group_unique_topk_mask_from_scores(
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
                    active_teacher_recall = _mask_recall(active_mask_diag, teacher_topk_mask)


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
                    fragaux_teacher_recall = _mask_recall(frag_source_mask, teacher_topk_mask)

                true_official_dense = _build_true_official_dense_for_batch(
                    batch,
                    official_metric_cfg,
                    device,
                )

                with autocast(enabled=amp_enabled, dtype=amp_dtype):
                    # ============================================================
                    # PASS 1: full-candidate selector
                    # ============================================================
                    res_full = model(**batch, selector_only_forward=True)

                    val_precursor_loss = _compute_precursor_loss_from_batch(batch, res_full)
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
                    val_selector_pos_rate = res_full['spect'].sum() * 0.0
                    val_selector_quality_mean = res_full['spect'].sum() * 0.0

                    if (
                        torch.is_tensor(selector_quality)
                        and torch.is_tensor(selector_pos_label)
                        and torch.is_tensor(selector_valid_mask)
                    ):
                        selector_logits_masked = selector_logits_for_loss.masked_fill(
                            selector_valid_mask <= 0.5, 0.0
                        )
                        bce_raw = F.binary_cross_entropy_with_logits(
                            selector_logits_masked,
                            selector_pos_label,
                            reduction='none',
                            pos_weight=selector_logits_for_loss.new_tensor([float(selector_pos_weight)]),
                        )
                        val_selector_bce = (
                            bce_raw * selector_valid_mask
                        ).sum() / selector_valid_mask.sum().clamp_min(1.0)

                        try:
                            gamma = float(os.environ.get("QUALITY_TARGET_GAMMA", "2.0"))
                        except Exception:
                            gamma = 2.0

                        target_dist = selector_quality.clamp_min(0.0) ** gamma
                        target_dist = target_dist * selector_valid_mask
                        target_dist = target_dist / target_dist.sum(dim=1, keepdim=True).clamp_min(1e-8)

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

                    val_target_pos_false_mass = res_full['spect'].sum() * 0.0
                    val_target_pos_overlap_exact = res_full['spect'].sum() * 0.0
                    if (
                        isinstance(selector_extra, dict)
                        and torch.is_tensor(selector_pos_label)
                        and torch.is_tensor(selector_extra.get('false_support_mass_exact', None))
                        and torch.is_tensor(selector_extra.get('overlap_intensity_exact', None))
                    ):
                        pos = selector_pos_label.float()
                        pos_den = pos.sum().clamp_min(1.0)
                        val_target_pos_false_mass = (
                            selector_extra['false_support_mass_exact'].to(
                                device=pos.device,
                                dtype=pos.dtype,
                            )
                            * pos
                        ).sum() / pos_den
                        val_target_pos_overlap_exact = (
                            selector_extra['overlap_intensity_exact'].to(
                                device=pos.device,
                                dtype=pos.dtype,
                            )
                            * pos
                        ).sum() / pos_den

                    val_selector_loss = (
                        float(selector_bce_weight) * val_selector_bce
                        + float(selector_kl_weight) * val_selector_kl
                    )
                    topk_candidate_mask_val = None
                    if os.environ.get("MODEL_TOPK_USE_ACTIVE_MASK", "0") == "1":
                        topk_candidate_mask_val = _get_active_candidate_mask_from_batch(
                            batch,
                            formulae_mask=batch.get('formulae_mask', None),
                        )

                    if os.environ.get("USE_GROUP_UNIQUE_MODEL_TOPK", "0") == "1":
                        model_topk_mask = _build_group_unique_topk_mask_from_scores(
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            group_id=batch.get("formulae_instance_group_id", None),
                            topk=model_topk_eval,
                            candidate_mask=topk_candidate_mask_val,
                        )
                    else:
                        model_topk_mask = _build_topk_mask_from_scores(
                            selector_logits_for_topk,
                            formulae_mask=batch.get('formulae_mask', None),
                            topk=model_topk_eval,
                            candidate_mask=topk_candidate_mask_val,
                        )

                    model_topk_teacher_recall = _mask_recall(model_topk_mask, teacher_topk_mask)

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
                    for k in k_list:
                        if os.environ.get("USE_GROUP_UNIQUE_MODEL_TOPK", "0") == "1":
                            selector_masks[k] = _build_group_unique_topk_mask_from_scores(
                                selector_logits_for_topk,
                                formulae_mask=batch.get('formulae_mask', None),
                                group_id=batch.get("formulae_instance_group_id", None),
                                topk=k,
                                candidate_mask=topk_candidate_mask_val,
                            )
                        else:
                            selector_masks[k] = _build_topk_mask_from_scores(
                                selector_logits_for_topk,
                                formulae_mask=batch.get('formulae_mask', None),
                                topk=k,
                                candidate_mask=topk_candidate_mask_val,
                            )

                    fragaux_model_topk_ratio_32 = float("nan")
                    fragaux_model_topk_ratio_64 = float("nan")
                    fragaux_model_topk_ratio_128 = float("nan")
                    fragaux_model_topk_ratio_256 = float("nan")

                    if torch.is_tensor(frag_source_mask):
                        fragaux_model_topk_ratio_32 = _mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(32, None),
                        )
                        fragaux_model_topk_ratio_64 = _mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(64, None),
                        )
                        fragaux_model_topk_ratio_128 = _mask_ratio_in_topk(
                            frag_source_mask,
                            selector_masks.get(128, None),
                        )
                        fragaux_model_topk_ratio_256 = _mask_ratio_in_topk(
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
                        rec_k = _mask_recall(mask_k, teacher_topk_mask)
                        prec_k = quality_metrics.get(
                            f'selector_precision_at_{k}',
                            _mask_precision(mask_k, teacher_topk_mask),
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

                        if k == 32:
                            val_selector_recall_32_vals.append(rec_k)
                            val_selector_precision_32_vals.append(prec_k)
                            val_selector_quality_mean_32_vals.append(q_mean_k)
                            if stats_k:
                                val_selected_true_hit_mass_32_vals.append(stats_k.get('pred_int_on_true', float('nan')))
                                val_selected_false_mass_32_vals.append(stats_k.get('false_support', float('nan')))
                        elif k == 64:
                            val_selector_recall_64_vals.append(rec_k)
                            val_selector_precision_64_vals.append(prec_k)
                            val_selector_quality_mean_64_vals.append(q_mean_k)
                            if stats_k:
                                val_selected_true_hit_mass_64_vals.append(stats_k.get('pred_int_on_true', float('nan')))
                                val_selected_false_mass_64_vals.append(stats_k.get('false_support', float('nan')))
                        elif k == 128:
                            val_selector_recall_128_vals.append(rec_k)
                            val_selector_precision_128_vals.append(prec_k)
                            val_selector_quality_mean_128_vals.append(q_mean_k)
                            if stats_k:
                                val_selected_true_hit_mass_128_vals.append(stats_k.get('pred_int_on_true', float('nan')))
                                val_selected_false_mass_128_vals.append(stats_k.get('false_support', float('nan')))
                        elif k == 256:
                            val_selector_recall_256_vals.append(rec_k)
                            val_selector_precision_256_vals.append(prec_k)
                            val_selector_quality_mean_256_vals.append(q_mean_k)
                            if stats_k:
                                val_selected_true_hit_mass_256_vals.append(stats_k.get('pred_int_on_true', float('nan')))
                                val_selected_false_mass_256_vals.append(stats_k.get('false_support', float('nan')))

                    if teacher_stats:
                        val_teacher_oracle_cos_vals.append(teacher_stats.get('official_cos', float('nan')))
                        val_teacher_oracle_false_support_vals.append(teacher_stats.get('false_support', float('nan')))
                        val_teacher_oracle_pred_int_on_true_vals.append(teacher_stats.get('pred_int_on_true', float('nan')))
                        val_teacher_oracle_pred_n_vals.append(teacher_stats.get('pred_n', float('nan')))

                    model_topk_stats_256 = compute_candidate_support_stats(
                        batch,
                        selector_masks.get(256, model_topk_mask),
                        official_bin_width=official_metric_cfg['bin_width'],
                        official_max_mz=official_metric_cfg['max_mz'],
                    )
                    if model_topk_stats_256:
                        val_model_topk_oracle_cos_256_vals.append(model_topk_stats_256.get('official_cos', float('nan')))
                        val_model_topk_oracle_false_support_256_vals.append(model_topk_stats_256.get('false_support', float('nan')))

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
                            "use_group_unique_teacher=", os.environ.get("USE_GROUP_UNIQUE_TEACHER_TOPK", "0"),
                            "use_group_unique_model=", os.environ.get("USE_GROUP_UNIQUE_MODEL_TOPK", "0"),
                            "use_group_unique_prune=", os.environ.get("USE_GROUP_UNIQUE_PRUNE", "0"),
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
                        group_unique=(os.environ.get("USE_GROUP_UNIQUE_PRUNE", "0") == "1"),
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
            # 注意：外层 is_best 用的是越大越好，所以 loss 要取负数
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
            # 默认仍然用 official cos
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
            f'val_model_topk_oracle_false_support@256={avg_val_model_topk_oracle_false_support_256:.4f}'
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
        history['train_use_rerank_delta'].append(avg_train_use_rerank_delta)
        history['val_selector_bce'].append(avg_val_selector_bce)
        history['val_selector_kl'].append(avg_val_selector_kl)
        history['val_selector_quality_mean'].append(avg_val_selector_quality_mean)
        history['val_selector_pos_rate'].append(avg_val_selector_pos_rate)
        history['val_target_pos_false_mass'].append(avg_val_target_pos_false_mass)
        history['val_target_pos_overlap_exact'].append(avg_val_target_pos_overlap_exact)
        history['val_target_pos_exact_support_mass'].append(avg_val_target_pos_exact_support_mass)
        history['val_target_strict_keep_rate'].append(avg_val_target_strict_keep_rate)
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
                f'⚠️ epoch={epoch+1}: no optimizer update was performed; '
                f'skip scheduler.step(). Check MAX_TRAIN_STEPS or non-finite loss.'
            )
        if (
            early_stop_patience > 0
            and (epoch + 1) >= early_stop_warmup_epochs
            and early_stop_wait >= early_stop_patience
        ):
            log(f'⏹️ early_stop: epoch={epoch+1} best_val_official_cos={best_val_official_cos:.6f}')
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
