# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Utility script for data preparation, cache building, model evaluation, or diagnostics around the main pipeline.

import argparse
import csv
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import random
import sys
import time
from collections import Counter, OrderedDict
from datetime import datetime, timezone
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rassp.featurize import msutil, create_mol_featurizer
from rassp.msutil import masscompute
from rassp import netutil
from rassp.data.nist20_reader import (
    load_nist20_records,
    records_to_dataframes,
)


# Function overview: _to_optional_float handles a specific workflow step in this module.
def _to_optional_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return None


# Function overview: _to_optional_str handles a specific workflow step in this module.
def _to_optional_str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


# Function overview: _infer_split handles a specific workflow step in this module.
def _infer_split(parquet_path):
    name = os.path.basename(parquet_path).lower()
    if 'train' in name:
        return 'train'
    if 'val' in name or 'valid' in name:
        return 'val'
    if 'test' in name:
        return 'test'
    return None


# Function overview: _parse_first_peak_from_tsv_row handles a specific workflow step in this module.
def _parse_first_peak_from_tsv_row(tsv_row):
    mzs_raw = str(tsv_row.get('mzs', ''))
    ints_raw = str(tsv_row.get('intensities', ''))
    mzs = [x for x in mzs_raw.split(',') if x]
    ints = [x for x in ints_raw.split(',') if x]
    if not mzs or not ints:
        return None, None
    try:
        return float(mzs[0]), float(ints[0])
    except Exception:
        return None, None


# Function overview: _verify_alignment handles a specific workflow step in this module.
def _verify_alignment(df_parquet, df_tsv_split, sample_n, seed):
    n = len(df_parquet)
    if len(df_tsv_split) != n:
        raise ValueError(f'Length mismatch: parquet={n}, tsv_split={len(df_tsv_split)}')

    if n <= 0:
        return

    if sample_n <= 0:
        return

    rng = random.Random(seed)
    idxs = {0, 1, 2, n // 2, n - 1}
    for _ in range(sample_n):
        idxs.add(rng.randrange(n))

    bad_rows = []
    for i in sorted(x for x in idxs if 0 <= x < n):
        p_row = df_parquet.iloc[i]
        t_row = df_tsv_split.iloc[i]

        if str(p_row.get('smiles', '')) != str(t_row.get('smiles', '')):
            bad_rows.append((i, 'smiles'))
            continue

        spect = np.asarray(p_row.get('spect', []))
        p0_mz = None
        p0_it = None
        if spect.size > 0:
            try:
                p0_mz = float(spect[0][0])
                p0_it = float(spect[0][1])
            except Exception:
                pass

        t0_mz, t0_it = _parse_first_peak_from_tsv_row(t_row)
        if p0_mz is None or t0_mz is None:
            bad_rows.append((i, 'peak_parse'))
            continue
        if abs(p0_mz - t0_mz) > 1e-4 or abs(p0_it - t0_it) > 1e-6:
            bad_rows.append((i, 'first_peak'))

    if bad_rows:
        i, kind = bad_rows[0]
        raise ValueError(f'Parquet/TSV alignment check failed at row={i}, reason={kind}, bad_count={len(bad_rows)}')


# Function overview: _parse_csv_list handles a specific workflow step in this module.
def _parse_csv_list(raw):
    if raw is None:
        return []
    vals = [x.strip() for x in str(raw).split(',')]
    return [x for x in vals if x]

def _load_identifier_whitelist(path, split):
    raw_path = str(path or '').strip()
    if not raw_path:
        return None
    with open(raw_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    vals = None
    if isinstance(payload, dict):
        if split in payload and isinstance(payload[split], (list, tuple)):
            vals = payload[split]
        elif 'identifiers' in payload and isinstance(payload['identifiers'], (list, tuple)):
            vals = payload['identifiers']
    elif isinstance(payload, (list, tuple)):
        vals = payload

    if vals is None:
        raise ValueError(f'Invalid identifier whitelist format: {raw_path}')

    out = set()
    for v in vals:
        s = _to_optional_str(v)
        if s:
            out.add(s)
    return out


# Function overview: _resolve_formula_atomicnos handles a specific workflow step in this module.
def _resolve_formula_atomicnos():
    base_default = [1, 6, 7, 8, 9, 15, 16, 17]
    tier2 = [11, 35, 53]

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
        for an in tier2:
            if an not in vals:
                vals.append(an)
    return vals


# Function overview: _mol_allowed_elements_ok handles a specific workflow step in this module.
def _mol_allowed_elements_ok(mol, formula_atomicnos):
    allowed = set(int(x) for x in formula_atomicnos)
    return all(atom.GetAtomicNum() in allowed for atom in mol.GetAtoms())


# Function overview: _heavy_atom_count handles a specific workflow step in this module.
def _heavy_atom_count(mol):
    try:
        return int(mol.GetNumHeavyAtoms())
    except Exception:
        return sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)


# Function overview: _formula_n_raw handles a specific workflow step in this module.
def _formula_n_raw(mol, formula_atomicnos):
    try:
        formula = masscompute.get_formula(mol)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
            formula = masscompute.get_formula(mol)
        except Exception:
            formula = {}
            for atom in mol.GetAtoms():
                an = int(atom.GetAtomicNum())
                formula[an] = int(formula.get(an, 0)) + 1
    out = 1
    for an in formula_atomicnos:
        try:
            count = int(formula.get(int(an), 0))
        except Exception:
            count = 0
        out *= (count + 1)
    return int(out)


# Function overview: _norm_text handles a specific workflow step in this module.
def _norm_text(raw):
    if raw is None:
        return None
    return str(raw).strip().lower()


# Function overview: _canonical_smiles_from_mol handles a specific workflow step in this module.
def _canonical_smiles_from_mol(mol):
    try:
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        return None


# Function overview: _feature_signature handles a specific workflow step in this module.
def _feature_signature(featurizer_config, cache_version):
    payload = {
        'cache_version': cache_version,
        'featurizer_config': featurizer_config,
    }
    data = pickle.dumps(payload, protocol=4)
    return hashlib.sha1(data).hexdigest()


# Function overview: _feature_equal handles a specific workflow step in this module.
def _feature_equal(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            return False
        for k in a:
            if not _feature_equal(a[k], b[k]):
                return False
        return True
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            if not _feature_equal(x, y):
                return False
        return True
    if isinstance(a, np.ndarray):
        if a.shape != b.shape:
            return False
        return bool(np.array_equal(a, b))
    try:
        return bool(a == b)
    except Exception:
        return False


_CACHE_COND2_STATE = {}


def _atomic_pickle_dump(obj, path, protocol=pickle.HIGHEST_PROTOCOL):
    tmp_path = f'{path}.tmp.{os.getpid()}.{time.time_ns()}'
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(obj, f, protocol=protocol)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


# Function overview: _append_jsonl appends one JSON object as a line.
def _append_jsonl(fp, obj):
    fp.write(json.dumps(obj, ensure_ascii=True) + '\n')


# Function overview: _safe_stat_dict computes robust summary stats for numeric lists.
def _safe_stat_dict(values, percentiles=(50, 90, 99)):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return None
    out = {
        'n': int(arr.size),
        'mean': float(np.mean(arr)),
    }
    for p in percentiles:
        out[f'p{int(p)}'] = float(np.percentile(arr, float(p)))
    return out


def _counter_to_dict(counter_obj):
    out = {}
    for k, v in counter_obj.items():
        key = str(k) if k is not None else '<missing>'
        out[key] = int(v)
    return dict(sorted(out.items(), key=lambda kv: kv[0]))


def _to_peak_matrix(spect_raw):
    if spect_raw is None:
        return np.zeros((0, 2), dtype=np.float32)
    try:
        if isinstance(spect_raw, np.ndarray) and spect_raw.dtype == object and spect_raw.ndim == 1:
            rows = []
            for x in spect_raw:
                xx = np.asarray(x, dtype=np.float32).reshape(-1)
                if xx.shape[0] >= 2:
                    rows.append([float(xx[0]), float(xx[1])])
            peaks = np.asarray(rows, dtype=np.float32)
        else:
            peaks = np.asarray(spect_raw, dtype=np.float32)
        if peaks.size <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            peaks = peaks.reshape(-1, 2)
        if peaks.shape[0] <= 0:
            return np.zeros((0, 2), dtype=np.float32)
        return peaks.astype(np.float32, copy=False)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

def _official_bin_indices(mz, bin_width):
    bw = float(max(1e-6, float(bin_width)))
    mode = str(os.environ.get("OFFICIAL_BIN_MODE", "floor")).strip().lower()

    mz_arr = np.asarray(mz, dtype=np.float64)
    if mode in ("round", "nearest", "nominal"):
        return np.rint(mz_arr / bw).astype(np.int64)
    return np.floor(mz_arr / bw + 1e-8).astype(np.int64)


def _build_true_official_sparse_from_spect(
    spect_raw,
    bin_width=0.01,
    max_mz=1005.0,
    exclude_precursor=False,
    precursor_mz=None,
):
    bw = float(max(1e-6, float(bin_width)))
    mz_max = float(max(float(max_mz), bw))

    peaks = _to_peak_matrix(spect_raw)
    if peaks.shape[0] <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    mz = peaks[:, 0].astype(np.float64)
    intensity = peaks[:, 1].astype(np.float64)
    valid = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (intensity > 0)
        & (mz >= 0.0)
        & (mz < mz_max)
    )
    mz = mz[valid]
    intensity = intensity[valid]
    if mz.size <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    idx = _official_bin_indices(mz, bw)
    if bool(exclude_precursor):
        try:
            pmz = float(precursor_mz)
            if np.isfinite(pmz) and pmz > 0:
                # Default 0 keeps historical exact-bin behavior.
                # For NIST nominal/rounded peaks, set e.g. 0.6 to remove precursor residual.
                tol_da = float(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.0"))

                # Also remove precursor isotope residuals M+1/M+2 when requested.
                isotope_n = int(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "0"))

                if tol_da > 0.0:
                    keep = np.ones_like(mz, dtype=bool)
                    for iso_k in range(max(0, isotope_n) + 1):
                        center = pmz + float(iso_k)
                        keep &= (np.abs(mz - center) > float(tol_da))
                else:
                    p_idx = int(_official_bin_indices(np.asarray([pmz], dtype=np.float64), bw)[0])
                    keep = idx != p_idx

                idx = idx[keep]
                intensity = intensity[keep]
                mz = mz[keep]
        except Exception:
            pass

    if idx.size <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    uniq_idx, inv = np.unique(idx, return_inverse=True)
    uniq_intensity = np.zeros((int(uniq_idx.shape[0]),), dtype=np.float64)
    np.add.at(uniq_intensity, inv, intensity)
    order = np.argsort(uniq_idx, kind='stable')
    out_idx = uniq_idx[order].astype(np.int64, copy=False)
    out_intensity = uniq_intensity[order].astype(np.float32, copy=False)
    return out_idx, out_intensity


def _build_true_topk_sparse(idx, intensity, k=20):
    kk = max(1, int(k))
    idx_arr = np.asarray(idx, dtype=np.int64).reshape(-1)
    int_arr = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if idx_arr.size <= 0 or int_arr.size <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    use_n = min(int(idx_arr.shape[0]), int(int_arr.shape[0]))
    idx_arr = idx_arr[:use_n]
    int_arr = int_arr[:use_n]
    valid = (idx_arr >= 0) & np.isfinite(int_arr) & (int_arr > 0)
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    idx_v = idx_arr[valid]
    int_v = int_arr[valid]
    order = np.argsort(-int_v, kind='stable')
    take = min(int(order.shape[0]), kk)
    if take <= 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    sel = order[:take]
    return idx_v[sel].astype(np.int64, copy=False), int_v[sel].astype(np.float32, copy=False)

def _build_candidate_official_agg(features):
    if not isinstance(features, dict):
        return None, None
    off_idx = features.get('formulae_peaks_official_idx', None)
    off_intensity = features.get('formulae_peaks_official_intensity', features.get('formulae_peaks_intensity', None))
    if off_idx is None or off_intensity is None:
        return None, None

    try:
        idx_arr = np.asarray(off_idx, dtype=np.int64)
        int_arr = np.asarray(off_intensity, dtype=np.float32)
    except Exception:
        return None, None

    if idx_arr.ndim == 3:
        idx_arr = idx_arr.reshape(idx_arr.shape[0], -1)
    if int_arr.ndim == 3:
        int_arr = int_arr.reshape(int_arr.shape[0], -1)
    if idx_arr.ndim != 2 or int_arr.ndim != 2:
        return None, None

    cand_n = min(int(idx_arr.shape[0]), int(int_arr.shape[0]))
    peak_n = min(int(idx_arr.shape[1]), int(int_arr.shape[1]))
    if cand_n <= 0 or peak_n <= 0:
        return None, None

    idx_use = idx_arr[:cand_n, :peak_n]
    int_use = int_arr[:cand_n, :peak_n]

    out_idx = np.full((cand_n, peak_n), -1, dtype=np.int64)
    out_intensity = np.zeros((cand_n, peak_n), dtype=np.float32)

    for ci in range(cand_n):
        idx_row = idx_use[ci]
        int_row = int_use[ci]
        valid = (idx_row >= 0) & np.isfinite(int_row) & (int_row > 0)
        if not np.any(valid):
            continue

        idx_v = idx_row[valid]
        int_v = int_row[valid].astype(np.float64)
        uniq_idx, inv = np.unique(idx_v, return_inverse=True)
        uniq_intensity = np.zeros((int(uniq_idx.shape[0]),), dtype=np.float64)
        np.add.at(uniq_intensity, inv, int_v)

        order = np.argsort(-uniq_intensity, kind='stable')
        uniq_idx = uniq_idx[order]
        uniq_intensity = uniq_intensity[order]

        keep = min(peak_n, int(uniq_idx.shape[0]))
        if keep <= 0:
            continue
        out_idx[ci, :keep] = uniq_idx[:keep].astype(np.int64, copy=False)
        out_intensity[ci, :keep] = uniq_intensity[:keep].astype(np.float32, copy=False)

    return out_idx, out_intensity


def filter_massspecgym_rows(df, df_tsv_split, formula_atomicnos, args, identifier_whitelist=None):
    """Apply a single, split-consistent filter policy before any sampling."""
    pre_filter_n = len(df)
    max_heavy_atoms = int(os.environ.get('MAX_HEAVY_ATOMS', '60'))
    # max_raw_unique_formulae = int(os.environ.get('MAX_RAW_UNIQUE_FORMULAE', str(int(args.max_formulae))))
    prefilter_max_raw_unique_formulae = int(getattr(args, 'prefilter_max_raw_unique_formulae', 0))
    hard_max_raw_unique_formulae = int(getattr(args, 'hard_max_raw_unique_formulae', 50000))

    allowed_adducts = _parse_csv_list(args.allowed_adducts)
    allowed_instruments = _parse_csv_list(args.allowed_instruments)
    allowed_adducts_norm = set(_norm_text(x) for x in allowed_adducts)
    allowed_instruments_norm = set(_norm_text(x) for x in allowed_instruments)

    all_indices = []
    row_complexity_meta = {}
    reject_stats = {
        'identifier_not_selected': 0,
        'ce_missing': 0,
        'adduct_not_allowed': 0,
        'instrument_not_allowed': 0,
        'precursor_mz_invalid': 0,
        'allowed_elements_not_ok': 0,
        'heavy_atom_overflow': 0,
        'formula_n_raw_overflow': 0,
        'sanitize_failed': 0,
        'charged_molecule': 0,
        'radical_molecule': 0,
        'multi_component': 0,
        'canonical_failed': 0,
        'empty_or_zero_spectrum': 0,
    }

    instrument_counter = Counter()
    adduct_counter = Counter()
    ce_missing_before_n = 0
    ce_missing_after_n = 0

    for i in range(pre_filter_n):
        row = df.iloc[i]
        trow = df_tsv_split.iloc[i]
        identifier_raw = _to_optional_str(trow.get('identifier', None))
        ce_raw = _to_optional_float(trow.get('collision_energy', None))
        adduct_raw = _to_optional_str(trow.get('adduct', None))
        instrument_raw = _to_optional_str(trow.get('instrument_type', None))
        precursor_raw = _to_optional_float(trow.get('precursor_mz', None))
        if ce_raw is None:
            ce_missing_before_n += 1

        mol = Chem.Mol(row['rdmol'])
        heavy_atom_n = _heavy_atom_count(mol)
        allowed_elements_ok = _mol_allowed_elements_ok(mol, formula_atomicnos)
        formula_n_raw = _formula_n_raw(mol, formula_atomicnos) if allowed_elements_ok else 0

        keep = True
        if identifier_whitelist is not None and (identifier_raw not in identifier_whitelist):
            reject_stats['identifier_not_selected'] += 1
            keep = False

        if keep and int(args.strict_molecule_policy) == 1:
            try:
                mol_s = Chem.Mol(mol)
                Chem.SanitizeMol(mol_s)
            except Exception:
                reject_stats['sanitize_failed'] += 1
                keep = False

            if keep:
                try:
                    formal_charge = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
                except Exception:
                    formal_charge = 1
                if formal_charge != 0:
                    reject_stats['charged_molecule'] += 1
                    keep = False

            if keep:
                try:
                    radical_e = int(sum(int(a.GetNumRadicalElectrons()) for a in mol.GetAtoms()))
                except Exception:
                    radical_e = 1
                if radical_e != 0:
                    reject_stats['radical_molecule'] += 1
                    keep = False

            if keep:
                try:
                    frag_n = int(len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)))
                except Exception:
                    frag_n = 2
                if frag_n != 1:
                    reject_stats['multi_component'] += 1
                    keep = False

            if keep:
                try:
                    c_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                    c_key = Chem.MolToInchiKey(mol)
                    if (not str(c_smi).strip()) or (not str(c_key).strip()):
                        raise ValueError('empty canonical keys')
                except Exception:
                    reject_stats['canonical_failed'] += 1
                    keep = False

            if keep:
                spect_raw = row.get('spect', None)
                try:
                    if isinstance(spect_raw, np.ndarray) and spect_raw.dtype == object and spect_raw.ndim == 1:
                        rows = []
                        for x in spect_raw:
                            xx = np.asarray(x, dtype=np.float32).reshape(-1)
                            if xx.shape[0] >= 2:
                                rows.append([float(xx[0]), float(xx[1])])
                        peaks = np.asarray(rows, dtype=np.float32)
                    else:
                        peaks = np.asarray(spect_raw, dtype=np.float32)
                    if peaks.size <= 0:
                        raise ValueError('empty')
                    if peaks.ndim != 2 or peaks.shape[1] != 2:
                        peaks = peaks.reshape(-1, 2)
                    if peaks.shape[0] <= 0:
                        raise ValueError('empty')
                    ints = peaks[:, 1]
                    if (not np.isfinite(ints).any()) or float(np.nansum(np.clip(ints, 0.0, None))) <= 0.0:
                        raise ValueError('all_zero')
                except Exception:
                    reject_stats['empty_or_zero_spectrum'] += 1
                    keep = False

        if keep and int(args.require_ce) == 1 and ce_raw is None:
            reject_stats['ce_missing'] += 1
            keep = False

        if keep and allowed_adducts_norm:
            if adduct_raw is None or _norm_text(adduct_raw) not in allowed_adducts_norm:
                reject_stats['adduct_not_allowed'] += 1
                keep = False

        if keep and allowed_instruments_norm:
            if instrument_raw is None or _norm_text(instrument_raw) not in allowed_instruments_norm:
                reject_stats['instrument_not_allowed'] += 1
                keep = False

        if keep and args.max_precursor_mz and args.max_precursor_mz > 0:
            if precursor_raw is None or precursor_raw > float(args.max_precursor_mz):
                reject_stats['precursor_mz_invalid'] += 1
                keep = False

        if keep and not allowed_elements_ok:
            reject_stats['allowed_elements_not_ok'] += 1
            keep = False

        if keep and max_heavy_atoms > 0 and heavy_atom_n > max_heavy_atoms:
            reject_stats['heavy_atom_overflow'] += 1
            keep = False

        # if keep and max_raw_unique_formulae > 0 and formula_n_raw > max_raw_unique_formulae:
        #     reject_stats['formula_n_raw_overflow'] += 1
        #     keep = False

        row_prefilter_overflow = 0
        if keep and prefilter_max_raw_unique_formulae > 0 and formula_n_raw > prefilter_max_raw_unique_formulae:
            row_prefilter_overflow = 1

        if keep and hard_max_raw_unique_formulae > 0 and formula_n_raw > hard_max_raw_unique_formulae:
            reject_stats['formula_n_raw_overflow'] += 1
            keep = False

        if keep:
            all_indices.append(i)
            row_complexity_meta[i] = {
                'identifier': identifier_raw,
                'allowed_elements_ok': bool(allowed_elements_ok),
                'heavy_atom_n': int(heavy_atom_n),
                'formula_n_raw': int(formula_n_raw),
                'prefilter_overflow_flag': int(row_prefilter_overflow),
            }
            instrument_counter[str(instrument_raw) if instrument_raw is not None else '<missing>'] += 1
            adduct_counter[str(adduct_raw) if adduct_raw is not None else '<missing>'] += 1
            if ce_raw is None:
                ce_missing_after_n += 1

    return {
        'all_indices': all_indices,
        'row_complexity_meta': row_complexity_meta,
        'reject_stats': reject_stats,
        'allowed_adducts': allowed_adducts,
        'allowed_instruments': allowed_instruments,
        'max_heavy_atoms': int(max_heavy_atoms),
        # 'max_raw_unique_formulae': int(max_raw_unique_formulae),
        'prefilter_max_raw_unique_formulae': int(prefilter_max_raw_unique_formulae),
        'hard_max_raw_unique_formulae': int(hard_max_raw_unique_formulae),  
        'pre_filter_n': int(pre_filter_n),
        'post_filter_n': int(len(all_indices)),
        'instrument_distribution_filtered': _counter_to_dict(instrument_counter),
        'adduct_distribution_filtered': _counter_to_dict(adduct_counter),
        'ce_missing_ratio_before_filter': float(ce_missing_before_n / max(1, pre_filter_n)),
        'ce_missing_ratio_after_filter': float(ce_missing_after_n / max(1, len(all_indices))) if len(all_indices) > 0 else float('nan'),
    }


def _run_three_stage_peak_audit(out_dir, selected_pairs, split, use_source_index_filenames, mol_n=50, csv_name='candidate_peak_audit.csv'):
    rows = []
    take_pairs = selected_pairs[: max(0, int(mol_n))]
    for local_i, src_i in take_pairs:
        file_idx = int(src_i) if bool(use_source_index_filenames) else int(local_i)
        pkl_path = os.path.join(out_dir, f'{file_idx}.pkl')
        if not os.path.isfile(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                obj = pickle.load(f)
        except Exception:
            continue

        features = obj.get('features', {}) if isinstance(obj, dict) else {}
        if not isinstance(features, dict):
            continue

        peaks = np.asarray(features.get('formulae_peaks', np.zeros((0, 0, 2), dtype=np.float32)), dtype=np.float32)
        off_idx = np.asarray(features.get('formulae_peaks_official_idx', np.zeros((0, 0), dtype=np.int64)), dtype=np.int64)
        off_int = np.asarray(
            features.get('formulae_peaks_official_intensity', features.get('formulae_peaks_intensity', np.zeros((0, 0), dtype=np.float32))),
            dtype=np.float32,
        )
        fmask = np.asarray(features.get('formulae_mask', np.ones((peaks.shape[0],), dtype=np.float32)), dtype=np.float32).reshape(-1)

        cand_n = min(
            int(peaks.shape[0]) if peaks.ndim >= 1 else 0,
            int(off_idx.shape[0]) if off_idx.ndim >= 1 else 0,
            int(off_int.shape[0]) if off_int.ndim >= 1 else 0,
            int(fmask.shape[0]),
        )
        if cand_n <= 0:
            continue

        sample_id = obj.get('identifier', f'{split}:{src_i}') if isinstance(obj, dict) else f'{split}:{src_i}'
        for ci in range(cand_n):
            if float(fmask[ci]) <= 0.5:
                continue
            raw = peaks[ci] if peaks.ndim == 3 else np.zeros((0, 2), dtype=np.float32)
            idx_row = off_idx[ci] if off_idx.ndim >= 2 else np.zeros((0,), dtype=np.int64)
            int_row = off_int[ci] if off_int.ndim >= 2 else np.zeros((0,), dtype=np.float32)

            if raw.ndim != 2 or raw.shape[-1] < 2:
                raw = np.zeros((0, 2), dtype=np.float32)

            raw_mz = raw[:, 0] if raw.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
            raw_it = raw[:, 1] if raw.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
            raw_mz_n = int(np.sum(np.isfinite(raw_mz)))
            raw_intensity_n = int(np.sum(np.isfinite(raw_it) & (raw_it > 0)))
            raw_peak_n = int(np.sum(np.isfinite(raw_mz) & np.isfinite(raw_it) & (raw_it > 0)))

            valid_official = (idx_row >= 0) & np.isfinite(int_row) & (int_row > 0)
            official_peak_n = int(np.sum(valid_official))
            unique_official_bin_n = int(np.unique(idx_row[valid_official]).shape[0]) if official_peak_n > 0 else 0
            final_valid_peak_n = int(np.sum(valid_official))

            rows.append(
                {
                    'split': str(split),
                    'sample_id': str(sample_id),
                    'row_idx': int(src_i),
                    'candidate_id': int(ci),
                    'raw_peak_n': int(raw_peak_n),
                    'raw_mz_n': int(raw_mz_n),
                    'raw_intensity_n': int(raw_intensity_n),
                    'official_peak_n': int(official_peak_n),
                    'unique_official_bin_n': int(unique_official_bin_n),
                    'final_valid_peak_n': int(final_valid_peak_n),
                }
            )

    csv_path = os.path.join(out_dir, csv_name)
    json_path = os.path.join(out_dir, 'candidate_peak_audit_summary.json')
    fieldnames = [
        'split',
        'sample_id',
        'row_idx',
        'candidate_id',
        'raw_peak_n',
        'raw_mz_n',
        'raw_intensity_n',
        'official_peak_n',
        'unique_official_bin_n',
        'final_valid_peak_n',
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    if len(rows) <= 0:
        summary = {
            'rows': 0,
            'candidate_peak_count_mean': float('nan'),
            'unique_bin_count_mean': float('nan'),
            'single_peak_candidate_ratio': float('nan'),
            'repeated_bin_ratio_mean': float('nan'),
            'csv_path': csv_path,
        }
    else:
        peak_arr = np.asarray([float(r['final_valid_peak_n']) for r in rows], dtype=np.float64)
        uniq_arr = np.asarray([float(r['unique_official_bin_n']) for r in rows], dtype=np.float64)
        off_arr = np.asarray([float(r['official_peak_n']) for r in rows], dtype=np.float64)
        repeated = np.maximum(0.0, off_arr - uniq_arr) / np.maximum(1.0, off_arr)
        repeated_ratio_arr = 1.0 - (uniq_arr / np.clip(peak_arr, 1.0, None))
        multi_peak_ratio = float(np.mean(peak_arr >= 2.0))
        peak_count_p25 = float(np.percentile(peak_arr, 25.0))
        summary = {
            'rows': int(len(rows)),
            'candidate_peak_count_mean': float(np.mean(peak_arr)),
            'candidate_peak_count_p25': peak_count_p25,
            'unique_bin_count_mean': float(np.mean(uniq_arr)),
            'single_peak_candidate_ratio': float(np.mean(peak_arr <= 1.0)),
            'multi_peak_candidate_ratio': multi_peak_ratio,
            'repeated_bin_ratio_mean': float(np.mean(repeated_ratio_arr)),
            'csv_path': csv_path,
            'summary_path': json_path,
        }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)
    summary['summary_path'] = json_path
    return summary


def _cache_condv2_process_row(task):
    local_i, src_i = task
    state = _CACHE_COND2_STATE
    df = state['df']
    df_tsv_split = state['df_tsv_split']
    featurizer = state['featurizer']
    spect_bin = state['spect_bin']
    out_dir = state['out_dir']
    args = state['args']
    row_complexity_meta = state['row_complexity_meta']
    dedup_enabled = state['dedup_enabled']
    struct_cache_dir = state['struct_cache_dir']
    feature_sig = state['feature_sig']
    use_source_index_filenames = state['use_source_index_filenames']

    file_idx = int(src_i) if use_source_index_filenames else int(local_i)
    out_path = os.path.join(out_dir, f'{file_idx}.pkl')
    err_path = os.path.join(out_dir, f'{file_idx}.err')

    if (not args.overwrite) and os.path.exists(out_path):
        return {
            'ok': True,
            'skipped': True,
            'src_i': int(src_i),
            'local_i': int(local_i),
            'file_idx': file_idx,
            'error_message': None,
        }

    row = df.iloc[src_i]
    trow = df_tsv_split.iloc[src_i]

    spect_raw = row.get('spect', None)
    ce_raw = _to_optional_float(trow.get('collision_energy', None))
    adduct_raw = _to_optional_str(trow.get('adduct', None))
    precursor_raw = _to_optional_float(trow.get('precursor_mz', None))
    precursor_formula_raw = _to_optional_str(trow.get('precursor_formula', None))
    instrument_raw = _to_optional_str(trow.get('instrument_type', None))
    fragmentation_method_raw = _to_optional_str(trow.get('fragmentation_method', None))
    ce_raw_text = _to_optional_str(trow.get('collision_energy_raw', None))
    ce_type_raw = _to_optional_str(trow.get('collision_energy_type', None))
    ce_parse_ok_raw = _to_optional_float(trow.get('collision_energy_parse_ok', None))

    mol = Chem.Mol(row['rdmol'])
    try:
        structure_cache_key = None
        structure_cache_source = 'per_row'
        if dedup_enabled:
            key_base = _canonical_smiles_from_mol(mol)
            if not key_base:
                key_base = _to_optional_str(row.get('smiles', None)) or f'row_{src_i}'

            structure_key_payload = {
                "feature_sig": str(feature_sig),
                "smiles": str(key_base),

                # 条件信息：同一结构不同 precursor/adduct 不应该复用同一 feature
                "adduct": str(adduct_raw),
                "precursor_formula": str(precursor_formula_raw),
                "precursor_mz_rounded": None if precursor_raw is None else round(float(precursor_raw), 4),

                # 元素顺序和维度：必须进入 key
                "formula_atomicnos": [int(x) for x in state.get("formula_atomicnos", [])],

                # official bin 设置：会影响 event_official_idx / formulae_peaks_official_idx
                "official_bin_width": float(args.official_bin_width),
                "official_max_mz": float(args.official_max_mz),
                "official_exclude_precursor": int(args.official_exclude_precursor),
                
                "official_bin_mode": os.environ.get("OFFICIAL_BIN_MODE", "floor"),
                "official_exclude_precursor_tol_da": os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.0"),
                "official_exclude_precursor_isotope_n": os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "0"),
                "formula_peak_renderer": os.environ.get("FORMULA_PEAK_RENDERER", "exact"),
                "num_peaks_per_formula": os.environ.get("NUM_PEAKS_PER_FORMULA", "32"),
                "formula_apply_adduct_shift": os.environ.get("FORMULA_APPLY_ADDUCT_SHIFT", "0"),
                "formula_adduct_mode": os.environ.get("FORMULA_ADDUCT_MODE", "none"),
                "formula_filter_dbe_negative": os.environ.get("FORMULA_FILTER_DBE_NEGATIVE", "1"),
                "formula_filter_dbe_parity": os.environ.get("FORMULA_FILTER_DBE_PARITY", "0"),
                "formula_proton_mass": os.environ.get("FORMULA_PROTON_MASS", "1.007276466812"),
                "formula_structure_guided": os.environ.get("FORMULA_STRUCTURE_GUIDED", "1"),
                "formula_structure_preselect": os.environ.get("FORMULA_STRUCTURE_PRESELECT", "0"),
                "formula_common_loss_mark": os.environ.get("FORMULA_COMMON_LOSS_MARK", "1"),
                "formula_structure_exact_mark": os.environ.get("FORMULA_STRUCTURE_EXACT_MARK", "1"),
                "formula_structure_h_window": os.environ.get("FORMULA_STRUCTURE_H_WINDOW", "4"),
                "formula_common_loss_h_window": os.environ.get("FORMULA_COMMON_LOSS_H_WINDOW", "3"),
                "formulae_official_intensity_mode": os.environ.get("FORMULAE_OFFICIAL_INTENSITY_MODE", "peakdist"),
                "formula_allow_label_overflow": os.environ.get("FORMULA_ALLOW_LABEL_OVERFLOW", "0"),
                "formula_overflow_chem_weight": os.environ.get("FORMULA_OVERFLOW_CHEM_WEIGHT", "0.35"),
                "formula_active_topk": os.environ.get("FORMULA_ACTIVE_TOPK", "300"),
                "formula_active_keep_all_source": os.environ.get("FORMULA_ACTIVE_KEEP_ALL_SOURCE", "1"),
                "formula_active_bin_width": os.environ.get("FORMULA_ACTIVE_BIN_WIDTH", os.environ.get("OFFICIAL_BIN_WIDTH", "0.01")),
                "formula_active_max_mz": os.environ.get("FORMULA_ACTIVE_MAX_MZ", os.environ.get("OFFICIAL_MAX_MZ", "1005.0")),
                "formula_multidepth_struct": os.environ.get("FORMULA_MULTIDEPTH_STRUCT", "0"),
                "formula_struct_depth": os.environ.get("FORMULA_STRUCT_DEPTH", "4"),
                "formula_struct_max_nodes": os.environ.get("FORMULA_STRUCT_MAX_NODES", "50000"),
                "formula_struct_branch_per_node": os.environ.get("FORMULA_STRUCT_BRANCH_PER_NODE", "64"),
                "formula_struct_allow_ring": os.environ.get("FORMULA_STRUCT_ALLOW_RING", "1"),
                "formula_struct_include_root": os.environ.get("FORMULA_STRUCT_INCLUDE_ROOT", "0"),
                "formula_annotation_library_path": os.environ.get("FORMULA_ANNOTATION_LIBRARY_PATH", ""),
                "use_fragment_local_aux": os.environ.get("USE_FRAGMENT_LOCAL_AUX", "0"),
                "frag_aux_h_shift_min": os.environ.get("FRAG_AUX_H_SHIFT_MIN", "-4"),
                "frag_aux_h_shift_max": os.environ.get("FRAG_AUX_H_SHIFT_MAX", "4"),
                "frag_aux_max_depth": os.environ.get("FRAG_AUX_MAX_DEPTH", "1"),
                "use_source_instance_candidates": os.environ.get("USE_SOURCE_INSTANCE_CANDIDATES", "0"),
                "source_instance_max_per_formula": os.environ.get("SOURCE_INSTANCE_MAX_PER_FORMULA", "3"),
                "source_instance_max_total": os.environ.get("SOURCE_INSTANCE_MAX_TOTAL", "2048"),
                "source_instance_teacher_bonus": os.environ.get("SOURCE_INSTANCE_TEACHER_BONUS", "0.0"),
                "frag_aux_max_depth2_bond_pairs": os.environ.get("FRAG_AUX_MAX_DEPTH2_BOND_PAIRS", "1200"),
                "frag_aux_max_sources": os.environ.get("FRAG_AUX_MAX_SOURCES", "100000"),
                "use_fragment_node_candidates": os.environ.get("USE_FRAGMENT_NODE_CANDIDATES", "0"),
                "fragment_node_max_n": os.environ.get("FRAGMENT_NODE_MAX_N", "4096"),
                "fragment_node_add_proton": os.environ.get("FRAGMENT_NODE_ADD_PROTON", "1"),
                "fragment_node_filter_precursor": os.environ.get("FRAGMENT_NODE_FILTER_PRECURSOR", "1"),
                'candidate_formula_rendering': {
                    'formula_peak_renderer': os.environ.get("FORMULA_PEAK_RENDERER", "exact"),
                    'num_peaks_per_formula': os.environ.get("NUM_PEAKS_PER_FORMULA", "32"),
                    'formula_apply_adduct_shift': os.environ.get("FORMULA_APPLY_ADDUCT_SHIFT", "0"),
                    'formula_adduct_mode': os.environ.get("FORMULA_ADDUCT_MODE", "none"),
                    'formula_filter_dbe_negative': os.environ.get("FORMULA_FILTER_DBE_NEGATIVE", "1"),
                    'formula_filter_dbe_parity': os.environ.get("FORMULA_FILTER_DBE_PARITY", "0"),
                    'formula_prior_use_dbe_negative': os.environ.get("FORMULA_PRIOR_USE_DBE_NEGATIVE", "1"),
                    'formula_prior_use_dbe_parity': os.environ.get("FORMULA_PRIOR_USE_DBE_PARITY", "0"),
                    'formula_proton_mass': os.environ.get("FORMULA_PROTON_MASS", "1.007276466812"),
                },
            }   
            structure_cache_key = hashlib.sha1(
                json.dumps(structure_key_payload, sort_keys=True).encode("utf-8")
            ).hexdigest()

            if os.path.isdir(struct_cache_dir):
                struct_file = os.path.join(struct_cache_dir, f'{structure_cache_key}.pkl')
                if os.path.exists(struct_file):
                    with open(struct_file, 'rb') as f:
                        payload = pickle.load(f)
                    payload_key = payload.get('structure_key_payload', None)
                    payload_key_ok = (payload_key == structure_key_payload)

                    if (
                        payload.get('feature_signature', None) == feature_sig
                        and payload.get('features', None) is not None
                        and payload_key_ok
                    ):
                        feat = payload['features']
                        structure_cache_source = 'disk'
                    else:
                        feat = featurizer(
                            mol,
                            spect_raw,
                            precursor_mz=precursor_raw,
                            precursor_formula=precursor_formula_raw,
                            adduct=adduct_raw,
                        )
                        structure_cache_source = 'compute'
                        with open(struct_file, 'wb') as f:
                            pickle.dump(
                                {
                                    'feature_signature': feature_sig,
                                    'structure_key_payload': structure_key_payload,
                                    'smiles': key_base,
                                    'features': feat,
                                },
                                f,
                                protocol=pickle.HIGHEST_PROTOCOL,
                            )
                else:
                    feat = featurizer(
                        mol,
                        spect_raw,
                        precursor_mz=precursor_raw,
                        precursor_formula=precursor_formula_raw,
                        adduct=adduct_raw,
                    )
                    structure_cache_source = 'compute'
                    with open(struct_file, 'wb') as f:
                        pickle.dump(
                            {
                                'feature_signature': feature_sig,
                                'structure_key_payload': structure_key_payload,
                                'smiles': key_base,
                                'features': feat,
                            },
                            f,
                            protocol=pickle.HIGHEST_PROTOCOL,
                        )
            else:
                feat = featurizer(
                    mol,
                    spect_raw,
                    precursor_mz=precursor_raw,
                    precursor_formula=precursor_formula_raw,
                    adduct=adduct_raw,
                )
        else:
            feat = featurizer(
                mol,
                spect_raw,
                precursor_mz=precursor_raw,
                precursor_formula=precursor_formula_raw,
                adduct=adduct_raw,
            )

        
        if isinstance(feat, dict):
            # 1) official target：按官方设置，通常 exclude precursor。
            # 这个用于你的 official dense spectral loss / official metric。
            true_idx, true_intensity = _build_true_official_sparse_from_spect(
                spect_raw=spect_raw,
                bin_width=float(args.official_bin_width),
                max_mz=float(args.official_max_mz),
                exclude_precursor=bool(int(args.official_exclude_precursor)),
                precursor_mz=precursor_raw,
            )

            top_idx, top_intensity = _build_true_topk_sparse(
                true_idx,
                true_intensity,
                k=int(args.true_topk),
            )

            feat['true_official_idx'] = true_idx.astype(np.int64, copy=False)
            feat['true_official_intensity'] = true_intensity.astype(np.float32, copy=False)
            feat['true_top20_official_idx'] = top_idx.astype(np.int64, copy=False)
            feat['true_top20_official_intensity'] = top_intensity.astype(np.float32, copy=False)

            # 2) all-target：不排除 precursor。
            # 这个只用于单独训练 precursor head，不参与 official no-precursor metric。
            true_all_idx, true_all_intensity = _build_true_official_sparse_from_spect(
                spect_raw=spect_raw,
                bin_width=float(args.official_bin_width),
                max_mz=float(args.official_max_mz),
                exclude_precursor=False,
                precursor_mz=precursor_raw,
            )

            feat['true_all_official_idx'] = true_all_idx.astype(np.int64, copy=False)
            feat['true_all_official_intensity'] = true_all_intensity.astype(np.float32, copy=False)

            # 3) precursor target：把 precursor bin 的真实强度单独拿出来。
            precursor_bin = -1
            precursor_intensity = 0.0
            precursor_prob_in_all = 0.0
            precursor_present = 0.0

            try:
                pmz = float(precursor_raw)
                bw = float(args.official_bin_width)
                if np.isfinite(pmz) and pmz > 0.0 and bw > 0.0:
                    precursor_bin = int(_official_bin_indices(np.asarray([pmz], dtype=np.float64), bw)[0])

                    idx_arr = np.asarray(true_all_idx, dtype=np.int64).reshape(-1)
                    int_arr = np.asarray(true_all_intensity, dtype=np.float32).reshape(-1)
                    use_n = min(int(idx_arr.shape[0]), int(int_arr.shape[0]))

                    if use_n > 0:
                        idx_arr = idx_arr[:use_n]
                        int_arr = int_arr[:use_n]
                        hit = idx_arr == int(precursor_bin)

                        if np.any(hit):
                            precursor_intensity = float(np.sum(int_arr[hit]))
                            precursor_present = 1.0

                        total_all = float(np.sum(np.clip(int_arr, 0.0, None)))
                        if total_all > 1e-12:
                            precursor_prob_in_all = float(precursor_intensity / total_all)
            except Exception:
                precursor_bin = -1
                precursor_intensity = 0.0
                precursor_prob_in_all = 0.0
                precursor_present = 0.0

            feat['true_precursor_bin'] = np.asarray([precursor_bin], dtype=np.int64)
            feat['true_precursor_intensity'] = np.asarray([precursor_intensity], dtype=np.float32)
            feat['true_precursor_prob_in_all'] = np.asarray([precursor_prob_in_all], dtype=np.float32)
            feat['true_precursor_present'] = np.asarray([precursor_present], dtype=np.float32)

            # 5) formula candidate official agg。
            agg_idx, agg_intensity = _build_candidate_official_agg(feat)
            if isinstance(agg_idx, np.ndarray) and isinstance(agg_intensity, np.ndarray):
                feat['formulae_peaks_official_idx_agg'] = agg_idx.astype(np.int64, copy=False)
                feat['formulae_peaks_official_intensity_agg'] = agg_intensity.astype(np.float32, copy=False)

        spect_dense = None
        try:
            if isinstance(spect_raw, np.ndarray) and spect_raw.dtype == object and spect_raw.ndim == 1:
                peaks = np.stack([np.asarray(x, dtype=np.float32) for x in spect_raw], axis=0)
            else:
                peaks = np.asarray(spect_raw, dtype=np.float32)

            if peaks.size > 0:
                if peaks.ndim != 2 or peaks.shape[1] != 2:
                    peaks = peaks.reshape(-1, 2)
                if peaks.ndim == 2 and peaks.shape[1] == 2 and peaks.shape[0] > 0:
                    _, _, binned = spect_bin.histogram(peaks[:, 0], peaks[:, 1])
                    spect_dense = np.asarray(binned, dtype=np.float32)
        except Exception:
            spect_dense = None

        missing_cond = []
        if ce_raw is None:
            missing_cond.append('collision_energy')
        if adduct_raw is None:
            missing_cond.append('adduct')
        if instrument_raw is None:
            missing_cond.append('instrument_type')
        if precursor_raw is None:
            missing_cond.append('precursor_mz')
        ce_missing = int(ce_raw is None)
        adduct_missing = int(adduct_raw is None)
        instrument_missing = int(instrument_raw is None)
        precursor_missing = int(precursor_raw is None)
        ms_level_missing = 0

                # Per-row cache diagnostics: candidate counts, support coverage, and condition quality.
        # Important:
        # - If features are loaded from structure cache, pff_obj.last_cache_diag may be stale/empty.
        # - Therefore prefer diagnostics persisted inside feat["pff_cache_diag"].
        pff_obj = getattr(featurizer, 'pff', None)
        pff_diag = {}

        feat_pff_diag = None
        if isinstance(feat, dict):
            feat_pff_diag = feat.get("pff_cache_diag", None)

        if isinstance(feat_pff_diag, dict):
            pff_diag = dict(feat_pff_diag)
        elif isinstance(getattr(pff_obj, 'last_cache_diag', None), dict):
            pff_diag = dict(getattr(pff_obj, 'last_cache_diag', {}))

        def _to_diag_int(v, default=0):
            try:
                if v is None:
                    return int(default)
                return int(v)
            except Exception:
                return int(default)

        def _to_diag_float(v, default=float('nan')):
            try:
                if v is None:
                    return float(default)
                vv = float(v)
                return vv if np.isfinite(vv) else float(default)
            except Exception:
                return float(default)

        formulae_mask = feat.get('formulae_mask', None)
        formulae_n_raw = _to_diag_int(feat.get('formulae_n_raw', pff_diag.get('n_formulae_raw', -1)), default=-1)

        if formulae_mask is not None:
            try:
                fmask = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)
                formulae_n_final = int((fmask > 0.5).sum())
            except Exception:
                formulae_n_final = _to_diag_int(pff_diag.get('n_formulae_final', min(max(0, formulae_n_raw), int(args.max_formulae))), default=0)
        else:
            formulae_n_final = _to_diag_int(pff_diag.get('n_formulae_final', min(max(0, formulae_n_raw), int(args.max_formulae))), default=0)

        off_idx = feat.get('formulae_peaks_official_idx', None)
        valid_formula_after_mass = None
        support_bins_after = np.zeros((0,), dtype=np.int64)
        support_peak_n_after = 0
        if off_idx is not None:
            try:
                off_idx_arr = np.asarray(off_idx, dtype=np.int64)
                if off_idx_arr.ndim == 3:
                    off_idx_arr = off_idx_arr.reshape(off_idx_arr.shape[0], -1)
                if off_idx_arr.ndim == 2:
                    use_n = min(int(off_idx_arr.shape[0]), int(max(0, formulae_n_final)))
                    off_use = off_idx_arr[:use_n]
                    valid_off = off_use >= 0
                    if valid_off.size > 0:
                        valid_formula_after_mass = np.any(valid_off, axis=1)
                        support_peak_n_after = int(valid_off.sum())
                        support_bins_after = np.unique(off_use[valid_off].astype(np.int64)) if np.any(valid_off) else np.zeros((0,), dtype=np.int64)
            except Exception:
                valid_formula_after_mass = None

        n_after_mass_mask_after = _to_diag_int(
            pff_diag.get('n_formulae_after_mass_mask_after_cap', None),
            default=int(np.sum(valid_formula_after_mass)) if valid_formula_after_mass is not None else int(formulae_n_final),
        )

        n_after_precursor_mask_after = None
        peaks_exact = feat.get('formulae_peaks', None)
        precursor_tol_da = _to_diag_float(os.environ.get('CACHE_DIAG_PRECURSOR_TOL_DA', '1.0'), default=1.0)
        if peaks_exact is not None and precursor_raw is not None:
            try:
                peaks_arr = np.asarray(peaks_exact, dtype=np.float32)
                if peaks_arr.ndim == 3 and peaks_arr.shape[-1] >= 2:
                    use_n = min(int(peaks_arr.shape[0]), int(max(0, formulae_n_final)))
                    peaks_use = peaks_arr[:use_n]
                    mz = peaks_use[..., 0]
                    it = peaks_use[..., 1]
                    valid_prec = (
                        np.isfinite(mz)
                        & np.isfinite(it)
                        & (it > 0)
                        & (mz <= (float(precursor_raw) + float(max(0.0, precursor_tol_da))))
                    )
                    n_after_precursor_mask_after = int(np.any(valid_prec, axis=1).sum())
            except Exception:
                n_after_precursor_mask_after = None

        n_after_precursor_mask_after = _to_diag_int(
            pff_diag.get('n_formulae_after_precursor_mask_after_cap', None),
            default=n_after_precursor_mask_after if n_after_precursor_mask_after is not None else n_after_mass_mask_after,
        )

        support_formula_n = _to_diag_int(
            pff_diag.get('n_formulae_after_precursor_mask_after_cap', None),
            default=n_after_precursor_mask_after,
        )
        support_peak_n = _to_diag_int(
            pff_diag.get('support_peak_n_after_cap', None),
            default=support_peak_n_after,
        )
        candidate_mz_unique_n = _to_diag_int(
            pff_diag.get('candidate_mz_unique_n_after_cap', None),
            default=int(support_bins_after.size),
        )

        official_bin_width = _to_diag_float(os.environ.get('CACHE_DIAG_BIN_WIDTH', '0.01'), default=0.01)
        official_max_mz = _to_diag_float(os.environ.get('CACHE_DIAG_MAX_MZ', '1500.0'), default=1500.0)
        if official_bin_width <= 0:
            official_bin_width = 0.01
        if official_max_mz <= official_bin_width:
            official_max_mz = 1500.0

        target_bins_dense = np.zeros((0,), dtype=np.int64)
        target_bin_intensity = {}
        try:
            diag_idx, diag_int = _build_true_official_sparse_from_spect(
                spect_raw=spect_raw,
                bin_width=float(official_bin_width),
                max_mz=float(official_max_mz),
                exclude_precursor=bool(int(args.official_exclude_precursor)),
                precursor_mz=precursor_raw,
            )

            diag_idx = np.asarray(diag_idx, dtype=np.int64).reshape(-1)
            diag_int = np.asarray(diag_int, dtype=np.float32).reshape(-1)
            use_n = min(int(diag_idx.shape[0]), int(diag_int.shape[0]))

            if use_n > 0:
                diag_idx = diag_idx[:use_n]
                diag_int = diag_int[:use_n]
                valid_diag = (diag_idx >= 0) & np.isfinite(diag_int) & (diag_int > 0)
                diag_idx = diag_idx[valid_diag]
                diag_int = diag_int[valid_diag]

                if diag_idx.size > 0:
                    target_bins_dense = np.asarray(sorted(set(int(x) for x in diag_idx.tolist())), dtype=np.int64)
                    for b, w in zip(diag_idx.tolist(), diag_int.tolist()):
                        bb = int(b)
                        target_bin_intensity[bb] = float(target_bin_intensity.get(bb, 0.0) + float(w))
        except Exception:
            target_bins_dense = np.zeros((0,), dtype=np.int64)
            target_bin_intensity = {}

        target_nonzero_bins = _to_diag_int(pff_diag.get('target_nonzero_bins', None), default=0)
        if target_nonzero_bins <= 0:
            target_nonzero_bins = int(target_bins_dense.size)

        support_coverage_after_cap = _to_diag_float(pff_diag.get('support_coverage_after_cap', None), default=float('nan'))
        if target_bins_dense.size > 0 and support_bins_after.size > 0:
            try:
                overlap_bins = np.intersect1d(target_bins_dense, support_bins_after, assume_unique=False)
                coverage_raw = float(overlap_bins.size) / float(max(1, int(target_bins_dense.size)))
                if (not np.isfinite(support_coverage_after_cap)) or support_coverage_after_cap <= 0.0:
                    support_coverage_after_cap = float(coverage_raw)
            except Exception:
                pass

        support_coverage_before_cap = _to_diag_float(
            pff_diag.get('support_coverage_before_cap', None),
            default=support_coverage_after_cap,
        )

                # ---- source / FIORA diagnostics from saved feature arrays ----
        src_arr = np.asarray(feat.get("formulae_source_flag", []), dtype=np.int64).reshape(-1)
        bd_arr = np.asarray(feat.get("formulae_break_depth", []), dtype=np.int64).reshape(-1)

        use_n_src = min(int(formulae_n_final), int(src_arr.shape[0])) if src_arr.size > 0 else 0
        if use_n_src > 0:
            src_use = src_arr[:use_n_src]
            bd_use = bd_arr[:use_n_src] if bd_arr.shape[0] >= use_n_src else np.zeros((use_n_src,), dtype=np.int64)

            structure_selected_after_feat = int(np.sum(src_use > 0))
            structure_single_after_feat = int(np.sum((src_use > 0) & (bd_use == 1)))
            structure_double_after_feat = int(np.sum((src_use > 0) & (bd_use == 2)))
            structure_deep_after_feat = int(np.sum((src_use > 0) & (bd_use >= 3)))
        else:
            structure_selected_after_feat = int(pff_diag.get("structure_selected_after_overflow", 0))
            structure_single_after_feat = int(pff_diag.get("structure_single_after_overflow", 0))
            structure_double_after_feat = int(pff_diag.get("structure_double_after_overflow", 0))
            structure_deep_after_feat = int(pff_diag.get("structure_deep_after_overflow", 0))

        cache_diag = {
            'sample_id': _to_optional_str(trow.get('identifier', None)) or f"{state['split']}:{int(src_i)}",
            'split': state['split'],
            'row_idx': int(src_i),
            'smiles': _to_optional_str(row.get('smiles', None)),
            'precursor_mz': _to_diag_float(precursor_raw, default=float('nan')),
            'ms_level': 2,
            'ce': _to_diag_float(ce_raw, default=float('nan')),
            'ce_missing': int(ce_missing),
            'adduct': adduct_raw,
            'adduct_missing': int(adduct_missing),
            'n_formulae_raw': int(formulae_n_raw),
            'n_formulae_after_mass_mask': int(n_after_mass_mask_after),
            'n_formulae_after_precursor_mask': int(n_after_precursor_mask_after),
            'n_formulae_final': int(formulae_n_final),
            'n_formulae_after_mass_mask_before_cap': _to_diag_int(pff_diag.get('n_formulae_after_mass_mask_before_cap', None), default=n_after_mass_mask_after),
            'n_formulae_after_precursor_mask_before_cap': _to_diag_int(pff_diag.get('n_formulae_after_precursor_mask_before_cap', None), default=n_after_precursor_mask_after),
            'overflow': int(_to_diag_int(pff_diag.get('overflow', None), default=int(formulae_n_raw > int(args.max_formulae)))),
            'max_formulae': int(args.max_formulae),
            'support_peak_n': int(support_peak_n),
            'support_formula_n': int(support_formula_n),
            'support_coverage_before_cap': float(support_coverage_before_cap),
            'support_coverage_after_cap': float(support_coverage_after_cap),
            'target_nonzero_bins': int(target_nonzero_bins),
            'candidate_mz_unique_n': int(candidate_mz_unique_n),
            'structure_selected_before_overflow': int(
                pff_diag.get('structure_selected_before_overflow', structure_selected_after_feat)
            ),
            'structure_single_before_overflow': int(
                pff_diag.get('structure_single_before_overflow', structure_single_after_feat)
            ),
            'structure_double_before_overflow': int(
                pff_diag.get('structure_double_before_overflow', structure_double_after_feat)
            ),
            'structure_deep_before_overflow': int(
                pff_diag.get('structure_deep_before_overflow', structure_deep_after_feat)
            ),
            'formula_supplement_before_overflow': int(
                pff_diag.get('formula_supplement_before_overflow', max(0, int(formulae_n_final) - structure_selected_after_feat))
            ),
            'structure_selected_after_overflow': int(structure_selected_after_feat),
            'structure_single_after_overflow': int(structure_single_after_feat),
            'structure_double_after_overflow': int(structure_double_after_feat),
            'structure_deep_after_overflow': int(structure_deep_after_feat),
            'overflow_used_label_greedy': int(
                pff_diag.get('overflow_used_label_greedy', 0)
            ),
            'active_candidate_n': int(
                pff_diag.get('active_candidate_n', 0)
            ),
            'active_topk_cfg': int(
                pff_diag.get('active_topk_cfg', int(os.environ.get('FORMULA_ACTIVE_TOPK', '300')))
            ),
            'active_source_candidate_n': int(
                pff_diag.get('active_source_candidate_n', 0)
            ),
            'active_ratio': float(
                pff_diag.get('active_ratio', float('nan'))
            ),
            'prior_score_mean': float(
                pff_diag.get('prior_score_mean', float('nan'))
            ),
            'prior_score_p90': float(
                pff_diag.get('prior_score_p90', float('nan'))
            ),
        }

        try:
            frag_aux = feat.get('formulae_frag_aux_feat', None)
            if isinstance(frag_aux, np.ndarray):
                cache_diag['formulae_frag_aux_dim'] = int(frag_aux.shape[-1]) if frag_aux.ndim >= 1 else 0
            elif frag_aux is None:
                cache_diag['formulae_frag_aux_dim'] = 0
        except Exception:
            cache_diag['formulae_frag_aux_dim'] = 0

        matched_intensity_ratio = float('nan')
        os_intensity_ratio = float('nan')
        try:
            total_int = float(sum(float(v) for v in target_bin_intensity.values()))
            if total_int > 0.0 and support_bins_after.size > 0:
                overlap_bins = np.intersect1d(target_bins_dense, support_bins_after, assume_unique=False)
                matched_int = 0.0
                for b in overlap_bins.tolist():
                    matched_int += float(target_bin_intensity.get(int(b), 0.0))
                matched_intensity_ratio = float(np.clip(matched_int / total_int, 0.0, 1.0))
                os_intensity_ratio = float(np.clip(1.0 - matched_intensity_ratio, 0.0, 1.0))
        except Exception:
            pass
        cache_diag['matched_intensity_ratio'] = float(matched_intensity_ratio)
        cache_diag['os_intensity_ratio'] = float(os_intensity_ratio)

        # ---- V3A fragment-node diagnostics ----
        try:
            fn_mask = np.asarray(feat.get("fragment_node_mask", []), dtype=np.float32).reshape(-1)
            fn_idx = np.asarray(feat.get("fragment_node_official_idx", []), dtype=np.int64).reshape(-1)
            fn_label = np.asarray(feat.get("fragment_node_label", []), dtype=np.float32).reshape(-1)
            fn_true_int = np.asarray(feat.get("fragment_node_true_intensity", []), dtype=np.float32).reshape(-1)
            fn_top20 = np.asarray(feat.get("fragment_node_label_top20", []), dtype=np.float32).reshape(-1)

            use_n = min(fn_mask.shape[0], fn_idx.shape[0])
            if use_n > 0:
                fn_mask_use = fn_mask[:use_n] > 0.5
                fn_idx_use = fn_idx[:use_n]

                fn_label_use = fn_label[:use_n] if fn_label.shape[0] >= use_n else np.zeros((use_n,), dtype=np.float32)
                fn_true_use = fn_true_int[:use_n] if fn_true_int.shape[0] >= use_n else np.zeros((use_n,), dtype=np.float32)
                fn_top20_use = fn_top20[:use_n] if fn_top20.shape[0] >= use_n else np.zeros((use_n,), dtype=np.float32)

                valid_node = fn_mask_use & (fn_idx_use >= 0)
                valid_bins = fn_idx_use[valid_node]
                valid_n = int(np.sum(fn_mask_use))

                unique_bins = (
                    np.asarray(sorted(set(int(x) for x in valid_bins.tolist())), dtype=np.int64)
                    if valid_bins.size > 0
                    else np.zeros((0,), dtype=np.int64)
                )

                if target_bins_dense.size > 0 and unique_bins.size > 0:
                    overlap_bins = np.intersect1d(target_bins_dense, unique_bins, assume_unique=False)
                    support_cov = float(len(overlap_bins) / max(1, int(target_bins_dense.size)))

                    total_int = float(sum(float(vv) for vv in target_bin_intensity.values()))
                    matched_int = 0.0
                    for b in overlap_bins.tolist():
                        matched_int += float(target_bin_intensity.get(int(b), 0.0))
                    intensity_cov = float(np.clip(matched_int / max(1e-12, total_int), 0.0, 1.0))
                else:
                    overlap_bins = np.zeros((0,), dtype=np.int64)
                    support_cov = float("nan")
                    intensity_cov = float("nan")

                # Node-level hit count: useful for class imbalance, but not a recall.
                true_hit_node_n = int(np.sum((fn_label_use > 0.5) & valid_node))
                true_hit_ratio = float(true_hit_node_n / max(1, valid_n))

                # Unique-bin hit count: useful for oracle coverage.
                true_hit_unique_bin_n = int(overlap_bins.size)
                true_hit_unique_bin_ratio = float(true_hit_unique_bin_n / max(1, int(unique_bins.size)))

                # Build true top20 bins from the cached top_idx computed above.
                try:
                    top_idx_arr = np.asarray(top_idx, dtype=np.int64).reshape(-1)
                    top_idx_arr = top_idx_arr[top_idx_arr >= 0]
                    target_top20_bins_dense = np.asarray(
                        sorted(set(int(x) for x in top_idx_arr.tolist())),
                        dtype=np.int64,
                    )
                except Exception:
                    target_top20_bins_dense = np.zeros((0,), dtype=np.int64)

                if target_top20_bins_dense.size > 0 and unique_bins.size > 0:
                    top20_overlap_bins = np.intersect1d(
                        target_top20_bins_dense,
                        unique_bins,
                        assume_unique=False,
                    )
                    top20_recall = float(top20_overlap_bins.size / max(1, int(target_top20_bins_dense.size)))
                else:
                    top20_overlap_bins = np.zeros((0,), dtype=np.int64)
                    top20_recall = float("nan")

                # Keep node-level top20 hit count separately for duplicate diagnostics.
                top20_hit_node_n = int(np.sum((fn_top20_use > 0.5) & valid_node))
                top20_hit_unique_bin_n = int(top20_overlap_bins.size)

                # Unique intensity sum must be <= 1.
                # Unique intensity coverage, normalized by total true intensity.
                # This should be numerically the same as fragment_node_intensity_coverage.
                total_int_for_unique = float(sum(float(vv) for vv in target_bin_intensity.values()))
                matched_int_unique = 0.0
                for b in overlap_bins.tolist():
                    matched_int_unique += float(target_bin_intensity.get(int(b), 0.0))

                true_int_sum_unique = float(
                    np.clip(
                        matched_int_unique / max(1e-12, total_int_for_unique),
                        0.0,
                        1.0,
                    )
                )

                # Node-level duplicated sum is only a duplicate diagnostic; it can exceed 1.
                true_int_sum_node = float(np.sum(fn_true_use[valid_node]))

            else:
                valid_n = 0
                unique_bins = np.zeros((0,), dtype=np.int64)
                support_cov = float("nan")
                intensity_cov = float("nan")
                true_hit_ratio = float("nan")
                true_hit_unique_bin_ratio = float("nan")
                top20_recall = float("nan")
                true_int_sum_unique = float("nan")
                true_int_sum_node = float("nan")
                true_hit_node_n = 0
                true_hit_unique_bin_n = 0
                top20_hit_node_n = 0
                top20_hit_unique_bin_n = 0

            cache_diag["fragment_node_n_valid"] = int(valid_n)
            cache_diag["fragment_node_unique_bin_n"] = int(unique_bins.size)
            cache_diag["fragment_node_support_coverage"] = float(support_cov)
            cache_diag["fragment_node_intensity_coverage"] = float(intensity_cov)

            # Existing name kept, but it is node-level positive ratio.
            cache_diag["fragment_node_true_hit_ratio"] = float(true_hit_ratio)

            # Fixed: this is now unique-bin top20 recall, so it must be <= 1.
            cache_diag["fragment_node_top20_recall"] = float(top20_recall)

            # Existing name kept, but fixed to unique-bin intensity coverage sum, so <= 1.
            cache_diag["fragment_node_true_int_sum"] = float(true_int_sum_unique)

            # Extra diagnostics to understand duplication.
            cache_diag["fragment_node_true_hit_node_n"] = int(true_hit_node_n)
            cache_diag["fragment_node_true_hit_unique_bin_n"] = int(true_hit_unique_bin_n)
            cache_diag["fragment_node_true_hit_unique_bin_ratio"] = float(true_hit_unique_bin_ratio)
            cache_diag["fragment_node_top20_hit_node_n"] = int(top20_hit_node_n)
            cache_diag["fragment_node_top20_hit_unique_bin_n"] = int(top20_hit_unique_bin_n)
            cache_diag["fragment_node_true_int_sum_node"] = float(true_int_sum_node)

        except Exception:
            cache_diag["fragment_node_n_valid"] = 0
            cache_diag["fragment_node_unique_bin_n"] = 0
            cache_diag["fragment_node_support_coverage"] = float("nan")
            cache_diag["fragment_node_intensity_coverage"] = float("nan")
            cache_diag["fragment_node_true_hit_ratio"] = float("nan")
            cache_diag["fragment_node_top20_recall"] = float("nan")
            cache_diag["fragment_node_true_int_sum"] = float("nan")
            cache_diag["fragment_node_true_hit_node_n"] = 0
            cache_diag["fragment_node_true_hit_unique_bin_n"] = 0
            cache_diag["fragment_node_true_hit_unique_bin_ratio"] = float("nan")
            cache_diag["fragment_node_top20_hit_node_n"] = 0
            cache_diag["fragment_node_top20_hit_unique_bin_n"] = 0
            cache_diag["fragment_node_true_int_sum_node"] = float("nan")

        meta = {
            'cache_version': args.cache_version,
            'featurize_ok': True,
            'error_type': None,
            'error_message': None,
            'condition_source': 'tsv_join' if not missing_cond else 'tsv_join_partial',
            'condition_missing_fields': missing_cond,
            'split': state['split'],
            'row_idx': int(src_i),
            'identifier': _to_optional_str(trow.get('identifier', None)),
            'mol_id': row.get('mol_id', None),
            'smiles': _to_optional_str(row.get('smiles', None)),
            'inchikey': _to_optional_str(trow.get('inchikey', row.get('inchi_key', None))),
            'spect': spect_raw,
            'collision_energy': ce_raw,
            'adduct': adduct_raw,
            'precursor_mz': precursor_raw,
            'ce_missing': ce_missing,
            'adduct_missing': adduct_missing,
            'instrument_missing': instrument_missing,
            'precursor_mz_missing': precursor_missing,
            'ms_level_missing': ms_level_missing,
            'precursor_mz_fallback': float(Descriptors.ExactMolWt(mol) + 1.0078),
            'precursor_formula': precursor_formula_raw,
            'instrument_type': instrument_raw,
            'ms_level': 2,
            'max_formulae': int(args.max_formulae),
            'formulae_n_raw': int(feat.get('formulae_n_raw', -1)),
            'allowed_elements_ok': bool(row_complexity_meta.get(src_i, {}).get('allowed_elements_ok', True)),
            'heavy_atom_n': int(row_complexity_meta.get(src_i, {}).get('heavy_atom_n', _heavy_atom_count(mol))),
            'peak_backend': str(getattr(getattr(featurizer, 'pff', None), 'last_peak_backend', 'unknown')),
            'formula_atomicnos': state['formula_atomicnos'],
            'structure_cache_key': structure_cache_key,
            'structure_cache_source': structure_cache_source,
            'structure_feature_signature': feature_sig,
            'cache_diag': cache_diag,
            'features': feat,
            'fragmentation_method': fragmentation_method_raw,
            'collision_energy_raw': ce_raw_text,
            'collision_energy_type': ce_type_raw,
            'collision_energy_parse_ok': None if ce_parse_ok_raw is None else int(ce_parse_ok_raw),
        }
        if spect_dense is not None:
            meta['spect_dense'] = spect_dense

        _atomic_pickle_dump(meta, out_path, protocol=pickle.HIGHEST_PROTOCOL)

        if os.path.exists(err_path):
            try:
                os.remove(err_path)
            except Exception:
                pass

        return {
            'ok': True,
            'skipped': False,
            'src_i': int(src_i),
            'local_i': int(local_i),
            'file_idx': file_idx,
            'cache_diag': cache_diag,
            'error_message': None,
        }

    except Exception as e:
        err_msg = str(e)
        err_payload = {
            'cache_version': args.cache_version,
            'featurize_ok': False,
            'error_type': type(e).__name__,
            'error_message': err_msg,
            'split': state['split'],
            'row_idx': int(src_i),
        }
        tmp_err_path = f'{err_path}.tmp.{os.getpid()}.{time.time_ns()}'
        try:
            with open(tmp_err_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(err_payload, ensure_ascii=True))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_err_path, err_path)
        finally:
            try:
                if os.path.exists(tmp_err_path):
                    os.remove(tmp_err_path)
            except Exception:
                pass
        return {
            'ok': False,
            'skipped': False,
            'src_i': int(src_i),
            'local_i': int(local_i),
            'file_idx': file_idx,
            'error_message': err_msg,
        }


# Function overview: main handles a specific workflow step in this module.
def main():
    # Stage 1: parse CLI options for split selection, filtering, dedup, and cache controls.
    p = argparse.ArgumentParser()
    p.add_argument('parquet', nargs='?', default='', help='Path to split parquet (train/val/test); optional when --input-format nist20')
    p.add_argument('--out-dir', default=None, help='Cache output directory')
    p.add_argument('--input-format', choices=['massspecgym', 'nist20'], default='massspecgym', help='Input source format used by Stage-2 loader')
    p.add_argument('--tsv', default='data/MassSpecGym.tsv', help='Path to MassSpecGym TSV with conditions')
    p.add_argument('--nist-mol-dir', default='', help='Path to hr_nist_msms.MOL directory (used when --input-format nist20)')
    p.add_argument('--nist-msp-path', default='', help='Path to hr_nist_msms.MSP file (used when --input-format nist20)')
    p.add_argument('--split-seed', type=int, default=42, help='Deterministic seed for NIST20 molecule-level split assignment')
    p.add_argument('--split-ratios', default='0.8,0.1,0.1', help='Train/val/test split ratios for NIST20, e.g. 0.8,0.1,0.1')
    p.add_argument('--split', default='auto', choices=['auto', 'train', 'val', 'test'], help='Split name for selecting fold rows from TSV')
    p.add_argument('--verify-align-samples', type=int, default=64, help='How many random rows to verify for parquet/TSV alignment')
    p.add_argument('--verify-seed', type=int, default=42, help='Seed used by alignment check')
    p.add_argument('--sample-rows', type=int, default=0, help='If >0, random sample this many rows to build a small debug cache')
    p.add_argument('--sample-seed', type=int, default=42, help='Sampling seed when sample-rows>0')
    p.add_argument('--max-rows', type=int, default=0, help='If >0, cap processed rows after sampling')
    p.add_argument('--subset-samples', type=int, default=0, help='Number of sampled vertex subsets to cache (0 means formula-only)')
    p.add_argument('--with-subsets', action='store_true', help='Enable subset feature generation during caching')
    p.add_argument('--with-mol-global-feat', action='store_true', help='Include global molecular descriptors in cached features')
    p.add_argument('--mol-global-features', default='exact_mol_wt,mol_logp,tpsa,num_hbd,num_hba,num_rot_bonds', help='Comma-separated global descriptor names')
    p.add_argument('--max-formulae', type=int, default=4096, help='Maximum number of explicit formulae to enumerate')
    p.add_argument(
    '--formulae-overflow-mode',
    default='coverage_topk',
    choices=['truncate', 'coverage_topk', 'coverage', 'coverage_hshift', 'topk_intensity', 'topk_mass', 'random'],
    )
    p.add_argument('--formulae-overflow-seed', type=int, default=0, help='Base seed for deterministic random overflow sampling')
    p.add_argument('--overwrite', action='store_true', help='Overwrite existing cached .pkl files')
    p.add_argument('--progress-every', type=int, default=50, help='Print textual progress every N rows')
    p.add_argument('--max-error-prints-per-type', type=int, default=3, help='Max times to print identical error messages')
    p.add_argument('--cache-version', default='condv2', help='Version string stored in cache metadata')
    p.add_argument('--official-bin-width', type=float, default=0.01, help='Official sparse bin width used for cached true_official_* fields')
    p.add_argument('--official-max-mz', type=float, default=1005.0, help='Official sparse max m/z (exclusive) for cached true_official_* fields')
    p.add_argument('--official-exclude-precursor', type=int, default=1, help='If 1, remove precursor bin from cached true_official_* fields')
    p.add_argument('--true-topk', type=int, default=20, help='Top-K true official bins to cache as true_top20_official_*')
    p.add_argument('--require-ce', type=int, default=1, help='If 1, drop rows with missing collision_energy')
    p.add_argument('--allowed-adducts', default='[M+H]+', help='Comma-separated adduct whitelist, e.g. "[M+H]+"')
    p.add_argument('--allowed-instruments', default='Orbitrap', help='Comma-separated instrument whitelist, e.g. "Orbitrap,QTOF"')
    p.add_argument(
        '--allowed-fragmentation-methods',
        default='HCD',
        help='Comma-separated fragmentation/activation method whitelist, e.g. "HCD,CID"; empty means all'
    )
    p.add_argument('--max-precursor-mz', type=float, default=0.0, help='If >0, keep only rows with precursor_mz <= value')
    p.add_argument('--identifier-whitelist-json', default='', help='Optional JSON list/dict of allowed identifier values for this split')
    p.add_argument('--strict-molecule-policy', type=int, default=1, help='If 1, enforce sanitize/neutral/no-radical/single-component/canonical/nonempty-spectrum checks')
    p.add_argument('--dedup-by-structure', type=int, default=1, help='When 1, reuse featurized structure features by canonical SMILES')
    p.add_argument('--shared-struct-cache-dir', default=None, help='Optional shared structure-feature cache directory (can be reused across splits)')
    p.add_argument('--struct-mem-cache-size', type=int, default=256, help='In-memory LRU size for loaded structure features')
    p.add_argument('--verify-structure-reuse-samples', type=int, default=0, help='Re-featurize and verify equality for first N reused rows (debug only)')
    p.add_argument('--allow-dedup-with-subsets', action='store_true', help='Allow structure dedup when subset sampling is enabled (may alter random subset behavior)')
    p.add_argument('--num-workers', type=int, default=1, help='Number of worker processes used to featurize rows in parallel')
    p.add_argument('--worker-chunksize', type=int, default=32, help='Chunk size for multiprocessing row dispatch')
    p.add_argument('--num-shards', type=int, default=1, help='Split the filtered row set into this many disjoint shards for parallel cache generation')
    p.add_argument('--shard-id', type=int, default=0, help='Shard index to process when num-shards > 1')
    p.add_argument('--use-source-index-filenames', type=int, default=0, help='If 1, write cache files as <source_row_idx>.pkl instead of local shard index')
    p.add_argument('--cache-diag-jsonl', default='cache_diag.jsonl', help='Per-row cache diagnostic JSONL filename (empty/none to disable)')
    p.add_argument('--peak-audit-mol-n', type=int, default=50, help='Audit the first N filtered molecules and export candidate peak counts')
    p.add_argument('--peak-audit-csv', default='candidate_peak_audit.csv', help='Peak audit csv filename')
    p.add_argument('--enforce-peak-quality-assert', type=int, default=1, help='If 1, fail cache generation when single-peak degeneration is detected')
    p.add_argument('--min-candidate-peak-count-mean', type=float, default=2.5)
    p.add_argument('--min-unique-bin-count-mean', type=float, default=2.5)
    p.add_argument('--max-single-peak-candidate-ratio', type=float, default=0.5)

    # 新增更有用的约束
    p.add_argument('--min-candidate-peak-count-p25', type=float, default=2.0)
    p.add_argument('--min-multi-peak-candidate-ratio', type=float, default=0.85)
    p.add_argument('--max-repeated-bin-ratio-mean', type=float, default=0.35)
    p.add_argument('--prefilter-max-raw-unique-formulae', type=int, default=0,
               help='If >0, hard-filter rows before featurization by raw formula count upper bound; 0 disables.')
    p.add_argument('--hard-max-raw-unique-formulae', type=int, default=50000,   
               help='Emergency hard kill threshold for absurdly large raw formula space; 0 disables.')

    args = p.parse_args()

    if float(args.official_bin_width) <= 0:
        raise ValueError(f'official-bin-width must be > 0, got {args.official_bin_width}')
    if float(args.official_max_mz) <= float(args.official_bin_width):
        raise ValueError(
            f'official-max-mz must be > official-bin-width, got max_mz={args.official_max_mz} bin_width={args.official_bin_width}'
        )
    if int(args.true_topk) <= 0:
        raise ValueError(f'true-topk must be > 0, got {args.true_topk}')

    pq = str(args.parquet or '').strip()
    if args.input_format == 'massspecgym':
        if not pq:
            raise ValueError('parquet path is required when --input-format massspecgym')
        if not os.path.exists(pq):
            raise FileNotFoundError(pq)
        if not os.path.exists(args.tsv):
            raise FileNotFoundError(args.tsv)
    else:
        if not os.path.isdir(str(args.nist_mol_dir or '').strip()):
            raise FileNotFoundError(f'nist mol dir not found: {args.nist_mol_dir}')
        if not os.path.isfile(str(args.nist_msp_path or '').strip()):
            raise FileNotFoundError(f'nist msp path not found: {args.nist_msp_path}')

    # Keep featurizer-side official binning in sync with cache builder args.
    os.environ['OFFICIAL_BIN_WIDTH'] = str(float(args.official_bin_width))
    os.environ['OFFICIAL_MAX_MZ'] = str(float(args.official_max_mz))
    os.environ.setdefault('CACHE_DIAG_BIN_WIDTH', str(float(args.official_bin_width)))
    os.environ.setdefault('CACHE_DIAG_MAX_MZ', str(float(args.official_max_mz)))

    renderer_mode_req = os.environ.get("FORMULA_PEAK_RENDERER", "exact").strip().lower()
    if renderer_mode_req == "masseval" and not getattr(masscompute, "HAS_MASSEVAL", False):
        raise RuntimeError(
            "FORMULA_PEAK_RENDERER=masseval was requested, but the compiled masseval backend "
            "was not found. Set FORMULA_PEAK_RENDERER=exact for the high-resolution NIST path, "
            "or build the masseval extension."
        )

    split = args.split
    if split == 'auto':
        split = _infer_split(pq) if pq else None
    if split not in ('train', 'val', 'test'):
        if args.input_format == 'nist20':
            raise ValueError('Unable to infer split in nist20 mode; pass --split train|val|test')
        raise ValueError(f'Unable to infer split from {pq}; pass --split train|val|test')
    identifier_whitelist = _load_identifier_whitelist(args.identifier_whitelist_json, split)
    if identifier_whitelist is not None:
        print(f'[cache-condv2] identifier_whitelist loaded: split={split} n={len(identifier_whitelist)}', flush=True)

    if args.out_dir:
        out_dir = args.out_dir
    elif pq:
        out_dir = pq + f'.cache_formula{args.max_formulae}_condv2'
    else:
        out_dir = os.path.join('data', f'nist20_{split}.cache_formula{args.max_formulae}_condv2')
    os.makedirs(out_dir, exist_ok=True)

    cache_diag_path = ''
    raw_diag = str(args.cache_diag_jsonl or '').strip()
    if raw_diag and raw_diag.lower() not in ('none', 'off', 'false', '0'):
        cache_diag_path = raw_diag if os.path.isabs(raw_diag) else os.path.join(out_dir, raw_diag)
        os.makedirs(os.path.dirname(cache_diag_path) or out_dir, exist_ok=True)
        print(f'[cache-condv2] cache_diag_jsonl={cache_diag_path}', flush=True)

    # Stage 2: load aligned row tables from MassSpecGym or NIST20 source.
    if args.input_format == 'massspecgym':
        print(f'[cache-condv2] loading parquet {pq}', flush=True)
        df = pd.read_parquet(pq)
        print(f'[cache-condv2] loading tsv {args.tsv}', flush=True)
        tsv_cols = [
            'identifier',
            'mzs',
            'intensities',
            'smiles',
            'inchikey',
            'fold',
            'collision_energy',
            'adduct',
            'precursor_mz',
            'precursor_formula',
            'instrument_type',
        ]
        df_tsv = pd.read_csv(args.tsv, sep='\t', usecols=tsv_cols)
        df_tsv_split = df_tsv[df_tsv['fold'] == split].reset_index(drop=True)
    else:
        print(f'[cache-condv2] loading nist mol_dir={args.nist_mol_dir}', flush=True)
        print(f'[cache-condv2] loading nist msp={args.nist_msp_path}', flush=True)
        nist_limit = int(args.sample_rows) if int(args.sample_rows) > 0 else 0
        records = load_nist20_records(
            mol_dir=args.nist_mol_dir,
            msp_path=args.nist_msp_path,
            split=split,
            seed=int(args.split_seed),
            split_ratios=args.split_ratios,
            limit=nist_limit,
            require_ce=int(args.require_ce),
            allowed_adducts=args.allowed_adducts,
            allowed_instruments=args.allowed_instruments,
            allowed_fragmentation_methods=args.allowed_fragmentation_methods,
            max_precursor_mz=float(args.max_precursor_mz),
            progress_every=2000,
        )
        df_rows, tsv_rows = records_to_dataframes(records, split=split)
        df = pd.DataFrame(df_rows).reset_index(drop=True)
        df_tsv_split = pd.DataFrame(tsv_rows).reset_index(drop=True)
        print(
            f'[cache-condv2] nist split={split} loaded_records={len(records)} '
            f'split_seed={int(args.split_seed)} split_ratios={args.split_ratios}',
            flush=True,
        )

    _verify_alignment(df, df_tsv_split, args.verify_align_samples, args.verify_seed)
    print(f'[cache-condv2] split={split} alignment_check=PASS rows={len(df)}', flush=True)

    # Stage 3: apply condition-domain filters before any sampling.
    formula_atomicnos = _resolve_formula_atomicnos()
    pre_filter_n = len(df)
    instrument_counter_before = Counter()
    adduct_counter_before = Counter()
    ce_missing_before_n = 0
    for i in range(pre_filter_n):
        trow = df_tsv_split.iloc[i]
        instrument_raw = _to_optional_str(trow.get('instrument_type', None))
        adduct_raw = _to_optional_str(trow.get('adduct', None))
        ce_raw = _to_optional_float(trow.get('collision_energy', None))
        instrument_counter_before[str(instrument_raw) if instrument_raw is not None else '<missing>'] += 1
        adduct_counter_before[str(adduct_raw) if adduct_raw is not None else '<missing>'] += 1
        if ce_raw is None:
            ce_missing_before_n += 1

    filter_result = filter_massspecgym_rows(
        df=df,
        df_tsv_split=df_tsv_split,
        formula_atomicnos=formula_atomicnos,
        args=args,
        identifier_whitelist=identifier_whitelist,
    )
    all_indices = filter_result['all_indices']
    row_complexity_meta = filter_result['row_complexity_meta']
    reject_stats = filter_result['reject_stats']
    allowed_adducts = filter_result['allowed_adducts']
    allowed_instruments = filter_result['allowed_instruments']
    max_heavy_atoms = filter_result['max_heavy_atoms']
    prefilter_max_raw_unique_formulae = filter_result.get('prefilter_max_raw_unique_formulae', 0)
    hard_max_raw_unique_formulae = filter_result.get('hard_max_raw_unique_formulae', 0) 
    post_filter_n = filter_result['post_filter_n']
    ce_missing_ratio_before_filter = filter_result['ce_missing_ratio_before_filter']
    ce_missing_ratio_after_filter = filter_result['ce_missing_ratio_after_filter']
    instrument_distribution_before_filter = _counter_to_dict(instrument_counter_before)
    adduct_distribution_before_filter = _counter_to_dict(adduct_counter_before)
    instrument_distribution_filtered = filter_result['instrument_distribution_filtered']
    adduct_distribution_filtered = filter_result['adduct_distribution_filtered']

    print(
        f'[cache-condv2] filters: require_ce={int(args.require_ce)} '
        f'allowed_adducts={allowed_adducts if allowed_adducts else "<all>"} '
        f'allowed_instruments={allowed_instruments if allowed_instruments else "<all>"} '
        f'max_precursor_mz={args.max_precursor_mz if args.max_precursor_mz > 0 else "<none>"} '
        f'strict_molecule_policy={int(args.strict_molecule_policy)} '
        f'identifier_whitelist_n={len(identifier_whitelist) if identifier_whitelist is not None else 0}',
        flush=True,
    )
    print(
        f'[cache-condv2] filter_result keep={post_filter_n}/{pre_filter_n} '
        f'max_heavy_atoms={max_heavy_atoms} '
        f'prefilter_max_raw_unique_formulae={prefilter_max_raw_unique_formulae} '
        f'hard_max_raw_unique_formulae={hard_max_raw_unique_formulae} '
        f'rejects={reject_stats}',
        flush=True,
    )
    print(
        f'[cache-condv2] ce_missing_ratio before={ce_missing_ratio_before_filter:.6f} '
        f'after={ce_missing_ratio_after_filter:.6f} ({ce_missing_before_n}/{pre_filter_n})',
        flush=True,
    )

    if not all_indices:
        raise RuntimeError('All rows were filtered out; relax filter settings.')

    if args.input_format != 'nist20':
        if args.sample_rows and args.sample_rows > 0 and args.sample_rows < len(all_indices):
            rng = random.Random(args.sample_seed)
            all_indices = sorted(rng.sample(all_indices, int(args.sample_rows)))
            print(f'[cache-condv2] sampled_rows={len(all_indices)} sample_seed={args.sample_seed}', flush=True)

    if args.max_rows and args.max_rows > 0:
        all_indices = all_indices[: int(args.max_rows)]

    num_shards = max(1, int(args.num_shards))
    shard_id = int(args.shard_id)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f'Invalid shard selection: shard_id={shard_id}, num_shards={num_shards}')
    if num_shards > 1:
        all_indices = [idx for idx in all_indices if (int(idx) % num_shards) == shard_id]
        print(
            f'[cache-condv2] shard_filter shard_id={shard_id}/{num_shards} kept_rows={len(all_indices)}',
            flush=True,
        )

    selected_indices_path = os.path.join(
        out_dir,
        'selected_row_indices.json' if num_shards == 1 else f'selected_row_indices.shard{shard_id}.json',
    )
    try:
        with open(selected_indices_path, 'w', encoding='utf-8') as f:
            json.dump({'split': split, 'rows': [int(x) for x in all_indices]}, f, ensure_ascii=True)
        print(f'[cache-condv2] selected_row_indices={selected_indices_path}', flush=True)
    except Exception as e:
        print(f'[cache-condv2] failed to write selected_row_indices: {e}', flush=True)

    # Stage 4: build featurizer config (must stay consistent with train/eval atomic space).
    official_bin_width = float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01'))
    official_max_mz = float(os.environ.get('OFFICIAL_MAX_MZ', '1005.0'))
    spect_bin_config = {
        'first_bin_center': 1.0,
        'bin_width': official_bin_width,
        'bin_number': int(np.floor(official_max_mz / official_bin_width)) + 1,
    }
    spect_bin = msutil.binutils.create_spectrum_bins(**spect_bin_config)

    feat_vert_args = netutil.dict_combine(netutil.default_feat_vert_args, {
        'feat_atomicno_onehot': formula_atomicnos
    })

    subset_samples = int(args.subset_samples)
    if args.with_subsets and subset_samples <= 0:
        subset_samples = 128
    mol_global_features = [x.strip() for x in str(args.mol_global_features).split(',') if x.strip()]

    renderer_mode = os.environ.get('FORMULA_PEAK_RENDERER', 'exact').strip().lower()
    if renderer_mode not in ('exact', 'masseval', 'legacy'):
        renderer_mode = 'exact'

    # exact renderer 本身就是 high-resolution exact isotope renderer。
    # 这里保持 use_highres=1，避免日志里出现 use_highres=0 造成误解。
    use_highres_default = '1' if renderer_mode in ('exact', 'masseval') else '0'

    featurizer_config = {
        'MAX_N': 128,
        'feat_vert_args': feat_vert_args,
        'adj_args': netutil.default_adj_args,
        'mol_args': {'global_features': mol_global_features if args.with_mol_global_feat else []},
        'vert_subset_samples_n': subset_samples,
        'subset_gen_config': {'name': 'break_and_rearrange', 'num_breaks': 3},
        'element_oh': feat_vert_args['feat_atomicno_onehot'],
        'explicit_formulae_config': {
            'formula_possible_atomicno': feat_vert_args['feat_atomicno_onehot'],
            'clip_mass': 1023,
            'use_highres': os.environ.get('FORMULA_USE_HIGHRES', use_highres_default) == '1',
            'max_formulae': int(args.max_formulae),
            'overflow_mode': str(args.formulae_overflow_mode).strip().lower(),
            'overflow_sample_seed': int(args.formulae_overflow_seed),
            'num_peaks_per_formula': int(os.environ.get('NUM_PEAKS_PER_FORMULA', '32')),
        },
    }
    featurizer = create_mol_featurizer(spect_bin, featurizer_config)

    dedup_enabled = bool(int(args.dedup_by_structure))
    if dedup_enabled and subset_samples > 0 and (not args.allow_dedup_with_subsets):
        print('[cache-condv2] dedup-by-structure disabled because subset sampling is enabled; '
              'use --allow-dedup-with-subsets to force.', flush=True)
        dedup_enabled = False

    num_workers = max(1, int(args.num_workers))
    worker_chunksize = max(1, int(args.worker_chunksize))
    if num_workers > 1 and dedup_enabled:
        print('[cache-condv2] dedup-by-structure disabled in multi-worker mode to avoid shared-cache contention.', flush=True)
        dedup_enabled = False

    feature_sig = _feature_signature(featurizer_config, args.cache_version)
    struct_cache_dir = None
    struct_mem_cache = OrderedDict()
    if dedup_enabled:
        struct_cache_dir = args.shared_struct_cache_dir or os.path.join(out_dir, '_struct_cache')
        os.makedirs(struct_cache_dir, exist_ok=True)

    n = len(all_indices)
    mode = 'Formula+Subset' if subset_samples > 0 else 'Formula-only'
    peak_backend_print = getattr(getattr(featurizer, 'pff', None), 'peak_backend', 'unknown')
    print(
        f'[cache-condv2] out={out_dir} rows={n} split={split} mode={mode} '
        f'max_formulae={args.max_formulae} overflow_mode={args.formulae_overflow_mode} '
        f'peak_backend={peak_backend_print}'
        f'official_bin_width={float(args.official_bin_width):.6f} official_max_mz={float(args.official_max_mz):.3f} '
        f'official_exclude_precursor={int(args.official_exclude_precursor)} true_topk={int(args.true_topk)} '
        f'use_highres={int(featurizer_config["explicit_formulae_config"]["use_highres"])} '
        f'with_mol_feat={int(args.with_mol_global_feat)} '
        f'dedup_by_structure={int(dedup_enabled)} '
        f'num_workers={num_workers} '
        f'formula_atomicnos={formula_atomicnos}',
        flush=True,
    )

    ok_count = 0
    fail_count = 0
    error_counts = {}
    suppressed_count = 0
    struct_hit_mem = 0
    struct_hit_disk = 0
    struct_miss_compute = 0
    reused_rows_checked = 0
    unique_struct_keys = set()
    # Stage 5: main cache loop with optional structure-level dedup reuse.
    start_ts = time.time()

    use_source_index_filenames = bool(int(args.use_source_index_filenames)) or (num_shards > 1)
    global _CACHE_COND2_STATE
    _CACHE_COND2_STATE = {
        'df': df,
        'df_tsv_split': df_tsv_split,
        'featurizer': featurizer,
        'spect_bin': spect_bin,
        'out_dir': out_dir,
        'args': args,
        'row_complexity_meta': row_complexity_meta,
        'dedup_enabled': dedup_enabled,
        'struct_cache_dir': struct_cache_dir,
        'feature_sig': feature_sig,
        'split': split,
        'formula_atomicnos': formula_atomicnos,
        'use_source_index_filenames': use_source_index_filenames,
    }

    selected_pairs = [(local_i, src_i) for local_i, src_i in enumerate(all_indices)]
    task_iter = iter(selected_pairs)
    if num_workers > 1:
        print(f'[cache-condv2] multiprocessing enabled: workers={num_workers} chunksize={worker_chunksize}', flush=True)
        try:
            ctx = mp.get_context('fork')
        except Exception:
            ctx = mp.get_context()
        pool = ctx.Pool(processes=num_workers)
        task_stream = pool.imap_unordered(_cache_condv2_process_row, task_iter, chunksize=worker_chunksize)
    else:
        pool = None
        task_stream = (_cache_condv2_process_row(task) for task in task_iter)

    completed = 0
    diag_file = None
    diag_acc = {
        'n_formulae_raw': [],
        'n_formulae_final': [],
        'n_formulae_after_mass_mask': [],
        'n_formulae_after_precursor_mask': [],
        'support_formula_n': [],
        'support_coverage_before_cap': [],
        'support_coverage_after_cap': [],
        'support_peak_n': [],
        'candidate_mz_unique_n': [],
        'os_intensity_ratio': [],
        'target_nonzero_bins': [],
        'overflow': [],
        'ce_missing': [],
        'adduct_missing': [],

        'structure_selected_before_overflow': [],
        'structure_single_before_overflow': [],
        'structure_double_before_overflow': [],
        'structure_deep_before_overflow': [],
        'formula_supplement_before_overflow': [],
        'structure_selected_after_overflow': [],
        'structure_single_after_overflow': [],
        'structure_double_after_overflow': [],
        'structure_deep_after_overflow': [],
        'overflow_used_label_greedy': [],
        'active_candidate_n': [],
        'active_topk_cfg': [],
        'active_source_candidate_n': [],
        'active_ratio': [],
        'prior_score_mean': [],
        'prior_score_p90': [],
        'fragment_node_n_valid': [],
        'fragment_node_unique_bin_n': [],
        'fragment_node_support_coverage': [],
        'fragment_node_intensity_coverage': [],
        'fragment_node_true_hit_ratio': [],
        'fragment_node_top20_recall': [],
        'fragment_node_true_int_sum': [],
        'fragment_node_true_hit_node_n': [],
        'fragment_node_true_hit_unique_bin_n': [],
        'fragment_node_true_hit_unique_bin_ratio': [],
        'fragment_node_top20_hit_node_n': [],
        'fragment_node_top20_hit_unique_bin_n': [],
        'fragment_node_true_int_sum_node': [],
    }
    diag_rows = 0
    diag_summary_payload = None
    peak_audit_summary = None
    try:
        if cache_diag_path:
            diag_file = open(cache_diag_path, 'w', encoding='utf-8')

        for result in tqdm(task_stream, total=n):
            completed += 1
            if result.get('ok', False):
                ok_count += 1
                cache_diag = result.get('cache_diag', None)
                if isinstance(cache_diag, dict):
                    diag_rows += 1
                    if diag_file is not None:
                        _append_jsonl(diag_file, cache_diag)
                    for k in diag_acc.keys():
                        if k in cache_diag:
                            diag_acc[k].append(cache_diag.get(k, None))
            else:
                fail_count += 1
                err_msg = str(result.get('error_message', 'unknown error'))
                error_counts[err_msg] = error_counts.get(err_msg, 0) + 1
                if error_counts[err_msg] <= args.max_error_prints_per_type:
                    print(f"[cache-condv2] row={result.get('src_i')} failed: {err_msg}", flush=True)
                else:
                    suppressed_count += 1

            if completed % max(1, args.progress_every) == 0:
                elapsed = time.time() - start_ts
                print(
                    f'[cache-condv2] processed={completed}/{n} ok={ok_count} fail={fail_count} elapsed={elapsed:.1f}s',
                    flush=True,
                )
    finally:
        if diag_file is not None:
            diag_file.flush()
            diag_file.close()
        if pool is not None:
            pool.close()
            pool.join()

    # Stage 6: write completion markers and aggregate diagnostics.
    index_name = 'index.txt' if num_shards == 1 else f'index.shard{shard_id}.txt'
    with open(os.path.join(out_dir, index_name), 'w', encoding='utf-8') as f:
        f.write(str(n))

    elapsed = time.time() - start_ts
    print(f'[cache-condv2] complete total={n} ok={ok_count} fail={fail_count} elapsed={elapsed:.1f}s', flush=True)

    if diag_rows > 0:
        raw_stats = _safe_stat_dict(diag_acc['n_formulae_raw'], percentiles=(50, 90, 99))
        final_stats = _safe_stat_dict(diag_acc['n_formulae_final'], percentiles=(50, 90, 99))
        support_formula_stats = _safe_stat_dict(diag_acc['support_formula_n'], percentiles=(50, 90))
        coverage_before_stats = _safe_stat_dict(diag_acc['support_coverage_before_cap'], percentiles=(50, 90))
        coverage_after_stats = _safe_stat_dict(diag_acc['support_coverage_after_cap'], percentiles=(50, 90))
        candidate_mz_stats = _safe_stat_dict(diag_acc['candidate_mz_unique_n'], percentiles=(50, 90))
        os_ratio_stats = _safe_stat_dict(diag_acc['os_intensity_ratio'], percentiles=(50, 90))

        structure_before_stats = _safe_stat_dict(diag_acc['structure_selected_before_overflow'], percentiles=(50, 90))
        structure_after_stats = _safe_stat_dict(diag_acc['structure_selected_after_overflow'], percentiles=(50, 90))
        structure_deep_before_stats = _safe_stat_dict(
            diag_acc.get('structure_deep_before_overflow', []),
            percentiles=(50, 90),
        )
        structure_deep_after_stats = _safe_stat_dict(
            diag_acc.get('structure_deep_after_overflow', []),
            percentiles=(50, 90),
        )
        active_candidate_stats = _safe_stat_dict(diag_acc['active_candidate_n'], percentiles=(50, 90))
        active_ratio_stats = _safe_stat_dict(diag_acc['active_ratio'], percentiles=(50, 90))
        prior_score_mean_stats = _safe_stat_dict(diag_acc['prior_score_mean'], percentiles=(50, 90))
        prior_score_p90_stats = _safe_stat_dict(diag_acc['prior_score_p90'], percentiles=(50, 90))
        fragment_node_n_valid_stats = _safe_stat_dict(diag_acc['fragment_node_n_valid'], percentiles=(50, 90))
        fragment_node_unique_bin_stats = _safe_stat_dict(diag_acc['fragment_node_unique_bin_n'], percentiles=(50, 90))
        fragment_node_support_cov_stats = _safe_stat_dict(diag_acc['fragment_node_support_coverage'], percentiles=(50, 90))
        fragment_node_intensity_cov_stats = _safe_stat_dict(diag_acc['fragment_node_intensity_coverage'], percentiles=(50, 90))
        fragment_node_hit_ratio_stats = _safe_stat_dict(diag_acc['fragment_node_true_hit_ratio'], percentiles=(50, 90))
        fragment_node_top20_recall_stats = _safe_stat_dict(diag_acc['fragment_node_top20_recall'], percentiles=(50, 90))
        fragment_node_true_int_sum_stats = _safe_stat_dict(diag_acc['fragment_node_true_int_sum'], percentiles=(50, 90))
        fragment_node_true_hit_node_n_stats = _safe_stat_dict(diag_acc['fragment_node_true_hit_node_n'], percentiles=(50, 90))
        fragment_node_true_hit_unique_bin_n_stats = _safe_stat_dict(diag_acc['fragment_node_true_hit_unique_bin_n'], percentiles=(50, 90))
        fragment_node_true_hit_unique_bin_ratio_stats = _safe_stat_dict(diag_acc['fragment_node_true_hit_unique_bin_ratio'], percentiles=(50, 90))
        fragment_node_top20_hit_node_n_stats = _safe_stat_dict(diag_acc['fragment_node_top20_hit_node_n'], percentiles=(50, 90))
        fragment_node_top20_hit_unique_bin_n_stats = _safe_stat_dict(diag_acc['fragment_node_top20_hit_unique_bin_n'], percentiles=(50, 90))
        fragment_node_true_int_sum_node_stats = _safe_stat_dict(diag_acc['fragment_node_true_int_sum_node'], percentiles=(50, 90))
        leak_arr = np.asarray(diag_acc['overflow_used_label_greedy'], dtype=np.float64)
        leak_arr = leak_arr[np.isfinite(leak_arr)]
        overflow_used_label_greedy_ratio = float(np.mean(leak_arr > 0.5)) if leak_arr.size > 0 else float('nan')
        overflow_arr = np.asarray(diag_acc['overflow'], dtype=np.float64)
        overflow_arr = overflow_arr[np.isfinite(overflow_arr)]
        overflow_ratio = float(np.mean(overflow_arr > 0.5)) if overflow_arr.size > 0 else float('nan')

        ce_missing_arr = np.asarray(diag_acc['ce_missing'], dtype=np.float64)
        ce_missing_arr = ce_missing_arr[np.isfinite(ce_missing_arr)]
        ce_missing_ratio = float(np.mean(ce_missing_arr > 0.5)) if ce_missing_arr.size > 0 else float('nan')

        adduct_missing_arr = np.asarray(diag_acc['adduct_missing'], dtype=np.float64)
        adduct_missing_arr = adduct_missing_arr[np.isfinite(adduct_missing_arr)]
        adduct_missing_ratio = float(np.mean(adduct_missing_arr > 0.5)) if adduct_missing_arr.size > 0 else float('nan')

        def _fmt(stats, p2=False):
            if not isinstance(stats, dict):
                return 'n/a'
            if p2:
                return (
                    f"mean={stats.get('mean', float('nan')):.4f} "
                    f"p50={stats.get('p50', float('nan')):.4f} "
                    f"p90={stats.get('p90', float('nan')):.4f}"
                )
            return (
                f"mean={stats.get('mean', float('nan')):.4f} "
                f"p50={stats.get('p50', float('nan')):.4f} "
                f"p90={stats.get('p90', float('nan')):.4f} "
                f"p99={stats.get('p99', float('nan')):.4f}"
            )

        print(f"[cache-condv2][diag] rows={diag_rows}", flush=True)
        print(f"[cache-condv2][diag] n_formulae_raw {_fmt(raw_stats)}", flush=True)
        print(f"[cache-condv2][diag] n_formulae_final {_fmt(final_stats)}", flush=True)
        print(f"[cache-condv2][diag] overflow_ratio={overflow_ratio:.6f}", flush=True)
        print(f"[cache-condv2][diag] support_formula_n {_fmt(support_formula_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] support_coverage_before_cap {_fmt(coverage_before_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] support_coverage_after_cap {_fmt(coverage_after_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] os_intensity_ratio {_fmt(os_ratio_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] candidate_mz_unique_n {_fmt(candidate_mz_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] ce_missing_ratio={ce_missing_ratio:.6f} adduct_missing_ratio={adduct_missing_ratio:.6f}", flush=True)
        print(f"[cache-condv2][diag] structure_selected_before_overflow {_fmt(structure_before_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] structure_selected_after_overflow {_fmt(structure_after_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] overflow_used_label_greedy_ratio={overflow_used_label_greedy_ratio:.6f}", flush=True)
        print(f"[cache-condv2][diag] active_candidate_n {_fmt(active_candidate_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] active_ratio {_fmt(active_ratio_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] prior_score_mean {_fmt(prior_score_mean_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] prior_score_p90 {_fmt(prior_score_p90_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_n_valid {_fmt(fragment_node_n_valid_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_unique_bin_n {_fmt(fragment_node_unique_bin_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_support_coverage {_fmt(fragment_node_support_cov_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_intensity_coverage {_fmt(fragment_node_intensity_cov_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_hit_ratio {_fmt(fragment_node_hit_ratio_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_top20_recall {_fmt(fragment_node_top20_recall_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_int_sum {_fmt(fragment_node_true_int_sum_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_hit_node_n {_fmt(fragment_node_true_hit_node_n_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_hit_unique_bin_n {_fmt(fragment_node_true_hit_unique_bin_n_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_hit_unique_bin_ratio {_fmt(fragment_node_true_hit_unique_bin_ratio_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_top20_hit_node_n {_fmt(fragment_node_top20_hit_node_n_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_top20_hit_unique_bin_n {_fmt(fragment_node_top20_hit_unique_bin_n_stats, p2=True)}", flush=True)
        print(f"[cache-condv2][diag] fragment_node_true_int_sum_node {_fmt(fragment_node_true_int_sum_node_stats, p2=True)}", flush=True)
        diag_summary_payload = {
            'rows': int(diag_rows),
            'n_formulae_raw': raw_stats,
            'n_formulae_final': final_stats,
            'overflow_ratio': overflow_ratio,
            'support_formula_n': support_formula_stats,
            'support_coverage_before_cap': coverage_before_stats,
            'support_coverage_after_cap': coverage_after_stats,
            'os_intensity_ratio': os_ratio_stats,
            'candidate_mz_unique_n': candidate_mz_stats,
            'ce_missing_ratio': ce_missing_ratio,
            'adduct_missing_ratio': adduct_missing_ratio,

            'structure_selected_before_overflow': structure_before_stats,
            'structure_selected_after_overflow': structure_after_stats,
            'structure_deep_before_overflow': structure_deep_before_stats,
            'structure_deep_after_overflow': structure_deep_after_stats,

            'overflow_used_label_greedy_ratio': overflow_used_label_greedy_ratio,
            'active_candidate_n': active_candidate_stats,
            'active_ratio': active_ratio_stats,
            'prior_score_mean': prior_score_mean_stats,
            'prior_score_p90': prior_score_p90_stats,
            'fragment_node_n_valid': fragment_node_n_valid_stats,
            'fragment_node_unique_bin_n': fragment_node_unique_bin_stats,
            'fragment_node_support_coverage': fragment_node_support_cov_stats,
            'fragment_node_intensity_coverage': fragment_node_intensity_cov_stats,
            'fragment_node_true_hit_ratio': fragment_node_hit_ratio_stats,
            'fragment_node_top20_recall': fragment_node_top20_recall_stats,
            'fragment_node_true_int_sum': fragment_node_true_int_sum_stats,
            'fragment_node_true_hit_node_n': fragment_node_true_hit_node_n_stats,
            'fragment_node_true_hit_unique_bin_n': fragment_node_true_hit_unique_bin_n_stats,
            'fragment_node_true_hit_unique_bin_ratio': fragment_node_true_hit_unique_bin_ratio_stats,
            'fragment_node_top20_hit_node_n': fragment_node_top20_hit_node_n_stats,
            'fragment_node_top20_hit_unique_bin_n': fragment_node_top20_hit_unique_bin_n_stats,
            'fragment_node_true_int_sum_node': fragment_node_true_int_sum_node_stats,
        }
        try:
            summary_path = os.path.join(out_dir, 'cache_diag_summary.json')
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(diag_summary_payload, f, ensure_ascii=True, indent=2)
            print(f'[cache-condv2] cache_diag_summary={summary_path}', flush=True)
        except Exception as e:
            print(f'[cache-condv2] failed to write cache_diag_summary: {e}', flush=True)

    try:
        peak_audit_summary = _run_three_stage_peak_audit(
            out_dir=out_dir,
            selected_pairs=selected_pairs,
            split=split,
            use_source_index_filenames=use_source_index_filenames,
            mol_n=int(args.peak_audit_mol_n),
            csv_name=str(args.peak_audit_csv),
        )
        print(
            '[cache-condv2][peak-audit] '
            f"rows={peak_audit_summary.get('rows', 0)} "
            f"candidate_peak_count_mean={peak_audit_summary.get('candidate_peak_count_mean', float('nan')):.6f} "
            f"unique_bin_count_mean={peak_audit_summary.get('unique_bin_count_mean', float('nan')):.6f} "
            f"single_peak_candidate_ratio={peak_audit_summary.get('single_peak_candidate_ratio', float('nan')):.6f}",
            flush=True,
        )
        print(
            f"[cache-condv2] peak_audit_csv={peak_audit_summary.get('csv_path', '')} "
            f"peak_audit_summary={peak_audit_summary.get('summary_path', '')}",
            flush=True,
        )
    except Exception as e:
        print(f'[cache-condv2] peak audit failed: {e}', flush=True)
        peak_audit_summary = {
            'rows': 0,
            'error': str(e),
        }

    peak_quality_assert_errors = []
    peak_quality_assert_pass = True
    if int(args.enforce_peak_quality_assert) == 1:
        cpcm = float(peak_audit_summary.get('candidate_peak_count_mean', float('nan')))
        cp25 = float(peak_audit_summary.get('candidate_peak_count_p25', float('nan')))
        ubcm = float(peak_audit_summary.get('unique_bin_count_mean', float('nan')))
        spcr = float(peak_audit_summary.get('single_peak_candidate_ratio', float('nan')))
        mpcr = float(peak_audit_summary.get('multi_peak_candidate_ratio', float('nan')))
        rbrm = float(peak_audit_summary.get('repeated_bin_ratio_mean', float('nan')))

        if (not np.isfinite(cpcm)) or (cpcm <= float(args.min_candidate_peak_count_mean)):
            peak_quality_assert_errors.append(
                f'candidate_peak_count_mean={cpcm:.6f} <= {float(args.min_candidate_peak_count_mean):.6f}'
            )

        if (not np.isfinite(cp25)) or (cp25 <= float(args.min_candidate_peak_count_p25)):
            peak_quality_assert_errors.append(
                f'candidate_peak_count_p25={cp25:.6f} <= {float(args.min_candidate_peak_count_p25):.6f}'
            )

        if (not np.isfinite(ubcm)) or (ubcm <= float(args.min_unique_bin_count_mean)):
            peak_quality_assert_errors.append(
                f'unique_bin_count_mean={ubcm:.6f} <= {float(args.min_unique_bin_count_mean):.6f}'
            )

        if (not np.isfinite(spcr)) or (spcr >= float(args.max_single_peak_candidate_ratio)):
            peak_quality_assert_errors.append(
                f'single_peak_candidate_ratio={spcr:.6f} >= {float(args.max_single_peak_candidate_ratio):.6f}'
            )

        if (not np.isfinite(mpcr)) or (mpcr <= float(args.min_multi_peak_candidate_ratio)):
            peak_quality_assert_errors.append(
                f'multi_peak_candidate_ratio={mpcr:.6f} <= {float(args.min_multi_peak_candidate_ratio):.6f}'
            )

        if (not np.isfinite(rbrm)) or (rbrm >= float(args.max_repeated_bin_ratio_mean)):
            peak_quality_assert_errors.append(
                f'repeated_bin_ratio_mean={rbrm:.6f} >= {float(args.max_repeated_bin_ratio_mean):.6f}'
            )

        if peak_quality_assert_errors:
            peak_quality_assert_pass = False

    observed_n_formulae_final_max = None
    if len(diag_acc.get('n_formulae_final', [])) > 0:
        arr_final = np.asarray(diag_acc['n_formulae_final'], dtype=np.float64)
        arr_final = arr_final[np.isfinite(arr_final)]
        if arr_final.size > 0:
            observed_n_formulae_final_max = int(np.max(arr_final))

    def _finite_or_none(v):
        try:
            vv = float(v)
            return vv if np.isfinite(vv) else None
        except Exception:
            return None

    manifest_name = 'cache_manifest.json' if num_shards == 1 else f'cache_manifest.shard{shard_id}.json'
    manifest_path = os.path.join(out_dir, manifest_name)
    manifest_payload = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'split': str(split),
        'input_format': str(args.input_format),
        'parquet': str(pq),
        'tsv': str(args.tsv),
        'nist_mol_dir': str(args.nist_mol_dir),
        'nist_msp_path': str(args.nist_msp_path),
        'split_seed': int(args.split_seed),
        'split_ratios': str(args.split_ratios),
        'pre_filter_rows': int(pre_filter_n),
        'post_filter_rows': int(post_filter_n),
        'final_retained_rows': int(n),
        'selected_row_indices_path': str(selected_indices_path),
        'num_shards': int(num_shards),
        'shard_id': int(shard_id),
        'filters': {
            'require_ce': int(args.require_ce),
            'allowed_adducts': [str(x) for x in allowed_adducts],
            'allowed_instruments': [str(x) for x in allowed_instruments],
            'max_precursor_mz': float(args.max_precursor_mz),
            'strict_molecule_policy': int(args.strict_molecule_policy),
            'identifier_whitelist_n': int(len(identifier_whitelist) if identifier_whitelist is not None else 0),
            'max_heavy_atoms': int(max_heavy_atoms),
            # 'max_raw_unique_formulae': int(max_raw_unique_formulae),
            'prefilter_max_raw_unique_formulae': int(prefilter_max_raw_unique_formulae),
            'hard_max_raw_unique_formulae': int(hard_max_raw_unique_formulae),
            'formula_atomicnos': [int(x) for x in formula_atomicnos],
        },
        'official_target_cache': {
            'official_bin_width': float(args.official_bin_width),
            'official_max_mz': float(args.official_max_mz),
            'official_exclude_precursor': int(args.official_exclude_precursor),
            'official_bin_mode': os.environ.get("OFFICIAL_BIN_MODE", "floor"),
            'official_exclude_precursor_tol_da': os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.0"),
            'official_exclude_precursor_isotope_n': os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "0"),
            'true_topk': int(args.true_topk),
            'candidate_agg_fields': [
                'formulae_peaks_official_idx_agg',
                'formulae_peaks_official_intensity_agg',
            ],
            'true_fields': [
                'true_official_idx',
                'true_official_intensity',
                'true_top20_official_idx',
                'true_top20_official_intensity',
            ],
        },
        'ce_missing': {
            'before_filter_ratio': _finite_or_none(ce_missing_ratio_before_filter),
            'after_filter_ratio': _finite_or_none(ce_missing_ratio_after_filter),
            'before_filter_missing_n': int(ce_missing_before_n),
        },
        'distribution': {
            'instrument_before_filter': instrument_distribution_before_filter,
            'instrument_after_filter': instrument_distribution_filtered,
            'adduct_before_filter': adduct_distribution_before_filter,
            'adduct_after_filter': adduct_distribution_filtered,
        },
        'reject_stats': reject_stats,
        'max_formulae': int(args.max_formulae),
        'max_formulae_effective': int(featurizer_config['explicit_formulae_config']['max_formulae']),
        'observed_n_formulae_final_max': observed_n_formulae_final_max,
        'cache_diag_summary': diag_summary_payload,
        'peak_audit_summary': peak_audit_summary,
        'peak_quality_assert': {
            'enabled': int(args.enforce_peak_quality_assert),
            'pass': bool(peak_quality_assert_pass),
            'errors': [str(x) for x in peak_quality_assert_errors],
            'thresholds': {
                'candidate_peak_count_mean_gt': float(args.min_candidate_peak_count_mean),
                'candidate_peak_count_p25_gt': float(args.min_candidate_peak_count_p25),
                'unique_bin_count_mean_gt': float(args.min_unique_bin_count_mean),
                'single_peak_candidate_ratio_lt': float(args.max_single_peak_candidate_ratio),
                'multi_peak_candidate_ratio_gt': float(args.min_multi_peak_candidate_ratio),
                'repeated_bin_ratio_mean_lt': float(args.max_repeated_bin_ratio_mean),
            }
        },
    }
    try:
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest_payload, f, ensure_ascii=True, indent=2)
        print(f'[cache-condv2] cache_manifest={manifest_path}', flush=True)
    except Exception as e:
        print(f'[cache-condv2] failed to write cache_manifest: {e}', flush=True)

    if int(args.enforce_peak_quality_assert) == 1 and (not peak_quality_assert_pass):
        raise RuntimeError('Peak quality assertion failed: ' + '; '.join(peak_quality_assert_errors))

    if dedup_enabled:
        print(
            f'[cache-condv2] structure-dedup stats: unique_structures={len(unique_struct_keys)} '
            f'mem_hits={struct_hit_mem} disk_hits={struct_hit_disk} computed={struct_miss_compute}',
            flush=True,
        )
    if error_counts:
        print('[cache-condv2] error summary (count x message):', flush=True)
        for msg, cnt in sorted(error_counts.items(), key=lambda kv: kv[1], reverse=True):
            print(f'  {cnt}x {msg}', flush=True)
        if suppressed_count > 0:
            print(f'[cache-condv2] suppressed repetitive errors: {suppressed_count}', flush=True)


if __name__ == '__main__':
    main()
