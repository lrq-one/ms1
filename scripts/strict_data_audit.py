#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem


def _to_opt_str(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    s = str(v).strip()
    return s if s else None


def _to_opt_float(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, (int, float)):
        try:
            vv = float(v)
            return vv if np.isfinite(vv) else None
        except Exception:
            return None
    s = str(v).strip()
    if not s:
        return None
    try:
        vv = float(s)
        return vv if np.isfinite(vv) else None
    except Exception:
        pass
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m is None:
        return None
    try:
        vv = float(m.group(0))
        return vv if np.isfinite(vv) else None
    except Exception:
        return None


def _safe_counter(counter_obj):
    items = sorted(counter_obj.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
    out = {}
    for k, v in items:
        out[str(k)] = int(v)
    return out


def _bucket_idx(v, edges):
    if v is None or (not np.isfinite(v)):
        return -1
    x = float(v)
    for i in range(len(edges) - 1):
        lo = float(edges[i])
        hi = float(edges[i + 1])
        if (x >= lo) and (x < hi):
            return int(i)
    return int(len(edges) - 2)


def _quantile_edges(values, n_bins):
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size <= 0:
        return [0.0, 1.0]
    qs = np.linspace(0.0, 1.0, int(max(2, n_bins + 1)))
    edges = [float(np.quantile(arr, q)) for q in qs]
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9
    return edges


def _estimate_formula_n_raw(mol, allowed_atomicnos):
    counts = {int(an): 0 for an in allowed_atomicnos}
    implicit_h_total = 0
    for atom in mol.GetAtoms():
        an = int(atom.GetAtomicNum())
        if an in counts:
            counts[an] += 1
        if an != 1:
            try:
                implicit_h_total += int(atom.GetNumImplicitHs()) + int(atom.GetNumExplicitHs())
            except Exception:
                pass
    if 1 in counts and implicit_h_total > 0:
        counts[1] += int(implicit_h_total)

    out = 1
    for an in allowed_atomicnos:
        out *= (int(counts.get(int(an), 0)) + 1)
    return int(out)


def _check_nonempty_spectrum(spect_raw):
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
            return False
        if peaks.ndim != 2 or peaks.shape[1] != 2:
            peaks = peaks.reshape(-1, 2)
        if peaks.shape[0] <= 0:
            return False
        ints = peaks[:, 1]
        if not np.isfinite(ints).any():
            return False
        return float(np.nansum(np.clip(ints, 0.0, None))) > 0.0
    except Exception:
        return False


def _sanitize_ok(mol):
    try:
        m2 = Chem.Mol(mol)
        Chem.SanitizeMol(m2)
        return True
    except Exception:
        return False


def _canonical_keys(mol):
    try:
        c_smi = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    except Exception:
        c_smi = None
    try:
        c_key = Chem.MolToInchiKey(mol)
    except Exception:
        c_key = None
    c_smi = _to_opt_str(c_smi)
    c_key = _to_opt_str(c_key)
    if (c_smi is None) or (c_key is None):
        return None, None
    return c_smi, c_key


def _align_check(df_parquet, df_tsv_split):
    if len(df_parquet) != len(df_tsv_split):
        raise ValueError(f'Length mismatch: parquet={len(df_parquet)} tsv_split={len(df_tsv_split)}')

    if len(df_parquet) <= 0:
        return

    idxs = [0, len(df_parquet) // 2, len(df_parquet) - 1]
    for i in idxs:
        p_smiles = _to_opt_str(df_parquet.iloc[i].get('smiles', None))
        t_smiles = _to_opt_str(df_tsv_split.iloc[i].get('smiles', None))
        if p_smiles != t_smiles:
            raise ValueError(f'Alignment failed at row={i}: parquet_smiles={p_smiles} tsv_smiles={t_smiles}')


def _build_policy(args, allowed_atomicnos):
    allowed_symbols = []
    pt = Chem.GetPeriodicTable()
    for an in allowed_atomicnos:
        try:
            allowed_symbols.append(str(pt.GetElementSymbol(int(an))))
        except Exception:
            pass

    return {
        'name': 'strict_data_policy_v1',
        'adduct_allowlist': [str(args.allowed_adduct)],
        'max_precursor_mz': float(args.max_precursor_mz),
        'require_collision_energy': True,
        'allowed_elements': allowed_symbols,
        'max_heavy_atoms': int(args.max_heavy_atoms),
        'require_neutral_charge': True,
        'require_no_radical_electrons': True,
        'require_single_molecule': True,
        'require_rdkit_sanitize_success': True,
        'require_canonical_smiles_inchikey': True,
        'require_nonempty_nonzero_spectrum': True,
        'do_not_merge_ce': True,
        'group_split_key': str(args.group_key),
        'mini_sampling': {
            'train_n': int(args.mini_train),
            'val_n': int(args.mini_val),
            'seed': int(args.mini_seed),
        },
    }


def _scan_split(split_name, pq_path, df_tsv_split, args, allowed_atomicnos):
    df = pd.read_parquet(pq_path)
    _align_check(df, df_tsv_split)

    raw_adduct = Counter()
    raw_instrument = Counter()
    raw_ce_missing = 0
    raw_precursor_missing = 0

    reject = Counter()
    kept = []

    allowed_set = set(int(x) for x in allowed_atomicnos)
    for i in range(len(df)):
        row = df.iloc[i]
        trow = df_tsv_split.iloc[i]

        identifier = _to_opt_str(trow.get('identifier', None)) or f'{split_name}:{i}'
        adduct = _to_opt_str(trow.get('adduct', None))
        ce = _to_opt_float(trow.get('collision_energy', None))
        precursor_mz = _to_opt_float(trow.get('precursor_mz', None))
        instrument = _to_opt_str(trow.get('instrument_type', None))

        raw_adduct[adduct or '__MISSING__'] += 1
        raw_instrument[instrument or '__MISSING__'] += 1
        if ce is None:
            raw_ce_missing += 1
        if precursor_mz is None:
            raw_precursor_missing += 1

        mol = Chem.Mol(row['rdmol'])
        keep = True

        if adduct != args.allowed_adduct:
            reject['adduct_not_allowed'] += 1
            keep = False

        if keep:
            if precursor_mz is None or (precursor_mz <= 0.0) or (precursor_mz > float(args.max_precursor_mz)):
                reject['precursor_mz_invalid'] += 1
                keep = False

        if keep and args.require_ce == 1 and ce is None:
            reject['ce_missing'] += 1
            keep = False

        if keep:
            allowed_elements_ok = all(int(atom.GetAtomicNum()) in allowed_set for atom in mol.GetAtoms())
            if not allowed_elements_ok:
                reject['allowed_elements_not_ok'] += 1
                keep = False

        heavy_atom_n = int(mol.GetNumHeavyAtoms()) if keep else int(mol.GetNumHeavyAtoms())
        if keep and heavy_atom_n > int(args.max_heavy_atoms):
            reject['heavy_atom_overflow'] += 1
            keep = False

        if keep and (not _sanitize_ok(mol)):
            reject['sanitize_failed'] += 1
            keep = False

        if keep:
            try:
                formal_charge = int(sum(int(a.GetFormalCharge()) for a in mol.GetAtoms()))
            except Exception:
                formal_charge = 1
            if formal_charge != 0:
                reject['charged_molecule'] += 1
                keep = False

        if keep:
            try:
                radical_e = int(sum(int(a.GetNumRadicalElectrons()) for a in mol.GetAtoms()))
            except Exception:
                radical_e = 1
            if radical_e != 0:
                reject['radical_molecule'] += 1
                keep = False

        if keep:
            try:
                frag_n = int(len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)))
            except Exception:
                frag_n = 2
            if frag_n != 1:
                reject['multi_component'] += 1
                keep = False

        c_smi, c_key = _canonical_keys(mol) if keep else (None, None)
        if keep and ((c_smi is None) or (c_key is None)):
            reject['canonical_failed'] += 1
            keep = False

        if keep and (not _check_nonempty_spectrum(row.get('spect', None))):
            reject['empty_or_zero_spectrum'] += 1
            keep = False

        if keep:
            try:
                ring_n = int(mol.GetRingInfo().NumRings())
            except Exception:
                ring_n = 0
            try:
                aromatic_n = int(sum(int(a.GetIsAromatic()) for a in mol.GetAtoms()))
            except Exception:
                aromatic_n = 0
            formula_n_raw_est = _estimate_formula_n_raw(mol, allowed_atomicnos)
            group_key = c_key if args.group_key == 'inchikey' else c_smi
            kept.append({
                'row_idx': int(i),
                'identifier': identifier,
                'group_key': group_key,
                'canonical_smiles': c_smi,
                'canonical_inchikey': c_key,
                'adduct': adduct,
                'precursor_mz': float(precursor_mz),
                'ce': float(ce) if ce is not None else None,
                'heavy_atom_n': int(heavy_atom_n),
                'ring_n': int(ring_n),
                'aromatic_atom_n': int(aromatic_n),
                'formula_n_raw_est': int(formula_n_raw_est),
                'instrument_type': instrument,
            })

    raw_summary = {
        'split': split_name,
        'spectra_n': int(len(df)),
        'molecule_group_n': int(len(set(_to_opt_str(v) for v in df_tsv_split.get('inchikey', pd.Series(dtype=str)).tolist()))),
        'adduct_counts': _safe_counter(raw_adduct),
        'instrument_counts': _safe_counter(raw_instrument),
        'ce_missing_n': int(raw_ce_missing),
        'ce_missing_ratio': float(raw_ce_missing / max(1, len(df))),
        'precursor_mz_missing_n': int(raw_precursor_missing),
        'precursor_mz_missing_ratio': float(raw_precursor_missing / max(1, len(df))),
        'ms_level_counts': {'2': int(len(df))},
    }

    filtered_summary = {
        'split': split_name,
        'kept_spectra_n': int(len(kept)),
        'kept_molecule_group_n': int(len(set(x['group_key'] for x in kept))),
        'reject_counts': _safe_counter(reject),
        'keep_ratio': float(len(kept) / max(1, len(df))),
    }

    return raw_summary, filtered_summary, kept


def _stratified_pick(records, target_n, seed):
    if target_n <= 0 or len(records) <= 0:
        return []
    if target_n >= len(records):
        return [r['identifier'] for r in records]

    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for r in records:
        by_stratum[r['stratum']].append(r)

    strata = sorted(by_stratum.keys(), key=lambda k: len(by_stratum[k]), reverse=True)
    alloc = {k: 0 for k in strata}

    base_take = min(target_n, len(strata))
    for k in strata[:base_take]:
        alloc[k] = 1

    remaining = target_n - base_take
    if remaining > 0:
        caps = {k: max(0, len(by_stratum[k]) - alloc[k]) for k in strata}
        total_cap = float(sum(caps.values()))
        if total_cap > 0:
            frac = []
            for k in strata:
                x = remaining * (float(caps[k]) / total_cap)
                a = int(math.floor(x))
                a = min(a, caps[k])
                alloc[k] += a
                frac.append((x - a, k))
            used = sum(alloc.values())
            leftover = target_n - used
            frac.sort(reverse=True)
            for _, k in frac:
                if leftover <= 0:
                    break
                if alloc[k] < len(by_stratum[k]):
                    alloc[k] += 1
                    leftover -= 1

    picked = []
    for k in strata:
        n_take = int(alloc[k])
        pool = list(by_stratum[k])
        if n_take <= 0:
            continue
        if n_take >= len(pool):
            picked.extend(pool)
        else:
            picked.extend(rng.sample(pool, n_take))

    if len(picked) > target_n:
        picked = picked[:target_n]
    if len(picked) < target_n:
        remain = [r for r in records if r['identifier'] not in set(x['identifier'] for x in picked)]
        if len(remain) > 0:
            need = min(target_n - len(picked), len(remain))
            picked.extend(rng.sample(remain, need))

    return [r['identifier'] for r in picked]


def _make_strata(records):
    if len(records) <= 0:
        return records

    precursor_edges = [0.0, 200.0, 400.0, 700.0, 1000.0, 1500.0001]
    heavy_edges = [0.0, 12.0, 20.0, 30.0, 45.0, 60.0001]
    ce_edges = [0.0, 10.0, 20.0, 35.0, 50.0, 70.0, 1e9]
    ring_edges = [0.0, 1.0, 3.0, 6.0, 1000.0]

    raw_vals = [math.log1p(float(r.get('formula_n_raw_est', 0))) for r in records]
    raw_edges = _quantile_edges(raw_vals, n_bins=4)

    out = []
    for r in records:
        p_bin = _bucket_idx(r.get('precursor_mz', None), precursor_edges)
        h_bin = _bucket_idx(float(r.get('heavy_atom_n', 0)), heavy_edges)
        ce_v = r.get('ce', None)
        ce_bin = _bucket_idx(ce_v, ce_edges)
        ring_bin = _bucket_idx(float(r.get('ring_n', 0)), ring_edges)
        raw_bin = _bucket_idx(math.log1p(float(r.get('formula_n_raw_est', 0))), raw_edges)
        rr = dict(r)
        rr['stratum'] = f'p{p_bin}|h{h_bin}|ce{ce_bin}|r{ring_bin}|f{raw_bin}'
        out.append(rr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='data/MassSpecGym.tsv')
    ap.add_argument('--train-parquet', default='data/massspecgym/massspecgym_train.parquet')
    ap.add_argument('--val-parquet', default='data/massspecgym/massspecgym_val.parquet')
    ap.add_argument('--test-parquet', default='data/massspecgym/massspecgym_test.parquet')
    ap.add_argument('--out-dir', default='outputs/strict_data_audit')
    ap.add_argument('--allowed-adduct', default='[M+H]+')
    ap.add_argument('--allowed-elements', default='H,C,O,N,P,S,F,Cl,Br,I,Se,Si')
    ap.add_argument('--max-precursor-mz', type=float, default=1500.0)
    ap.add_argument('--max-heavy-atoms', type=int, default=60)
    ap.add_argument('--require-ce', type=int, default=1)
    ap.add_argument('--group-key', default='inchikey', choices=['inchikey', 'smiles'])
    ap.add_argument('--mini-train', type=int, default=256)
    ap.add_argument('--mini-val', type=int, default=128)
    ap.add_argument('--mini-seed', type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    pt = Chem.GetPeriodicTable()
    allowed_atomicnos = []
    for sym in [x.strip() for x in str(args.allowed_elements).split(',') if x.strip()]:
        try:
            an = int(pt.GetAtomicNumber(sym))
        except Exception:
            an = 0
        if an > 0 and an not in allowed_atomicnos:
            allowed_atomicnos.append(an)
    if len(allowed_atomicnos) <= 0:
        raise ValueError('allowed-elements parsed to empty set')

    policy = _build_policy(args, allowed_atomicnos)

    tsv_cols = ['identifier', 'smiles', 'inchikey', 'fold', 'collision_energy', 'adduct', 'precursor_mz', 'instrument_type']
    df_tsv = pd.read_csv(args.tsv, sep='\t', usecols=tsv_cols)
    split_to_tsv = {
        'train': df_tsv[df_tsv['fold'] == 'train'].reset_index(drop=True),
        'val': df_tsv[df_tsv['fold'] == 'val'].reset_index(drop=True),
        'test': df_tsv[df_tsv['fold'] == 'test'].reset_index(drop=True),
    }

    split_paths = {
        'train': args.train_parquet,
        'val': args.val_parquet,
        'test': args.test_parquet,
    }

    raw_by_split = {}
    filtered_by_split = {}
    kept_by_split = {}

    for split in ['train', 'val', 'test']:
        raw_summary, filtered_summary, kept = _scan_split(
            split,
            split_paths[split],
            split_to_tsv[split],
            args,
            allowed_atomicnos,
        )
        raw_by_split[split] = raw_summary
        filtered_by_split[split] = filtered_summary
        kept_by_split[split] = kept

    raw_total_spectra = sum(int(v['spectra_n']) for v in raw_by_split.values())
    raw_total_mol = len(set(
        _to_opt_str(x) for split in ['train', 'val', 'test'] for x in split_to_tsv[split]['inchikey'].tolist()
    ))

    raw_data_summary = {
        'policy_name': policy['name'],
        'by_split': raw_by_split,
        'total_spectra_n': int(raw_total_spectra),
        'total_molecule_group_n': int(raw_total_mol),
    }

    filtered_total_spectra = sum(int(v['kept_spectra_n']) for v in filtered_by_split.values())
    filtered_total_mol = len(set(
        rec['group_key']
        for split in ['train', 'val', 'test']
        for rec in kept_by_split[split]
    ))
    filtered_data_summary = {
        'policy': policy,
        'by_split': filtered_by_split,
        'total_kept_spectra_n': int(filtered_total_spectra),
        'total_kept_molecule_group_n': int(filtered_total_mol),
    }

    group_sets = {
        split: set(rec['group_key'] for rec in kept_by_split[split])
        for split in ['train', 'val', 'test']
    }
    train_val_overlap = sorted(group_sets['train'].intersection(group_sets['val']))
    train_test_overlap = sorted(group_sets['train'].intersection(group_sets['test']))
    val_test_overlap = sorted(group_sets['val'].intersection(group_sets['test']))

    split_leakage_report = {
        'group_key': args.group_key,
        'group_counts': {split: int(len(group_sets[split])) for split in ['train', 'val', 'test']},
        'train_val_overlap_n': int(len(train_val_overlap)),
        'train_test_overlap_n': int(len(train_test_overlap)),
        'val_test_overlap_n': int(len(val_test_overlap)),
        'train_val_overlap_examples': train_val_overlap[:20],
        'train_test_overlap_examples': train_test_overlap[:20],
        'val_test_overlap_examples': val_test_overlap[:20],
    }

    train_records = _make_strata(kept_by_split['train'])
    val_records = _make_strata(kept_by_split['val'])
    mini_train_ids = _stratified_pick(train_records, int(args.mini_train), int(args.mini_seed))
    mini_val_ids = _stratified_pick(val_records, int(args.mini_val), int(args.mini_seed) + 17)

    mini_payload = {
        'train': mini_train_ids,
        'val': mini_val_ids,
        'meta': {
            'train_n': int(len(mini_train_ids)),
            'val_n': int(len(mini_val_ids)),
            'seed': int(args.mini_seed),
            'strategy': 'stratified_precursor_heavy_ce_ring_formula',
        },
    }

    out_paths = {
        'policy': os.path.join(args.out_dir, 'strict_data_policy.json'),
        'raw': os.path.join(args.out_dir, 'raw_data_summary.json'),
        'filtered': os.path.join(args.out_dir, 'filtered_data_summary.json'),
        'leakage': os.path.join(args.out_dir, 'split_leakage_report.json'),
        'mini_ids': os.path.join(args.out_dir, 'mini_identifier_whitelist.json'),
    }

    with open(out_paths['policy'], 'w', encoding='utf-8') as f:
        json.dump(policy, f, ensure_ascii=True, indent=2)
    with open(out_paths['raw'], 'w', encoding='utf-8') as f:
        json.dump(raw_data_summary, f, ensure_ascii=True, indent=2)
    with open(out_paths['filtered'], 'w', encoding='utf-8') as f:
        json.dump(filtered_data_summary, f, ensure_ascii=True, indent=2)
    with open(out_paths['leakage'], 'w', encoding='utf-8') as f:
        json.dump(split_leakage_report, f, ensure_ascii=True, indent=2)
    with open(out_paths['mini_ids'], 'w', encoding='utf-8') as f:
        json.dump(mini_payload, f, ensure_ascii=True, indent=2)

    print(f"[strict_data_audit] policy={out_paths['policy']}")
    print(f"[strict_data_audit] raw={out_paths['raw']}")
    print(f"[strict_data_audit] filtered={out_paths['filtered']}")
    print(f"[strict_data_audit] leakage={out_paths['leakage']}")
    print(f"[strict_data_audit] mini_ids={out_paths['mini_ids']}")
    print(
        f"[strict_data_audit] kept_spectra train={filtered_by_split['train']['kept_spectra_n']} "
        f"val={filtered_by_split['val']['kept_spectra_n']} test={filtered_by_split['test']['kept_spectra_n']}"
    )


if __name__ == '__main__':
    main()
