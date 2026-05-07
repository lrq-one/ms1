# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Feature engineering utilities that transform molecules/spectra into model-ready tensors.

import pandas as pd
import numpy as np
import sklearn.metrics
import torch
from numba import jit
import scipy.spatial
from rdkit import Chem
from rdkit.Chem import AllChem
import networkx as nx
import rdkit.Chem.Descriptors
import sys
import os
import hashlib
import re

from . import util
from .util import get_nos_coords, get_nos
try:
    from rassp.msutil import masscompute
except Exception:
    masscompute = None

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*') 


# Helper: convert sparse peaks to a dense fixed-width spectrum.
def _sparse_spect_to_dense(sparse_spect, spect_bin_config):
    if sparse_spect is None or spect_bin_config is None:
        return None
    try:
        spect_arr = np.asarray(sparse_spect, dtype=np.float32)
    except Exception:
        return None
    if spect_arr.size == 0:
        return np.zeros((int(spect_bin_config.get('bin_number', 0)),), dtype=np.float32)
    if spect_arr.ndim == 1 and spect_arr.shape[0] == int(spect_bin_config.get('bin_number', 0)):
        return spect_arr.astype(np.float32, copy=False)
    try:
        spect_arr = spect_arr.reshape(-1, 2)
    except Exception:
        return None
    try:
        from rassp.msutil import binutils

        spect_bin = binutils.create_spectrum_bins(**spect_bin_config)
        _, _, dense = spect_bin.histogram(spect_arr[:, 0], spect_arr[:, 1])
        return np.asarray(dense, dtype=np.float32)
    except Exception:
        return None


# Helper: resolve the formula atomic-number ordering used by the candidate vector.
def _resolve_formula_atomicnos(formula_atomicnos=None):
    if formula_atomicnos is not None:
        out = []
        for an in formula_atomicnos:
            try:
                an_i = int(an)
            except Exception:
                continue
            if an_i > 0 and an_i not in out:
                out.append(an_i)
        if out:
            return out

    raw = os.environ.get('FORMULA_ATOMICNOS', '').strip()
    if raw:
        out = []
        for token in raw.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                an_i = int(token)
            except Exception:
                continue
            if an_i > 0 and an_i not in out:
                out.append(an_i)
        if out:
            return out

    base_default = [1, 6, 7, 8, 9, 15, 16, 17]
    if os.environ.get('FORMULA_ATOMICNOS_TIER2', '0') == '1':
        for an in [11, 35, 53]:
            if an not in base_default:
                base_default.append(an)
    return base_default


# Helper: convert a simple molecular formula string into a count vector.
def _formula_string_to_counts(formula_str, formula_atomicnos):
    if not formula_str:
        return None

    atomicnos = _resolve_formula_atomicnos(formula_atomicnos)
    if not atomicnos:
        return None

    periodic = Chem.GetPeriodicTable()
    symbol_to_idx = {}
    for idx, an in enumerate(atomicnos):
        try:
            symbol = periodic.GetElementSymbol(int(an))
        except Exception:
            continue
        if symbol and symbol not in symbol_to_idx:
            symbol_to_idx[symbol] = idx

    counts = np.zeros((len(atomicnos),), dtype=np.float32)
    for sym, num_raw in re.findall(r'([A-Z][a-z]?)(\d*)', str(formula_str).strip()):
        if sym not in symbol_to_idx:
            continue
        try:
            cnt = int(num_raw) if num_raw else 1
        except Exception:
            cnt = 1
        counts[symbol_to_idx[sym]] += float(cnt)
    return counts


# Helper: compute a simple DBE heuristic from a count vector.
def _candidate_dbe_from_counts(counts, formula_atomicnos):
    atomicnos = _resolve_formula_atomicnos(formula_atomicnos)
    if not atomicnos:
        return None, None, None

    counts = np.asarray(counts, dtype=np.float32)
    if counts.ndim == 1:
        counts = counts.reshape(1, -1)
    if counts.shape[-1] < len(atomicnos):
        pad = np.zeros((counts.shape[0], len(atomicnos) - counts.shape[-1]), dtype=np.float32)
        counts = np.concatenate([counts, pad], axis=-1)
    elif counts.shape[-1] > len(atomicnos):
        counts = counts[:, :len(atomicnos)]

    lookup = {int(an): idx for idx, an in enumerate(atomicnos)}

    def _col(an):
        idx = lookup.get(int(an), None)
        if idx is None or idx >= counts.shape[1]:
            return np.zeros((counts.shape[0],), dtype=np.float32)
        return counts[:, idx]

    c = _col(6)
    h = _col(1)
    n = _col(7)
    p = _col(15)
    halogen = _col(9) + _col(17) + _col(35) + _col(53)
    dbe_num = (2.0 * c) + 2.0 + n + p - h - halogen
    dbe = 0.5 * dbe_num
    parity_violation = (np.mod(np.abs(np.rint(dbe_num)).astype(np.int64), 2) != 0).astype(np.float32)
    is_negative = (dbe < 0).astype(np.float32)
    return dbe.astype(np.float32), is_negative.astype(np.float32), parity_violation.astype(np.float32)


# Helper: score candidates by how much real-spectrum intensity they cover.
def _score_candidate_coverage_with_h_shift(masses, spect_dense, spect_bin_config=None, hydrogen_shift_bins=0):
    if masses is None or spect_dense is None:
        return None

    cand = np.asarray(masses, dtype=np.float32)
    if cand.ndim == 2 and cand.shape[-1] == 2:
        cand = cand[:, None, :]
    if cand.ndim != 3 or cand.shape[-1] < 2:
        return None

    true_dense = np.asarray(spect_dense, dtype=np.float32).reshape(-1)
    if true_dense.size == 0:
        return np.zeros((cand.shape[0],), dtype=np.float32)

    first_bin_center = 1.0
    bin_width = 1.0
    if isinstance(spect_bin_config, dict):
        first_bin_center = float(spect_bin_config.get('first_bin_center', first_bin_center))
        bin_width = float(spect_bin_config.get('bin_width', bin_width))

    peak_mass = cand[..., 0]
    peak_intensity = np.maximum(cand[..., 1], 0.0)
    valid_peak = np.isfinite(peak_mass) & np.isfinite(peak_intensity) & (peak_intensity > 0)
    peak_bin = np.floor((peak_mass - first_bin_center + (bin_width / 2.0)) / bin_width + 1e-8).astype(np.int64)

    score = np.zeros((cand.shape[0],), dtype=np.float32)
    max_shift = max(0, int(hydrogen_shift_bins))
    n_bins = int(true_dense.shape[0])
    for shift in range(-max_shift, max_shift + 1):
        shifted = peak_bin + int(shift)
        valid = valid_peak & (shifted >= 0) & (shifted < n_bins)
        shifted_safe = np.clip(shifted, 0, n_bins - 1)
        contrib = np.where(valid, true_dense[shifted_safe] * peak_intensity, 0.0).sum(axis=1)
        score = np.maximum(score, contrib.astype(np.float32))

    return score


# Function overview: mol_to_nums_adj handles a specific workflow step in this module.
def mol_to_nums_adj(mol, MAX_ATOM_N=None):
    """
    Return padded atomic numbers and a bond-order adjacency matrix.

    Legacy callers in atom_features.py and edge_features.py depend on this
    two-tuple interface.
    """
    atomic_nos = get_nos(mol).astype(np.uint8, copy=False)
    atom_n = int(len(atomic_nos))

    if MAX_ATOM_N is None:
        MAX_ATOM_N = atom_n
    else:
        MAX_ATOM_N = int(MAX_ATOM_N)
        if MAX_ATOM_N < 0:
            MAX_ATOM_N = atom_n

    atom_limit = min(atom_n, MAX_ATOM_N)
    atomic_nos_pad = np.zeros((MAX_ATOM_N,), dtype=np.uint8)
    atomic_nos_pad[:atom_limit] = atomic_nos[:atom_limit]

    adj = np.zeros((MAX_ATOM_N, MAX_ATOM_N), dtype=np.float32)
    for bond in mol.GetBonds():
        a_i = bond.GetBeginAtomIdx()
        a_j = bond.GetEndAtomIdx()
        if a_i >= MAX_ATOM_N or a_j >= MAX_ATOM_N:
            continue
        bond_order = float(bond.GetBondTypeAsDouble())
        adj[a_i, a_j] = bond_order
        adj[a_j, a_i] = bond_order

    return atomic_nos_pad, adj


# Function overview: feat_mol_adj handles a specific workflow step in this module.
def feat_mol_adj(mol, split_weights=None, edge_weighted=False, norm_adj=True, add_identity=True, mat_power=1):
    """
    Build bond-order adjacency channels for a molecule.

    Returns a C x N x N array, where C is the number of split weights when
    provided, otherwise 1.
    """
    _, adj_np = mol_to_nums_adj(mol)
    adj = torch.as_tensor(adj_np, dtype=torch.float32)
    atom_n = int(adj.shape[0])

    if split_weights is None or len(split_weights) == 0:
        base = adj if edge_weighted else (adj > 0).float()
        adj_out = base.unsqueeze(0)
    else:
        split_adj = torch.zeros((len(split_weights), atom_n, atom_n), dtype=torch.float32)
        for i, weight in enumerate(split_weights):
            weight = float(weight)
            mask = (adj == weight).float()
            split_adj[i] = mask * weight if edge_weighted else mask
        adj_out = split_adj

    if add_identity:
        eye = torch.eye(atom_n, dtype=adj_out.dtype)
        adj_out = adj_out + eye.unsqueeze(0)

    if norm_adj:
        normed = []
        for i in range(adj_out.shape[0]):
            a = adj_out[i]
            deg = torch.sum(a, dim=0)
            deg = torch.clamp(deg, min=1e-8)
            d_12 = 1.0 / torch.sqrt(deg)
            a_norm = d_12.reshape(atom_n, 1) * a * d_12.reshape(1, atom_n)
            if isinstance(mat_power, list):
                for p in mat_power:
                    normed.append(torch.matrix_power(a_norm, int(p)))
            else:
                if int(mat_power) > 1:
                    a_norm = torch.matrix_power(a_norm, int(mat_power))
                normed.append(a_norm)
        adj_out = torch.stack(normed, dim=0)
    elif isinstance(mat_power, list):
        raised = []
        for i in range(adj_out.shape[0]):
            for p in mat_power:
                raised.append(torch.matrix_power(adj_out[i], int(p)))
        adj_out = torch.stack(raised, dim=0)
    elif int(mat_power) > 1:
        adj_out = torch.stack([torch.matrix_power(adj_out[i], int(mat_power)) for i in range(adj_out.shape[0])], dim=0)

    return adj_out.detach().cpu().float()

# Function overview: feat_tensor_mol handles a specific workflow step in this module.
def feat_tensor_mol(mol, feat_distances=False,
                    feat_r_pow = None,
                    feat_r_max = None,
                    feat_r_onehot_tholds = [],
                    feat_r_gaussian_filters = [],
                    conf_embed_mol = False,
                    conf_opt_mmff = False,
                    conf_opt_uff = False,
                    
                    is_in_ring=False,
                    is_in_ring_size = None, 
                    MAX_POW_M = 2.0,
                    conf_idx = 0,
                    add_identity=False,
                    edge_type_tuples = [],
                    adj_pow_bin = [],
                    adj_pow_scale = [],
                    graph_props_config = {},
                    frag_props_config = {}, 
                    columb_mat = False,
                    dihedral_mat = False, 
                    dihedral_sincos_mat = False, 
                    norm_mat=False, mat_power=1):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    mol_init = mol
    if conf_embed_mol:
        mol_change = Chem.Mol(mol)
        try:
            Chem.AllChem.EmbedMolecule(mol_change)
            if conf_opt_mmff:
                Chem.AllChem.MMFFOptimizeMolecule(mol_change)
            elif conf_opt_uff:
                Chem.AllChem.UFFOptimizeMolecule(mol_change)
            if mol_change.GetNumConformers() > 0:
                mol = mol_change
        except Exception as e:
            print('error generating conformer', e)

        
    assert mol.GetNumConformers() > 0
    if feat_distances or feat_r_pow:
        atomic_nos, coords = get_nos_coords(mol, conf_idx)
    else:
        atomic_nos = get_nos(mol)
    ATOM_N = len(atomic_nos)

    if feat_distances:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = np.abs((a - a.T))
        c = np.swapaxes(b, 2, 1)
        res_mats.append(c)
    if feat_r_pow is not None:
        pos = coords
        a = pos.T.reshape(1, 3, -1)
        b = (a - a.T)**2
        c = np.swapaxes(b, 2, 1)
        d = np.sqrt(np.sum(c, axis=2))
        e = (np.eye(d.shape[0]) + d)[:, :, np.newaxis]
        if feat_r_max is not None:
            d[d >= feat_r_max] = 0.0
                       
        for p in feat_r_pow:
            e_pow = e**p
            if (e_pow > MAX_POW_M).any():
                e_pow = np.minimum(e_pow, MAX_POW_M)

            res_mats.append(e_pow)
        for th in feat_r_onehot_tholds:
            e_oh = (e <= th).astype(np.float32)
            res_mats.append(e_oh)

        for mu, sigma in feat_r_gaussian_filters:
            
            e_val = np.exp(-(e - mu)**2/(2*sigma**2))
            res_mats.append(e_val)
            
    if len(edge_type_tuples) > 0:
        a = np.zeros((ATOM_N, ATOM_N, len(edge_type_tuples)))
        for et_i, et in enumerate(edge_type_tuples):
            for b in mol.GetBonds():
                a_i = b.GetBeginAtomIdx()
                a_j = b.GetEndAtomIdx()
                if set(et) == set([atomic_nos[a_i], atomic_nos[a_j]]):
                    a[a_i, a_j, et_i] = 1
                    a[a_j, a_i, et_i] = 1
        res_mats.append(a)
        
    if is_in_ring:
        a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
        for b in mol.GetBonds():
            a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
            a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
        res_mats.append(a)
        
    if is_in_ring_size is not None:
        for rs in is_in_ring_size:
            a = np.zeros((ATOM_N, ATOM_N, 1), dtype=np.float32)
            for b in mol.GetBonds():
                if b.IsInRingSize(rs):
                    a[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] = 1
                    a[b.GetEndAtomIdx(), b.GetBeginAtomIdx()] = 1
            res_mats.append(a)
            
    if columb_mat:
        res_mats.append(np.expand_dims(get_columb_mat(mol, conf_idx), -1))

    if dihedral_mat:
        res_mats.append(np.expand_dims(get_dihedral_angles(mol, conf_idx), -1))
        
    if dihedral_sincos_mat:
        res_mats.append(get_dihedral_sincos(mol, conf_idx))
        
    if len(graph_props_config) > 0:
        res_mats.append(get_graph_props(mol, **graph_props_config))

def build_formulae_aux_feat(
    formulae_feats,
    formulae_peaks_mass_idx=None,
    formulae_peaks_intensity=None,
    formulae_mask=None,
    formula_atomicnos=None,
    precursor_mz=None,
    precursor_formula=None,
    parent_formula_counts=None,
    formulae_peaks_official_idx=None,
    formulae_peaks_official_intensity=None,
):
    """
    Build compact per-candidate formula descriptors for FormulaNet-style scoring.

    Fixed 15-D layout:
    0 atom_sum
    1 nonzero_element_count
    2 exact_mass
    3 DBE
    4 negative_DBE_flag
    5 DBE_parity_violation
    6 neutral_loss_mass
    7 neutral_loss_atom_count
    8 neutral_loss_nonzero_element_count
    9 halogen_count
    10 isotope_entropy
    11 mono_fraction
    12 M+1_fraction
    13 M+2_fraction
    14 precursor_gap
    """
    feats = np.asarray(formulae_feats, dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(-1, 1)
    elif feats.ndim == 0:
        feats = feats.reshape(1, 1)

    cand_n = int(feats.shape[0]) if feats.ndim > 0 else 0
    if cand_n <= 0:
        return np.zeros((0, 15), dtype=np.float32)

    formula_atomicnos = _resolve_formula_atomicnos(formula_atomicnos)
    atomic_count_n = len(formula_atomicnos)
    formula_counts = feats[:, :atomic_count_n] if feats.shape[1] >= atomic_count_n else feats
    if formula_counts.shape[1] < atomic_count_n:
        pad = np.zeros((cand_n, atomic_count_n - formula_counts.shape[1]), dtype=np.float32)
        formula_counts = np.concatenate([formula_counts, pad], axis=-1)

    formula_atom_sum = formula_counts.sum(axis=-1).astype(np.float32)
    formula_nonzero = (formula_counts > 0).sum(axis=-1).astype(np.float32)

    isotope_entropy = np.zeros((cand_n,), dtype=np.float32)
    mono_fraction = np.zeros((cand_n,), dtype=np.float32)
    m1_fraction = np.zeros((cand_n,), dtype=np.float32)
    m2_fraction = np.zeros((cand_n,), dtype=np.float32)
    precursor_gap = np.zeros((cand_n,), dtype=np.float32)

    def _reshape_peak_tensor(x, dtype, fill_value=0.0):
        arr = np.asarray(x, dtype=dtype)
        if arr.ndim == 0:
            arr = arr.reshape(1, 1)
        elif arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        elif arr.ndim > 2:
            arr = arr.reshape(arr.shape[0], -1)

        if arr.shape[0] < cand_n:
            pad = np.full((cand_n - arr.shape[0], arr.shape[1]), fill_value=fill_value, dtype=arr.dtype)
            arr = np.concatenate([arr, pad], axis=0)
        elif arr.shape[0] > cand_n:
            arr = arr[:cand_n]
        return arr

    idx_src = None
    int_src = None
    if formulae_peaks_official_idx is not None and formulae_peaks_official_intensity is not None:
        idx_src = _reshape_peak_tensor(formulae_peaks_official_idx, np.float32, fill_value=-1.0)
        int_src = _reshape_peak_tensor(formulae_peaks_official_intensity, np.float32, fill_value=0.0)
    elif formulae_peaks_mass_idx is not None and formulae_peaks_intensity is not None:
        idx_src = _reshape_peak_tensor(formulae_peaks_mass_idx, np.float32, fill_value=-1.0)
        int_src = _reshape_peak_tensor(formulae_peaks_intensity, np.float32, fill_value=0.0)

    if idx_src is not None and int_src is not None and idx_src.shape[1] == int_src.shape[1] and idx_src.shape[1] > 0:
        valid = np.isfinite(idx_src) & (idx_src >= 0.0) & np.isfinite(int_src) & (int_src > 0.0)
        if formulae_mask is not None:
            mask_vec = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)
            if mask_vec.shape[0] < cand_n:
                mask_vec = np.concatenate([mask_vec, np.zeros((cand_n - mask_vec.shape[0],), dtype=np.float32)], axis=0)
            valid = valid & (mask_vec[:cand_n] > 0.5)[:, None]

        w = np.where(valid, np.maximum(int_src, 0.0), 0.0).astype(np.float64)
        w_sum = np.clip(w.sum(axis=-1, keepdims=True), 1e-12, None)
        p = w / w_sum

        with np.errstate(divide='ignore', invalid='ignore'):
            plogp = np.where(p > 0.0, p * np.log(np.clip(p, 1e-12, None)), 0.0)
        entropy = -plogp.sum(axis=-1)
        peak_n = valid.sum(axis=-1).astype(np.float64)
        entropy_denom = np.log(np.maximum(peak_n, 2.0))
        entropy_denom = np.where(entropy_denom > 1e-12, entropy_denom, 1.0)
        isotope_entropy = np.where(peak_n > 1.0, entropy / entropy_denom, 0.0).astype(np.float32)

        base_idx = np.where(valid, idx_src, np.inf).min(axis=-1)
        base_idx = np.where(np.isfinite(base_idx), base_idx, 0.0)
        shift = np.rint(idx_src - base_idx[:, None]).astype(np.int64)

        mono_fraction = np.where(valid & (shift == 0), p, 0.0).sum(axis=-1).astype(np.float32)
        m1_fraction = np.where(valid & (shift == 1), p, 0.0).sum(axis=-1).astype(np.float32)
        m2_fraction = np.where(valid & (shift == 2), p, 0.0).sum(axis=-1).astype(np.float32)

    pt = Chem.GetPeriodicTable()
    atomic_weights = np.asarray([pt.GetMostCommonIsotopeMass(int(an)) for an in formula_atomicnos], dtype=np.float32)
    formula_mass = formula_counts @ atomic_weights if atomic_weights.size > 0 else np.zeros((cand_n,), dtype=np.float32)

    parent_counts = None
    if parent_formula_counts is not None:
        parent_counts = np.asarray(parent_formula_counts, dtype=np.float32).reshape(-1)
    elif precursor_formula:
        parent_counts = _formula_string_to_counts(precursor_formula, formula_atomicnos)
    if parent_counts is None:
        parent_counts = np.zeros((atomic_count_n,), dtype=np.float32)
    if parent_counts.shape[0] < atomic_count_n:
        parent_counts = np.concatenate([parent_counts, np.zeros((atomic_count_n - parent_counts.shape[0],), dtype=np.float32)], axis=0)
    elif parent_counts.shape[0] > atomic_count_n:
        parent_counts = parent_counts[:atomic_count_n]

    # Parent mass should be neutral mass for neutral-loss features.
    # For [M+H]+ data, precursor_mz is ion mass, while formula_mass is neutral formula mass.
    # Prefer precursor_formula-derived neutral mass whenever available.
    parent_formula_mass = None
    try:
        if atomic_weights.size > 0 and parent_counts is not None and float(np.sum(parent_counts)) > 0.0:
            parent_formula_mass = float(parent_counts @ atomic_weights)
    except Exception:
        parent_formula_mass = None

    try:
        proton_mass = float(os.environ.get('FORMULA_PROTON_MASS', '1.007276466812'))
    except Exception:
        proton_mass = 1.007276466812

    parent_ion_mass = None
    if precursor_mz is not None:
        try:
            parent_ion_mass = float(precursor_mz)
            if not np.isfinite(parent_ion_mass) or parent_ion_mass <= 0.0:
                parent_ion_mass = None
        except Exception:
            parent_ion_mass = None

    if parent_formula_mass is not None and np.isfinite(parent_formula_mass) and parent_formula_mass > 0.0:
        parent_neutral_mass = float(parent_formula_mass)
    elif parent_ion_mass is not None:
        # Main clean branch is [M+H]+, so subtract proton to recover neutral parent mass.
        parent_neutral_mass = max(0.0, float(parent_ion_mass) - float(proton_mass))
    else:
        parent_neutral_mass = 0.0

    neutral_loss_mass = np.asarray(np.maximum(parent_neutral_mass  - formula_mass, 0.0), dtype=np.float32)
    neutral_loss_counts = np.maximum(parent_counts.reshape(1, -1) - formula_counts, 0.0).astype(np.float32)
    neutral_loss_nonzero_atoms = (neutral_loss_counts > 0).sum(axis=-1).astype(np.float32)
    neutral_loss_total_atoms = neutral_loss_counts.sum(axis=-1).astype(np.float32)
    formula_halogen_count = np.zeros((cand_n,), dtype=np.float32)
    for an in (9, 17, 35, 53):
        idx = formula_atomicnos.index(an) if an in formula_atomicnos else None
        if idx is None:
            continue
        formula_halogen_count += formula_counts[:, idx].astype(np.float32)

    dbe, is_dbe_negative, parity_violation = _candidate_dbe_from_counts(formula_counts, formula_atomicnos)
    if dbe is None:
        dbe = np.zeros((cand_n,), dtype=np.float32)
        is_dbe_negative = np.zeros((cand_n,), dtype=np.float32)
        parity_violation = np.zeros((cand_n,), dtype=np.float32)

    try:
        pmz_val = float(precursor_mz) if precursor_mz is not None else float('nan')
    except Exception:
        pmz_val = float('nan')
    # For formula-level candidate features, gap should also be neutral-mass based.
    # Candidate masses are neutral formula masses; ion shift is applied separately to peaks.
    precursor_gap = np.abs(float(parent_neutral_mass) - formula_mass).astype(np.float32)

    aux = np.stack(
        [
            formula_atom_sum,
            formula_nonzero,
            formula_mass,
            dbe,
            is_dbe_negative,
            parity_violation,
            neutral_loss_mass,
            neutral_loss_total_atoms,
            neutral_loss_nonzero_atoms,
            formula_halogen_count,
            isotope_entropy,
            mono_fraction,
            m1_fraction,
            m2_fraction,
            precursor_gap,
        ],
        axis=-1,
    ).astype(np.float32)

    if formulae_mask is not None:
        mask = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)
        if mask.shape[0] >= aux.shape[0]:
            mask = mask[: aux.shape[0]]
        else:
            pad = np.ones((aux.shape[0] - mask.shape[0],), dtype=np.float32)
            mask = np.concatenate([mask, pad], axis=0)
        aux = aux * (mask > 0.5).astype(np.float32)[:, None]

    if os.environ.get("DEBUG_AUX_FEAT", "0") == "1":
        try:
            dbg_max = max(0, int(os.environ.get("DEBUG_AUX_FEAT_MAX", "5")))
        except Exception:
            dbg_max = 5
        dbg_cnt = int(getattr(build_formulae_aux_feat, "_dbg_cnt", 0))
        if dbg_cnt < dbg_max:
            print(
                "[DEBUG_AUX_FEAT]",
                "aux_shape=", aux.shape,
                "aux_min=", float(np.min(aux)) if aux.size > 0 else float('nan'),
                "aux_max=", float(np.max(aux)) if aux.size > 0 else float('nan'),
                "aux_mean=", float(np.mean(aux)) if aux.size > 0 else float('nan'),
                "row0_first8=", aux[0, :8].tolist() if aux.ndim == 2 and aux.shape[0] > 0 else "NA",
            )
            setattr(build_formulae_aux_feat, "_dbg_cnt", dbg_cnt + 1)

    return aux.astype(np.float32)


# Function overview: whole_molecule_features handles a specific workflow step in this module.
def whole_molecule_features(full_record, atom_type_counts=False):
    """
    return a vector of features for the full molecule 
    """
    out_feat = []
    if atom_type_counts:
        atom_counts = np.zeros(32, dtype=np.float32)
        for atom in full_record['rdmol'].GetAtoms():
            atom_counts[atom.GetAtomicNum()] +=1

        out_feat.append(atom_counts)

    if len(out_feat) == 0:
        return torch.Tensor([])
    return torch.Tensor(np.concatenate(out_feat).astype(np.float32))


# Function overview: mol_global_features handles a specific workflow step in this module.
def mol_global_features(mol, feature_names=None):
    """
    Lightweight global molecular descriptors for optional conditioning.

    Supported names:
    - exact_mol_wt
    - mol_logp
    - tpsa
    - num_hbd
    - num_hba
    - num_rot_bonds
    """
    if feature_names is None:
        feature_names = []
    names = [str(x).strip().lower() for x in feature_names if str(x).strip()]
    if len(names) == 0:
        return np.zeros((0,), dtype=np.float32)

    out = []
    for name in names:
        if name == 'exact_mol_wt':
            out.append(float(rdkit.Chem.Descriptors.ExactMolWt(mol)))
        elif name == 'mol_logp':
            out.append(float(rdkit.Chem.Descriptors.MolLogP(mol)))
        elif name == 'tpsa':
            out.append(float(rdkit.Chem.Descriptors.TPSA(mol)))
        elif name == 'num_hbd':
            out.append(float(rdkit.Chem.Descriptors.NumHDonors(mol)))
        elif name == 'num_hba':
            out.append(float(rdkit.Chem.Descriptors.NumHAcceptors(mol)))
        elif name == 'num_rot_bonds':
            out.append(float(rdkit.Chem.Descriptors.NumRotatableBonds(mol)))
        else:
            raise ValueError(f"Unknown mol global feature: {name}")

    return np.asarray(out, dtype=np.float32)


# Function overview: get_columb_mat handles a specific workflow step in this module.
def get_columb_mat(mol, conf_idx = 0):
    """
    from 
    https://github.com/cameronus/coulomb-matrix/blob/master/generate.py

    """

    n_atoms = mol.GetNumAtoms()
    m = np.zeros((n_atoms, n_atoms), dtype=np.float32)
    z, xyz = get_nos_coords(mol, conf_idx)
    
    for r in range(n_atoms):
      for c in range(n_atoms):
          if r == c:
              m[r][c] = 0.5 * z[r] ** 2.4
          elif r < c:
              v = z[r] * z[c] / np.linalg.norm(np.array(xyz[r]) - np.array(xyz[c])) * 0.52917721092
              m[r][c] = v
              m[c][r] = v
    return m

# Function overview: dist_mat handles a specific workflow step in this module.
def dist_mat(mol,
             conf_idx = 0,
             feat_distance_pow = [{'pow' : 1,
                                   'max' : 10,
                                   'min' : 0,
                                   'offset' : 0.1}],
             mmff_opt_conf = False, 
             ):
    """
    Return matrix features for molecule
    
    """
    res_mats = []
    if mmff_opt_conf:
        Chem.AllChem.EmbedMolecule(mol)
        Chem.AllChem.MMFFOptimizeMolecule(mol)
    atomic_nos, coords = get_nos_coords(mol, conf_idx)
    ATOM_N = len(atomic_nos)

    pos = coords
    a = pos.T.reshape(1, 3, -1)
    b = np.abs((a - a.T))
    c = np.swapaxes(b, 2, 1)
    c = np.sqrt((c**2).sum(axis=-1))
    dist_mat = torch.Tensor(c).unsqueeze(-1).numpy() # ugh i am sorry
    for d in feat_distance_pow:
        power = d.get('pow', 1)
        max_val = d.get('max', 10000)
        min_val = d.get('min', 0)
        offset = d.get('offset', 0)

        v = (dist_mat + offset) ** power
        v = np.clip(v, a_min = min_val,
                    a_max = max_val)
        res_mats.append(v)

    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)

    assert np.isfinite(M).all()
    return M

# Function overview: mol_to_nx handles a specific workflow step in this module.
def mol_to_nx(mol):
    g = nx.Graph()
    g.add_nodes_from(range(mol.GetNumAtoms()))
    g.add_edges_from([(b.GetBeginAtomIdx(), b.GetEndAtomIdx(), 
     {'weight' : b.GetBondTypeAsDouble()}) for b in mol.GetBonds()])
    return g

w_lut = {1.0 : 0, 1.5 : 1, 2.0: 2, 3.0: 3}

# Function overview: get_min_path_length handles a specific workflow step in this module.
def get_min_path_length(g):
    N = len(g.nodes)
    out = np.zeros((N, N), dtype=np.int32)
    sp = nx.shortest_path(g)
    for i, j in sp.items():
        for jj, path in j.items():
            out[i, jj] = len(path)
    return out

# Function overview: get_bond_path_counts handles a specific workflow step in this module.
def get_bond_path_counts(g):
    N = len(g.nodes)
    out = np.zeros((N, N, 4), dtype=np.int32)
    sp = nx.shortest_path(g)   
    
    for i, j in sp.items():
        for jj, path in j.items():
            for a, b in zip(path[:-1], path[1:]):
                w = g.edges[a, b]['weight']
                
                out[i, jj, w_lut[w]] +=1
                
    return out

# Function overview: get_cycle_counts handles a specific workflow step in this module.
def get_cycle_counts(g, cycle_size_max = 10):
    cb = nx.cycle_basis(g)
    N = len(g.nodes)
    M = cycle_size_max - 2
    cycle_mat = np.zeros((N, N, M), dtype=np.float32)
    for c in nx.cycle_basis(g):
        x = np.zeros(N)
        x[c] = 1
        if len(c) <= cycle_size_max:
            
            cycle_mat[:, :, len(c)-3] += np.outer(x, x)
    return cycle_mat

# Function overview: get_dihedral_angles handles a specific workflow step in this module.
def get_dihedral_angles(mol, conf_idx=0):
    c = mol.GetConformers()[conf_idx]

    atom_n = mol.GetNumAtoms()

    out = np.zeros((atom_n, atom_n), dtype=np.float32)
    for i in range(atom_n):
        for j in range(i+1, atom_n):
            
            sp = Chem.rdmolops.GetShortestPath(mol, i, j)
            if len(sp) < 4:
                dh = 0
            else:
                try:
                    dh = Chem.rdMolTransforms.GetDihedralDeg(c, sp[0], sp[1], sp[-2], sp[-1])
                except ValueError:
                    dh = 0

            if not np.isfinite(dh):
                print(f"WARNING {dh} is not finite between {sp}")
                dh = 0
            
            out[i, j] = dh
            out[j, i] = dh

    return out

# Function overview: get_dihedral_sincos handles a specific workflow step in this module.
def get_dihedral_sincos(mol, conf_idx=0):
    c = mol.GetConformers()[conf_idx]

    atom_n = mol.GetNumAtoms()

    out = np.zeros((atom_n, atom_n, 2), dtype=np.float32)
    for i in range(atom_n):
        for j in range(i+1, atom_n):
            
            sp = Chem.rdmolops.GetShortestPath(mol, i, j)
            if len(sp) < 4:
                dh = 0
            else:
                try:
                    dh = Chem.rdMolTransforms.GetDihedralRad(c, sp[0], sp[1], sp[-2], sp[-1])
                except ValueError:
                    dh = 0

            if not np.isfinite(dh):
                print(f"WARNING {dh} is not finite between {sp}")
                dh = 0
            
            dh_sin = np.sin(dh)
            dh_cos = np.cos(dh)
            out[i, j, 0] = dh_sin
            out[j, i, 0] = dh_sin
            out[i, j, 1] = dh_cos
            out[j, i, 1] = dh_cos

    return out

# Function overview: get_graph_props handles a specific workflow step in this module.
def get_graph_props(mol, min_path_length=False,
                    bond_path_counts=False,
                    cycle_counts = False,
                    cycle_size_max=9, 
                    
                    ):
    g = mol_to_nx(mol)

    out = []
    if min_path_length:
        out.append(np.expand_dims(get_min_path_length(g), -1))

    if bond_path_counts:
        out.append(get_bond_path_counts(g))

    if cycle_counts:
        out.append(get_cycle_counts(g, cycle_size_max=cycle_size_max))

    if len(out) == 0:
        return None
    out = np.concatenate(out, axis=-1)

    assert np.isfinite(out).all()
    return out

# Function overview: pad handles a specific workflow step in this module.
def pad(M, MAX_N):
    """
    Pad M with shape N x N x C  to MAX_N x MAX_N x C
    """
    N, _, C = M.shape
    X = np.zeros((MAX_N, MAX_N, C),
                 dtype=M.dtype)

    for c in range(C):
        X[:N, :N, c] = M[:, :, c]
    return X
        
# Function overview: get_geom_props handles a specific workflow step in this module.
def get_geom_props(mol,
              dist_mat_mean = False,
              dist_mat_std = False):
    """
    returns geometry features for mol
    
    """
    res_mats = []

    Ds = np.stack([Chem.rdmolops.Get3DDistanceMatrix(mol, c.GetId()) for c in mol.GetConformers()], -1)

    M = None

    if dist_mat_mean:
        D_mean = np.mean(Ds, -1)

        res_mats.append(np.expand_dims(D_mean.astype(np.float32), -1))
        
    if dist_mat_std:
        D_std = np.std(Ds, -1)

        res_mats.append(np.expand_dims(D_std.astype(np.float32), -1))
        
    if len(res_mats) > 0:
        M = np.concatenate(res_mats, 2)


    return M

# Function overview: recon_features_edge handles a specific workflow step in this module.
def recon_features_edge(mol,
                        graph_recon_config = {}, 
                        geom_recon_config = {},
                        
                        ):

    p = []
    p.append(get_graph_props(mol, **graph_recon_config))
    p.append(get_geom_props(mol, **geom_recon_config))

    a_sub = [a for a in p if a is not None]
    if len(a_sub) == 0:
        return np.zeros((mol.GetNumAtoms(),
                         mol.GetNumAtoms(),
                         0), dtype=np.float32)
    return np.concatenate(a_sub, -1)

# Function overview: get_single_bond_fragment_masses handles a specific workflow step in this module.
def get_single_bond_fragment_masses(mol, **config):
    """
    Return a N x N x 2 matrix of the weights of the fragments 
    if that bond were cut
    """

    N = mol.GetNumAtoms()
    out_masses = np.zeros((N, N, 2), dtype=np.float32)
    for bi, b in enumerate(mol.GetBonds()):

        mol_f = Chem.FragmentOnBonds(mol, (bi,))
        atomic_masses = np.array([int(a.GetMass()) for a in mol_f.GetAtoms()])

        out_frags = []
        frags = Chem.GetMolFrags(mol_f, sanitizeFrags=False, 
                                 fragsMolAtomMapping=out_frags)
        frag_masses = [np.sum(atomic_masses[np.array(f)]) for f in frags][:2]
        if len(frag_masses) == 1:
            frag_masses.append(frag_masses[0])
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()            
        out_masses[i, j, :] = frag_masses
        out_masses[j, i, :] = frag_masses[::-1]
    return out_masses    


_ANNOT_LIB_CACHE = {}

def _load_annotation_library_from_env():
    path = str(os.environ.get("FORMULA_ANNOTATION_LIBRARY_PATH", "") or "").strip()
    if not path:
        return None
    if path in _ANNOT_LIB_CACHE:
        return _ANNOT_LIB_CACHE[path]
    try:
        import json
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
    except Exception:
        _ANNOT_LIB_CACHE[path] = None
        return None

    product = obj.get("product", {}) if isinstance(obj, dict) else {}
    neutral_loss = obj.get("neutral_loss", {}) if isinstance(obj, dict) else {}

    payload = {
        "product": product if isinstance(product, dict) else {},
        "neutral_loss": neutral_loss if isinstance(neutral_loss, dict) else {},
    }
    _ANNOT_LIB_CACHE[path] = payload
    return payload

def _counts_to_formula_string_from_vector(counts, formula_atomicnos):
    counts = np.asarray(counts, dtype=np.int64).reshape(-1)
    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]
    pt = Chem.GetPeriodicTable()

    sym_counts = {}
    for i, an in enumerate(formula_atomicnos):
        if i >= counts.shape[0]:
            break
        c = int(counts[i])
        if c <= 0:
            continue
        try:
            sym = pt.GetElementSymbol(int(an))
        except Exception:
            continue
        sym_counts[sym] = sym_counts.get(sym, 0) + c

    if not sym_counts:
        return ""

    order = []
    if "C" in sym_counts:
        order.append("C")
    if "H" in sym_counts:
        order.append("H")
    order += sorted([k for k in sym_counts if k not in ("C", "H")])

    s = ""
    for k in order:
        s += k
        if sym_counts[k] != 1:
            s += str(sym_counts[k])
    return s

def _annotation_library_features_for_candidates(
    formulae_arr,
    formula_atomicnos,
    precursor_formula=None,
):
    """
    Train-only annotation library features.
    Safe if FORMULA_ANNOTATION_LIBRARY_PATH was built from train split only.

    Return [N,4]:
      product_hit
      product_logfreq
      loss_hit
      loss_logfreq
    """
    formulae_arr = np.asarray(formulae_arr, dtype=np.int64)
    if formulae_arr.ndim != 2 or formulae_arr.shape[0] <= 0:
        return np.zeros((0, 4), dtype=np.float32)

    lib = _load_annotation_library_from_env()
    if lib is None:
        return np.zeros((formulae_arr.shape[0], 4), dtype=np.float32)

    prod = lib.get("product", {})
    loss = lib.get("neutral_loss", {})

    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]

    parent_counts = None
    if precursor_formula:
        parent_counts = _formula_string_to_counts(precursor_formula, formula_atomicnos)
        if parent_counts is not None:
            parent_counts = np.asarray(parent_counts, dtype=np.int64).reshape(-1)
            if parent_counts.shape[0] < len(formula_atomicnos):
                parent_counts = np.concatenate(
                    [parent_counts, np.zeros((len(formula_atomicnos) - parent_counts.shape[0],), dtype=np.int64)],
                    axis=0,
                )
            parent_counts = parent_counts[:len(formula_atomicnos)]

    out = np.zeros((formulae_arr.shape[0], 4), dtype=np.float32)

    for i in range(formulae_arr.shape[0]):
        cc = formulae_arr[i, :len(formula_atomicnos)]
        f = _counts_to_formula_string_from_vector(cc, formula_atomicnos)

        if f in prod:
            out[i, 0] = 1.0
            try:
                out[i, 1] = float(prod[f].get("log_intensity", 0.0))
            except Exception:
                out[i, 1] = 0.0

        if parent_counts is not None:
            lc = parent_counts - cc
            if np.all(lc >= 0):
                lf = _counts_to_formula_string_from_vector(lc, formula_atomicnos)
                if lf in loss:
                    out[i, 2] = 1.0
                    try:
                        out[i, 3] = float(loss[lf].get("log_intensity", 0.0))
                    except Exception:
                        out[i, 3] = 0.0

    # logfreq 压缩到较稳的尺度
    for col in (1, 3):
        x = out[:, col]
        pos = x > 0
        if np.any(pos):
            mx = float(np.percentile(x[pos], 95))
            if mx > 1e-8:
                out[:, col] = np.clip(x / mx, 0.0, 1.0)

    return out.astype(np.float32)

def _candidate_formula_prior_score_basic(
    formulae_arr,
    formula_atomicnos,
    precursor_mz=None,
    precursor_formula=None,
):
    """
    No-label / no-spectrum prior score for formula candidates.
    Used when we cannot rely on target spectrum.
    """
    formulae_arr = np.asarray(formulae_arr, dtype=np.float32)
    if formulae_arr.ndim != 2 or formulae_arr.shape[0] <= 0:
        return np.zeros((0,), dtype=np.float32)

    try:
        aux = build_formulae_aux_feat(
            formulae_feats=formulae_arr,
            formulae_peaks_mass_idx=None,
            formulae_peaks_intensity=None,
            formula_atomicnos=formula_atomicnos,
            precursor_mz=precursor_mz,
            precursor_formula=precursor_formula,
        ).astype(np.float32)
    except Exception:
        return np.zeros((formulae_arr.shape[0],), dtype=np.float32)

    neg_dbe = aux[:, 4]
    parity_bad = aux[:, 5]
    nl_mass = aux[:, 6]
    precursor_gap = aux[:, 14]

    score = np.zeros((aux.shape[0],), dtype=np.float32)

    # 1) DBE prior.
    # For charged fragment ions, parity violation is common and should not be penalized by default.
    # Negative DBE is still a useful hard/soft chemical sanity signal.
    use_prior_dbe_negative = os.environ.get(
        "FORMULA_PRIOR_USE_DBE_NEGATIVE",
        os.environ.get("FORMULA_FILTER_DBE_NEGATIVE", "1"),
    ) == "1"

    use_prior_dbe_parity = os.environ.get(
        "FORMULA_PRIOR_USE_DBE_PARITY",
        os.environ.get("FORMULA_FILTER_DBE_PARITY", "0"),
    ) == "1"

    if use_prior_dbe_negative:
        score += 0.80 * (neg_dbe <= 0.5).astype(np.float32)
        score -= 2.00 * (neg_dbe > 0.5).astype(np.float32)

    if use_prior_dbe_parity:
        score += 0.20 * (parity_bad <= 0.5).astype(np.float32)
        score -= 0.50 * (parity_bad > 0.5).astype(np.float32)

    # 2) common neutral-loss bonus
    common_losses = np.asarray(
        [18.0106, 17.0265, 27.9949, 43.9898, 30.0106, 28.0313],
        dtype=np.float32,
    )  # H2O, NH3, CO, CO2, CH2O, C2H4
    diff = np.abs(nl_mass[:, None] - common_losses[None, :])
    min_diff = diff.min(axis=1)
    nl_bonus = np.exp(-(min_diff ** 2) / (2.0 * (0.8 ** 2))).astype(np.float32)
    score += 1.2 * nl_bonus


    # 2.5) train-only annotation product/loss library prior.
    # This is label-free for val/test as long as the JSON is built from train split only.
    try:
        lib_feat = _annotation_library_features_for_candidates(
            formulae_arr=formulae_arr,
            formula_atomicnos=formula_atomicnos,
            precursor_formula=precursor_formula,
        )
        if lib_feat.shape[0] == score.shape[0]:
            product_hit = lib_feat[:, 0]
            product_logfreq = lib_feat[:, 1]
            loss_hit = lib_feat[:, 2]
            loss_logfreq = lib_feat[:, 3]

            score += 1.0 * product_hit
            score += 0.5 * product_logfreq
            score += 0.8 * loss_hit
            score += 0.4 * loss_logfreq
    except Exception:
        pass

    # Exact formula-level neutral-loss bonus when precursor formula is available.
    # This is label-free and only uses molecular formula.
    try:
        parent_counts_exact = _formula_string_to_counts(precursor_formula, formula_atomicnos)
    except Exception:
        parent_counts_exact = None

    if parent_counts_exact is not None:
        formula_atomicnos_resolved = _resolve_formula_atomicnos(formula_atomicnos)
        use_d = min(int(formulae_arr.shape[1]), int(len(formula_atomicnos_resolved)))
        formula_counts = np.asarray(formulae_arr[:, :use_d], dtype=np.float32)

        pc = np.asarray(parent_counts_exact[:use_d], dtype=np.float32)
        loss_counts = pc.reshape(1, -1) - formula_counts
        valid_loss_counts = np.all(loss_counts >= -1e-6, axis=1)

        common_loss_formulas = [
            "H2O", "NH3", "CO", "CO2", "CH2O", "HCN", "C2H4", "H2S", "SO2"
        ]

        exact_loss_match = np.zeros((formulae_arr.shape[0],), dtype=np.float32)
        for lf in common_loss_formulas:
            lc = _formula_string_to_counts(lf, formula_atomicnos_resolved)
            if lc is None:
                continue
            lc = np.asarray(lc[:use_d], dtype=np.float32)
            if lc.shape[0] != use_d:
                continue
            hit = np.all(np.abs(loss_counts - lc.reshape(1, -1)) <= 1e-6, axis=1)
            exact_loss_match = np.maximum(exact_loss_match, hit.astype(np.float32))

        score += 1.0 * exact_loss_match * valid_loss_counts.astype(np.float32)

    # 3) prefer not too far from precursor (light penalty)
    try:
        pmz = float(precursor_mz)
    except Exception:
        pmz = float('nan')
    if np.isfinite(pmz) and pmz > 0.0:
        gap_norm = np.clip(precursor_gap / max(pmz, 1e-6), 0.0, 1.0)
        score -= 0.5 * gap_norm.astype(np.float32)

    return score.astype(np.float32)

def _enumerate_break_connected_subsets(
    mol,
    max_single=96,
    max_double=48,
    allow_ring=False,
    max_seed_bonds=16,
    min_heavy_atoms=2,
):
    """
    Enumerate connected atom subsets from restricted 1-break / 2-break cuts.

    Returns:
        subsets: list[np.ndarray[int64]]
        depths: list[int]      # 1 or 2
        ring_flags: list[int]  # 1 if any cut bond is ring bond else 0
    """
    atom_n = int(mol.GetNumAtoms())
    if atom_n <= 1:
        return [], [], []

    g = mol_to_nx(mol)

    bond_rows = []
    for b in mol.GetBonds():
        bo = float(b.GetBondTypeAsDouble())

        # FIORA-style: structure event 不再只限 single bond。
        # 对 formula supplement 的 2-break 可以仍然保守，但 single structural mark 不该只看 bo=1。
        if os.environ.get("FORMULA_STRUCTURE_SINGLE_ONLY", "0") == "1":
            if bo != 1.0:
                continue

        in_ring = int(b.IsInRing())
        if (not allow_ring) and in_ring == 1:
            continue

        a0 = mol.GetAtomWithIdx(int(b.GetBeginAtomIdx()))
        a1 = mol.GetAtomWithIdx(int(b.GetEndAtomIdx()))

        # Do not use X-H bonds as structural fragmentation seeds.
        # They produce trivial H-loss / near-precursor fragments and pollute the structure bucket.
        if int(a0.GetAtomicNum()) == 1 or int(a1.GetAtomicNum()) == 1:
            continue

        hetero_score = float(a0.GetAtomicNum() != 6) + float(a1.GetAtomicNum() != 6)
        aromatic_penalty = float(a0.GetIsAromatic()) + float(a1.GetIsAromatic())

        # higher score = more likely cut candidate
        score = (2.0 * hetero_score) - (1.5 * aromatic_penalty) - (2.0 * in_ring)

        bond_rows.append(
            (
                float(score),
                int(b.GetIdx()),
                int(b.GetBeginAtomIdx()),
                int(b.GetEndAtomIdx()),
                int(in_ring),
            )
        )

    if len(bond_rows) <= 0:
        return [], [], []

    bond_rows = sorted(bond_rows, key=lambda x: (-x[0], x[1]))
    if int(max_seed_bonds) <= 0:
        seed_bonds = bond_rows
    else:
        seed_bonds = bond_rows[: max(1, int(max_seed_bonds))]

    seen = set()
    subsets = []
    depths = []
    ring_flags = []

    def _heavy_atom_count(nodes):
        cnt = 0
        for ai in nodes:
            if mol.GetAtomWithIdx(int(ai)).GetAtomicNum() > 1:
                cnt += 1
        return cnt

    def _add_comp(comp_nodes, depth, ring_flag):
        comp_nodes = tuple(sorted(int(x) for x in comp_nodes))
        if len(comp_nodes) <= 1 or len(comp_nodes) >= atom_n:
            return
        if _heavy_atom_count(comp_nodes) < int(min_heavy_atoms):
            return
        if comp_nodes in seen:
            return
        seen.add(comp_nodes)
        subsets.append(np.asarray(comp_nodes, dtype=np.int64))
        depths.append(int(depth))
        ring_flags.append(int(ring_flag))

    # single cuts
    single_cnt = 0
    for _, _, u, v, rf in seed_bonds:
        if single_cnt >= int(max_single):
            break
        g2 = g.copy()
        if g2.has_edge(u, v):
            g2.remove_edge(u, v)
        for comp in nx.connected_components(g2):
            _add_comp(comp, 1, rf)
            single_cnt += 1
            if single_cnt >= int(max_single):
                break

    # double cuts (restricted)
    double_cnt = 0
    for i in range(len(seed_bonds)):
        if double_cnt >= int(max_double):
            break
        for j in range(i + 1, len(seed_bonds)):
            if double_cnt >= int(max_double):
                break

            _, _, u1, v1, rf1 = seed_bonds[i]
            _, _, u2, v2, rf2 = seed_bonds[j]

            g2 = g.copy()
            if g2.has_edge(u1, v1):
                g2.remove_edge(u1, v1)
            if g2.has_edge(u2, v2):
                g2.remove_edge(u2, v2)

            for comp in nx.connected_components(g2):
                _add_comp(comp, 2, max(rf1, rf2))
                double_cnt += 1
                if double_cnt >= int(max_double):
                    break

    return subsets, depths, ring_flags


def _enumerate_multidepth_structural_subsets(
    mol,
    max_depth=4,
    max_nodes=50000,
    allow_ring=True,
    min_heavy_atoms=1,
    max_branch_per_node=64,
    include_root=False,
):
    """
    Multi-depth connected heavy-atom subset enumerator.

    This is NOT a FraGNNet model implementation:
    - it does not build a Fragment GNN;
    - it does not define P(n), P(f|n), or P(n|f);
    - it only marks candidate formulae with structural evidence.

    Node = connected heavy-atom subgraph.
    Expansion = remove one heavy-heavy bond from the current subgraph.
    Dedup = atom-set key.
    """
    atom_n = int(mol.GetNumAtoms())
    if atom_n <= 1:
        return [], [], []

    heavy_nodes = [
        int(a.GetIdx())
        for a in mol.GetAtoms()
        if int(a.GetAtomicNum()) > 1
    ]
    if len(heavy_nodes) <= 0:
        return [], [], []

    g_full = mol_to_nx(mol)
    g_heavy = g_full.subgraph(heavy_nodes).copy()

    seen = set()
    subsets = []
    depths = []
    ring_flags = []

    def _add(nodes, depth, ring_flag):
        if len(seen) >= int(max_nodes):
            return False

        key = tuple(sorted(int(x) for x in nodes))
        if len(key) <= 0:
            return False
        if len(key) < int(min_heavy_atoms):
            return False

        # 对 no-precursor official target 来说，root 不是主要候选；
        # 但仍然允许展开 root，只是不一定把 root 自身标记成结构候选。
        if key in seen:
            return False

        seen.add(key)

        if bool(include_root) or int(depth) > 0:
            subsets.append(np.asarray(key, dtype=np.int64))
            depths.append(int(depth))
            ring_flags.append(int(ring_flag))

        return True

    def _bond_priority(u, v):
        try:
            bond = mol.GetBondBetweenAtoms(int(u), int(v))
            if bond is None:
                return -999.0

            a0 = mol.GetAtomWithIdx(int(u))
            a1 = mol.GetAtomWithIdx(int(v))

            hetero = float(a0.GetAtomicNum() != 6) + float(a1.GetAtomicNum() != 6)
            aromatic = float(a0.GetIsAromatic()) + float(a1.GetIsAromatic())
            ring = float(bond.IsInRing())
            bo = float(bond.GetBondTypeAsDouble())

            # 这不是 FraGNNet 原样复刻：这里加入了你自己的局部化学优先级。
            # 优先异原子邻域、非芳香、非环，限制每个节点分支数，避免 DAG 爆炸。
            return 2.0 * hetero + 0.25 * bo - 1.0 * aromatic - 0.7 * ring
        except Exception:
            return 0.0

    def _recurse(nodes, depth, ring_flag):
        if int(depth) > int(max_depth):
            return
        if len(seen) >= int(max_nodes):
            return

        nodes = tuple(sorted(int(x) for x in nodes))
        added = _add(nodes, depth, ring_flag)

        # 如果已经见过该 atom-set，就不继续重复展开。
        if not added:
            return

        if int(depth) >= int(max_depth):
            return

        sub_g = g_heavy.subgraph(nodes).copy()
        edges = list(sub_g.edges())
        if len(edges) <= 0:
            return

        edges = sorted(edges, key=lambda e: -_bond_priority(e[0], e[1]))
        edges = edges[: max(1, int(max_branch_per_node))]

        for u, v in edges:
            try:
                bond = mol.GetBondBetweenAtoms(int(u), int(v))
                is_ring = int(bond.IsInRing()) if bond is not None else 0
            except Exception:
                is_ring = 0

            if (not bool(allow_ring)) and is_ring:
                continue

            g2 = sub_g.copy()
            if g2.has_edge(u, v):
                g2.remove_edge(u, v)

            comps = list(nx.connected_components(g2))
            if len(comps) <= 1:
                continue

            for comp in comps:
                comp = tuple(sorted(int(x) for x in comp))
                if len(comp) <= 0 or len(comp) >= len(nodes):
                    continue
                _recurse(
                    comp,
                    int(depth) + 1,
                    max(int(ring_flag), int(is_ring)),
                )

    _recurse(tuple(sorted(heavy_nodes)), depth=0, ring_flag=0)

    return subsets, depths, ring_flags

def _select_formula_candidates_from_subsets(
    mol,
    formulae_arr,
    subsets,
    subset_depths,
    subset_ring_flags,
    formula_atomicnos,
    candidate_prior_score=None,
    topn_per_subset=8,
    h_window=4,
):
    """
    Map structure-grounded atom subsets to formula-only candidates by heavy-atom counts.

    Returns:
        uniq_idx: np.ndarray[int64]
        depth_arr: np.ndarray[int8]
        ring_arr: np.ndarray[int8]
    """
    formulae_arr = np.asarray(formulae_arr, dtype=np.int16)
    if formulae_arr.ndim != 2 or formulae_arr.shape[0] <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.int8),
        )

    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]
    lut = {int(an): i for i, an in enumerate(formula_atomicnos)}

    heavy_cols = [i for i, z in enumerate(formula_atomicnos) if int(z) != 1]
    if len(heavy_cols) <= 0:
        heavy_cols = list(range(int(formulae_arr.shape[1])))

    h_col = None
    for i, z in enumerate(formula_atomicnos):
        if int(z) == 1:
            h_col = int(i)
            break

    cand_heavy = formulae_arr[:, heavy_cols]
    cand_h = formulae_arr[:, h_col].astype(np.int16) if h_col is not None and h_col < formulae_arr.shape[1] else None

    if candidate_prior_score is None:
        candidate_prior_score = np.zeros((formulae_arr.shape[0],), dtype=np.float32)
    else:
        candidate_prior_score = np.asarray(candidate_prior_score, dtype=np.float32).reshape(-1)
        if candidate_prior_score.shape[0] != formulae_arr.shape[0]:
            candidate_prior_score = np.zeros((formulae_arr.shape[0],), dtype=np.float32)

    chosen = {}
    topn_per_subset = max(1, int(topn_per_subset))

    for subset_nodes, depth, ring_flag in zip(subsets, subset_depths, subset_ring_flags):
        target_counts = np.zeros((len(formula_atomicnos),), dtype=np.int16)
        for ai in subset_nodes.tolist():
            atom = mol.GetAtomWithIdx(int(ai))
            an = int(atom.GetAtomicNum())
            pos = lut.get(an, None)
            if pos is not None:
                target_counts[pos] += 1

        target_heavy = target_counts[heavy_cols]
        matched = np.where(np.all(cand_heavy == target_heavy[None, :], axis=1))[0].astype(np.int64)
        if matched.size <= 0:
            continue

        # If explicit H is available in the structural subset, keep a small H-shift window.
        # This prevents losing valid proton/hydrogen rearrangement variants while avoiding
        # completely unconstrained H variants.
        h_penalty = np.zeros((matched.shape[0],), dtype=np.float32)
        if h_col is not None and cand_h is not None and h_col < target_counts.shape[0]:
            target_h = int(target_counts[h_col])
            if target_h > 0:
                h_diff_all = np.abs(cand_h[matched].astype(np.int16) - int(target_h)).astype(np.float32)
                h_ok = h_diff_all <= float(max(0, int(h_window)))
                if np.any(h_ok):
                    matched = matched[h_ok]
                    h_penalty = h_diff_all[h_ok]
                else:
                    h_penalty = h_diff_all

        score = candidate_prior_score[matched] - (0.05 * h_penalty)
        order = np.argsort(-score, kind='stable')
        keep = matched[order[:topn_per_subset]]

        for cid in keep.tolist():
            cid = int(cid)
            if cid not in chosen:
                chosen[cid] = (int(depth), int(ring_flag))
            else:
                old_depth, old_ring = chosen[cid]
                chosen[cid] = (min(old_depth, int(depth)), max(old_ring, int(ring_flag)))

    if len(chosen) <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.int8),
        )

    uniq_idx = np.asarray(sorted(chosen.keys()), dtype=np.int64)
    depth_arr = np.asarray([chosen[int(k)][0] for k in uniq_idx.tolist()], dtype=np.int8)
    ring_arr = np.asarray([chosen[int(k)][1] for k in uniq_idx.tolist()], dtype=np.int8)
    return uniq_idx, depth_arr, ring_arr


def _make_formula_key_index(formulae_arr):
    formulae_arr = np.asarray(formulae_arr, dtype=np.int16)
    out = {}
    if formulae_arr.ndim != 2:
        return out
    for i in range(int(formulae_arr.shape[0])):
        key = tuple(int(x) for x in formulae_arr[i].tolist())
        out.setdefault(key, []).append(int(i))
    return out

def _subset_formula_counts_with_h(mol, subset_nodes, formula_atomicnos):
    """
    Compute formula counts for an atom subset.

    If explicit H atoms exist in mol, count them only when present in subset.
    If no explicit H atoms exist, add implicit/explicit-H properties from included heavy atoms.
    """
    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]
    lut = {int(an): i for i, an in enumerate(formula_atomicnos)}
    counts = np.zeros((len(formula_atomicnos),), dtype=np.int16)

    has_explicit_h_atoms = any(int(a.GetAtomicNum()) == 1 for a in mol.GetAtoms())
    subset_nodes = [int(x) for x in subset_nodes]

    for ai in subset_nodes:
        atom = mol.GetAtomWithIdx(int(ai))
        an = int(atom.GetAtomicNum())
        pos = lut.get(an, None)
        if pos is not None:
            counts[pos] += 1

        # If hydrogens are implicit, include implicit-H contribution on included heavy atoms.
        if (not has_explicit_h_atoms) and an != 1:
            h_pos = lut.get(1, None)
            if h_pos is not None:
                try:
                    h_extra = int(atom.GetNumImplicitHs()) + int(atom.GetNumExplicitHs())
                except Exception:
                    h_extra = 0
                if h_extra > 0:
                    counts[h_pos] += int(h_extra)

    return counts

def _select_exact_structural_formula_candidates(
    mol,
    formulae_arr,
    subsets,
    subset_depths,
    subset_ring_flags,
    formula_atomicnos,
    h_window=4,
):
    """
    Exact structure-derived formula marking.

    Unlike heavy-only matching, this computes formula counts from connected
    structural fragments, then searches formula candidates with small H-shift.
    """
    formulae_arr = np.asarray(formulae_arr, dtype=np.int16)
    if formulae_arr.ndim != 2 or formulae_arr.shape[0] <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.int8),
        )

    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]
    key_to_ids = _make_formula_key_index(formulae_arr)

    h_col = None
    for i, z in enumerate(formula_atomicnos):
        if int(z) == 1:
            h_col = int(i)
            break

    chosen = {}
    h_window = max(0, int(h_window))

    for subset_nodes, depth, ring_flag in zip(subsets, subset_depths, subset_ring_flags):
        base_counts = _subset_formula_counts_with_h(
            mol=mol,
            subset_nodes=subset_nodes,
            formula_atomicnos=formula_atomicnos,
        )

        for dh in range(-h_window, h_window + 1):
            cc = base_counts.copy()
            if h_col is not None:
                new_h = int(cc[h_col]) + int(dh)
                if new_h < 0:
                    continue
                cc[h_col] = int(new_h)

            key = tuple(int(x) for x in cc.tolist())
            ids = key_to_ids.get(key, [])
            if not ids:
                continue

            for cid in ids:
                cid = int(cid)
                if cid not in chosen:
                    chosen[cid] = (int(depth), int(ring_flag))
                else:
                    old_depth, old_ring = chosen[cid]
                    chosen[cid] = (min(old_depth, int(depth)), max(old_ring, int(ring_flag)))

    if len(chosen) <= 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int8),
            np.zeros((0,), dtype=np.int8),
        )

    idx = np.asarray(sorted(chosen.keys()), dtype=np.int64)
    depth_arr = np.asarray([chosen[int(i)][0] for i in idx.tolist()], dtype=np.int8)
    ring_arr = np.asarray([chosen[int(i)][1] for i in idx.tolist()], dtype=np.int8)
    return idx, depth_arr, ring_arr

def _select_common_loss_formula_candidates(
    formulae_arr,
    precursor_formula,
    formula_atomicnos,
    h_window=3,
):
    """
    Label candidates that equal precursor_formula - common_neutral_loss (+/- H shift).

    This is label-free. It uses only precursor molecular formula.
    """
    formulae_arr = np.asarray(formulae_arr, dtype=np.int16)
    if formulae_arr.ndim != 2 or formulae_arr.shape[0] <= 0:
        return np.zeros((0,), dtype=np.int64)

    formula_atomicnos = [int(x) for x in _resolve_formula_atomicnos(formula_atomicnos)]
    parent_counts = _formula_string_to_counts(precursor_formula, formula_atomicnos)
    if parent_counts is None:
        return np.zeros((0,), dtype=np.int64)

    parent_counts = np.asarray(parent_counts, dtype=np.int16)
    if parent_counts.shape[0] < len(formula_atomicnos):
        pad = np.zeros((len(formula_atomicnos) - parent_counts.shape[0],), dtype=np.int16)
        parent_counts = np.concatenate([parent_counts, pad], axis=0)
    parent_counts = parent_counts[:len(formula_atomicnos)]

    key_to_ids = _make_formula_key_index(formulae_arr)

    h_col = None
    for i, z in enumerate(formula_atomicnos):
        if int(z) == 1:
            h_col = int(i)
            break

    common_loss_formulas = [
        "H2O", "NH3", "CO", "CO2", "CH2O", "HCN", "C2H4", "H2S", "SO2",
        "CH4", "C2H2", "C2H6", "NO", "NO2",
    ]

    out = set()
    h_window = max(0, int(h_window))

    for lf in common_loss_formulas:
        loss_counts = _formula_string_to_counts(lf, formula_atomicnos)
        if loss_counts is None:
            continue

        loss_counts = np.asarray(loss_counts, dtype=np.int16)
        if loss_counts.shape[0] < len(formula_atomicnos):
            pad = np.zeros((len(formula_atomicnos) - loss_counts.shape[0],), dtype=np.int16)
            loss_counts = np.concatenate([loss_counts, pad], axis=0)
        loss_counts = loss_counts[:len(formula_atomicnos)]

        base = parent_counts - loss_counts
        if np.any(base < 0):
            continue

        for dh in range(-h_window, h_window + 1):
            cc = base.copy()
            if h_col is not None:
                new_h = int(cc[h_col]) + int(dh)
                if new_h < 0:
                    continue
                cc[h_col] = int(new_h)

            ids = key_to_ids.get(tuple(int(x) for x in cc.tolist()), [])
            for cid in ids:
                out.add(int(cid))

    if len(out) <= 0:
        return np.zeros((0,), dtype=np.int64)
    return np.asarray(sorted(out), dtype=np.int64)



# Class overview: PossibleFormulaFeaturizer encapsulates a reusable component in this module.
class PossibleFormulaFeaturizer:

    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(
        self,
        formula_possible_atomicno=[1, 6, 7, 8, 9, 15, 16, 17],
        featurize_mode='numerical',
        num_peaks_per_formula=16, 
        max_formulae=4096,
        overflow_mode='truncate',
        overflow_sample_seed=0,
        clip_mass=511, 
        use_highres=True, 
        formula_max_num_atomicno=20,
    ):

        # Ensure masscompute is available when we need to instantiate enumerators
        global masscompute
        if masscompute is None:
            import importlib
            try:
                masscompute = importlib.import_module('rassp.msutil.masscompute')
            except Exception as e:
                raise ImportError("rassp.msutil.masscompute (and its optional dependency 'diskcache') is required for PossibleFormulaFeaturizer. Install 'diskcache' or avoid using this featurizer.") from e

        self.ffe = masscompute.FragmentFormulaPeakEnumerator(
            formula_possible_atomicno,
            use_highres=use_highres,
            max_peak_num=num_peaks_per_formula,
        )
        self.peak_backend = getattr(self.ffe, 'peak_backend', 'formula_isotope')
            
        self.formula_possible_atomicno = formula_possible_atomicno
        self.featurize_mode = featurize_mode
        self.formula_max_atomicno =formula_max_num_atomicno
        self.max_formulae = max_formulae
        self.num_peaks_per_formula = num_peaks_per_formula

        if isinstance(formula_max_num_atomicno, int):
            # constant int
            formula_max_num_atomicno = [formula_max_num_atomicno] * len(formula_possible_atomicno)
        # create one-hot luts
        running_pos = 0
        self.oh_an_lut = {}
        self.oh_an_max = {}
        for an, m in zip(formula_possible_atomicno, formula_max_num_atomicno):
            self.oh_an_lut[an] = running_pos
            self.oh_an_max[an] = m + 1
            running_pos += m + 1
        self.oh_max = running_pos
        self.oh_offsets = np.array([self.oh_an_lut[a] for a in self.formula_possible_atomicno])

        self.clip_mass = clip_mass
        self.overflow_mode = str(overflow_mode).strip().lower()
        self.overflow_sample_seed = int(overflow_sample_seed)
        self.last_formulae_n_raw = 0
        self.last_formulae_n_kept = 0
        self.last_formulae_truncated = 0
        self.last_formulae_mask = np.zeros((self.max_formulae,), dtype=np.float32)
        self.last_peak_backend = str(self.peak_backend)
        self.last_cache_diag = {}
        self.use_safe_filters = os.environ.get('FORMULA_SAFE_FILTERS', '1') == '1'
        self.safe_filter_precursor = os.environ.get('FORMULA_FILTER_PRECURSOR_CAP', '1') == '1'
        self.safe_filter_dbe_negative = os.environ.get('FORMULA_FILTER_DBE_NEGATIVE', '1') == '1'
        self.safe_filter_dbe_parity = os.environ.get('FORMULA_FILTER_DBE_PARITY', '0') == '1'
        self.safe_filter_dbe = self.safe_filter_dbe_negative or self.safe_filter_dbe_parity
        self.safe_filter_empty = os.environ.get('FORMULA_FILTER_EMPTY_PEAKS', '1') == '1'
        try:
            self.safe_precursor_tol_da = float(os.environ.get('FORMULA_PRECURSOR_CAP_TOL_DA', '1.0'))
        except Exception:
            self.safe_precursor_tol_da = 1.0
        # NIST high-res product peak annotations are already product-ion m/z.
        # Do not add precursor adduct/proton shift by default.
        self.apply_adduct_shift = os.environ.get('FORMULA_APPLY_ADDUCT_SHIFT', '0') == '1'
        self.adduct_mode = os.environ.get('FORMULA_ADDUCT_MODE', 'none').strip().lower() or 'none'
        try:
            self.proton_mass = float(os.environ.get('FORMULA_PROTON_MASS', '1.007276466812'))
        except Exception:
            self.proton_mass = 1.007276466812
        # Verbose logs are noisy during training; keep them opt-in.
        self.verbose = os.environ.get('RASSP_PFF_VERBOSE', '0') == '1'

        self.use_structure_prior = os.environ.get('FORMULA_STRUCTURE_GUIDED', '1') == '1'
        self.structure_allow_ring = os.environ.get('FORMULA_STRUCTURE_ALLOW_RING', '0') == '1'
        self.structure_preselect = os.environ.get('FORMULA_STRUCTURE_PRESELECT', '0') == '1'
        
        # Multi-depth structural evidence generator.
        # This is a candidate-evidence module, not a FraGNNet-style Fragment GNN.
        self.multidepth_struct_enable = os.environ.get('FORMULA_MULTIDEPTH_STRUCT', '0') == '1'

        try:
            self.multidepth_struct_depth = int(os.environ.get('FORMULA_STRUCT_DEPTH', '4'))
        except Exception:
            self.multidepth_struct_depth = 4

        try:
            self.multidepth_struct_max_nodes = int(os.environ.get('FORMULA_STRUCT_MAX_NODES', '50000'))
        except Exception:
            self.multidepth_struct_max_nodes = 50000

        try:
            self.multidepth_struct_branch = int(os.environ.get('FORMULA_STRUCT_BRANCH_PER_NODE', '64'))
        except Exception:
            self.multidepth_struct_branch = 64

        self.multidepth_struct_allow_ring = os.environ.get('FORMULA_STRUCT_ALLOW_RING', '1') == '1'
        self.multidepth_struct_include_root = os.environ.get('FORMULA_STRUCT_INCLUDE_ROOT', '0') == '1'


        self.structure_exact_mark = os.environ.get('FORMULA_STRUCTURE_EXACT_MARK', '1') == '1'
        self.common_loss_mark = os.environ.get('FORMULA_COMMON_LOSS_MARK', '1') == '1'
        try:
            self.common_loss_h_window = int(os.environ.get('FORMULA_COMMON_LOSS_H_WINDOW', '3'))
        except Exception:
            self.common_loss_h_window = 3

        try:
            self.structure_single_keep = int(os.environ.get('FORMULA_STRUCTURE_SINGLE_KEEP', '96'))
        except Exception:
            self.structure_single_keep = 96

        try:
            self.structure_double_keep = int(os.environ.get('FORMULA_STRUCTURE_DOUBLE_KEEP', '48'))
        except Exception:
            self.structure_double_keep = 48

        try:
            self.structure_seed_bonds = int(os.environ.get('FORMULA_STRUCTURE_SEED_BONDS', '16'))
        except Exception:
            self.structure_seed_bonds = 16

        try:
            self.structure_topn_per_subset = int(os.environ.get('FORMULA_STRUCTURE_PER_SUBSET_TOPN', '8'))
        except Exception:
            self.structure_topn_per_subset = 2

        try:
            self.formula_supplement_keep = int(os.environ.get('FORMULA_FORMULA_SUPPLEMENT_KEEP', '1536'))
        except Exception:
            self.formula_supplement_keep = 1536

        try:
            self.preselect_min_keep = int(os.environ.get('FORMULA_PRESELECT_MIN_KEEP', '1536'))
        except Exception:
            self.preselect_min_keep = 1536

        try:
            self.structure_h_window = int(os.environ.get('FORMULA_STRUCTURE_H_WINDOW', '4'))
        except Exception:
            self.structure_h_window = 4

        self.last_formulae_source_flag = np.zeros((self.max_formulae,), dtype=np.int8)
        self.last_formulae_break_depth = np.zeros((self.max_formulae,), dtype=np.int8)
        self.last_formulae_ring_cut_flag = np.zeros((self.max_formulae,), dtype=np.int8)    
        self.last_formulae_prior_score = np.zeros((self.max_formulae,), dtype=np.float32)
        self.last_formulae_active_mask = np.zeros((self.max_formulae,), dtype=np.float32)

        # 正式 cache 默认绝不允许 overflow 用真实谱 greedy coverage。
        # 只有 oracle / upper-bound 诊断时才手动设成 1。
        self.allow_label_overflow = os.environ.get('FORMULA_ALLOW_LABEL_OVERFLOW', '0') == '1'

        try:
            self.active_topk = int(os.environ.get('FORMULA_ACTIVE_TOPK', '300'))
        except Exception:
            self.active_topk = 300

        self.active_keep_all_source = os.environ.get('FORMULA_ACTIVE_KEEP_ALL_SOURCE', '1') == '1'

        # 诊断：正式 cache 里这个应该永远是 0
        self.last_overflow_used_label_greedy = 0


    # Helper: resolve adduct mass shift for candidate peak masses.
    def _resolve_adduct_shift(self, adduct=None):
        mode = str(self.adduct_mode).strip().lower()
        if mode in ('none', 'neutral'):
            return 0.0
        if mode == 'mh_only':
            return float(self.proton_mass)

        raw = str(adduct or '').replace(' ', '').strip().upper()
        lut = {
            '[M+H]+': float(self.proton_mass),
            'M+H': float(self.proton_mass),
            '[M]+': 0.0,
            'M+': 0.0,
            '[M+NA]+': 22.989218,
            '[M+K]+': 38.963158,
        }
        if raw in lut:
            return float(lut[raw])

        # In positive mode datasets with missing adduct, default to protonated mass.
        return float(self.proton_mass)

    # Helper: apply conservative hard filters to remove obviously invalid candidates.
    def _apply_safe_formula_filters(self, formulae, masses, precursor_mz=None):
        formulae_arr = np.asarray(formulae)
        masses_arr = np.asarray(masses, dtype=np.float32)

        cand_n = int(formulae_arr.shape[0]) if formulae_arr.ndim > 0 else 0
        stats = {
            'enabled': int(self.use_safe_filters),
            'n_before': int(cand_n),
            'n_after': int(cand_n),
            'removed_total': 0,
            'removed_empty_peak': 0,
            'removed_precursor_cap': 0,
            'removed_invalid_dbe_parity': 0,
            'fallback_keep_one': 0,
        }

        if (not self.use_safe_filters) or cand_n <= 0:
            return formulae_arr, masses_arr, stats

        if masses_arr.ndim == 2 and masses_arr.shape[-1] == 2:
            masses_arr = masses_arr[:, None, :]
        if masses_arr.ndim != 3 or masses_arr.shape[-1] < 2:
            masses_arr = np.zeros((cand_n, max(1, int(self.num_peaks_per_formula)), 2), dtype=np.float32)

        peak_mass = np.asarray(masses_arr[..., 0], dtype=np.float32)
        peak_int = np.asarray(masses_arr[..., 1], dtype=np.float32)
        valid_peak = np.isfinite(peak_mass) & np.isfinite(peak_int) & (peak_int > 0.0)

        keep = np.ones((cand_n,), dtype=bool)

        if self.safe_filter_empty:
            has_peak = valid_peak.any(axis=1)
            stats['removed_empty_peak'] = int((~has_peak).sum())
            keep &= has_peak

        if self.safe_filter_precursor:
            try:
                pmz = float(precursor_mz)
            except Exception:
                pmz = float('nan')
            if np.isfinite(pmz) and pmz > 0.0:
                cap = float(pmz + max(0.0, float(self.safe_precursor_tol_da)))
                has_precursor_compatible_peak = (valid_peak & (peak_mass <= cap)).any(axis=1)
                stats['removed_precursor_cap'] = int((~has_precursor_compatible_peak).sum())
                keep &= has_precursor_compatible_peak

        if self.safe_filter_dbe:
            dbe, is_dbe_negative, parity_violation = _candidate_dbe_from_counts(
                formulae_arr,
                self.formula_possible_atomicno,
            )
            if dbe is not None:
                valid_dbe = np.ones((cand_n,), dtype=bool)

                if self.safe_filter_dbe_negative:
                    valid_dbe &= (is_dbe_negative <= 0.5)

                if self.safe_filter_dbe_parity:
                    valid_dbe &= (parity_violation <= 0.5)

                stats['removed_invalid_dbe_parity'] = int((~valid_dbe).sum())
                keep &= valid_dbe

        kept_idx = np.where(keep)[0].astype(np.int64)
        if kept_idx.size <= 0:
            stats['fallback_keep_one'] = 1
            try:
                score = np.asarray(peak_int, dtype=np.float32).sum(axis=1)
                best = int(np.argmax(score)) if score.size > 0 else 0
            except Exception:
                best = 0
            kept_idx = np.asarray([max(0, min(best, cand_n - 1))], dtype=np.int64)

        stats['n_after'] = int(kept_idx.size)
        stats['removed_total'] = int(max(0, cand_n - kept_idx.size))
        return formulae_arr[kept_idx], masses_arr[kept_idx], stats

    def _candidate_plausibility_score(
        self,
        formulae_arr,
        masses_arr=None,
        precursor_mz=None,
        precursor_formula=None,
        source_flag_arr=None,
        break_depth_arr=None,
        ring_cut_arr=None,
    ):
        """
        Label-free candidate prior.
        只使用分子式、precursor、结构来源标记，不使用真实谱。
        """
        try:
            base = _candidate_formula_prior_score_basic(
                formulae_arr=formulae_arr,
                formula_atomicnos=self.formula_possible_atomicno,
                precursor_mz=precursor_mz,
                precursor_formula=precursor_formula,
            )
        except Exception:
            formulae_arr = np.asarray(formulae_arr)
            base = np.zeros((int(formulae_arr.shape[0]),), dtype=np.float32)

        score = np.asarray(base, dtype=np.float32).reshape(-1).copy()
        n = int(score.shape[0])
        if n <= 0:
            return score

        # 结构来源：不是硬删，只是提权
        if source_flag_arr is not None:
            sf = np.asarray(source_flag_arr, dtype=np.int8).reshape(-1)
            if sf.shape[0] == n:
                is_struct = ((sf == 1) | (sf == 3)).astype(np.float32)
                is_common_loss = ((sf == 2) | (sf == 3)).astype(np.float32)
                score += 1.20 * is_struct
                score += 1.00 * is_common_loss
                score += 0.40 * (is_struct * is_common_loss)

        # 断裂深度：浅层更可信，深层仍然给小幅奖励。
        if break_depth_arr is not None:
            bd = np.asarray(break_depth_arr, dtype=np.int8).reshape(-1)
            if bd.shape[0] == n:
                score += 0.80 * (bd == 1).astype(np.float32)
                score += 0.55 * (bd == 2).astype(np.float32)
                score += 0.35 * (bd == 3).astype(np.float32)
                score += 0.20 * (bd >= 4).astype(np.float32)

        # 环切先惩罚，不硬删
        if ring_cut_arr is not None:
            rc = np.asarray(ring_cut_arr, dtype=np.int8).reshape(-1)
            if rc.shape[0] == n:
                score -= 0.40 * (rc > 0).astype(np.float32)

        # 支持峰数量过少/过空的候选略降；支持丰富一点的略加分
        if masses_arr is not None:
            try:
                cand = np.asarray(masses_arr, dtype=np.float32)
                if cand.ndim == 2 and cand.shape[-1] == 2:
                    cand = cand[:, None, :]
                if cand.ndim == 3 and cand.shape[0] == n and cand.shape[-1] >= 2:
                    mz = cand[..., 0]
                    inten = cand[..., 1]
                    valid = np.isfinite(mz) & np.isfinite(inten) & (inten > 0) & (mz >= 0)
                    peak_n = valid.sum(axis=1).astype(np.float32)
                    score += 0.05 * np.log1p(peak_n)
                    score -= 1.00 * (peak_n <= 0).astype(np.float32)
            except Exception:
                pass

        score = np.where(np.isfinite(score), score, -1e9).astype(np.float32)
        return score

    def _candidate_support_signatures(self, masses_arr):
        """
        Label-free support signature.
        用 candidate 自己投影出来的 m/z support 去重；不看真实谱。
        """
        try:
            bw = float(os.environ.get('FORMULA_ACTIVE_BIN_WIDTH', os.environ.get('OFFICIAL_BIN_WIDTH', '0.01')))
        except Exception:
            bw = 0.01
        bw = float(max(1e-6, bw))

        try:
            max_mz = float(os.environ.get('FORMULA_ACTIVE_MAX_MZ', os.environ.get('OFFICIAL_MAX_MZ', '1005.0')))
        except Exception:
            max_mz = 1005.0
        max_mz = float(max(bw, max_mz))

        cand = np.asarray(masses_arr, dtype=np.float32)
        if cand.ndim == 2 and cand.shape[-1] == 2:
            cand = cand[:, None, :]
        if cand.ndim != 3 or cand.shape[-1] < 2:
            return [tuple() for _ in range(int(cand.shape[0]) if cand.ndim > 0 else 0)]

        mz = cand[..., 0]
        inten = cand[..., 1]
        valid = (
            np.isfinite(mz)
            & np.isfinite(inten)
            & (inten > 0)
            & (mz >= 0)
            & (mz < max_mz)
        )

        out = []
        for i in range(int(cand.shape[0])):
            idx = np.floor(mz[i][valid[i]] / bw + 1e-8).astype(np.int64)
            if idx.size <= 0:
                out.append(tuple())
            else:
                out.append(tuple(np.unique(idx).astype(np.int64).tolist()))
        return out

    def _build_active_candidate_mask(
        self,
        formulae_arr,
        masses_arr,
        precursor_mz=None,
        precursor_formula=None,
        source_flag_arr=None,
        break_depth_arr=None,
        ring_cut_arr=None,
    ):
        """
        从 full pool 中标出 active pool。
        full pool 仍保留；active mask 给模型或诊断重点使用。
        """
        formulae_arr = np.asarray(formulae_arr)
        n = int(formulae_arr.shape[0]) if formulae_arr.ndim > 0 else 0
        if n <= 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

        score = self._candidate_plausibility_score(
            formulae_arr=formulae_arr,
            masses_arr=masses_arr,
            precursor_mz=precursor_mz,
            precursor_formula=precursor_formula,
            source_flag_arr=source_flag_arr,
            break_depth_arr=break_depth_arr,
            ring_cut_arr=ring_cut_arr,
        )

        topk = max(1, min(int(self.active_topk), n))
        active = np.zeros((n,), dtype=np.float32)

        sf = None
        if source_flag_arr is not None:
            sf = np.asarray(source_flag_arr, dtype=np.int8).reshape(-1)
            if sf.shape[0] != n:
                sf = None

        # 结构来源候选不要轻易扔掉，因为它们是“高 precision 但 coverage 不足”的那部分
        if self.active_keep_all_source and sf is not None:
            active[sf > 0] = 1.0

        # support signature 去重：同一个 support 只保留 prior 最高的候选
        sigs = self._candidate_support_signatures(masses_arr)
        order = np.argsort(-score, kind='stable')

        selected_by_sig = []
        seen = set()
        for idx in order.tolist():
            sig = sigs[int(idx)] if int(idx) < len(sigs) else tuple()
            if len(sig) <= 0:
                continue
            if sig in seen:
                continue
            seen.add(sig)
            selected_by_sig.append(int(idx))

        for idx in selected_by_sig:
            if int(active.sum()) >= topk:
                break
            active[int(idx)] = 1.0

        # 兜底：如果全空，直接取 prior topK
        if float(active.sum()) <= 0:
            active[order[:topk]] = 1.0

        return active.astype(np.float32), score.astype(np.float32)

    # Function overview: _resolve_overflow_indices handles a specific workflow step in this module.
    def _resolve_overflow_indices(
        self,
        mol,
        formulae_n,
        formulae=None,
        masses=None,
        spect_dense=None,
        spect_bin_config=None,
        precursor_mz=None,
        precursor_formula=None,
        source_flag=None,
        break_depth=None,
        ring_cut_flag=None,
    ):
        self.last_overflow_used_label_greedy = 0
        if formulae_n <= self.max_formulae:
            return np.arange(formulae_n, dtype=np.int64), 0

        def _topk_keep_by_score(score_1d):
            try:
                score = np.asarray(score_1d, dtype=np.float32).reshape(-1)
            except Exception:
                return None
            if score.size != int(formulae_n):
                return None
            score = np.where(np.isfinite(score), score, -np.inf)
            if not np.isfinite(score).any():
                return None
            keep_local = np.argsort(-score, kind='stable')[:self.max_formulae].astype(np.int64)
            return np.sort(keep_local)
        def _candidate_plausibility_score(
            formulae_arr,
            masses_arr,
            source_flag_arr=None,
            break_depth_arr=None,
            ring_cut_arr=None,
        ):
            return self._candidate_plausibility_score(
                formulae_arr=formulae_arr,
                masses_arr=masses_arr,
                precursor_mz=precursor_mz,
                precursor_formula=precursor_formula,
                source_flag_arr=source_flag_arr,
                break_depth_arr=break_depth_arr,
                ring_cut_arr=ring_cut_arr,
            )
        
        def _candidate_support_bins_from_masses(masses_arr, first_bin_center=1.0, bin_width=1.0, n_bins=None):
            cand = np.asarray(masses_arr, dtype=np.float32)
            if cand.ndim == 2 and cand.shape[-1] == 2:
                cand = cand[:, None, :]
            if cand.ndim != 3 or cand.shape[-1] < 2:
                return None, None, None

            peak_mass = cand[..., 0]
            peak_int = np.maximum(cand[..., 1], 0.0)
            valid = np.isfinite(peak_mass) & np.isfinite(peak_int) & (peak_int > 0)

            peak_bin = np.floor((peak_mass - first_bin_center + (bin_width / 2.0)) / bin_width + 1e-8).astype(np.int64)

            support_bins = []
            support_int_sum = np.zeros((cand.shape[0],), dtype=np.float32)
            support_hit_n = np.zeros((cand.shape[0],), dtype=np.int64)
            for i in range(cand.shape[0]):
                idx_i = peak_bin[i][valid[i]]
                int_i = peak_int[i][valid[i]]
                if n_bins is not None:
                    keep = (idx_i >= 0) & (idx_i < int(n_bins))
                    idx_i = idx_i[keep]
                    int_i = int_i[keep]
                if idx_i.size <= 0:
                    support_bins.append(np.zeros((0,), dtype=np.int64))
                    support_int_sum[i] = 0.0
                    support_hit_n[i] = 0
                    continue
                support_bins.append(np.unique(idx_i.astype(np.int64)))
                support_int_sum[i] = float(np.sum(int_i))
                support_hit_n[i] = int(idx_i.shape[0])
            return support_bins, support_int_sum, support_hit_n

        def _greedy_coverage_keep(masses_arr, spect_dense_arr, spect_bin_config=None, max_keep=4096):
            if masses_arr is None or spect_dense_arr is None:
                return None

            true_dense = np.asarray(spect_dense_arr, dtype=np.float32).reshape(-1)
            if true_dense.size <= 0:
                return None

            first_bin_center = 1.0
            bin_width = 1.0
            if isinstance(spect_bin_config, dict):
                first_bin_center = float(spect_bin_config.get('first_bin_center', first_bin_center))
                bin_width = float(spect_bin_config.get('bin_width', bin_width))

            target_bins = np.where(np.isfinite(true_dense) & (true_dense > 0))[0].astype(np.int64)
            if target_bins.size <= 0:
                return None
            target_bin_set = set(int(x) for x in target_bins.tolist())
            target_weight = {int(i): float(true_dense[int(i)]) for i in target_bins.tolist()}

            parsed = _candidate_support_bins_from_masses(
                masses_arr,
                first_bin_center=first_bin_center,
                bin_width=bin_width,
                n_bins=int(true_dense.shape[0]),
            )
            if parsed is None:
                return None
            cand_bins, cand_int_sum, cand_peak_n = parsed
            if cand_bins is None or len(cand_bins) != int(formulae_n):
                return None

            chem_score = _candidate_plausibility_score(formulae, masses_arr)
            if chem_score is None or chem_score.shape[0] != int(formulae_n):
                chem_score = np.zeros((int(formulae_n),), dtype=np.float32)

            try:
                chem_weight = float(os.environ.get('FORMULA_OVERFLOW_CHEM_WEIGHT', '0.35'))
            except Exception:
                chem_weight = 0.35
            chem_weight = float(max(0.0, chem_weight))

            true_peak_n = int(target_bins.size)
            top20_true_idx = np.argsort(-true_dense, kind='stable')[: min(20, int(true_dense.size))]
            top20_true_bin_set = set(int(b) for b in top20_true_idx.tolist() if true_dense[int(b)] > 0)
            min_hit_bins = 1 if true_peak_n <= 4 else 2

            # 1) peak-hit prefilter
            hit_idx = []
            hit_score = []
            for i, bins_i in enumerate(cand_bins):
                if bins_i is None or bins_i.size <= 0:
                    continue
                bins_hit = [int(b) for b in bins_i.tolist() if int(b) in target_bin_set]
                if len(bins_hit) < min_hit_bins:
                    continue

                int_gain = float(sum(np.sqrt(max(target_weight.get(int(b), 0.0), 0.0)) for b in bins_hit))
                count_gain = float(len(bins_hit))
                top20_gain = float(sum(1 for b in bins_hit if int(b) in top20_true_bin_set))
                gain = (1.0 * int_gain) + (0.8 * count_gain) + (1.2 * top20_gain) + (chem_weight * float(chem_score[i]))

                hit_idx.append(int(i))
                hit_score.append(gain)

            if len(hit_idx) <= 0:
                return None

            # 2) support signature 去重：同 support 只保留一个 gain 最大的
            best_by_sig = {}
            for i, g in zip(hit_idx, hit_score):
                sig = tuple(int(x) for x in cand_bins[i].tolist())
                old = best_by_sig.get(sig, None)
                if old is None or g > old[1]:
                    best_by_sig[sig] = (int(i), float(g))

            try:
                dbg_flag = os.environ.get("DEBUG_OVERFLOW_SIGNATURE", "0") == "1"
            except Exception:
                dbg_flag = False

            candidate_order = [v[0] for v in best_by_sig.values()]

            if dbg_flag:
                print(
                    "[DEBUG_OVERFLOW_SIGNATURE]",
                    f"formulae_n={int(formulae_n)}",
                    f"hit_idx_n={len(hit_idx)}",
                    f"signature_n={len(candidate_order)}",
                    f"max_keep={int(max_keep)}",
                )
            if len(candidate_order) <= int(max_keep):
                return np.sort(np.asarray(candidate_order, dtype=np.int64))

            # 3) greedy marginal coverage keep
            selected = []
            covered = set()
            selected_set = set()
            while len(selected) < int(max_keep):
                best_i = None
                best_gain = -1.0
                best_tie = -1.0

                for i in candidate_order:
                    if i in selected_set:
                        continue

                    bins_i = cand_bins[i]
                    if bins_i is None or bins_i.size <= 0:
                        continue

                    new_bins = [int(b) for b in bins_i.tolist() if int(b) in target_bin_set and int(b) not in covered]
                    if len(new_bins) < min_hit_bins:
                        continue

                    int_gain = float(sum(np.sqrt(max(target_weight.get(int(b), 0.0), 0.0)) for b in new_bins))
                    count_gain = float(len(new_bins))
                    top20_gain = float(sum(1 for b in new_bins if int(b) in top20_true_bin_set))
                    gain = (1.0 * int_gain) + (0.8 * count_gain) + (1.2 * top20_gain) + (chem_weight * float(chem_score[i]))

                    tie = float(cand_int_sum[i]) + 0.05 * float(cand_peak_n[i])

                    if (gain > best_gain) or (gain == best_gain and tie > best_tie):
                        best_i = int(i)
                        best_gain = float(gain)
                        best_tie = float(tie)

                if best_i is None:
                    break

                selected.append(int(best_i))
                selected_set.add(int(best_i))
                for b in cand_bins[best_i].tolist():
                    bb = int(b)
                    if bb in target_bin_set:
                        covered.add(bb)

            # 4) 不够再按 hit_score 补齐
            if len(selected) < int(max_keep):
                residual = [i for i in candidate_order if i not in selected_set]
                residual_score = []
                for i in residual:
                    bins_i = cand_bins[i]
                    bins_hit = [int(b) for b in bins_i.tolist() if int(b) in target_bin_set]
                    if len(bins_hit) < min_hit_bins:
                        residual_score.append(-1e9)
                        continue
                    int_gain = float(sum(np.sqrt(max(target_weight.get(int(b), 0.0), 0.0)) for b in bins_hit))
                    count_gain = float(len(bins_hit))
                    top20_gain = float(sum(1 for b in bins_hit if int(b) in top20_true_bin_set))
                    gain = (1.0 * int_gain) + (0.8 * count_gain) + (1.2 * top20_gain) + (chem_weight * float(chem_score[i]))
                    residual_score.append(gain)     

                if len(residual) > 0:
                    order = np.argsort(-np.asarray(residual_score, dtype=np.float32), kind='stable')
                    for j in order.tolist():
                        if len(selected) >= int(max_keep):
                            break
                        if residual_score[j] <= -1e8:
                            continue
                        selected.append(int(residual[j]))

            if len(selected) <= 0:
                return None
            return np.sort(np.asarray(selected[: int(max_keep)], dtype=np.int64))

        mode = self.overflow_mode
        if mode == 'raise':
            raise ValueError(
                f"molecule {Chem.MolToSmiles(mol)} has {formulae_n}, more than the limit of {self.max_formulae} "
                f"(naive count={self.num_unique_f(mol)})"
            )

        if mode in ('random', 'sample'):
            smi = Chem.MolToSmiles(mol)
            seed_bytes = f"{smi}|{self.overflow_sample_seed}".encode('utf-8')
            seed = int(hashlib.sha1(seed_bytes).hexdigest()[:8], 16)
            rng = np.random.default_rng(seed)
            keep = np.sort(rng.choice(formulae_n, size=self.max_formulae, replace=False).astype(np.int64))
            return keep, 1

        if mode in ('topk_mass', 'mass', 'mz') and masses is not None:
            try:
                mass = np.asarray(masses[..., 0], dtype=np.float32)
                if mass.ndim == 1:
                    score = mass
                else:
                    score = mass.max(axis=1)
                keep = np.argsort(-score, kind='stable')[:self.max_formulae].astype(np.int64)
                keep = np.sort(keep)
                return keep, 1
            except Exception:
                pass

        if mode in ('topk_prob', 'topk_intensity', 'topk', 'importance', 'importance_sampling') and masses is not None:
            try:
                intensity = np.asarray(masses[..., 1], dtype=np.float32)
                if intensity.ndim == 1:
                    score = intensity
                else:
                    score = intensity.sum(axis=1)
                keep = np.argsort(-score, kind='stable')[:self.max_formulae].astype(np.int64)
                keep = np.sort(keep)
                return keep, 1
            except Exception:
                pass

        if mode in ('coverage_topk', 'coverage', 'coverage_hshift') and masses is not None:
            # 只有 oracle / upper-bound 诊断允许用真实谱 greedy coverage。
            # 正式 cache 默认 FORMULA_ALLOW_LABEL_OVERFLOW=0，因此即使 spect_dense 意外非空，也不会泄露。
            if spect_dense is not None and bool(getattr(self, 'allow_label_overflow', False)):
                try:
                    keep = _greedy_coverage_keep(
                        masses_arr=masses,
                        spect_dense_arr=spect_dense,
                        spect_bin_config=spect_bin_config,
                        max_keep=self.max_formulae,
                    )
                    if keep is not None:
                        self.last_overflow_used_label_greedy = 1
                        return keep, 1
                except Exception:
                    pass

            # 正式无泄露路径：只用分子式、precursor、结构来源、common loss、DBE 等 no-label prior。
            try:
                score = self._candidate_plausibility_score(
                    formulae_arr=formulae,
                    masses_arr=masses,
                    precursor_mz=precursor_mz,
                    precursor_formula=precursor_formula,
                    source_flag_arr=source_flag,
                    break_depth_arr=break_depth,
                    ring_cut_arr=ring_cut_flag,
                )
                keep = _topk_keep_by_score(score)
                if keep is not None:
                    return keep, 1
            except Exception:
                pass

        if mode in ('truncate', 'head', 'deterministic') and masses is not None:
            reorder_before_truncate = os.environ.get('FORMULA_TRUNCATE_REORDER', '1') == '1'
            if reorder_before_truncate:
                # 1) 正式路径：优先 no-label prior，不看真实谱。
                try:
                    score = self._candidate_plausibility_score(
                        formulae_arr=formulae,
                        masses_arr=masses,
                        precursor_mz=precursor_mz,
                        precursor_formula=precursor_formula,
                        source_flag_arr=source_flag,
                        break_depth_arr=break_depth,
                        ring_cut_arr=ring_cut_flag,
                    )
                    keep = _topk_keep_by_score(score)
                    if keep is not None:
                        return keep, 1
                except Exception:
                    pass

                # 2) 只有 oracle / upper-bound 诊断才允许用真实谱 greedy。
                if spect_dense is not None and bool(getattr(self, 'allow_label_overflow', False)):
                    try:
                        keep = _greedy_coverage_keep(
                            masses_arr=masses,
                            spect_dense_arr=spect_dense,
                            spect_bin_config=spect_bin_config,
                            max_keep=self.max_formulae,
                        )
                        if keep is not None:
                            self.last_overflow_used_label_greedy = 1
                            return keep, 1
                    except Exception:
                        pass

                # 3) 最后兜底：旧逻辑。只有 prior 全失败才会走这里。
                try:
                    intensity = np.asarray(masses[..., 1], dtype=np.float32)
                    mass = np.asarray(masses[..., 0], dtype=np.float32)
                    if intensity.ndim == 1:
                        inten_score = intensity
                    else:
                        inten_score = intensity.sum(axis=1)
                    if mass.ndim == 1:
                        mass_score = mass
                    else:
                        mass_score = mass.max(axis=1)

                    score = np.asarray(inten_score, dtype=np.float32) + (
                        1e-4 * np.asarray(mass_score, dtype=np.float32)
                    )
                    keep = _topk_keep_by_score(score)
                    if keep is not None:
                        return keep, 1
                except Exception:
                    pass

        keep = np.arange(self.max_formulae, dtype=np.int64)
        return keep, 1
    # Function overview: num_unique_f handles a specific workflow step in this module.
    def num_unique_f(self, m):
        f = util.get_formula(Chem.AddHs(Chem.Mol(m)))
        return np.prod([v + 1 for v in f.values()])

    # mol -> self.ffe.get_frag_formulae(...) -> formulae, masses -> multipeak归一化 -> overflow截断 -> pad到max_formulae -> 返回 
    # 候选峰先生成，再归一化，再截断
    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(
        self,
        mol,
        formulae=None,
        masses=None,
        return_n_raw=False,
        sparse_spect=None,
        spect_bin_config=None,
        adduct=None,
        precursor_mz=None,
        precursor_formula=None,

    ):
        import time
        
        if formulae is None or masses is None:
            t0 = time.time()
            if self.verbose:
                print(f"[PossibleFormulaFeaturizer] computing fragment formulae for mol (atoms={mol.GetNumAtoms()})...")
            formulae, masses = self.ffe.get_frag_formulae(mol)
            self.last_peak_backend = getattr(self.ffe, 'peak_backend', 'unknown')
            t1 = time.time()
            if self.verbose:
                print(f"[PossibleFormulaFeaturizer] fragment enumeration took {t1-t0:.3f}s | formulae={len(formulae)} | masses shape={getattr(masses,'shape',str(type(masses)))}")

        masses = masses[:, :self.num_peaks_per_formula]

        if self.apply_adduct_shift:
            try:
                adduct_shift = float(self._resolve_adduct_shift(adduct))
            except Exception:
                adduct_shift = 0.0
            if abs(adduct_shift) > 1e-12:
                masses = np.array(masses, dtype=np.float32, copy=True)
                masses[:, :, 0] = masses[:, :, 0] + float(adduct_shift)

        # Normalize each candidate's peak distribution after backend/fallback generation.
        masses[:, :, 1] = masses[:, :, 1] / np.clip(
            np.sum(masses[:, :, 1], axis=1).reshape(-1, 1),
            a_min=0.01,
            a_max=1.0,
        )

        src_formulae_n_raw = int(len(formulae))
        self.last_formulae_n_raw = src_formulae_n_raw
        masses_src = np.asarray(masses, dtype=np.float32)

        filter_stats = {
            'enabled': 0,
            'n_before': int(src_formulae_n_raw),
            'n_after': int(src_formulae_n_raw),
            'removed_total': 0,
            'removed_empty_peak': 0,
            'removed_precursor_cap': 0,
            'removed_invalid_dbe_parity': 0,
            'fallback_keep_one': 0,
        }
        if self.use_safe_filters:
            formulae, masses, filter_stats = self._apply_safe_formula_filters(
                formulae,
                masses,
                precursor_mz=precursor_mz,
            )

        src_formulae_n = int(len(formulae))

        source_flag = np.zeros((src_formulae_n,), dtype=np.int8)
        break_depth = np.zeros((src_formulae_n,), dtype=np.int8)
        ring_cut_flag = np.zeros((src_formulae_n,), dtype=np.int8)
        struct_diag = {
            'structure_selected_before_overflow': 0,
            'structure_single_before_overflow': 0,
            'structure_double_before_overflow': 0,
            'formula_supplement_before_overflow': 0,
            'structure_selected_after_overflow': 0,
            'structure_single_after_overflow': 0,
            'structure_double_after_overflow': 0,
            'structure_deep_before_overflow': 0,
            'structure_deep_after_overflow': 0,
        }
        try:
            prior_score = _candidate_formula_prior_score_basic(
                formulae_arr=formulae,
                formula_atomicnos=self.formula_possible_atomicno,
                precursor_mz=precursor_mz,
                precursor_formula=precursor_formula,
            )
        except Exception:
            prior_score = np.zeros((src_formulae_n,), dtype=np.float32)

        # ===== structure-guided annotation / optional preselection (no true spectrum) =====
        if self.use_structure_prior and src_formulae_n > 0:
            # prior_score 已经在上方统一初始化，并且可能已经叠加 FIORA prior。
            # 这里不要重新计算，否则会覆盖 FIORA formula prior。
            prior_score = np.asarray(prior_score, dtype=np.float32)

            if self.multidepth_struct_enable:
                subsets, depths, ring_flags = _enumerate_multidepth_structural_subsets(
                    mol=mol,
                    max_depth=int(self.multidepth_struct_depth),
                    max_nodes=int(self.multidepth_struct_max_nodes),
                    allow_ring=bool(self.multidepth_struct_allow_ring),
                    min_heavy_atoms=1,
                    max_branch_per_node=int(self.multidepth_struct_branch),
                    include_root=bool(self.multidepth_struct_include_root),
                )
            else:
                subsets, depths, ring_flags = _enumerate_break_connected_subsets(
                    mol,
                    max_single=self.structure_single_keep,
                    max_double=self.structure_double_keep,
                    allow_ring=self.structure_allow_ring,
                    max_seed_bonds=self.structure_seed_bonds,
                    min_heavy_atoms=2,
                )

            struct_idx, struct_depth, struct_ring = _select_formula_candidates_from_subsets(
                mol=mol,
                formulae_arr=formulae,
                subsets=subsets,
                subset_depths=depths,
                subset_ring_flags=ring_flags,
                formula_atomicnos=self.formula_possible_atomicno,
                candidate_prior_score=prior_score,
                topn_per_subset=self.structure_topn_per_subset,
                h_window=self.structure_h_window,
            )

            if struct_idx.size > 0:
                source_flag[struct_idx] = 1
                break_depth[struct_idx] = struct_depth
                ring_cut_flag[struct_idx] = struct_ring

            # More precise exact structural marking: connected component formula +/- H shift.
            if self.structure_exact_mark:
                exact_idx, exact_depth, exact_ring = _select_exact_structural_formula_candidates(
                    mol=mol,
                    formulae_arr=formulae,
                    subsets=subsets,
                    subset_depths=depths,
                    subset_ring_flags=ring_flags,
                    formula_atomicnos=self.formula_possible_atomicno,
                    h_window=self.structure_h_window,
                )
                if exact_idx.size > 0:
                    source_flag[exact_idx] = np.maximum(source_flag[exact_idx], 1)
                    # keep better / shallower break depth
                    old_depth = break_depth[exact_idx]
                    new_depth = exact_depth
                    merged_depth = np.where(old_depth > 0, np.minimum(old_depth, new_depth), new_depth)
                    break_depth[exact_idx] = merged_depth.astype(np.int8)
                    ring_cut_flag[exact_idx] = np.maximum(ring_cut_flag[exact_idx], exact_ring).astype(np.int8)

            # Common neutral-loss candidates: precursor - common loss +/- H shift.
            if self.common_loss_mark:
                loss_idx = _select_common_loss_formula_candidates(
                    formulae_arr=formulae,
                    precursor_formula=precursor_formula,
                    formula_atomicnos=self.formula_possible_atomicno,
                    h_window=self.common_loss_h_window,
                )
                if loss_idx.size > 0:
                    # 2 = common-loss source, 3 = both structure and common-loss
                    source_flag[loss_idx] = np.where(source_flag[loss_idx] > 0, 3, 2).astype(np.int8)

            struct_diag['structure_selected_before_overflow'] = int(np.sum(source_flag > 0))
            struct_diag['structure_single_before_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 1)))
            struct_diag['structure_double_before_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 2)))
            struct_diag['structure_deep_before_overflow'] = int(np.sum((source_flag > 0) & (break_depth >= 3)))

            # Optional hard preselection. Keep OFF for V1c.
            if self.structure_preselect:
                struct_all_idx = np.where(source_flag > 0)[0].astype(np.int64)

                residual = np.where(source_flag <= 0)[0].astype(np.int64)
                supp_keep_n = max(0, int(self.formula_supplement_keep))
                if residual.size > 0 and supp_keep_n > 0:
                    order = np.argsort(-prior_score[residual], kind='stable')
                    supp_idx = residual[order[:supp_keep_n]]
                else:
                    supp_idx = np.zeros((0,), dtype=np.int64)

                pre_keep = np.unique(np.concatenate([struct_all_idx, supp_idx], axis=0)).astype(np.int64)

                min_keep = min(int(self.max_formulae), max(0, int(self.preselect_min_keep)))
                if pre_keep.size < min_keep:
                    used = np.zeros((src_formulae_n,), dtype=bool)
                    used[pre_keep] = True
                    rest = np.where(~used)[0].astype(np.int64)
                    if rest.size > 0:
                        order = np.argsort(-prior_score[rest], kind='stable')
                        extra = rest[order[: max(0, int(min_keep - pre_keep.size))]]
                        pre_keep = np.unique(np.concatenate([pre_keep, extra], axis=0)).astype(np.int64)

                if pre_keep.size > 0:
                    formulae = formulae[pre_keep]
                    masses = masses[pre_keep]
                    source_flag = source_flag[pre_keep]
                    break_depth = break_depth[pre_keep]
                    ring_cut_flag = ring_cut_flag[pre_keep]
                    src_formulae_n = int(len(formulae))

            struct_diag['formula_supplement_before_overflow'] = int(np.sum(source_flag <= 0))
        # ===== end structure-guided block =====

        spect_dense = _sparse_spect_to_dense(sparse_spect, spect_bin_config)
        keep_idx, truncated = self._resolve_overflow_indices(
            mol,
            src_formulae_n,
            formulae=formulae,
            masses=masses,
            spect_dense=spect_dense,
            spect_bin_config=spect_bin_config,
            precursor_mz=precursor_mz,
            precursor_formula=precursor_formula,
            source_flag=source_flag,
            break_depth=break_depth,
            ring_cut_flag=ring_cut_flag,
        )

        formulae = formulae[keep_idx]
        masses = masses[keep_idx]
        source_flag = source_flag[keep_idx]
        break_depth = break_depth[keep_idx]
        ring_cut_flag = ring_cut_flag[keep_idx]
        kept_n = int(len(formulae))
        final_active_mask, final_prior_score = self._build_active_candidate_mask(
            formulae_arr=formulae,
            masses_arr=masses,
            precursor_mz=precursor_mz,
            precursor_formula=precursor_formula,
            source_flag_arr=source_flag,
            break_depth_arr=break_depth,
            ring_cut_arr=ring_cut_flag,
        )
        self.last_formulae_n_kept = kept_n
        self.last_formulae_truncated = int(truncated)
        self.last_formulae_source_flag = np.zeros((self.max_formulae,), dtype=np.int8)
        self.last_formulae_break_depth = np.zeros((self.max_formulae,), dtype=np.int8)
        self.last_formulae_ring_cut_flag = np.zeros((self.max_formulae,), dtype=np.int8)
        self.last_formulae_prior_score = np.zeros((self.max_formulae,), dtype=np.float32)
        self.last_formulae_active_mask = np.zeros((self.max_formulae,), dtype=np.float32)
        self.last_formulae_source_flag[:kept_n] = source_flag[:kept_n]
        self.last_formulae_break_depth[:kept_n] = break_depth[:kept_n]
        self.last_formulae_ring_cut_flag[:kept_n] = ring_cut_flag[:kept_n]
        self.last_formulae_prior_score[:kept_n] = final_prior_score[:kept_n]
        self.last_formulae_active_mask[:kept_n] = final_active_mask[:kept_n]
        struct_diag['structure_selected_after_overflow'] = int(np.sum(source_flag > 0))
        struct_diag['structure_single_after_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 1)))
        struct_diag['structure_double_after_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 2)))
        struct_diag['structure_deep_after_overflow'] = int(np.sum((source_flag > 0) & (break_depth >= 3)))

        self.last_cache_diag['structure_selected_after_overflow'] = int(np.sum(source_flag > 0))
        self.last_cache_diag['structure_single_after_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 1)))
        self.last_cache_diag['structure_double_after_overflow'] = int(np.sum((source_flag > 0) & (break_depth == 2)))
        self.last_cache_diag['safe_filter_enabled'] = int(filter_stats.get('enabled', 0))
        self.last_cache_diag['safe_n_before'] = int(filter_stats.get('n_before', 0))
        self.last_cache_diag['safe_n_after'] = int(filter_stats.get('n_after', 0))
        self.last_cache_diag['safe_removed_total'] = int(filter_stats.get('removed_total', 0))
        self.last_cache_diag['safe_removed_empty_peak'] = int(filter_stats.get('removed_empty_peak', 0))
        self.last_cache_diag['safe_removed_precursor_cap'] = int(filter_stats.get('removed_precursor_cap', 0))
        self.last_cache_diag['safe_removed_invalid_dbe_parity'] = int(filter_stats.get('removed_invalid_dbe_parity', 0))
        self.last_cache_diag['safe_fallback_keep_one'] = int(filter_stats.get('fallback_keep_one', 0))
        # Cache diagnostics used by cache_featurizer_condv2 to localize
        # candidate/support issues before running full training.
        try:
            diag_bin_width = float(os.environ.get('CACHE_DIAG_BIN_WIDTH', '0.01'))
        except Exception:
            diag_bin_width = 0.01
        diag_bin_width = float(max(1e-6, diag_bin_width))

        try:
            diag_max_mz = float(os.environ.get('CACHE_DIAG_MAX_MZ', '1005.0'))
        except Exception:
            diag_max_mz = 1005.0
        diag_max_mz = float(max(diag_bin_width, diag_max_mz))
        try:
            precursor_tol_da = float(os.environ.get('CACHE_DIAG_PRECURSOR_TOL_DA', '1.0'))
        except Exception:
            precursor_tol_da = 1.0
        precursor_limit = None
        try:
            pmz = float(precursor_mz)
            if np.isfinite(pmz) and pmz > 0:
                precursor_limit = pmz + float(max(0.0, precursor_tol_da))
        except Exception:
            precursor_limit = None

        target_bins = np.zeros((0,), dtype=np.int64)
        if sparse_spect is not None:
            try:
                spect_arr = np.asarray(sparse_spect, dtype=np.float32)
                if spect_arr.size > 0:
                    spect_arr = spect_arr.reshape(-1, 2)
                    mz_t = spect_arr[:, 0]
                    it_t = spect_arr[:, 1]
                    valid_t = (
                        np.isfinite(mz_t)
                        & np.isfinite(it_t)
                        & (it_t > 0)
                        & (mz_t >= 0)
                        & (mz_t < diag_max_mz)
                    )
                    if np.any(valid_t):
                        t_idx = np.floor(mz_t[valid_t] / diag_bin_width + 1e-8).astype(np.int64)
                        target_bins = np.unique(t_idx)
            except Exception:
                target_bins = np.zeros((0,), dtype=np.int64)

        # Helper: summarize candidate support statistics for one candidate-peak tensor.
        def _summarize_candidate_block(mass_block):
            arr = np.asarray(mass_block, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[-1] == 2:
                arr = arr[:, None, :]
            if arr.ndim != 3 or arr.shape[-1] < 2:
                return {
                    'n_formulae_after_mass_mask': 0,
                    'n_formulae_after_precursor_mask': 0,
                    'support_peak_n': 0,
                    'candidate_mz_unique_n': 0,
                    'support_bins': np.zeros((0,), dtype=np.int64),
                    'coverage': float('nan'),
                }

            mz = arr[..., 0]
            inten = arr[..., 1]
            valid = (
                np.isfinite(mz)
                & np.isfinite(inten)
                & (inten > 0)
                & (mz >= 0)
                & (mz < diag_max_mz)
            )
            has_mass = valid.any(axis=1)

            if precursor_limit is not None:
                has_prec = (valid & (mz <= float(precursor_limit))).any(axis=1)
            else:
                has_prec = has_mass.copy()

            if np.any(valid):
                support_bins = np.unique(np.floor(mz[valid] / diag_bin_width + 1e-8).astype(np.int64))
            else:
                support_bins = np.zeros((0,), dtype=np.int64)

            if target_bins.size > 0:
                overlap = np.intersect1d(target_bins, support_bins, assume_unique=False)
                coverage = float(overlap.size) / float(max(1, target_bins.size))
            else:
                coverage = float('nan')

            return {
                'n_formulae_after_mass_mask': int(has_mass.sum()),
                'n_formulae_after_precursor_mask': int(has_prec.sum()),
                'support_peak_n': int(valid.sum()),
                'candidate_mz_unique_n': int(support_bins.size),
                'support_bins': support_bins,
                'coverage': float(coverage),
            }

        before_block = _summarize_candidate_block(masses_src)
        after_block = _summarize_candidate_block(masses)
        self.last_cache_diag = {
            'n_formulae_raw': int(src_formulae_n_raw),
            'n_formulae_after_safe_filter': int(filter_stats.get('n_after', len(formulae))),
            'n_formulae_final': int(kept_n),
            'overflow': int(truncated),
            'max_formulae': int(self.max_formulae),
            'overflow_used_label_greedy': int(getattr(self, 'last_overflow_used_label_greedy', 0)),
            'safe_filter_enabled': int(filter_stats.get('enabled', 0)),
            'safe_filter_removed_total': int(filter_stats.get('removed_total', 0)),
            'safe_filter_removed_empty_peak': int(filter_stats.get('removed_empty_peak', 0)),
            'safe_filter_removed_precursor_cap': int(filter_stats.get('removed_precursor_cap', 0)),
            'safe_filter_removed_invalid_dbe_parity': int(filter_stats.get('removed_invalid_dbe_parity', 0)),
            'safe_filter_fallback_keep_one': int(filter_stats.get('fallback_keep_one', 0)),
            'n_formulae_after_mass_mask_before_cap': int(before_block['n_formulae_after_mass_mask']),
            'n_formulae_after_mass_mask_after_cap': int(after_block['n_formulae_after_mass_mask']),
            'n_formulae_after_precursor_mask_before_cap': int(before_block['n_formulae_after_precursor_mask']),
            'n_formulae_after_precursor_mask_after_cap': int(after_block['n_formulae_after_precursor_mask']),
            'support_peak_n_before_cap': int(before_block['support_peak_n']),
            'support_peak_n_after_cap': int(after_block['support_peak_n']),
            'candidate_mz_unique_n_before_cap': int(before_block['candidate_mz_unique_n']),
            'candidate_mz_unique_n_after_cap': int(after_block['candidate_mz_unique_n']),
            'support_coverage_before_cap': float(before_block['coverage']),
            'support_coverage_after_cap': float(after_block['coverage']),
            'target_nonzero_bins': int(target_bins.size),
            'diag_bin_width': float(diag_bin_width),
            'diag_max_mz': float(diag_max_mz),
            'structure_selected_before_overflow': int(struct_diag.get('structure_selected_before_overflow', 0)),
            'structure_single_before_overflow': int(struct_diag.get('structure_single_before_overflow', 0)),
            'structure_double_before_overflow': int(struct_diag.get('structure_double_before_overflow', 0)),
            'formula_supplement_before_overflow': int(struct_diag.get('formula_supplement_before_overflow', 0)),
            'structure_selected_after_overflow': int(struct_diag.get('structure_selected_after_overflow', 0)),
            'structure_single_after_overflow': int(struct_diag.get('structure_single_after_overflow', 0)),
            'structure_double_after_overflow': int(struct_diag.get('structure_double_after_overflow', 0)),
            'structure_deep_before_overflow': int(struct_diag.get('structure_deep_before_overflow', 0)),
            'structure_deep_after_overflow': int(struct_diag.get('structure_deep_after_overflow', 0)),
            'fiora_formula_mark_before_overflow': int(struct_diag.get('fiora_formula_mark_before_overflow', 0)),
            'active_candidate_n': int(np.sum(final_active_mask > 0.5)),
            'active_topk_cfg': int(self.active_topk),
            'active_source_candidate_n': int(np.sum((final_active_mask > 0.5) & (source_flag > 0))),
            'active_ratio': float(np.sum(final_active_mask > 0.5) / max(1, kept_n)),
            'prior_score_mean': float(np.mean(final_prior_score)) if final_prior_score.size > 0 else float('nan'),
            'prior_score_p90': float(np.percentile(final_prior_score, 90.0)) if final_prior_score.size > 0 else float('nan'),
        }

        self.last_formulae_mask = np.zeros((self.max_formulae,), dtype=np.float32)
        self.last_formulae_mask[:kept_n] = 1.0

        masses = np.pad(masses, ((0, self.max_formulae - kept_n),
                                 (0, 0),
                                 (0, 0)))

        # keep padded region fully silent (no dummy peak), mask handles invalid entries downstream
        masses[kept_n:, :, 1] = 0.0
                        
        if np.max(masses) > 511:
            s = f"DEBUG mass too large, mol={Chem.MolToSmiles(mol)} weight={ Chem.Descriptors.ExactMolWt(mol)}, peak_mass={np.max(masses)}"
            if self.clip_mass > 0:
                # print debug and clip; clipping can be expensive for very large arrays
                if self.verbose:
                    print(f"[PossibleFormulaFeaturizer] {s} -> clipping to {self.clip_mass}")
                masses = np.clip(masses, 0, self.clip_mass)
            else:
                raise ValueError(s)

        if self.featurize_mode == 'numerical':
            formulae_feat = formulae.astype(np.float32)
            formulae_feat = np.pad(formulae_feat, 
                                   [(0, self.max_formulae - kept_n), 
                                    (0, 0)])

            if return_n_raw:
                return formulae_feat, masses, int(self.last_formulae_n_raw)
            return formulae_feat, masses 
            
        elif self.featurize_mode == 'onehot' :
            formulae_feat = np.zeros((self.max_formulae, 
                                      self.oh_max), dtype=np.float32)

            util.fast_multi_onehot(formulae, self.oh_offsets, formulae_feat, accum=False)
            if return_n_raw:
                return formulae_feat, masses, int(self.last_formulae_n_raw)
            return formulae_feat, masses
        
        elif self.featurize_mode == 'onehot_accum' :
            formulae_feat = np.zeros((self.max_formulae, 
                                      self.oh_max), dtype=np.float32)

            util.fast_multi_onehot(formulae, self.oh_offsets, formulae_feat, accum=True)
            if return_n_raw:
                return formulae_feat, masses, int(self.last_formulae_n_raw)
            return formulae_feat, masses

        else:
            raise ValueError(f"Unknown featurize mode {self.featurize_mode}")
