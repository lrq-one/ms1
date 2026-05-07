# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Feature engineering utilities that transform molecules/spectra into model-ready tensors.

import numpy as np
import torch
import torch.utils.data
import os
from rdkit import Chem

from . import atom_features
from . import molecule_features
from .fragment_source_features import aggregate_source_features_for_formulae, FRAGMENT_LOCAL_AUX_DIM
from . import util
from .. import msutil
import importlib


# Ensure commonly used msutil submodules are loaded and attached to the msutil package
try:
    _binutils = importlib.import_module('rassp.msutil.binutils')
    msutil.binutils = _binutils
except Exception:
    pass
try:
    _vertsub = importlib.import_module('rassp.msutil.vertsubsetgen')
    msutil.vertsubsetgen = _vertsub
except Exception:
    pass


# Function overview: create_mol_featurizer handles a specific workflow step in this module.
def create_mol_featurizer(spect_bin_config, featurizer_config):

    return MolFeaturizer(bin_config = spect_bin_config,
                         **featurizer_config)

# Function overview: create_pred_featurizer handles a specific workflow step in this module.
def create_pred_featurizer(spect_bin_config, pred_featurizer_config):

    return PredFeaturizer(bin_config = spect_bin_config,
                         **pred_featurizer_config)


def _official_bin_indices(mz, bin_width):
    bw = float(max(1e-6, float(bin_width)))
    mode = str(os.environ.get("OFFICIAL_BIN_MODE", "floor")).strip().lower()

    mz_arr = np.asarray(mz, dtype=np.float64)
    if mode in ("round", "nearest", "nominal"):
        return np.rint(mz_arr / bw).astype(np.int64)
    return np.floor(mz_arr / bw + 1e-8).astype(np.int64)

def _build_official_peak_tensors(
    formulae_peaks,
    formulae_mask,
    official_bin_width,
    official_max_mz,
    fallback_intensity,
    mode,
):
    """Build per-candidate official-bin index/intensity tensors.

    mode=raw: keep historical behavior (official idx + fallback intensity tensor).
    mode=peakdist: aggregate repeated official bins and L1-normalize per candidate.
    """
    peaks = np.asarray(formulae_peaks, dtype=np.float32)
    if peaks.ndim != 3 or peaks.shape[-1] < 2:
        m = int(peaks.shape[0]) if peaks.ndim >= 1 else 0
        k = int(peaks.shape[1]) if peaks.ndim >= 2 else 0
        return (
            np.full((m, k), -1, dtype=np.int64),
            np.zeros((m, k), dtype=np.float32),
        )

    cand_n = int(peaks.shape[0])
    peak_n = int(peaks.shape[1])
    bw = float(max(1e-6, official_bin_width))
    max_mz = float(max(bw, official_max_mz))

    if formulae_mask is None:
        cand_mask = np.ones((cand_n,), dtype=bool)
    else:
        cand_mask = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)
        if cand_mask.shape[0] < cand_n:
            pad = np.zeros((cand_n - cand_mask.shape[0],), dtype=np.float32)
            cand_mask = np.concatenate([cand_mask, pad], axis=0)
        cand_mask = cand_mask[:cand_n] > 0.5

    mz = peaks[..., 0]
    inten = peaks[..., 1]
    off_idx = _official_bin_indices(mz, bw)
    valid = (
        np.isfinite(mz)
        & np.isfinite(inten)
        & (inten > 0)
        & (mz >= 0)
        & (mz < max_mz)
        & cand_mask[:, None]
    )

    off_idx_out = np.full((cand_n, peak_n), -1, dtype=np.int64)
    off_int_out = np.zeros((cand_n, peak_n), dtype=np.float32)

    mode_norm = str(mode or "raw").strip().lower()
    if mode_norm not in {"raw", "peakdist"}:
        mode_norm = "raw"

    if mode_norm == "raw":
        fb = np.asarray(fallback_intensity, dtype=np.float32)
        if fb.shape != off_int_out.shape:
            fb2 = np.zeros_like(off_int_out)
            use_m = min(int(fb2.shape[0]), int(fb.shape[0]))
            use_k = min(int(fb2.shape[1]), int(fb.shape[1])) if fb.ndim >= 2 else 0
            if use_m > 0 and use_k > 0:
                fb2[:use_m, :use_k] = fb[:use_m, :use_k]
            fb = fb2
        off_idx_out[valid] = off_idx[valid]
        off_int_out[valid] = np.maximum(inten[valid], 0.0)

        row_sum = off_int_out.sum(axis=1, keepdims=True)
        valid_row = row_sum > 1e-12
        off_int_out = np.where(valid_row, off_int_out / np.where(valid_row, row_sum, 1.0), 0.0).astype(np.float32)
        return off_idx_out, off_int_out

    for ci in range(cand_n):
        if not bool(cand_mask[ci]):
            continue
        row_valid = valid[ci]
        if not np.any(row_valid):
            continue
        idx_row = off_idx[ci, row_valid].astype(np.int64, copy=False)
        int_row = inten[ci, row_valid].astype(np.float64, copy=False)
        uniq, inv = np.unique(idx_row, return_inverse=True)
        agg = np.zeros((uniq.shape[0],), dtype=np.float64)
        np.add.at(agg, inv, np.maximum(int_row, 0.0))
        total = float(np.sum(agg))
        if total > 1e-12:
            agg = agg / total
        keep_n = min(int(peak_n), int(uniq.shape[0]))
        if keep_n <= 0:
            continue
        off_idx_out[ci, :keep_n] = uniq[:keep_n]
        off_int_out[ci, :keep_n] = agg[:keep_n].astype(np.float32)

    return off_idx_out, off_int_out


def _build_true_official_intensity_map_from_sparse(
    sparse_spect,
    bin_width,
    max_mz,
    exclude_precursor=False,
    precursor_mz=None,
    precursor_tol_da=0.05,
    precursor_isotope_n=2,
):
    """
    Build normalized true official-bin intensity map from raw sparse spectrum.
    Returns:
      target_intensity_map: dict[int bin -> float normalized intensity]
      target_top20_bins: set[int]
    """
    out = {}
    top20_bins = set()

    if sparse_spect is None:
        return out, top20_bins

    try:
        arr = np.asarray(sparse_spect, dtype=np.float32)
        if arr.size == 0:
            return out, top20_bins
        arr = arr.reshape(-1, 2)
    except Exception:
        return out, top20_bins

    bw = float(max(1e-6, float(bin_width)))
    max_mz = float(max(bw, float(max_mz)))

    mz = arr[:, 0].astype(np.float64)
    inten = arr[:, 1].astype(np.float64)

    valid = (
        np.isfinite(mz)
        & np.isfinite(inten)
        & (inten > 0)
        & (mz >= 0)
        & (mz < max_mz)
    )

    if bool(exclude_precursor) and precursor_mz is not None:
        try:
            pmz = float(precursor_mz)
            if np.isfinite(pmz):
                remove = np.zeros_like(valid, dtype=bool)
                iso_step = 1.0033548378
                for k in range(0, int(precursor_isotope_n) + 1):
                    remove |= np.abs(mz - (pmz + k * iso_step)) <= float(precursor_tol_da)
                valid &= ~remove
        except Exception:
            pass

    if not np.any(valid):
        return out, top20_bins

    mz = mz[valid]
    inten = inten[valid]

    bins = _official_bin_indices(mz, bw)
    for b, w in zip(bins.tolist(), inten.tolist()):
        bb = int(b)
        if bb < 0:
            continue
        out[bb] = float(out.get(bb, 0.0) + max(0.0, float(w)))

    total = float(sum(out.values()))
    if total <= 1e-12:
        return {}, set()

    out = {int(k): float(v / total) for k, v in out.items() if v > 0}

    # top20 by normalized true intensity
    top_items = sorted(out.items(), key=lambda kv: kv[1], reverse=True)[:20]
    top20_bins = set(int(k) for k, _ in top_items)

    return out, top20_bins

# Class overview: MolFeaturizer encapsulates a reusable component in this module.
class MolFeaturizer(torch.utils.data.Dataset):
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(
            self, 
            MAX_N,
            bin_config, 
            feat_vert_args = {}, 
            adj_args = {},
            mol_args = {},
            explicit_formulae_config = {},
            formula_frag_count_config = {},
            max_conf_sample = 1, 
            spect_assign = True,
            extra_features = None,
            sparse_spect = False, 
            sparse_peak_num = 128,
            round_mass_to_int = True,
            removeHs = False,
            element_oh = [],
            subset_gen_config = {},
            vert_subset_samples_n = 0, 
            MAX_EDGE_N=64,
            spect_input_sparse = False,         
        ):
        self.MAX_N = MAX_N

        self.feat_vert_args = feat_vert_args
        self.adj_args = adj_args
        self.mol_args = mol_args
        self.global_feature_names = []
        if isinstance(self.mol_args, dict):
            gnames = self.mol_args.get('global_features', [])
            if isinstance(gnames, str):
                gnames = [x.strip() for x in gnames.split(',') if x.strip()]
            if gnames is None:
                gnames = []
            self.global_feature_names = list(gnames)

        self.spect_bin_config = bin_config
        
        self.formula_frag_count_config = formula_frag_count_config

        self.spect_assign = spect_assign
        self.extra_features = extra_features
        self.max_conf_sample = max_conf_sample

        self.sparse_spect = sparse_spect
        self.sparse_peak_num = sparse_peak_num

        self.round_mass_to_int= round_mass_to_int

        if explicit_formulae_config != {}:
            self.pff = molecule_features.PossibleFormulaFeaturizer(**explicit_formulae_config)
        else:
            self.pff = None

        self.removeHs = removeHs

        self.element_oh = element_oh

        self.vert_subset_samples_n = vert_subset_samples_n
        self.MAX_EDGE_N = MAX_EDGE_N

        self.spect_input_sparse = spect_input_sparse

        self.official_bin_width = max(1e-6, float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01')))
        self.official_max_mz = max(self.official_bin_width, float(os.environ.get('OFFICIAL_MAX_MZ', '1005.0')))

        self._debug_feat = os.environ.get("DEBUG_FEAT", "0") == "1"
        try:
            self._debug_feat_max = max(0, int(os.environ.get("DEBUG_FEAT_MAX", "5")))
        except Exception:
            self._debug_feat_max = 5
        self._debug_feat_cnt = 0

        self.mp2b = msutil.binutils.create_peaks_to_bins(self.spect_bin_config)

        self.subset_gen_config = subset_gen_config
        
    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, mol, sparse_spect=None, precursor_mz=None, precursor_formula=None, adduct=None):

        # Output contract for downstream trainer/model:
        # - graph inputs: vect_feat / adj / input_mask (+ optional mol_feat)
        # - candidate formula inputs: formulae_features / formulae_peaks_* / formulae_mask
        # - selector helper: formulae_aux_feat (compact per-candidate summary)
        # The training script expects these names in prepare_batch_cpu().

        if self.removeHs:
            mol = Chem.RemoveHs(mol)
        
        f_vect = atom_features.feat_tensor_atom(mol, conf_idx=0, 
                                                **self.feat_vert_args)
        DATA_N = f_vect.shape[0]
        
        vect_feat = np.zeros((self.MAX_N, f_vect.shape[1]), dtype=np.float32)
        vect_feat[:DATA_N] = f_vect
        
        adj_nopad = molecule_features.feat_mol_adj(mol, **self.adj_args)
        adj = torch.zeros((adj_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj[:, :adj_nopad.shape[1], :adj_nopad.shape[2]] = adj_nopad

        adj_oh_nopad = molecule_features.feat_mol_adj(mol, split_weights=[1.0, 1.5, 2.0, 3.0], 
                                                      edge_weighted=False, norm_adj=False, add_identity=False)

        adj_oh = torch.zeros((adj_oh_nopad.shape[0], self.MAX_N, self.MAX_N))
        adj_oh[:, :adj_oh_nopad.shape[1], :adj_oh_nopad.shape[2]] = adj_oh_nopad
                
        atomicnos = util.get_nos(mol)

        # input mask
        input_mask = torch.zeros(self.MAX_N) 
        input_mask[:DATA_N] = 1.0
        
        v = {
            'vect_feat' : vect_feat, 
            'adj' : adj,
            'adj_oh' : adj_oh,
            'input_mask' : input_mask, 
        }

        if len(self.global_feature_names) > 0:
            v['mol_feat'] = molecule_features.mol_global_features(mol, self.global_feature_names).astype(np.float32)

        if self.sparse_spect:
            if sparse_spect is None:
                bins = np.zeros((self.sparse_peak_num,), dtype=np.int64)
                probs = np.zeros((self.sparse_peak_num,), dtype=np.float32)
            else:
                spect_arr = np.asarray(sparse_spect, dtype=np.float32)
                if spect_arr.size == 0:
                    spect_out = np.zeros((self.spect_bin_config.get_num_bins(),), dtype=np.float32)
                else:
                    spect_arr = spect_arr.reshape(-1, 2)
                    _, _, spect_out = self.spect_bin_config.histogram(spect_arr[:, 0], spect_arr[:, 1])
                    spect_out = spect_out.astype(np.float32)
                sort_idx = np.argsort(spect_out)[::-1]
                bins = np.arange(len(spect_out), dtype=np.int64)[sort_idx][:self.sparse_peak_num]
                probs = spect_out[sort_idx][:self.sparse_peak_num].astype(np.float32)

            v['spect_peak_mass'] = bins
            v['spect_peak_prob'] = probs

        if self.pff is not None:
            # Keep cache generation label-free: do NOT pass target spectrum into PFF here.
            formulae_feats, formulae_peaks, formulae_n_raw = self.pff(
                mol,
                return_n_raw=True,
                sparse_spect=None,
                spect_bin_config=None,
                adduct=adduct,
                precursor_mz=precursor_mz,
                precursor_formula=precursor_formula,
            )
            assert np.max(formulae_feats) <= 255

            # Bin the mass peaks into the app
            formulae_peaks_mass_idx, formulae_peaks_intensity =\
                self.mp2b(formulae_peaks) 

            formulae_mask = np.asarray(getattr(self.pff, 'last_formulae_mask', np.ones((formulae_feats.shape[0],), dtype=np.float32)), dtype=np.float32)
            formulae_n_kept = int(np.sum(formulae_mask > 0.5))

            # Explicitly invalidate padded candidates in projection tensors.
            formulae_peaks_mass_idx = formulae_peaks_mass_idx.astype(np.int64, copy=False)
            formulae_peaks_intensity = formulae_peaks_intensity.astype(np.float32, copy=False)
            if formulae_n_kept < formulae_peaks_mass_idx.shape[0]:
                formulae_peaks_mass_idx[formulae_n_kept:, ...] = -1
                formulae_peaks_intensity[formulae_n_kept:, ...] = 0.0

            # Keep exact-peak tensor consistent for any downstream diagnostics.
            formulae_peaks = formulae_peaks.astype(np.float32, copy=False)
            if formulae_n_kept < formulae_peaks.shape[0]:
                formulae_peaks[formulae_n_kept:, :, 1] = 0.0

            # Build official-bin tensors used by official projection/targets.
            official_mode = os.environ.get('FORMULAE_OFFICIAL_INTENSITY_MODE', 'peakdist').strip().lower()
            formulae_peaks_official_idx, formulae_peaks_official_intensity = _build_official_peak_tensors(
                formulae_peaks=formulae_peaks,
                formulae_mask=formulae_mask,
                official_bin_width=self.official_bin_width,
                official_max_mz=self.official_max_mz,
                fallback_intensity=formulae_peaks_intensity,
                mode=official_mode,
            )

            formulae_aux_feat = molecule_features.build_formulae_aux_feat(
                formulae_feats,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity,
                formulae_mask,
                formula_atomicnos=getattr(self.pff, 'formula_possible_atomicno', None),
                precursor_mz=precursor_mz,
                precursor_formula=precursor_formula,
                formulae_peaks_official_idx=formulae_peaks_official_idx,
                formulae_peaks_official_intensity=formulae_peaks_official_intensity,
            )

            formulae_frag_aux_feat = None
            if os.environ.get('USE_FRAGMENT_LOCAL_AUX', '0') == '1':
                try:
                    formulae_frag_aux_feat = aggregate_source_features_for_formulae(
                        mol=mol,
                        formulae_features=formulae_feats,
                        formula_atomicnos=getattr(self.pff, 'formula_possible_atomicno', None),
                        h_shift_min=int(os.environ.get('FRAG_AUX_H_SHIFT_MIN', '-4')),
                        h_shift_max=int(os.environ.get('FRAG_AUX_H_SHIFT_MAX', '4')),
                    )
                except Exception as e:
                    if os.environ.get('DEBUG_FRAGMENT_LOCAL_AUX', '0') == '1':
                        print('[FRAG_AUX_ERROR]', repr(e))
                    formulae_frag_aux_feat = None

            if formulae_frag_aux_feat is None:
                formulae_frag_aux_feat = np.zeros((formulae_feats.shape[0], FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)
            elif os.environ.get('DEBUG_FRAGMENT_LOCAL_AUX', '0') == '1':
                print('[FRAG_AUX_DIM]', formulae_frag_aux_feat.shape, flush=True)
            
            formulae_source_flag_raw = np.asarray(
                getattr(self.pff, 'last_formulae_source_flag', np.zeros((formulae_feats.shape[0],), dtype=np.int8)),
                dtype=np.int8,
            ).reshape(-1, 1)
            formulae_source_flag = formulae_source_flag_raw.astype(np.int8, copy=True)
            formulae_break_depth = np.asarray(
                getattr(self.pff, 'last_formulae_break_depth', np.zeros((formulae_feats.shape[0],), dtype=np.int8)),
                dtype=np.float32,
            ).reshape(-1, 1)

            formulae_ring_cut_flag = np.asarray(
                getattr(self.pff, 'last_formulae_ring_cut_flag', np.zeros((formulae_feats.shape[0],), dtype=np.int8)),
                dtype=np.float32,
            ).reshape(-1, 1)
            formulae_prior_score = np.asarray(
                getattr(self.pff, 'last_formulae_prior_score', np.zeros((formulae_feats.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 1)

            formulae_active_mask = np.asarray(
                getattr(self.pff, 'last_formulae_active_mask', np.zeros((formulae_feats.shape[0],), dtype=np.float32)),
                dtype=np.float32,
            ).reshape(-1, 1)

            # prior score 做行内标准化，避免绝对尺度影响模型。
            try:
                valid_prior = (formulae_mask.reshape(-1, 1) > 0.5)
                ps = formulae_prior_score.copy()
                if np.any(valid_prior):
                    m = float(np.mean(ps[valid_prior]))
                    s = float(np.std(ps[valid_prior]))
                    if s > 1e-6:
                        ps = (ps - m) / s
                    else:
                        ps = ps * 0.0
                formulae_prior_score_z = ps.astype(np.float32)
            except Exception:
                formulae_prior_score_z = np.zeros_like(formulae_prior_score, dtype=np.float32)
            is_any_source = (formulae_source_flag_raw > 0).astype(np.float32)
            is_struct_source = ((formulae_source_flag_raw == 1) | (formulae_source_flag_raw == 3)).astype(np.float32)
            is_common_loss_source = ((formulae_source_flag_raw == 2) | (formulae_source_flag_raw == 3)).astype(np.float32)
            formulae_break_depth_norm = formulae_break_depth / 2.0

            struct_aux = np.concatenate(
                [
                    is_any_source,
                    is_struct_source,
                    is_common_loss_source,
                    formulae_break_depth_norm,
                    formulae_ring_cut_flag,
                    formulae_prior_score_z,
                    formulae_active_mask,
                ],
                axis=-1,
            ).astype(np.float32)

            if struct_aux.shape[0] == formulae_aux_feat.shape[0]:
                formulae_aux_feat = np.concatenate([formulae_aux_feat.astype(np.float32), struct_aux], axis=-1)
            

            # -------------------------------------------------------------------------
            # Optional V2: source-instance candidate expansion.
            #
            # Candidate axis becomes:
            #   [source-instance slots, original formula fallback slots]
            #
            # Source slots are fixed-length for stable batching:
            #   SOURCE_INSTANCE_MAX_TOTAL valid-or-padding source rows
            #   + original formula rows
            # -------------------------------------------------------------------------
            use_source_instance_candidates = os.environ.get("USE_SOURCE_INSTANCE_CANDIDATES", "0").strip() == "1"
            def _v2_set_first_rows(arr, n_rows, values):
                """
                Safely set the first n_rows of arr with values.
                Supports 1D or 2D column arrays.
                """
                if arr is None or n_rows <= 0:
                    return arr
                out = np.asarray(arr).copy()
                vals = np.asarray(values)

                if out.ndim == 1:
                    out[:n_rows] = vals.reshape(-1)[:n_rows]
                elif out.ndim == 2:
                    if out.shape[1] == 1:
                        out[:n_rows, 0] = vals.reshape(-1)[:n_rows]
                    else:
                        # Set first column only; keep remaining duplicated fallback values.
                        out[:n_rows, 0] = vals.reshape(-1)[:n_rows]
                else:
                    # For higher dims, do not modify.
                    pass

                return out

            
            def _v2_source_orig_pad(arr, src_idx_actual, src_idx_pad):
                """
                Build candidate axis as:
                [actual source rows copied from arr]
                + [original formula rows]
                + [padding source rows copied from arr[0]]

                This keeps valid candidates as a contiguous prefix:
                valid sources + valid original formulas, then all invalid padding.
                """
                if arr is None:
                    return None

                arr = np.asarray(arr)
                if arr.shape[0] <= 0:
                    return arr

                parts = []

                if src_idx_actual.shape[0] > 0:
                    parts.append(arr[src_idx_actual])

                parts.append(arr)

                if src_idx_pad.shape[0] > 0:
                    parts.append(arr[src_idx_pad])

                return np.concatenate(parts, axis=0)

            if use_source_instance_candidates:
                try:
                    from .fragment_source_features import build_source_instance_records_for_formulae

                    max_per_formula = int(os.environ.get("SOURCE_INSTANCE_MAX_PER_FORMULA", "3"))
                    max_total_source_instances = int(os.environ.get("SOURCE_INSTANCE_MAX_TOTAL", "2048"))
                    max_depth = int(os.environ.get("FRAG_AUX_MAX_DEPTH", "2"))
                    max_depth2_bond_pairs = int(os.environ.get("FRAG_AUX_MAX_DEPTH2_BOND_PAIRS", "1200"))
                    max_sources = int(os.environ.get("FRAG_AUX_MAX_SOURCES", "100000"))

                    orig_formulae_feats = np.asarray(formulae_feats)
                    orig_formulae_mask = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)
                    orig_M = int(orig_formulae_feats.shape[0])

                    source_records = build_source_instance_records_for_formulae(
                        mol=mol,
                        formulae_features=orig_formulae_feats,
                        formula_atomicnos=getattr(self.pff, "formula_possible_atomicno", None),
                        h_shift_min=int(os.environ.get("FRAG_AUX_H_SHIFT_MIN", "-4")),
                        h_shift_max=int(os.environ.get("FRAG_AUX_H_SHIFT_MAX", "4")),
                        max_depth=max_depth,
                        max_per_formula=max_per_formula,
                        max_total_source_instances=max_total_source_instances,
                        max_depth2_bond_pairs=max_depth2_bond_pairs,
                        max_sources=max_sources,
                    )

                    n_actual_src = min(len(source_records), max_total_source_instances)
                    n_src_slots = int(max_total_source_instances)
                    n_pad_src = max(0, n_src_slots - n_actual_src)

                    # Actual source rows first; padding source rows go to the very end.
                    # Layout:
                    #   [actual source rows] + [original formula rows] + [source padding rows]
                    src_idx_actual = np.zeros((n_actual_src,), dtype=np.int64)
                    src_frag_aux_actual = np.zeros((n_actual_src, formulae_frag_aux_feat.shape[-1]), dtype=np.float32)
                    src_depth_actual = np.zeros((n_actual_src,), dtype=np.float32)
                    src_h_shift_actual = np.zeros((n_actual_src,), dtype=np.float32)
                    src_ring_cut_actual = np.zeros((n_actual_src,), dtype=np.float32)
                    src_is_source_actual = np.ones((n_actual_src,), dtype=np.float32)

                    if n_actual_src > 0:
                        src_idx_actual = np.asarray(
                            [int(r["orig_idx"]) for r in source_records[:n_actual_src]],
                            dtype=np.int64,
                        )
                        src_idx_actual = np.clip(src_idx_actual, 0, max(0, orig_M - 1))

                        src_frag_aux_actual = np.stack(
                            [np.asarray(r["frag_aux"], dtype=np.float32) for r in source_records[:n_actual_src]],
                            axis=0,
                        )

                        src_depth_actual = np.asarray(
                            [float(r.get("source_depth", 0.0)) for r in source_records[:n_actual_src]],
                            dtype=np.float32,
                        )

                        src_h_shift_actual = np.asarray(
                            [float(r.get("source_h_shift", 0.0)) for r in source_records[:n_actual_src]],
                            dtype=np.float32,
                        )

                        src_ring_cut_actual = np.asarray(
                            [float(r.get("source_ring_cut", 0.0)) for r in source_records[:n_actual_src]],
                            dtype=np.float32,
                        )

                    # Padding rows are invalid and placed after original formulas.
                    # They duplicate formula 0 only to preserve tensor shape.
                    src_idx_pad = np.zeros((n_pad_src,), dtype=np.int64)
                    src_frag_aux_pad = np.zeros((n_pad_src, formulae_frag_aux_feat.shape[-1]), dtype=np.float32)
                    src_depth_pad = np.zeros((n_pad_src,), dtype=np.float32)
                    src_h_shift_pad = np.zeros((n_pad_src,), dtype=np.float32)
                    src_ring_cut_pad = np.zeros((n_pad_src,), dtype=np.float32)
                    src_is_source_pad = np.zeros((n_pad_src,), dtype=np.float32)

                    # Build all new arrays into temp vars first.
                    new_formulae_feats = _v2_source_orig_pad(orig_formulae_feats, src_idx_actual, src_idx_pad)

                    new_formulae_frag_aux_feat = np.concatenate(
                        [
                            src_frag_aux_actual.astype(np.float32),
                            np.asarray(formulae_frag_aux_feat, dtype=np.float32),
                            src_frag_aux_pad.astype(np.float32),
                        ],
                        axis=0,
                    )

                    if formulae_aux_feat is not None:
                        new_formulae_aux_feat = _v2_source_orig_pad(formulae_aux_feat, src_idx_actual, src_idx_pad)
                        if new_formulae_aux_feat.ndim == 2 and new_formulae_aux_feat.shape[1] >= 7 and n_actual_src > 0:
                            # last 7 dims: any_source, struct_source, common_loss, break_depth, ring_cut, prior_z, active
                            new_formulae_aux_feat[:n_actual_src, -7] = 1.0
                            new_formulae_aux_feat[:n_actual_src, -6] = 1.0
                            new_formulae_aux_feat[:n_actual_src, -4] = np.maximum(
                                new_formulae_aux_feat[:n_actual_src, -4],
                                np.clip(src_depth_actual / 4.0, 0.0, 1.0)
                            )
                            new_formulae_aux_feat[:n_actual_src, -3] = np.maximum(
                                new_formulae_aux_feat[:n_actual_src, -3],
                                src_ring_cut_actual
                            )
                    else:
                        new_formulae_aux_feat = None

                    # Peak/template arrays: duplicate source rows from original formula rows.
                    if "formulae_peaks" in locals() and formulae_peaks is not None:
                        new_formulae_peaks = _v2_source_orig_pad(formulae_peaks, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks = None

                    if "formulae_peaks_mass_idx" in locals() and formulae_peaks_mass_idx is not None:
                        new_formulae_peaks_mass_idx = _v2_source_orig_pad(formulae_peaks_mass_idx, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_mass_idx = None

                    if "formulae_peaks_intensity" in locals() and formulae_peaks_intensity is not None:
                        new_formulae_peaks_intensity = _v2_source_orig_pad(formulae_peaks_intensity, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_intensity = None

                    if "formulae_peaks_official_idx" in locals() and formulae_peaks_official_idx is not None:
                        new_formulae_peaks_official_idx = _v2_source_orig_pad(formulae_peaks_official_idx, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_official_idx = None

                    if "formulae_peaks_official_intensity" in locals() and formulae_peaks_official_intensity is not None:
                        new_formulae_peaks_official_intensity = _v2_source_orig_pad(formulae_peaks_official_intensity, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_official_intensity = None

                    if "formulae_peaks_official_idx_agg" in locals() and formulae_peaks_official_idx_agg is not None:
                        new_formulae_peaks_official_idx_agg = _v2_source_orig_pad(formulae_peaks_official_idx_agg, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_official_idx_agg = None

                    if "formulae_peaks_official_intensity_agg" in locals() and formulae_peaks_official_intensity_agg is not None:
                        new_formulae_peaks_official_intensity_agg = _v2_source_orig_pad(formulae_peaks_official_intensity_agg, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_peaks_official_intensity_agg = None

                    # Optional candidate arrays.
                    if "formulae_prior_score" in locals() and formulae_prior_score is not None:
                        new_formulae_prior_score = _v2_source_orig_pad(formulae_prior_score, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_prior_score = None

                    if "formulae_active_mask" in locals() and formulae_active_mask is not None:
                        new_formulae_active_mask = _v2_source_orig_pad(formulae_active_mask, src_idx_actual, src_idx_pad)
                    else:
                        new_formulae_active_mask = None

                    if "formulae_source_flag" in locals() and formulae_source_flag is not None:
                        new_formulae_source_flag = _v2_source_orig_pad(formulae_source_flag, src_idx_actual, src_idx_pad)
                        new_formulae_source_flag = _v2_set_first_rows(
                            new_formulae_source_flag,
                            n_actual_src,
                            np.ones((n_actual_src,), dtype=np.float32),
                        )
                    else:
                        new_formulae_source_flag = None

                    if "formulae_break_depth" in locals() and formulae_break_depth is not None:
                        new_formulae_break_depth = _v2_source_orig_pad(
                            formulae_break_depth,
                            src_idx_actual,
                            src_idx_pad,
                        )
                        new_formulae_break_depth = _v2_set_first_rows(
                            new_formulae_break_depth,
                            n_actual_src,
                            src_depth_actual,
                        )
                    else:
                        new_formulae_break_depth = None

                    if "formulae_ring_cut_flag" in locals() and formulae_ring_cut_flag is not None:
                        new_formulae_ring_cut_flag = _v2_source_orig_pad(
                            formulae_ring_cut_flag,
                            src_idx_actual,
                            src_idx_pad,
                        )
                        new_formulae_ring_cut_flag = _v2_set_first_rows(
                            new_formulae_ring_cut_flag,
                            n_actual_src,
                            src_ring_cut_actual,
                        )
                    else:
                        new_formulae_ring_cut_flag = None

                    # Mask and diagnostic fields.
                    src_mask_actual = np.ones((n_actual_src,), dtype=np.float32)
                    src_mask_pad = np.zeros((n_pad_src,), dtype=np.float32)

                    new_formulae_mask = np.concatenate(
                        [
                            src_mask_actual,
                            orig_formulae_mask.astype(np.float32),
                            src_mask_pad,
                        ],
                        axis=0,
                    )

                    new_formulae_instance_is_source = np.concatenate(
                        [
                            src_is_source_actual.astype(np.float32),
                            np.zeros((orig_M,), dtype=np.float32),
                            src_is_source_pad.astype(np.float32),
                        ],
                        axis=0,
                    )

                    new_formulae_instance_group_id = np.concatenate(
                        [
                            src_idx_actual.astype(np.int32),
                            np.arange(orig_M, dtype=np.int32),
                            src_idx_pad.astype(np.int32),
                        ],
                        axis=0,
                    )

                    new_formulae_instance_depth = np.concatenate(
                        [
                            src_depth_actual.astype(np.float32),
                            np.zeros((orig_M,), dtype=np.float32),
                            src_depth_pad.astype(np.float32),
                        ],
                        axis=0,
                    )

                    new_formulae_instance_h_shift = np.concatenate(
                        [
                            src_h_shift_actual.astype(np.float32),
                            np.zeros((orig_M,), dtype=np.float32),
                            src_h_shift_pad.astype(np.float32),
                        ],
                        axis=0,
                    )

                    # Sanity checks before committing.
                    new_M = int(new_formulae_feats.shape[0])
                    for _name, _arr in [
                        ("formulae_mask", new_formulae_mask),
                        ("formulae_frag_aux_feat", new_formulae_frag_aux_feat),
                        ("formulae_instance_is_source", new_formulae_instance_is_source),
                        ("formulae_instance_group_id", new_formulae_instance_group_id),
                        ("formulae_instance_depth", new_formulae_instance_depth),
                        ("formulae_instance_h_shift", new_formulae_instance_h_shift),
                    ]:
                        if int(np.asarray(_arr).shape[0]) != new_M:
                            raise ValueError(f"V2 source-instance shape mismatch: {_name} shape={np.asarray(_arr).shape} expected first dim={new_M}")

                    # Commit only after everything succeeded.
                    formulae_feats = new_formulae_feats
                    formulae_mask = new_formulae_mask
                    formulae_frag_aux_feat = new_formulae_frag_aux_feat
                    if new_formulae_aux_feat is not None:
                        formulae_aux_feat = new_formulae_aux_feat

                    if new_formulae_peaks is not None:
                        formulae_peaks = new_formulae_peaks
                    if new_formulae_peaks_mass_idx is not None:
                        formulae_peaks_mass_idx = new_formulae_peaks_mass_idx
                    if new_formulae_peaks_intensity is not None:
                        formulae_peaks_intensity = new_formulae_peaks_intensity
                    if new_formulae_peaks_official_idx is not None:
                        formulae_peaks_official_idx = new_formulae_peaks_official_idx
                    if new_formulae_peaks_official_intensity is not None:
                        formulae_peaks_official_intensity = new_formulae_peaks_official_intensity
                    if new_formulae_peaks_official_idx_agg is not None:
                        formulae_peaks_official_idx_agg = new_formulae_peaks_official_idx_agg
                    if new_formulae_peaks_official_intensity_agg is not None:
                        formulae_peaks_official_intensity_agg = new_formulae_peaks_official_intensity_agg

                    if new_formulae_prior_score is not None:
                        formulae_prior_score = new_formulae_prior_score
                    if new_formulae_active_mask is not None:
                        formulae_active_mask = new_formulae_active_mask
                    if new_formulae_source_flag is not None:
                        formulae_source_flag = new_formulae_source_flag
                    if new_formulae_break_depth is not None:
                        formulae_break_depth = new_formulae_break_depth
                    if new_formulae_ring_cut_flag is not None:
                        formulae_ring_cut_flag = new_formulae_ring_cut_flag

                    formulae_instance_is_source = new_formulae_instance_is_source
                    formulae_instance_group_id = new_formulae_instance_group_id
                    formulae_instance_depth = new_formulae_instance_depth
                    formulae_instance_h_shift = new_formulae_instance_h_shift

                    if os.environ.get("DEBUG_SOURCE_INSTANCE_CANDIDATES", "0").strip() == "1":
                        print(
                            "[SOURCE_INSTANCE]",
                            "n_actual_source=", int(n_actual_src),
                            "n_source_slots=", int(n_src_slots),
                            "orig_M=", int(orig_M),
                            "new_M=", int(new_M),
                            "source_ratio_valid=", float(src_mask_actual.sum() / max(1.0, new_formulae_mask.sum())),
                        )

                except Exception as e:
                    # Critical: do not leave partially expanded arrays behind.
                    if os.environ.get("DEBUG_SOURCE_INSTANCE_CANDIDATES", "0").strip() == "1":
                        print("[SOURCE_INSTANCE_ERROR]", repr(e), flush=True)

                    if use_source_instance_candidates:
                        raise RuntimeError(f"V2 source-instance expansion failed: {repr(e)}")

                    orig_M = int(np.asarray(formulae_feats).shape[0])
                    formulae_instance_is_source = np.zeros((orig_M,), dtype=np.float32)
                    formulae_instance_group_id = np.arange(orig_M, dtype=np.int32)
                    formulae_instance_depth = np.zeros((orig_M,), dtype=np.float32)
                    formulae_instance_h_shift = np.zeros((orig_M,), dtype=np.float32)
            else:
                orig_M = int(np.asarray(formulae_feats).shape[0])
                formulae_instance_is_source = np.zeros((orig_M,), dtype=np.float32)
                formulae_instance_group_id = np.arange(orig_M, dtype=np.int32)
                formulae_instance_depth = np.zeros((orig_M,), dtype=np.float32)
                formulae_instance_h_shift = np.zeros((orig_M,), dtype=np.float32)

            v['formulae_features'] = formulae_feats.astype(np.uint8)
            v['formulae_peaks'] = formulae_peaks
            v['formulae_peaks_mass_idx'] = formulae_peaks_mass_idx.astype(np.int64)
            v['formulae_peaks_intensity'] = formulae_peaks_intensity
            v['formulae_peaks_official_idx'] = formulae_peaks_official_idx.astype(np.int64)
            v['formulae_peaks_official_intensity'] = formulae_peaks_official_intensity.astype(np.float32)
            v['formulae_aux_feat'] = formulae_aux_feat.astype(np.float32)
            v['formulae_frag_aux_feat'] = formulae_frag_aux_feat.astype(np.float32)
            v['formulae_mask'] = np.asarray(formulae_mask, dtype=np.float32).reshape(-1)

            # V2 后 formulae_n_kept 必须来自当前 mask，而不是 PFF 的旧 last_formulae_n_kept。
            v['formulae_n_kept'] = np.int64(int(np.asarray(v['formulae_mask']).reshape(-1).sum()))

            v['formulae_prior_score'] = formulae_prior_score.reshape(-1).astype(np.float32)
            v['formulae_active_mask'] = formulae_active_mask.reshape(-1).astype(np.float32)
            v['formulae_n_raw'] = np.int64(formulae_n_raw)
            v['formulae_instance_is_source'] = formulae_instance_is_source.astype(np.float32)
            v['formulae_instance_group_id'] = formulae_instance_group_id.astype(np.int32)
            v['formulae_instance_depth'] = formulae_instance_depth.astype(np.float32)
            v['formulae_instance_h_shift'] = formulae_instance_h_shift.astype(np.float32)
            v['formulae_source_flag'] = np.asarray(formulae_source_flag).reshape(-1).astype(np.int8)
            v['formulae_break_depth'] = np.asarray(formulae_break_depth).reshape(-1).astype(np.float32)
            v['formulae_ring_cut_flag'] = np.asarray(formulae_ring_cut_flag).reshape(-1).astype(np.float32)
            if hasattr(self.pff, 'last_formulae_truncated'):
                v['formulae_truncated'] = np.int64(getattr(self.pff, 'last_formulae_truncated'))

            # -------------------------------------------------------------------------
            # Strong candidate-axis sanity check.
            # Do not allow malformed V2 cache rows to be written.
            # -------------------------------------------------------------------------
            cand_M = int(np.asarray(v['formulae_features']).shape[0])

            for _k in [
                'formulae_mask',
                'formulae_aux_feat',
                'formulae_frag_aux_feat',
                'formulae_peaks',
                'formulae_peaks_mass_idx',
                'formulae_peaks_intensity',
                'formulae_peaks_official_idx',
                'formulae_peaks_official_intensity',
                'formulae_prior_score',
                'formulae_active_mask',
                'formulae_source_flag',
                'formulae_break_depth',
                'formulae_ring_cut_flag',
                'formulae_instance_is_source',
                'formulae_instance_group_id',
                'formulae_instance_depth',
                'formulae_instance_h_shift',
            ]:
                if _k in v:
                    _arr = np.asarray(v[_k])
                    if _arr.ndim < 1 or int(_arr.shape[0]) != cand_M:
                        raise RuntimeError(
                            f'candidate axis mismatch: {_k} shape={_arr.shape} expected first dim={cand_M}'
                        )

            # Optional agg fields, only if your local code writes them.
            for _k in [
                'formulae_peaks_official_idx_agg',
                'formulae_peaks_official_intensity_agg',
            ]:
                if _k in v:
                    _arr = np.asarray(v[_k])
                    if _arr.ndim < 1 or int(_arr.shape[0]) != cand_M:
                        raise RuntimeError(
                            f'candidate axis mismatch: {_k} shape={_arr.shape} expected first dim={cand_M}'
                        )

            # In V2, valid candidates must be a contiguous prefix:
            # [actual source] + [valid fallback] + [invalid fallback/source padding]
            if use_source_instance_candidates:
                _mask = np.asarray(v['formulae_mask'], dtype=np.float32).reshape(-1)
                _valid_n = int((_mask > 0.5).sum())

                if _valid_n > 0:
                    _prefix_ok = bool(
                        np.all(_mask[:_valid_n] > 0.5)
                        and np.all(_mask[_valid_n:] <= 0.5)
                    )
                else:
                    _prefix_ok = True

                if not _prefix_ok:
                    raise RuntimeError(
                        f'V2 candidate mask is not a contiguous valid prefix: valid_n={_valid_n}, M={cand_M}'
                    )

                if int(v['formulae_n_kept']) != _valid_n:
                    raise RuntimeError(
                        f'formulae_n_kept mismatch: formulae_n_kept={int(v["formulae_n_kept"])} valid_n={_valid_n}'
                    )

            # Persist PFF diagnostics into features so structure-dedup cached rows
            # can still report correct source/FIORA stats.
            if isinstance(getattr(self.pff, "last_cache_diag", None), dict):
                v["pff_cache_diag"] = dict(getattr(self.pff, "last_cache_diag", {}))
            
            if self._debug_feat and self._debug_feat_cnt < self._debug_feat_max:
                ff = v.get('formulae_features', None)
                fp = v.get('formulae_peaks', None)
                fpm = v.get('formulae_peaks_mass_idx', None)
                fpo = v.get('formulae_peaks_official_idx', None)
                fm = v.get('formulae_mask', None)
                mol_id_dbg = 'NA'
                try:
                    if mol is not None and mol.HasProp('_Name'):
                        mol_id_dbg = mol.GetProp('_Name')
                except Exception:
                    mol_id_dbg = 'NA'
                print(
                    "[DEBUG_FEAT]",
                    "mol_id=", mol_id_dbg,
                    "formulae_features_shape=", None if ff is None else np.asarray(ff).shape,
                    "formulae_peaks_shape=", None if fp is None else np.asarray(fp).shape,
                    "formulae_peaks_mass_idx_shape=", None if fpm is None else np.asarray(fpm).shape,
                    "formulae_peaks_official_idx_shape=", None if fpo is None else np.asarray(fpo).shape,
                    "formulae_mask_shape=", None if fm is None else np.asarray(fm).shape,
                    "n_raw=", v.get('formulae_n_raw', None),
                    "n_kept=", v.get('formulae_n_kept', None),
                    "truncated=", v.get('formulae_truncated', None),
                    "mask_valid_n=", None if fm is None else int(np.asarray(fm).sum()),
                )
                self._debug_feat_cnt += 1
        

        # ---------------------------------------------------------------------
        # V3A: fragment-node candidates.
        #
        # This is only cache/oracle stage. It does not replace formulae_* fields.
        # It adds a new candidate axis:
        #   fragment_node_* [N]
        #
        # Key difference from V2:
        #   V2 copied formula peak templates.
        #   V3A stores one candidate per actual fragment source / h-shift instance.
        # ---------------------------------------------------------------------
        if os.environ.get("USE_FRAGMENT_NODE_CANDIDATES", "0").strip() == "1":
            try:
                from .fragment_source_features import build_fragment_node_candidate_tensors

                if self.pff is not None and getattr(self.pff, "formula_possible_atomicno", None) is not None:
                    frag_formula_atomicnos = [int(x) for x in list(getattr(self.pff, "formula_possible_atomicno"))]
                else:
                    raw_atomicnos = os.environ.get("FORMULA_ATOMICNOS", "1,6,7,8,9,15,16,17,35,53,11")
                    frag_formula_atomicnos = [int(x.strip()) for x in raw_atomicnos.split(",") if x.strip()]

                fragment_max_nodes = int(os.environ.get("FRAGMENT_NODE_MAX_N", "4096"))
                fragment_add_proton = os.environ.get("FRAGMENT_NODE_ADD_PROTON", "1").strip() == "1"
                fragment_filter_precursor = os.environ.get("FRAGMENT_NODE_FILTER_PRECURSOR", "1").strip() == "1"

                fragment_node = build_fragment_node_candidate_tensors(
                    mol=mol,
                    formula_atomicnos=frag_formula_atomicnos,
                    official_bin_width=self.official_bin_width,
                    official_max_mz=self.official_max_mz,
                    precursor_mz=precursor_mz,
                    h_shift_min=int(os.environ.get("FRAG_AUX_H_SHIFT_MIN", "-4")),
                    h_shift_max=int(os.environ.get("FRAG_AUX_H_SHIFT_MAX", "4")),
                    max_depth=int(os.environ.get("FRAG_AUX_MAX_DEPTH", "2")),
                    max_depth2_bond_pairs=int(os.environ.get("FRAG_AUX_MAX_DEPTH2_BOND_PAIRS", "1200")),
                    max_sources=int(os.environ.get("FRAG_AUX_MAX_SOURCES", "100000")),
                    max_nodes=fragment_max_nodes,
                    add_proton=fragment_add_proton,
                    proton_mass=float(os.environ.get("FORMULA_PROTON_MASS", "1.007276466812")),
                    filter_precursor=fragment_filter_precursor,
                    precursor_tol_da=float(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.05")),
                )

                target_map, target_top20_bins = _build_true_official_intensity_map_from_sparse(
                    sparse_spect=sparse_spect,
                    bin_width=self.official_bin_width,
                    max_mz=self.official_max_mz,
                    exclude_precursor=os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR", "1").strip() == "1",
                    precursor_mz=precursor_mz,
                    precursor_tol_da=float(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_TOL_DA", "0.05")),
                    precursor_isotope_n=int(os.environ.get("OFFICIAL_EXCLUDE_PRECURSOR_ISOTOPE_N", "2")),
                )

                fn_idx = np.asarray(fragment_node["fragment_node_official_idx"], dtype=np.int64).reshape(-1)
                fn_mask = np.asarray(fragment_node["fragment_node_mask"], dtype=np.float32).reshape(-1)

                fn_label = np.zeros_like(fn_mask, dtype=np.float32)
                fn_true_intensity = np.zeros_like(fn_mask, dtype=np.float32)
                fn_label_top20 = np.zeros_like(fn_mask, dtype=np.float32)

                for ii, bb in enumerate(fn_idx.tolist()):
                    if ii >= fn_label.shape[0]:
                        break
                    if fn_mask[ii] <= 0.5:
                        continue
                    b = int(bb)
                    if b in target_map:
                        fn_label[ii] = 1.0
                        fn_true_intensity[ii] = float(target_map.get(b, 0.0))
                    if b in target_top20_bins:
                        fn_label_top20[ii] = 1.0

                # Per-bin duplicate count among valid fragment nodes.
                # This is important for later training: many nodes can hit the same official bin.
                fn_bin_dup_count = np.ones_like(fn_mask, dtype=np.float32)
                try:
                    valid_bin_mask = (fn_mask > 0.5) & (fn_idx >= 0)
                    valid_bins_for_dup = fn_idx[valid_bin_mask]
                    if valid_bins_for_dup.size > 0:
                        uniq_b, cnt_b = np.unique(valid_bins_for_dup.astype(np.int64), return_counts=True)
                        cnt_map = {int(b): float(c) for b, c in zip(uniq_b.tolist(), cnt_b.tolist())}
                        for ii, bb in enumerate(fn_idx.tolist()):
                            if ii >= fn_bin_dup_count.shape[0]:
                                break
                            if fn_mask[ii] <= 0.5 or int(bb) < 0:
                                continue
                            fn_bin_dup_count[ii] = float(cnt_map.get(int(bb), 1.0))
                except Exception:
                    fn_bin_dup_count = np.ones_like(fn_mask, dtype=np.float32)

                fragment_node_true_intensity_share = np.where(
                    fn_label > 0.5,
                    fn_true_intensity / np.maximum(fn_bin_dup_count, 1.0),
                    0.0,
                ).astype(np.float32)

                for k, arr in fragment_node.items():
                    v[k] = arr

                v["fragment_node_label"] = fn_label.astype(np.float32)
                v["fragment_node_true_intensity"] = fn_true_intensity.astype(np.float32)
                v["fragment_node_label_top20"] = fn_label_top20.astype(np.float32)
                v["fragment_node_bin_dup_count"] = fn_bin_dup_count.astype(np.float32)
                v["fragment_node_true_intensity_share"] = fragment_node_true_intensity_share.astype(np.float32)
                v["fragment_node_n_valid"] = np.asarray([int((fn_mask > 0.5).sum())], dtype=np.int64)

                if os.environ.get("DEBUG_FRAGMENT_NODE_CANDIDATES", "0").strip() == "1":
                    valid_n = int((fn_mask > 0.5).sum())
                    hit_n = int(((fn_label > 0.5) & (fn_mask > 0.5)).sum())
                    top20_hit_node_n = int(((fn_label_top20 > 0.5) & (fn_mask > 0.5)).sum())

                    try:
                        top20_unique_bins = set(
                            int(x) for x in fn_idx[
                                (fn_label_top20 > 0.5)
                                & (fn_mask > 0.5)
                                & (fn_idx >= 0)
                            ].tolist()
                        )
                        top20_hit_unique_bin_n = len(top20_unique_bins)
                    except Exception:
                        top20_hit_unique_bin_n = 0

                    print(
                        "[FRAGMENT_NODE]",
                        "valid_n=", valid_n,
                        "hit_node_n=", hit_n,
                        "top20_hit_node_n=", top20_hit_node_n,
                        "top20_hit_unique_bin_n=", top20_hit_unique_bin_n,
                        "target_bins=", len(target_map),
                        "target_top20_bins=", len(target_top20_bins),
                        flush=True,
                    )

            except Exception as e:
                if os.environ.get("DEBUG_FRAGMENT_NODE_CANDIDATES", "0").strip() == "1":
                    print("[FRAGMENT_NODE_ERROR]", repr(e), flush=True)
                if os.environ.get("STRICT_FRAGMENT_NODE_CANDIDATES", "1").strip() == "1":
                    raise RuntimeError(f"V3 fragment-node candidate build failed: {repr(e)}")

        # atomicno one-hot matrix
        if len(self.element_oh ) > 0 :
            element_oh_mat = np.zeros((self.MAX_N, len(self.element_oh)), dtype=np.float32)
            for ei, e in enumerate(self.element_oh):
                element_oh_mat[:DATA_N, ei] = (atomicnos == e)

            v['vert_element_oh'] = element_oh_mat

        return v


# Class overview: PredFeaturizer encapsulates a reusable component in this module.
class PredFeaturizer:
    """
    create output data to predict
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, bin_config, **kwargs):
        self.spect_bin_config = bin_config

    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, mol, sparse_spect):
        """
        sparse_spect is N x 2 of (mz, intensity) 
        note that there can be duplicate m/zs

        """
        record_spect = np.stack(sparse_spect)
        spect_idx, spect_p, spect_out = self.spect_bin_config.histogram(record_spect[:, 0],
                                                                        record_spect[:, 1])
        spect_out = spect_out.astype(np.float32)
        return {'spect': spect_out}

def _fallback_local_connected_subsets(mol, num_samples=None):
    """
    Safe pure-Python fallback when vertsubsetgen backend is unavailable.

    Returns an array of shape [S, N_atoms] with 0/1 masks.
    We use low-noise local connected subsets:
      - single atoms
      - bond pairs
      - 1-hop neighborhoods
      - ring atom sets
      - 2-hop neighborhoods (only if still need more)
    """
    atom_n = int(mol.GetNumAtoms())
    if atom_n <= 0:
        return np.zeros((0, 0), dtype=np.float32)

    max_samples = int(num_samples) if (num_samples is not None and int(num_samples) > 0) else 128
    subsets = []
    seen = set()

    def add_subset(idxs):
        idxs = sorted(set(int(i) for i in idxs if 0 <= int(i) < atom_n))
        if len(idxs) <= 0:
            return
        key = tuple(idxs)
        if key in seen:
            return
        seen.add(key)
        mask = np.zeros((atom_n,), dtype=np.float32)
        mask[idxs] = 1.0
        subsets.append(mask)

    # 1) single atoms
    for i in range(atom_n):
        add_subset([i])
        if len(subsets) >= max_samples:
            return np.stack(subsets, axis=0)

    # 2) bond pairs
    for b in mol.GetBonds():
        add_subset([b.GetBeginAtomIdx(), b.GetEndAtomIdx()])
        if len(subsets) >= max_samples:
            return np.stack(subsets, axis=0)

    # build adjacency list
    nbrs = {i: set() for i in range(atom_n)}
    for b in mol.GetBonds():
        a = int(b.GetBeginAtomIdx())
        c = int(b.GetEndAtomIdx())
        nbrs[a].add(c)
        nbrs[c].add(a)

    # 3) 1-hop neighborhoods
    for i in range(atom_n):
        add_subset([i] + sorted(nbrs[i]))
        if len(subsets) >= max_samples:
            return np.stack(subsets, axis=0)

    # 4) rings
    try:
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            add_subset(list(ring))
            if len(subsets) >= max_samples:
                return np.stack(subsets, axis=0)
    except Exception:
        pass

    # 5) 2-hop neighborhoods
    for i in range(atom_n):
        two_hop = set([i])
        for j in nbrs[i]:
            two_hop.add(j)
            for k in nbrs[j]:
                two_hop.add(k)
        add_subset(sorted(two_hop))
        if len(subsets) >= max_samples:
            return np.stack(subsets, axis=0)

    if len(subsets) <= 0:
        return np.zeros((0, atom_n), dtype=np.float32)

    return np.stack(subsets, axis=0)

def _build_expl_formula_multi_idx_from_subset_counts(
    atom_subsets,
    vert_element_oh,
    formulae_features,
    anchor_idx=None,
    formulae_peaks_official_idx=None,
    formula_atomicnos=None,
    topr=16,
):
    """
    Build explanation raw formula id list from subset-derived element counts,
    not from anchor expansion.

    atom_subsets: [S, A] float/bool
    vert_element_oh: [A, E]
    formulae_features: [M, E]
    anchor_idx: [S] or None, used only as fallback / light tie-break
    formulae_peaks_official_idx: [M, K] or None
    formula_atomicnos: list[int]
    topr: int

    return:
        out: [S, R] int64, padded with -1
    """
    atom_subsets = np.asarray(atom_subsets, dtype=np.float32)
    vert_element_oh = np.asarray(vert_element_oh, dtype=np.float32)
    formulae_features = np.asarray(formulae_features, dtype=np.int16)

    subset_n = int(atom_subsets.shape[0])
    cand_n = int(formulae_features.shape[0])
    feat_d = int(formulae_features.shape[1])
    topr = max(1, int(topr))

    out = np.full((subset_n, topr), -1, dtype=np.int64)
    if subset_n <= 0 or cand_n <= 0 or feat_d <= 0:
        return out

    if anchor_idx is None:
        anchor_idx = np.full((subset_n,), -1, dtype=np.int64)
    else:
        anchor_idx = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
        if anchor_idx.shape[0] < subset_n:
            pad = np.full((subset_n - anchor_idx.shape[0],), -1, dtype=np.int64)
            anchor_idx = np.concatenate([anchor_idx, pad], axis=0)
        anchor_idx = anchor_idx[:subset_n]

    if formula_atomicnos is None:
        formula_atomicnos = list(range(feat_d))
    try:
        formula_atomicnos = [int(x) for x in formula_atomicnos]
    except Exception:
        formula_atomicnos = list(range(feat_d))

    # current vert_element_oh comes from atom list, so H is usually unreliable
    # unless hydrogens are explicit; therefore match on heavy-element counts.
    heavy_cols = [i for i, z in enumerate(formula_atomicnos[:feat_d]) if int(z) != 1]
    if len(heavy_cols) <= 0:
        heavy_cols = list(range(feat_d))

    use_atom_d = min(int(atom_subsets.shape[1]), int(vert_element_oh.shape[0]))
    use_feat_d = min(int(vert_element_oh.shape[1]), feat_d)

    subset_counts_full = np.rint(
        atom_subsets[:, :use_atom_d] @ vert_element_oh[:use_atom_d, :use_feat_d]
    ).astype(np.int16)

    if use_feat_d < feat_d:
        pad = np.zeros((subset_n, feat_d - use_feat_d), dtype=np.int16)
        subset_counts_full = np.concatenate([subset_counts_full, pad], axis=1)

    subset_heavy = subset_counts_full[:, heavy_cols]
    cand_heavy = formulae_features[:, heavy_cols]

    peak_idx = None
    if formulae_peaks_official_idx is not None:
        peak_idx = np.asarray(formulae_peaks_official_idx, dtype=np.int64)
        if peak_idx.ndim != 2 or int(peak_idx.shape[0]) != cand_n:
            peak_idx = None

    for si in range(subset_n):
        target_heavy = subset_heavy[si]

        # strict heavy-element exact match
        matched = np.where(np.all(cand_heavy == target_heavy[None, :], axis=1))[0].astype(np.int64)

        # fallback to anchor only if subset-derived exact heavy match is empty
        ai = int(anchor_idx[si])
        if matched.size <= 0:
            if 0 <= ai < cand_n:
                out[si, 0] = ai
            continue

        score = np.zeros((matched.shape[0],), dtype=np.float64)

        # prefer candidates with richer official peak support
        if peak_idx is not None:
            valid_peak_n = (peak_idx[matched] >= 0).sum(axis=1).astype(np.float64)
            score += valid_peak_n

            uniq_peak_n = []
            for cid in matched:
                row = peak_idx[int(cid)]
                row = row[row >= 0]
                uniq_peak_n.append(float(np.unique(row).shape[0]))
            score += np.asarray(uniq_peak_n, dtype=np.float64) * 0.25

        # light tie-break for current anchor, but do NOT let it dominate
        if 0 <= ai < cand_n:
            score += (matched == ai).astype(np.float64) * 0.5

        order = np.argsort(-score, kind='stable')
        keep = matched[order[:topr]]
        out[si, :keep.shape[0]] = keep

    return out


def _aggregate_expl_subset_peaks_from_multi_idx(
    expl_multi_idx,
    formulae_peaks_official_idx,
    formulae_peaks_official_intensity,
    max_k=32,
):
    """
    Aggregate official-bin peaks across all raw formula ids assigned to a subset.

    expl_multi_idx: [S, R]
    formulae_peaks_official_idx: [M, K]
    formulae_peaks_official_intensity: [M, K]

    return:
        subset_idx: [S, max_k] int64
        subset_int: [S, max_k] float32
    """
    multi_idx = np.asarray(expl_multi_idx, dtype=np.int64)
    peak_idx = np.asarray(formulae_peaks_official_idx, dtype=np.int64)
    peak_int = np.asarray(formulae_peaks_official_intensity, dtype=np.float32)

    subset_n = int(multi_idx.shape[0])
    cand_n = int(peak_idx.shape[0])
    max_k = max(1, int(max_k))

    out_idx = np.full((subset_n, max_k), -1, dtype=np.int64)
    out_int = np.zeros((subset_n, max_k), dtype=np.float32)

    if subset_n <= 0 or cand_n <= 0:
        return out_idx, out_int

    for si in range(subset_n):
        ids = multi_idx[si]
        ids = ids[(ids >= 0) & (ids < cand_n)]
        if ids.size <= 0:
            continue

        agg = {}
        # each candidate contributes equally after its own per-candidate normalization
        for cid in np.unique(ids):
            row_idx = peak_idx[int(cid)]
            row_int = peak_int[int(cid)]

            valid = (row_idx >= 0) & np.isfinite(row_int) & (row_int > 0)
            if not np.any(valid):
                continue

            idx_v = row_idx[valid].astype(np.int64, copy=False)
            int_v = row_int[valid].astype(np.float64, copy=False)
            total = float(np.sum(int_v))
            if total > 1e-12:
                int_v = int_v / total

            for bi, wi in zip(idx_v, int_v):
                agg[int(bi)] = agg.get(int(bi), 0.0) + float(wi)

        if len(agg) <= 0:
            continue

        items = sorted(agg.items(), key=lambda kv: (-kv[1], kv[0]))[:max_k]
        idx_keep = np.asarray([kv[0] for kv in items], dtype=np.int64)
        int_keep = np.asarray([kv[1] for kv in items], dtype=np.float32)

        total = float(np.sum(int_keep))
        if total > 1e-12:
            int_keep = int_keep / total

        out_idx[si, :idx_keep.shape[0]] = idx_keep
        out_int[si, :int_keep.shape[0]] = int_keep

    return out_idx, out_int

def _build_expl_formula_multi_idx_from_anchor(
    anchor_idx,
    formulae_features,
    formulae_peaks_official_idx=None,
    formula_atomicnos=None,
    topr=8,
):
    """
    Build multi-match raw formula index list for explanation subsets.

    anchor_idx: [S] int64
        current single best-match formula id per subset
    formulae_features: [M, E]
        all formula candidates
    formulae_peaks_official_idx: [M, K] or None
        optional, used only to rank same-heavy-formula candidates
    formula_atomicnos: list[int]
        candidate formula element order
    topr: int
        max number of raw formula ids kept per subset

    return:
        out: [S, top_r] int64, padded with -1
    """
    anchor_idx = np.asarray(anchor_idx, dtype=np.int64).reshape(-1)
    formulae_features = np.asarray(formulae_features, dtype=np.int16)
    cand_n = int(formulae_features.shape[0])
    topr = max(1, int(topr))

    out = np.full((anchor_idx.shape[0], topr), -1, dtype=np.int64)
    if cand_n <= 0:
        return out

    if formula_atomicnos is None:
        formula_atomicnos = list(range(int(formulae_features.shape[1])))
    try:
        formula_atomicnos = [int(x) for x in formula_atomicnos]
    except Exception:
        formula_atomicnos = list(range(int(formulae_features.shape[1])))

    # ignore H for explanation matching v2; keep heavy-atom composition the same
    heavy_cols = [i for i, z in enumerate(formula_atomicnos) if int(z) != 1]
    if len(heavy_cols) <= 0:
        heavy_cols = list(range(int(formulae_features.shape[1])))

    all_heavy = formulae_features[:, heavy_cols]

    peak_idx = None
    if formulae_peaks_official_idx is not None:
        peak_idx = np.asarray(formulae_peaks_official_idx, dtype=np.int64)
        if peak_idx.ndim != 2 or int(peak_idx.shape[0]) != cand_n:
            peak_idx = None

    for si, ai in enumerate(anchor_idx):
        ai = int(ai)
        if ai < 0 or ai >= cand_n:
            continue

        target_heavy = all_heavy[ai]
        matched = np.where(np.all(all_heavy == target_heavy[None, :], axis=1))[0].astype(np.int64)

        if matched.size <= 0:
            out[si, 0] = ai
            continue

        # rank candidates: anchor first, then by official-peak overlap with anchor
        score = np.zeros((matched.shape[0],), dtype=np.int64)
        score += (matched == ai).astype(np.int64) * 100000

        if peak_idx is not None:
            anchor_peaks = peak_idx[ai]
            anchor_peaks = anchor_peaks[anchor_peaks >= 0]
            if anchor_peaks.size > 0:
                for mj, cid in enumerate(matched):
                    cand_peaks = peak_idx[int(cid)]
                    cand_peaks = cand_peaks[cand_peaks >= 0]
                    if cand_peaks.size > 0:
                        score[mj] += int(np.isin(cand_peaks, anchor_peaks).sum())

        order = np.argsort(-score)
        keep = matched[order[:topr]]
        out[si, :keep.shape[0]] = keep

    return out

# Function overview: create_subset_generator handles a specific workflow step in this module.
def create_subset_generator(name, **config):
    ln = str(name).lower()

    if ln in ('bandr', 'break_and_rearrange', 'breakandrearrange', 'break-and-rearrange'):
        def run(mol, num_samples=None):
            # 1) fast compiled path if available
            has_fast_ext = getattr(msutil.vertsubsetgen, 'fast', None) is not None
            if mol.GetNumAtoms() <= 64 and has_fast_ext and hasattr(msutil.vertsubsetgen, 'BreakAndRearrangeFast'):
                try:
                    return msutil.vertsubsetgen.BreakAndRearrangeFast(**config)(mol)
                except Exception:
                    pass

            # 2) python/normal backend if available
            if hasattr(msutil.vertsubsetgen, 'EnumerateBreaks'):
                try:
                    enum_breaks = msutil.vertsubsetgen.EnumerateBreaks(**config)
                    return enum_breaks(mol, num_samples=num_samples)
                except Exception:
                    pass

            # 3) last-resort local fallback
            return _fallback_local_connected_subsets(mol, num_samples=num_samples)
        return run

    elif ln in ('b', 'enumeratebreaks', 'enumerate_breaks'):
        def run(mol, num_samples=None):
            if hasattr(msutil.vertsubsetgen, 'EnumerateBreaks'):
                try:
                    return msutil.vertsubsetgen.EnumerateBreaks(**config)(mol, num_samples=num_samples)
                except Exception:
                    pass
            return _fallback_local_connected_subsets(mol, num_samples=num_samples)
        return run

    else:
        raise NotImplementedError(f"unknown subset generator {name}")