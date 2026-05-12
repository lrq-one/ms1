# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Mass-spectrometry-specific helper utilities for binning, formula/mass operations, and evaluation helpers.

import importlib
import math
import os

import numpy as np
from rdkit import Chem

try:
    import diskcache as dc
except Exception:
    dc = None

masseval = None
for module_name in ('rassp.masseval', 'rassp.msutil.masseval'):
    try:
        masseval = importlib.import_module(module_name)
        break
    except Exception:
        masseval = None

USE_DC = False
HAS_MASSEVAL = masseval is not None


# Helper: parse boolean-like environment switches (1/0/true/false/on/off).
def _env_flag(name, default='1'):
    raw = os.environ.get(name, default)
    if raw is None:
        return default not in ('0', 'false', 'False', 'no', 'off')
    return str(raw).strip().lower() not in ('0', 'false', 'no', 'off')


# Helper: parse integer environment values with safe fallback.
def _env_int(name, default=0):
    raw = os.environ.get(name, '').strip()
    if not raw:
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def require_masseval_backend():
    if not HAS_MASSEVAL:
        raise RuntimeError(
            'rassp.msutil.masscompute requires the compiled masseval backend, but it was not found. '
            'Install/build the extension or set REQUIRE_MASSEVAL_BACKEND=0 only for local debugging.'
        )

# Recursive enumerator: Cartesian count expansion for per-element atom counts.
def count_seq(sequences):
    if len(sequences) == 1:
        return [[i] for i in range(sequences[0]+1)]
    
    subseq = count_seq(sequences[1:])
    
    j = sequences[0]
    out = []
    for i in range(j+1):
        out += [[i] + s for s in subseq]
    return out

# Build molecular formula dictionary (atomic number -> count), with optional implicit-H inclusion.
"""
1）get_formula(mol)

它做的是：

统计分子里每种元素的个数
默认会把 implicit H 也算进去
受环境变量 FORMULA_INCLUDE_IMPLICIT_H 和 FORMULA_IMPLICIT_H_CAP 控制。

这一步很重要，因为你后面所有 formula candidate 的上界，都是从这里来的。

你要看懂什么

这里的意思是：

你的候选公式空间，不是从谱里来的，而是从母体分子元素计数笛卡尔展开出来的。

所以如果这里的 H 统计就和你预期不一致，后面所有 candidate 都会偏。
"""
def get_formula(mol, include_implicit_h=None):
    """
    Return a dictionary of atomicno:num

    By default we include implicit hydrogens. If the molecule already
    contains explicit H atoms, do NOT also add GetNumExplicitHs() from
    heavy atoms, otherwise H can be double-counted.
    """
    if include_implicit_h is None:
        include_implicit_h = _env_flag('FORMULA_INCLUDE_IMPLICIT_H', '1')

    out = {}
    implicit_h_total = 0
    has_explicit_h_atoms = any(int(a.GetAtomicNum()) == 1 for a in mol.GetAtoms())

    for a in mol.GetAtoms():
        an = a.GetAtomicNum()
        out[an] = out.get(an, 0) + 1

        if include_implicit_h and an != 1:
            try:
                h_extra = int(a.GetNumImplicitHs())
                if not has_explicit_h_atoms:
                    h_extra += int(a.GetNumExplicitHs())
            except Exception:
                h_extra = 0
            if h_extra > 0:
                implicit_h_total += h_extra

    if include_implicit_h and implicit_h_total > 0:
        h_cap = max(0, _env_int('FORMULA_IMPLICIT_H_CAP', 0))
        if h_cap > 0:
            implicit_h_total = min(implicit_h_total, h_cap)
        out[1] = out.get(1, 0) + implicit_h_total

    return out   
# Enumerate all fragment formula count vectors and their naive isotope masses.
class FragmentFormulaEnumerator:
    """
    Enumerate all possible formulae and their naive weights

    """
    
    # Initialize periodic-table mass lookup for configured candidate elements.
    def __init__(self, formula_possible_atomicnos):

        """
        formula_possible_atomicnos: possible atomicnos in the formula, in order
        #formula_max_atoms: maximum number of atoms in the formula
        """
       
        
        self.formula_possible_atomicnos = formula_possible_atomicnos
        #self.formula_max_atoms = formula_max_atoms
        pt = Chem.GetPeriodicTable()
        self.atomic_weights = np.array([pt.GetMostCommonIsotopeMass(a) \
                                      for a in self.formula_possible_atomicnos])
        
    # Enumerate formula combinations for one molecule and compute per-formula mass.
    def get_frag_formulae(self, mol, weights='naive'):
        f = get_formula(mol)
        
        seq = [ f.get(a, 0) for a in self.formula_possible_atomicnos  ]
        frag_formulae = count_seq(seq)
        frag_formulae = np.array(frag_formulae, dtype=np.int32)

        if weights == 'naive':
           frag_formulae_masses = frag_formulae @ self.atomic_weights
        else:
           raise NotImplementedError("Have not yet implemented proper isotopes")
        
        return frag_formulae, frag_formulae_masses 

# Enumerate fragment formulas and corresponding peak arrays (fallback or masseval backend).
class FragmentFormulaPeakEnumerator:
    """
    Enumerate all possible formulae and their peak weights. 

    """
    
    # Initialize naive enumerator + optional disk cache + backend mode.
    def __init__(self, formula_possible_atomicnos,
                 use_highres=True, max_peak_num=16):

        """
        formula_possible_atomicnos: possible atomicnos in the formula, in order
        """
        self.formula_possible_atomicnos = formula_possible_atomicnos
        self.naive_enum = FragmentFormulaEnumerator(formula_possible_atomicnos)
        self.atomicno_to_pos = {int(an): i for i, an in enumerate(self.formula_possible_atomicnos)}

        if USE_DC and dc is not None:
            self.dc = dc.Cache("FragFormulaPeakEnumerator")

        self.use_highres = use_highres
        self.max_peak_num = max_peak_num
        renderer_mode = os.environ.get(
            'FORMULA_PEAK_RENDERER',
            'exact'
        ).strip().lower()
        if renderer_mode not in ('exact', 'masseval', 'legacy'):
            renderer_mode = 'exact'
        self.renderer_mode = renderer_mode

        # Keep legacy renderer for regression comparisons.
        self.isotope_shift_da = 1.0033548378
        self.plus1_abundance = {
            1: 0.00015,
            6: 0.0107,
            7: 0.00364,
            8: 0.00038,
            16: 0.00790,
            17: 0.0,
            35: 0.0,
            53: 0.0,
        }
        self.plus2_abundance = {
            8: 0.00205,
            16: 0.04210,
            17: 0.24220,
            35: 0.49310,
            53: 0.0,
        }

        self._pt = Chem.GetPeriodicTable()
        self._atomic_weights = np.asarray(
            [self._pt.GetMostCommonIsotopeMass(int(an)) for an in self.formula_possible_atomicnos],
            dtype=np.float64,
        )
        self._build_exact_isotope_model()

        if self.renderer_mode == 'masseval':
            if HAS_MASSEVAL:
                self.peak_backend = 'masseval_highres' if self.use_highres else 'masseval_nostruct'
            else:
                if _env_flag('REQUIRE_MASSEVAL_BACKEND', '1'):
                    require_masseval_backend()
                self.renderer_mode = 'exact'
                self.peak_backend = 'formula_exact_isotope'
        elif self.renderer_mode == 'legacy':
            self.peak_backend = 'formula_legacy_isotope'
        else:
            self.renderer_mode = 'exact'
            self.peak_backend = 'formula_exact_isotope'

    # Build isotope channel tables used by exact truncated isotopologue renderer.
    def _build_exact_isotope_model(self):
        isotope_abundance = {
            # atomic_number: [(isotope_number, natural_abundance)]
            1: [(2, 0.000115)],
            6: [(13, 0.0107)],
            7: [(15, 0.00364)],
            8: [(17, 0.00038), (18, 0.00205)],
            9: [],
            15: [],
            16: [(33, 0.00790), (34, 0.04210)],
            17: [(37, 0.24220)],
            35: [(81, 0.49310)],
            53: [],
        }

        e_n = len(self.formula_possible_atomicnos)
        self._no_sub_prob = np.ones((e_n,), dtype=np.float64)

        channels = []
        for pos, an_raw in enumerate(self.formula_possible_atomicnos):
            an = int(an_raw)
            defs = isotope_abundance.get(an, [])
            if len(defs) <= 0:
                self._no_sub_prob[pos] = 1.0
                continue

            p_total = float(sum(float(p) for _, p in defs if float(p) > 0.0))
            p_total = min(max(p_total, 0.0), 0.999999)
            no_sub = max(1e-12, 1.0 - p_total)
            self._no_sub_prob[pos] = no_sub

            base_iso = int(self._pt.GetMostCommonIsotope(an))
            base_mass = float(self._pt.GetMostCommonIsotopeMass(an))
            for iso_num_raw, p_raw in defs:
                p = float(p_raw)
                if p <= 0.0:
                    continue
                iso_num = int(iso_num_raw)
                iso_mass = float(self._pt.GetMassForIsotope(an, iso_num))
                nominal_shift = int(iso_num - base_iso)
                if nominal_shift <= 0:
                    continue
                channels.append(
                    {
                        'pos': int(pos),
                        'atomicno': int(an),
                        'abundance': p,
                        'ratio': p / no_sub,
                        'nominal_shift': nominal_shift,
                        'mass_delta': float(iso_mass - base_mass),
                    }
                )

        self._channels = channels
        if len(channels) > 0:
            self._ch_pos = np.asarray([c['pos'] for c in channels], dtype=np.int64)
            self._ch_ratio = np.asarray([c['ratio'] for c in channels], dtype=np.float64)
            self._ch_delta = np.asarray([c['mass_delta'] for c in channels], dtype=np.float64)
            self._ch_nominal = np.asarray([c['nominal_shift'] for c in channels], dtype=np.int64)
        else:
            self._ch_pos = np.zeros((0,), dtype=np.int64)
            self._ch_ratio = np.zeros((0,), dtype=np.float64)
            self._ch_delta = np.zeros((0,), dtype=np.float64)
            self._ch_nominal = np.zeros((0,), dtype=np.int64)

    # Exact renderer: monoisotopic mass + truncated isotopologue expansion (0/1/2 substitutions).
    def _exact_isotope_peaks(self, formulae, masses):
        formulae_arr = np.asarray(formulae, dtype=np.float64)
        cand_n = int(formulae_arr.shape[0]) if formulae_arr.ndim >= 2 else 0
        peak_n = max(1, min(int(self.max_peak_num), 16))

        peaks = np.zeros((cand_n, peak_n, 2), dtype=np.float32)
        if cand_n <= 0:
            return peaks

        base_mass = np.asarray(masses, dtype=np.float64).reshape(-1)
        if base_mass.shape[0] != cand_n:
            base_mass = (formulae_arr @ self._atomic_weights).reshape(-1)

        # p0: probability mass of monoisotopic channel (no heavy substitutions).
        log_p0 = np.zeros((cand_n,), dtype=np.float64)
        for pos in range(formulae_arr.shape[1]):
            no_sub = float(self._no_sub_prob[pos]) if pos < self._no_sub_prob.shape[0] else 1.0
            if no_sub <= 0.0:
                continue
            log_p0 += formulae_arr[:, pos] * math.log(no_sub)
        p0 = np.exp(log_p0)

        raw_mass_cols = [base_mass]
        raw_weight_cols = [p0]

        ch_n = int(self._ch_pos.shape[0])
        # One heavy-isotope substitution terms.
        for i in range(ch_n):
            pos_i = int(self._ch_pos[i])
            n_i = formulae_arr[:, pos_i]
            w_i = p0 * n_i * float(self._ch_ratio[i])
            raw_mass_cols.append(base_mass + float(self._ch_delta[i]))
            raw_weight_cols.append(w_i)

        # Two-substitution terms cover the M+2 regime and a compact fine-structure approximation.
        for i in range(ch_n):
            pos_i = int(self._ch_pos[i])
            ratio_i = float(self._ch_ratio[i])
            delta_i = float(self._ch_delta[i])
            n_i = formulae_arr[:, pos_i]

            for j in range(i, ch_n):
                pos_j = int(self._ch_pos[j])
                ratio_j = float(self._ch_ratio[j])
                delta_j = float(self._ch_delta[j])

                if i == j:
                    comb = 0.5 * n_i * (n_i - 1.0)
                elif pos_i == pos_j:
                    comb = n_i * (n_i - 1.0)
                else:
                    comb = n_i * formulae_arr[:, pos_j]

                w_ij = p0 * comb * ratio_i * ratio_j
                raw_mass_cols.append(base_mass + delta_i + delta_j)
                raw_weight_cols.append(w_ij)

        raw_mass = np.stack(raw_mass_cols, axis=1)
        raw_weight = np.stack(raw_weight_cols, axis=1)

        raw_weight = np.where(np.isfinite(raw_weight) & (raw_weight > 0.0), raw_weight, 0.0)
        raw_mass = np.where(np.isfinite(raw_mass), raw_mass, 0.0)

        raw_k = int(raw_weight.shape[1])
        keep_k = min(peak_n, raw_k)
        if keep_k <= 0:
            return peaks

        if keep_k < raw_k:
            keep_idx = np.argpartition(raw_weight, kth=raw_k - keep_k, axis=1)[:, raw_k - keep_k :]
        else:
            keep_idx = np.tile(np.arange(raw_k, dtype=np.int64), (cand_n, 1))

        keep_mass = np.take_along_axis(raw_mass, keep_idx, axis=1)
        keep_weight = np.take_along_axis(raw_weight, keep_idx, axis=1)

        # Keep deterministic order for downstream debugging and aux-feature stability.
        sort_idx = np.argsort(keep_mass, axis=1, kind='mergesort')
        keep_mass = np.take_along_axis(keep_mass, sort_idx, axis=1)
        keep_weight = np.take_along_axis(keep_weight, sort_idx, axis=1)

        peaks[:, :keep_k, 0] = keep_mass.astype(np.float32)
        peaks[:, :keep_k, 1] = keep_weight.astype(np.float32)

        denom = np.sum(peaks[:, :, 1], axis=1, keepdims=True)
        valid = denom > 1e-12
        peaks[:, :, 1] = np.where(valid, peaks[:, :, 1] / np.where(valid, denom, 1.0), 0.0)

        empty_rows = np.where(~valid.reshape(-1))[0]
        if empty_rows.size > 0:
            peaks[empty_rows, 0, 0] = base_mass[empty_rows].astype(np.float32)
            peaks[empty_rows, 0, 1] = 1.0

        return peaks

    # Legacy helper: build a truncated isotope envelope from +1/+2 abundance approximation.
    def _legacy_isotope_envelope(self, formulae, peak_n):
        cand_n = int(formulae.shape[0])
        if cand_n <= 0:
            return np.zeros((0, peak_n), dtype=np.float32)

        counts = np.asarray(formulae, dtype=np.float64)
        lam1 = np.zeros((cand_n,), dtype=np.float64)
        lam2 = np.zeros((cand_n,), dtype=np.float64)

        for an, p in self.plus1_abundance.items():
            pos = self.atomicno_to_pos.get(int(an), None)
            if pos is None or pos >= counts.shape[1]:
                continue
            lam1 += counts[:, pos] * float(p)

        for an, p in self.plus2_abundance.items():
            pos = self.atomicno_to_pos.get(int(an), None)
            if pos is None or pos >= counts.shape[1]:
                continue
            lam2 += counts[:, pos] * float(p)

        lam1 = np.clip(lam1, 0.0, None)
        lam2 = np.clip(lam2, 0.0, None)

        coeff = np.zeros((cand_n, peak_n), dtype=np.float64)
        for k in range(int(peak_n)):
            row = np.zeros((cand_n,), dtype=np.float64)
            for j in range((k // 2) + 1):
                a = k - (2 * j)
                row += (
                    (np.power(lam1, a) / float(math.factorial(a)))
                    * (np.power(lam2, j) / float(math.factorial(j)))
                )
            coeff[:, k] = row

        coeff = np.clip(coeff, 0.0, None)
        denom = coeff.sum(axis=1, keepdims=True)
        denom = np.where(denom > 1e-12, denom, 1.0)
        coeff = coeff / denom
        return coeff.astype(np.float32)

    # Legacy renderer: formula-conditioned isotope envelope (historical approximation).
    def _legacy_get_frag_formulae(self, mol):
        formulae, masses = self.naive_enum.get_frag_formulae(mol)
        masses = np.asarray(masses, dtype=np.float32).reshape(-1)

        peak_n = max(1, min(int(self.max_peak_num), 6))
        peaks = np.zeros((formulae.shape[0], peak_n, 2), dtype=np.float32)
        if formulae.shape[0] <= 0:
            return formulae, peaks

        shift = np.arange(peak_n, dtype=np.float32) * float(self.isotope_shift_da)
        peaks[:, :, 0] = masses[:, None] + shift[None, :]
        peaks[:, :, 1] = self._legacy_isotope_envelope(formulae, peak_n)
        return formulae, peaks

    # Exact renderer entrypoint: enumerate formulae and compute exact-isotope peaks.
    def _exact_get_frag_formulae(self, mol):
        formulae, masses = self.naive_enum.get_frag_formulae(mol)
        masses = np.asarray(masses, dtype=np.float64).reshape(-1)
        peaks = self._exact_isotope_peaks(formulae, masses)
        return formulae, peaks

    # masseval renderer path preserved for compatibility/ablation.
    def _masseval_get_frag_formulae(self, formula_dict):
        # The compiled masseval backend in this repo uses a limited atomic-number
        # axis (historically up to 19). If the current molecule contains requested
        # elements beyond that axis (e.g., Br=35, I=53), call exact renderer directly
        # before entering masseval to avoid extension-layer index errors.
        for an in self.formula_possible_atomicnos:
            try:
                an_i = int(an)
            except Exception:
                continue
            if an_i > 19 and int(formula_dict.get(an_i, 0)) > 0:
                return None

        if self.use_highres:
            np_out = masseval.py_get_all_frag_spect_highres(
                formula_dict,
                MAX_PEAK_NUM=self.max_peak_num,
            )
        else:
            np_out = masseval.py_get_all_frag_spect_np_nostruct(
                formula_dict,
                MAX_PEAK_NUM=self.max_peak_num,
            )

        formula_arr = np.asarray(np_out['formula'])
        backend_cols = int(formula_arr.shape[1]) if formula_arr.ndim >= 2 else 0

        # If the compiled backend cannot represent one of the requested elements
        # and the current molecule actually contains that element, fallback to exact renderer.
        for an in self.formula_possible_atomicnos:
            try:
                an_i = int(an)
            except Exception:
                continue
            if an_i >= backend_cols and int(formula_dict.get(an_i, 0)) > 0:
                return None

        formulae = np.zeros((formula_arr.shape[0], len(self.formula_possible_atomicnos)), dtype=formula_arr.dtype)
        for j, an in enumerate(self.formula_possible_atomicnos):
            try:
                an_i = int(an)
            except Exception:
                continue
            if 0 <= an_i < backend_cols:
                formulae[:, j] = formula_arr[:, an_i]

        peaks = np_out['peaks']
        mass = np.asarray(peaks['mass'], dtype=np.float32)
        intensity = np.asarray(peaks['intensity'], dtype=np.float32)

        # Safety guard:
        # Some masseval builds return nominal integer masses even when the path is named "highres".
        # This is fatal for 0.01 Da NIST product-ion tasks, so fallback to exact isotope renderer.
        try:
            bw = float(os.environ.get("OFFICIAL_BIN_WIDTH", "0.01"))
        except Exception:
            bw = 0.01

        if bool(self.use_highres) and bw <= 0.02:
            valid_mass = mass[np.isfinite(mass) & (mass > 0)]
            if valid_mass.size > 100:
                frac = np.abs(valid_mass - np.round(valid_mass))
                integer_ratio = float(np.mean(frac < 1e-4))
                if integer_ratio > 0.95:
                    return None

        mp = np.stack([mass, intensity], -1)
        return formulae, mp

    # Main entry: enumerate formulas and peaks, using cache/backend when available.
    def get_frag_formulae(self, mol):
    
        formula_dict = get_formula(mol)
        if USE_DC and dc is not None:
            key = f"{self.renderer_mode}|{int(self.use_highres)}|{int(self.max_peak_num)}|{str(formula_dict)}"
            if key in self.dc:
                return self.dc[key]

        if self.renderer_mode == 'legacy':
            out = self._legacy_get_frag_formulae(mol)
        elif self.renderer_mode == 'masseval':
            out = self._masseval_get_frag_formulae(formula_dict)
            if out is None:
                out = self._exact_get_frag_formulae(mol)
                self.peak_backend = 'formula_exact_isotope_fallback'
        else:
            out = self._exact_get_frag_formulae(mol)

        if USE_DC:
            self.dc[key] = out
        
        return out


