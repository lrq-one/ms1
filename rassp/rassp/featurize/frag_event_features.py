# rassp/featurize/frag_event_features.py
import os
import re
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

_FORMULA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def _resolve_atomicnos(formula_atomicnos):
    if formula_atomicnos is not None:
        return [int(x) for x in formula_atomicnos]

    raw = os.environ.get("FORMULA_ATOMICNOS", "").strip()
    if raw:
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    return [1, 6, 7, 8, 9, 15, 16, 17]

def _formula_to_counts(formula_str, atomicnos):
    pt = Chem.GetPeriodicTable()
    sym_to_idx = {}
    for i, an in enumerate(atomicnos):
        sym_to_idx[pt.GetElementSymbol(int(an))] = i

    out = np.zeros((len(atomicnos),), dtype=np.float32)
    if not formula_str:
        return out

    for sym, n_raw in _FORMULA_RE.findall(str(formula_str)):
        if sym not in sym_to_idx:
            continue
        n = int(n_raw) if n_raw else 1
        out[sym_to_idx[sym]] += float(n)
    return out

def _exact_mass_from_counts(counts, atomicnos):
    pt = Chem.GetPeriodicTable()
    weights = np.asarray([pt.GetMostCommonIsotopeMass(int(an)) for an in atomicnos], dtype=np.float64)
    return float(np.asarray(counts, dtype=np.float64) @ weights)

def _fiora_mode_masses():
    # Match FIORA constants.py as closely as possible.
    h_plus = Descriptors.ExactMolWt(Chem.MolFromSmiles("[H+]"))
    h_minus = Descriptors.ExactMolWt(Chem.MolFromSmiles("[H-]"))
    h2 = Descriptors.ExactMolWt(Chem.MolFromSmiles("[HH]"))

    return [
        ("[M+H]+", float(h_plus), 0),
        ("[M]+", 0.0, 1),
        ("[M-H]+", -float(h_minus), 2),
        ("[M-2H]+", -float(h2), 3),
        ("[M-3H]+", -float(h2) - float(h_minus), 4),
    ]

def _components_after_removing_bond(mol, u, v):
    """
    Return two atom-id components after removing bond (u, v).

    关键点：
    - 不生成 RDKit fragment mol
    - 不 sanitize
    - 不让 RDKit 在断键后重新补 implicit H
    """
    n = int(mol.GetNumAtoms())
    nbrs = {i: set() for i in range(n)}

    for b in mol.GetBonds():
        a = int(b.GetBeginAtomIdx())
        c = int(b.GetEndAtomIdx())

        if (a == int(u) and c == int(v)) or (a == int(v) and c == int(u)):
            continue

        nbrs[a].add(c)
        nbrs[c].add(a)

    seen = set()
    comps = []

    for start in range(n):
        if start in seen:
            continue

        stack = [start]
        seen.add(start)
        comp = []

        while stack:
            x = stack.pop()
            comp.append(x)
            for y in nbrs[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)

        comps.append(sorted(comp))

    if len(comps) != 2:
        return []

    return comps


def _subset_counts_from_original_mol(mol, atom_ids, atomicnos):
    """
    从原始分子 atom subset 直接计算 fragment formula。

    关键点：
    - 如果 H 是显式 atom，就只统计 subset 里的 H atom
    - 如果 H 不是显式 atom，就统计原分子 heavy atom 原本带的 implicit/explicit H
    - 不统计断键后 RDKit sanitize 新补出来的 H
    """
    lut = {int(an): i for i, an in enumerate(atomicnos)}
    counts = np.zeros((len(atomicnos),), dtype=np.float32)

    atom_ids = [int(x) for x in atom_ids]
    has_explicit_h_atoms = any(int(a.GetAtomicNum()) == 1 for a in mol.GetAtoms())

    for ai in atom_ids:
        atom = mol.GetAtomWithIdx(int(ai))
        an = int(atom.GetAtomicNum())

        pos = lut.get(an, None)
        if pos is not None:
            counts[pos] += 1.0

        if (not has_explicit_h_atoms) and an != 1:
            h_pos = lut.get(1, None)
            if h_pos is not None:
                try:
                    h_extra = int(atom.GetNumImplicitHs()) + int(atom.GetNumExplicitHs())
                except Exception:
                    h_extra = 0
                if h_extra > 0:
                    counts[h_pos] += float(h_extra)

    return counts.astype(np.float32)


def _break_bond_into_two_fragments(mol, u, v):
    """
    FIORA-style:
    - copy mol
    - use atom isotope as temporary original atom id
    - remove one bond
    - sanitize
    - get two connected fragment mols
    """
    m = Chem.Mol(mol)
    for i, a in enumerate(m.GetAtoms()):
        a.SetIsotope(int(i))

    em = Chem.EditableMol(m)
    em.RemoveBond(int(u), int(v))
    new_mol = em.GetMol()
    Chem.SanitizeMol(new_mol)

    frags = Chem.GetMolFrags(new_mol, asMols=True, sanitizeFrags=True)
    if len(frags) != 2:
        return []

    out = []
    for frag in frags:
        atom_ids = []
        for a in frag.GetAtoms():
            atom_ids.append(int(a.GetIsotope()))
            a.SetIsotope(0)

        atom_ids = sorted(set(atom_ids))
        formula = rdMolDescriptors.CalcMolFormula(frag)
        neutral_mass = float(Descriptors.ExactMolWt(frag))
        out.append(
            {
                "mol": frag,
                "atom_ids": atom_ids,
                "formula": formula,
                "neutral_mass": neutral_mass,
            }
        )

    return out

def build_fiora_event_candidates(
    mol,
    formula_atomicnos=None,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    precursor_mz=None,
    precursor_formula=None,
    max_events=2048,
    allow_ring=False,
    include_precursor=False,
):
    """
    Build padded FIORA-style event candidates.

    Output fields:
      event_formulae_features: [E, A]
      event_neutral_loss_features: [E, A]
      event_mz: [E]
      event_official_idx: [E]
      event_official_intensity: [E]
      event_bond_idx: [E]
      event_bond_begin: [E]
      event_bond_end: [E]
      event_side: [E] 0/1, precursor=2
      event_h_mode: [E] 0..4, precursor=-1
      event_source_flag: [E] 1=single_bond_event, 9=precursor
      event_aux_feat: [E, 16]
      event_mask: [E]
    """
    atomicnos = _resolve_atomicnos(formula_atomicnos)
    max_events = int(max_events)
    bw = float(max(1e-8, official_bin_width))
    max_mz = float(max(official_max_mz, bw))

    pt = Chem.GetPeriodicTable()
    parent_counts = _formula_to_counts(precursor_formula, atomicnos) if precursor_formula else None
    if parent_counts is None or float(parent_counts.sum()) <= 0:
        try:
            parent_formula = rdMolDescriptors.CalcMolFormula(mol)
            parent_counts = _formula_to_counts(parent_formula, atomicnos)
        except Exception:
            parent_counts = np.zeros((len(atomicnos),), dtype=np.float32)

    parent_neutral_mass = _exact_mass_from_counts(parent_counts, atomicnos)
    if precursor_mz is not None:
        try:
            pmz = float(precursor_mz)
        except Exception:
            pmz = float("nan")
    else:
        pmz = float("nan")

    rows = []
    ion_modes = _fiora_mode_masses()

    for bond in mol.GetBonds():
        u = int(bond.GetBeginAtomIdx())
        v = int(bond.GetEndAtomIdx())

        if bond.IsInRing() and not bool(allow_ring):
            continue

        # FIORA skips ring; additionally skip explicit X-H breaks.
        if mol.GetAtomWithIdx(u).GetAtomicNum() == 1 or mol.GetAtomWithIdx(v).GetAtomicNum() == 1:
            continue

        mass_mode = os.environ.get("FIORA_EVENT_MASS_MODE", "rdkit_sanitize").strip().lower()

        frag_items = []
        if mass_mode in ("rdkit", "rdkit_sanitize", "fiora"):
            try:
                frag_items = _break_bond_into_two_fragments(mol, u, v)
            except Exception:
                frag_items = []
        else:
            try:
                frag_comps = _components_after_removing_bond(mol, u, v)
            except Exception:
                frag_comps = []

            for atom_ids in frag_comps:
                formula_counts_tmp = _subset_counts_from_original_mol(
                    mol=mol,
                    atom_ids=atom_ids,
                    atomicnos=atomicnos,
                )
                neutral_mass_tmp = _exact_mass_from_counts(formula_counts_tmp, atomicnos)
                frag_items.append(
                    {
                        "atom_ids": [int(x) for x in atom_ids],
                        "formula": None,
                        "formula_counts": formula_counts_tmp.astype(np.float32),
                        "neutral_mass": float(neutral_mass_tmp),
                    }
                )

        if len(frag_items) != 2:
            continue

        for side_i, frag_item in enumerate(frag_items):
            atom_ids = [int(x) for x in frag_item.get("atom_ids", [])]

            if "formula_counts" in frag_item:
                formula_counts = np.asarray(frag_item["formula_counts"], dtype=np.float32)
            else:
                formula_counts = _formula_to_counts(
                    frag_item.get("formula", None),
                    atomicnos,
                )

            if float(formula_counts.sum()) <= 0:
                continue

            neutral_mass = float(frag_item.get("neutral_mass", _exact_mass_from_counts(formula_counts, atomicnos)))

            neutral_loss_counts = np.maximum(parent_counts - formula_counts, 0.0).astype(np.float32)
            neutral_loss_mass = max(0.0, parent_neutral_mass - neutral_mass)

            atom_frac = float(len(atom_ids) / max(1, mol.GetNumAtoms()))
            heavy_n = sum(
                1 for ai in atom_ids
                if mol.GetAtomWithIdx(int(ai)).GetAtomicNum() > 1
            )
            heavy_frac = float(heavy_n / max(1, mol.GetNumHeavyAtoms()))

            bond_type = float(bond.GetBondTypeAsDouble())
            is_aromatic = float(bond.GetIsAromatic())
            is_conjugated = float(bond.GetIsConjugated())
            in_ring = float(bond.IsInRing())

            for mode_name, mass_shift, h_mode_idx in ion_modes:
                mz = neutral_mass + float(mass_shift)
                if not np.isfinite(mz) or mz <= 0.0 or mz >= max_mz:
                    continue

                official_idx = int(np.floor(mz / bw + 1e-8))
                precursor_gap = abs(pmz - mz) if np.isfinite(pmz) else 0.0
                mz_frac = float(mz / pmz) if np.isfinite(pmz) and pmz > 0 else 0.0
                small_frag_score = float(1.0 - np.clip(mz_frac, 0.0, 1.0))

                aux = np.asarray(
                    [
                        mz / 1000.0,
                        neutral_mass / 1000.0,
                        neutral_loss_mass / 1000.0,
                        precursor_gap / 1000.0,
                        mz_frac,
                        small_frag_score,
                        atom_frac,
                        heavy_frac,
                        bond_type / 3.0,
                        is_aromatic,
                        is_conjugated,
                        in_ring,
                        float(side_i),
                        float(h_mode_idx) / 4.0,
                        float(formula_counts.sum()) / 128.0,
                        float(neutral_loss_counts.sum()) / 128.0,
                    ],
                    dtype=np.float32,
                )

                rows.append(
                    {
                        "formula_counts": formula_counts.astype(np.float32),
                        "neutral_loss_counts": neutral_loss_counts.astype(np.float32),
                        "mz": float(mz),
                        "official_idx": int(official_idx),
                        "official_intensity": 1.0,
                        "bond_idx": int(bond.GetIdx()),
                        "bond_begin": int(u),
                        "bond_end": int(v),
                        "side": int(side_i),
                        "h_mode": int(h_mode_idx),
                        "source_flag": 1,
                        "break_depth": 1,
                        "ring_cut_flag": int(bond.IsInRing()),
                        "aux": aux,
                    }
                )

    # Optional precursor candidate. Keep it separate in training/eval if official target excludes precursor.
    if include_precursor and np.isfinite(pmz) and pmz > 0 and pmz < max_mz:
        official_idx = int(np.floor(pmz / bw + 1e-8))
        aux = np.zeros((16,), dtype=np.float32)
        aux[0] = pmz / 1000.0
        aux[1] = parent_neutral_mass / 1000.0
        aux[4] = 1.0
        rows.append(
            {
                "formula_counts": parent_counts.astype(np.float32),
                "neutral_loss_counts": np.zeros_like(parent_counts, dtype=np.float32),
                "mz": float(pmz),
                "official_idx": int(official_idx),
                "official_intensity": 1.0,
                "bond_idx": -1,
                "bond_begin": -1,
                "bond_end": -1,
                "side": 2,
                "h_mode": -1,
                "source_flag": 9,
                "break_depth": 0,
                "ring_cut_flag": 0,
                "aux": aux,
            }
        )

    # Do not top-seed bonds. Only truncate if absolutely above max_events.
    if len(rows) > max_events:
        # Prefer real fragment events over precursor, lower m/z high-CE-friendly events are not forced here.
        rows = rows[:max_events]

    E = max_events
    A = len(atomicnos)

    event_formulae_features = np.zeros((E, A), dtype=np.float32)
    event_neutral_loss_features = np.zeros((E, A), dtype=np.float32)
    event_mz = np.zeros((E,), dtype=np.float32)
    event_official_idx = np.full((E,), -1, dtype=np.int64)
    event_official_intensity = np.zeros((E,), dtype=np.float32)
    event_bond_idx = np.full((E,), -1, dtype=np.int64)
    event_bond_begin = np.full((E,), -1, dtype=np.int64)
    event_bond_end = np.full((E,), -1, dtype=np.int64)
    event_side = np.full((E,), -1, dtype=np.int64)
    event_h_mode = np.full((E,), -1, dtype=np.int64)
    event_source_flag = np.zeros((E,), dtype=np.int8)
    event_break_depth = np.zeros((E,), dtype=np.int8)
    event_ring_cut_flag = np.zeros((E,), dtype=np.int8)
    event_aux_feat = np.zeros((E, 16), dtype=np.float32)
    event_mask = np.zeros((E,), dtype=np.float32)

    for i, r in enumerate(rows):
        event_formulae_features[i] = r["formula_counts"]
        event_neutral_loss_features[i] = r["neutral_loss_counts"]
        event_mz[i] = r["mz"]
        event_official_idx[i] = r["official_idx"]
        event_official_intensity[i] = r["official_intensity"]
        event_bond_idx[i] = r["bond_idx"]
        event_bond_begin[i] = r["bond_begin"]
        event_bond_end[i] = r["bond_end"]
        event_side[i] = r["side"]
        event_h_mode[i] = r["h_mode"]
        event_source_flag[i] = r["source_flag"]
        event_break_depth[i] = r["break_depth"]
        event_ring_cut_flag[i] = r["ring_cut_flag"]
        event_aux_feat[i] = r["aux"]
        event_mask[i] = 1.0

    return {
        "event_formulae_features": event_formulae_features,
        "event_neutral_loss_features": event_neutral_loss_features,
        "event_mz": event_mz,
        "event_official_idx": event_official_idx,
        "event_official_intensity": event_official_intensity,
        "event_bond_idx": event_bond_idx,
        "event_bond_begin": event_bond_begin,
        "event_bond_end": event_bond_end,
        "event_side": event_side,
        "event_h_mode": event_h_mode,
        "event_source_flag": event_source_flag,
        "event_break_depth": event_break_depth,
        "event_ring_cut_flag": event_ring_cut_flag,
        "event_aux_feat": event_aux_feat,
        "event_mask": event_mask,
        "event_n": np.int64(len(rows)),
    }