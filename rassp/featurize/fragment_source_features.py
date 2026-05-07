import os
import math
from collections import defaultdict, deque

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]

COMMON_ATOMS = [6, 7, 8, 15, 16, 9, 17, 35, 53]
FRAGMENT_LOCAL_AUX_DIM = 10 + (4 + 3 + len(COMMON_ATOMS) + 10 + 3 + 1 + 1) * 2


def _safe_float(x, default=0.0):
    try:
        x = float(x)
        if math.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def atom_total_h(atom):
    h = 0
    try:
        h += int(atom.GetTotalNumHs())
    except Exception:
        pass
    return max(0, h)


def formula_vec_from_atom_ids(mol, atom_ids, formula_atomicnos, h_shift=0):
    pos = {int(z): i for i, z in enumerate(formula_atomicnos)}
    out = np.zeros((len(formula_atomicnos),), dtype=np.int16)

    h_count = 0
    for ai in atom_ids:
        atom = mol.GetAtomWithIdx(int(ai))
        z = int(atom.GetAtomicNum())
        if z in pos:
            out[pos[z]] += 1
        h_count += atom_total_h(atom)

    if 1 in pos:
        out[pos[1]] = max(0, h_count + int(h_shift))

    return out


def get_brics_bond_set(mol):
    out = {}
    try:
        for item in BRICS.FindBRICSBonds(mol):
            bond_atoms, envs = item
            a, b = int(bond_atoms[0]), int(bond_atoms[1])
            key = tuple(sorted((a, b)))
            out[key] = tuple(str(x) for x in envs)
    except Exception:
        pass
    return out


def connected_components_after_cut(mol, cut_bond_idx):
    bond = mol.GetBondWithIdx(int(cut_bond_idx))
    a = bond.GetBeginAtomIdx()
    b = bond.GetEndAtomIdx()

    n = mol.GetNumAtoms()
    adj = [[] for _ in range(n)]
    for bd in mol.GetBonds():
        if int(bd.GetIdx()) == int(cut_bond_idx):
            continue
        u = bd.GetBeginAtomIdx()
        v = bd.GetEndAtomIdx()
        adj[u].append(v)
        adj[v].append(u)

    seen = [False] * n
    comps = []

    for start in [a, b]:
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []
        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    q.append(y)
        comps.append(sorted(comp))

    if len(comps) != 2:
        return []
    return comps

def connected_components_after_cuts(mol, cut_bond_indices):
    """
    Remove multiple bonds and return all connected components.

    Used for depth=2 simultaneous cuts:
      cut two bonds -> components can be 2 or 3 fragments.
    """
    cut_set = set(int(x) for x in cut_bond_indices)

    n = int(mol.GetNumAtoms())
    adj = [[] for _ in range(n)]

    for bd in mol.GetBonds():
        if int(bd.GetIdx()) in cut_set:
            continue
        u = int(bd.GetBeginAtomIdx())
        v = int(bd.GetEndAtomIdx())
        adj[u].append(v)
        adj[v].append(u)

    seen = [False] * n
    comps = []

    for start in range(n):
        if seen[start]:
            continue
        q = deque([start])
        seen[start] = True
        comp = []

        while q:
            x = q.popleft()
            comp.append(x)
            for y in adj[x]:
                if not seen[y]:
                    seen[y] = True
                    q.append(y)

        comps.append(sorted(comp))

    # If cuts do not split the graph, no useful source.
    if len(comps) <= 1:
        return []

    return comps

def connected_components_within_atom_set_after_cut(mol, atom_ids, cut_bond_idx):
    """
    Cut one bond inside a current atom subset and return connected components
    restricted to that subset.

    This is used for recursive fragment-node enumeration:
      full molecule -> cut -> fragment -> cut again -> smaller fragment
    """
    atom_set = set(int(x) for x in atom_ids)
    if len(atom_set) <= 1:
        return []

    cut_bond_idx = int(cut_bond_idx)

    nbs = {int(a): [] for a in atom_set}

    for bd in mol.GetBonds():
        bi = int(bd.GetIdx())
        if bi == cut_bond_idx:
            continue

        u = int(bd.GetBeginAtomIdx())
        v = int(bd.GetEndAtomIdx())

        if u not in atom_set or v not in atom_set:
            continue

        nbs[u].append(v)
        nbs[v].append(u)

    seen = set()
    comps = []

    for start in sorted(atom_set):
        if start in seen:
            continue

        q = deque([start])
        seen.add(start)
        comp = []

        while q:
            x = q.popleft()
            comp.append(x)
            for y in nbs.get(x, []):
                if y not in seen:
                    seen.add(y)
                    q.append(y)

        comps.append(sorted(comp))

    if len(comps) <= 1:
        return []

    return comps


def _bond_record_map(mol, brics_info):
    """
    Map bond idx -> heavy-heavy bond record.
    """
    out = {}
    for rec in _heavy_bond_records(mol, brics_info):
        out[int(rec["idx"])] = rec
    return out


def _subgraph_cut_bond_candidates(mol, atom_ids, bond_record_by_idx):
    """
    Return heavy-heavy bonds fully inside the current atom subset.
    """
    atom_set = set(int(x) for x in atom_ids)
    out = []

    for bd in mol.GetBonds():
        idx = int(bd.GetIdx())
        u = int(bd.GetBeginAtomIdx())
        v = int(bd.GetEndAtomIdx())

        if u not in atom_set or v not in atom_set:
            continue

        rec = bond_record_by_idx.get(idx, None)
        if rec is None:
            continue

        out.append(rec)

    return out


def _recursive_bond_priority(rec):
    """
    Higher priority bonds are expanded first.
    Keeps recursion bounded and chemically biased.
    """
    local = np.asarray(rec.get("local_feat", []), dtype=np.float32)
    stability = float(local[-1]) if local.ndim == 1 and local.size > 0 else 0.0

    is_brics = float(rec.get("is_brics", 0.0))
    ring_cut = float(rec.get("ring_cut", 0.0))

    # Prefer BRICS / hetero / stable contexts, penalize ring cuts.
    return (
        is_brics,
        stability,
        -ring_cut,
    )


def _avg_local_feats(local_feats):
    if len(local_feats) <= 0:
        local_d = (int(FRAGMENT_LOCAL_AUX_DIM) - 10) // 2
        return np.zeros((local_d,), dtype=np.float32)

    arr = np.stack(
        [np.asarray(x, dtype=np.float32).reshape(-1) for x in local_feats],
        axis=0,
    )
    return np.mean(arr, axis=0).astype(np.float32)

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return int(default)


def _heavy_bond_records(mol, brics_info):
    """
    Return heavy-heavy bond records with local bond features precomputed.
    """
    records = []
    for bond in mol.GetBonds():
        a = bond.GetBeginAtom()
        b = bond.GetEndAtom()
        if int(a.GetAtomicNum()) == 1 or int(b.GetAtomicNum()) == 1:
            continue

        idx = int(bond.GetIdx())
        bond_key = tuple(sorted((int(bond.GetBeginAtomIdx()), int(bond.GetEndAtomIdx()))))

        records.append({
            "idx": idx,
            "bond_key": bond_key,
            "is_brics": 1.0 if bond_key in brics_info else 0.0,
            "ring_cut": 1.0 if bond.IsInRing() else 0.0,
            "local_feat": local_bond_feature(mol, bond, brics_info=brics_info),
        })

    return records


def enumerate_depth2_bond_sources(
    mol,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
    max_depth2_bond_pairs=1200,
    max_sources=100000,
):
    """
    Depth=2 source enumeration.

    It cuts two heavy-heavy bonds at once and treats every resulting
    connected component as a possible fragment source.

    This is still formula-level aggregated source information, not
    source-instance candidate generation.
    """
    brics_info = get_brics_bond_set(mol)
    bond_records = _heavy_bond_records(mol, brics_info=brics_info)

    out = []
    seen = set()

    n_bonds = len(bond_records)
    if n_bonds < 2:
        return out

    pair_count = 0
    total_atoms = int(mol.GetNumAtoms())

    for i in range(n_bonds):
        rec_i = bond_records[i]
        for j in range(i + 1, n_bonds):
            rec_j = bond_records[j]

            pair_count += 1
            if pair_count > int(max_depth2_bond_pairs):
                return out

            cut_idxs = [int(rec_i["idx"]), int(rec_j["idx"])]
            comps = connected_components_after_cuts(mol, cut_idxs)
            if len(comps) <= 1:
                continue

            # Combine the two local bond features into a path/cut-pair feature.
            local_feat = 0.5 * (
                np.asarray(rec_i["local_feat"], dtype=np.float32)
                + np.asarray(rec_j["local_feat"], dtype=np.float32)
            )

            is_brics = 1.0 if (rec_i["is_brics"] > 0.5 or rec_j["is_brics"] > 0.5) else 0.0
            ring_cut = 1.0 if (rec_i["ring_cut"] > 0.5 or rec_j["ring_cut"] > 0.5) else 0.0

            for comp in comps:
                comp = list(comp)
                if len(comp) <= 0:
                    continue

                # Skip the whole molecule; keep small fragments because low-m/z ions matter.
                if len(comp) >= total_atoms:
                    continue

                atom_tuple = tuple(int(x) for x in sorted(comp))

                for hs in range(int(h_shift_min), int(h_shift_max) + 1):
                    fv = formula_vec_from_atom_ids(
                        mol,
                        comp,
                        formula_atomicnos,
                        h_shift=hs,
                    )
                    if np.any(fv < 0):
                        continue

                    formula_key = tuple(int(x) for x in fv.tolist())
                    dedup_key = (formula_key, atom_tuple, int(hs), int(rec_i["idx"]), int(rec_j["idx"]))
                    if dedup_key in seen:
                        continue
                    seen.add(dedup_key)

                    out.append({
                        "formula_key": formula_key,
                        "atom_ids": comp,
                        "cut_bond_idx": tuple(cut_idxs),
                        "depth": 2,
                        "h_shift": int(hs),
                        "is_brics": is_brics,
                        "ring_cut": ring_cut,
                        "local_feat": local_feat,
                        'source_type': 2,
                    })

                    if len(out) >= int(max_sources):
                        return out

    return out


def enumerate_recursive_subgraph_sources(
    mol,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
    max_depth=3,
    max_branch_per_node=8,
    max_nodes=200000,
    min_heavy_atoms=1,
):
    """
    V3B recursive connected-subgraph fragment enumeration.

    This differs from current depth=2:
      current depth=2 cuts two bonds in the original molecule at once;
      this recursively cuts inside the already generated fragment.

    It creates true fragment-path candidates closer to a fragmentation DAG.
    """
    brics_info = get_brics_bond_set(mol)
    bond_record_by_idx = _bond_record_map(mol, brics_info)

    heavy_nodes = [
        int(a.GetIdx())
        for a in mol.GetAtoms()
        if int(a.GetAtomicNum()) > 1
    ]

    if len(heavy_nodes) <= 0:
        return []

    total_heavy_n = len(heavy_nodes)

    out = []
    seen_record = set()
    seen_expand = set()

    def _emit_fragment(atom_ids, cut_path, depth, local_path):
        atom_ids = sorted(int(x) for x in atom_ids)

        if len(atom_ids) < int(min_heavy_atoms):
            return

        if len(atom_ids) >= total_heavy_n:
            return

        atom_tuple = tuple(atom_ids)
        cut_tuple = tuple(int(x) for x in cut_path)

        local_feat = _avg_local_feats(local_path)

        is_brics = 0.0
        ring_cut = 0.0
        for ci in cut_tuple:
            rec = bond_record_by_idx.get(int(ci), None)
            if rec is None:
                continue
            is_brics = max(is_brics, float(rec.get("is_brics", 0.0)))
            ring_cut = max(ring_cut, float(rec.get("ring_cut", 0.0)))

        for hs in range(int(h_shift_min), int(h_shift_max) + 1):
            fv = formula_vec_from_atom_ids(
                mol,
                atom_ids,
                formula_atomicnos,
                h_shift=hs,
            )
            if np.any(fv < 0):
                continue

            formula_key = tuple(int(x) for x in fv.tolist())
            rec_key = (atom_tuple, cut_tuple, int(hs))

            if rec_key in seen_record:
                continue

            seen_record.add(rec_key)

            out.append({
                "formula_key": formula_key,
                "atom_ids": list(atom_ids),
                "cut_bond_idx": cut_tuple,
                "depth": int(depth),
                "h_shift": int(hs),
                "is_brics": float(is_brics),
                "ring_cut": float(ring_cut),
                "local_feat": local_feat,
                'source_type': 3,
            })

            if len(out) >= int(max_nodes):
                return

    def _recurse(atom_ids, depth, cut_path, local_path):
        if len(out) >= int(max_nodes):
            return

        atom_ids = tuple(sorted(int(x) for x in atom_ids))
        expand_key = (atom_ids, int(depth), tuple(int(x) for x in cut_path))

        if expand_key in seen_expand:
            return
        seen_expand.add(expand_key)

        if int(depth) >= int(max_depth):
            return

        candidates = _subgraph_cut_bond_candidates(
            mol,
            atom_ids,
            bond_record_by_idx,
        )

        if len(candidates) <= 0:
            return

        candidates = sorted(candidates, key=_recursive_bond_priority, reverse=True)
        candidates = candidates[:max(1, int(max_branch_per_node))]

        for rec in candidates:
            cut_idx = int(rec["idx"])

            comps = connected_components_within_atom_set_after_cut(
                mol,
                atom_ids,
                cut_idx,
            )

            if len(comps) <= 1:
                continue

            new_cut_path = tuple(list(cut_path) + [cut_idx])
            new_local_path = list(local_path) + [np.asarray(rec["local_feat"], dtype=np.float32)]

            for comp in comps:
                if len(out) >= int(max_nodes):
                    return

                comp = sorted(int(x) for x in comp)
                if len(comp) <= 0:
                    continue

                # Do not emit the unchanged current fragment.
                if tuple(comp) == tuple(atom_ids):
                    continue

                new_depth = int(depth) + 1

                _emit_fragment(
                    comp,
                    cut_path=new_cut_path,
                    depth=new_depth,
                    local_path=new_local_path,
                )

                _recurse(
                    comp,
                    depth=new_depth,
                    cut_path=new_cut_path,
                    local_path=new_local_path,
                )

    _recurse(
        heavy_nodes,
        depth=0,
        cut_path=tuple(),
        local_path=[],
    )

    return out

def enumerate_fragment_sources(
    mol,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
    max_depth=1,
    max_depth2_bond_pairs=1200,
    max_sources=100000,
):
    """
    Unified source enumerator.

    depth=1: current single-bond source.
    depth=2: additional simultaneous two-bond source.
    """
    sources = enumerate_single_bond_sources(
        mol,
        formula_atomicnos=formula_atomicnos,
        h_shift_min=h_shift_min,
        h_shift_max=h_shift_max,
    )

    if int(max_depth) >= 2:
        depth2_sources = enumerate_depth2_bond_sources(
            mol,
            formula_atomicnos=formula_atomicnos,
            h_shift_min=h_shift_min,
            h_shift_max=h_shift_max,
            max_depth2_bond_pairs=max_depth2_bond_pairs,
            max_sources=max(1, int(max_sources) - len(sources)),
        )
        sources.extend(depth2_sources)

    if len(sources) > int(max_sources):
        sources = sources[:int(max_sources)]

    return sources

def local_bond_feature(mol, bond, brics_info=None):
    a = bond.GetBeginAtom()
    b = bond.GetEndAtom()
    ai, bi = int(a.GetIdx()), int(b.GetIdx())
    key = tuple(sorted((ai, bi)))

    feat = []

    bt = bond.GetBondType()
    feat.extend([1.0 if bt == x else 0.0 for x in BOND_TYPES])
    feat.append(1.0 if bond.IsInRing() else 0.0)
    feat.append(1.0 if bond.GetIsAromatic() else 0.0)
    feat.append(1.0 if bond.GetIsConjugated() else 0.0)

    za, zb = int(a.GetAtomicNum()), int(b.GetAtomicNum())
    for z in COMMON_ATOMS:
        feat.append(float((za == z) or (zb == z)))

    for atom in [a, b]:
        feat.append(float(atom.GetDegree()) / 4.0)
        feat.append(float(atom.GetFormalCharge()))
        feat.append(1.0 if atom.GetIsAromatic() else 0.0)
        feat.append(1.0 if atom.IsInRing() else 0.0)
        feat.append(float(atom_total_h(atom)) / 4.0)

    hetero_nei = 0
    aromatic_nei = 0
    ring_nei = 0
    for atom in [a, b]:
        for nb in atom.GetNeighbors():
            z = int(nb.GetAtomicNum())
            if z not in (1, 6):
                hetero_nei += 1
            if nb.GetIsAromatic():
                aromatic_nei += 1
            if nb.IsInRing():
                ring_nei += 1

    feat.append(min(hetero_nei, 6) / 6.0)
    feat.append(min(aromatic_nei, 6) / 6.0)
    feat.append(min(ring_nei, 6) / 6.0)

    is_brics = brics_info is not None and key in brics_info
    feat.append(1.0 if is_brics else 0.0)

    stability = 0.0
    if a.GetIsAromatic() or b.GetIsAromatic():
        stability += 0.4
    if za in (7, 8, 15, 16) or zb in (7, 8, 15, 16):
        stability += 0.3
    if bond.GetIsConjugated():
        stability += 0.2
    if bond.IsInRing():
        stability -= 0.1
    feat.append(stability)

    return np.asarray(feat, dtype=np.float32)


def enumerate_single_bond_sources(
    mol,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
):
    brics_info = get_brics_bond_set(mol)
    out = []

    for bond in mol.GetBonds():
        if bond.GetBeginAtom().GetAtomicNum() == 1 or bond.GetEndAtom().GetAtomicNum() == 1:
            continue

        comps = connected_components_after_cut(mol, bond.GetIdx())
        if len(comps) != 2:
            continue

        bond_feat = local_bond_feature(mol, bond, brics_info=brics_info)
        bond_key = tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))

        for comp in comps:
            comp = list(comp)
            for hs in range(int(h_shift_min), int(h_shift_max) + 1):
                fv = formula_vec_from_atom_ids(mol, comp, formula_atomicnos, h_shift=hs)
                if np.any(fv < 0):
                    continue
                out.append({
                    'formula_key': tuple(int(x) for x in fv.tolist()),
                    'atom_ids': comp,
                    'cut_bond_idx': int(bond.GetIdx()),
                    'depth': 1,
                    'h_shift': int(hs),
                    'is_brics': 1.0 if bond_key in brics_info else 0.0,
                    'ring_cut': 1.0 if bond.IsInRing() else 0.0,
                    'local_feat': bond_feat,
                    'source_type': 1,
                })

    return out


def aggregate_source_features_for_formulae(
    mol,
    formulae_features,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
    max_source_count_clip=16,
):
    formulae_features = np.asarray(formulae_features, dtype=np.int16)
    M = int(formulae_features.shape[0])
    if M <= 0:
        return np.zeros((0, FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)

    if formula_atomicnos is None:
        return np.zeros((M, FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)

    try:
        formula_atomicnos = [int(x) for x in list(formula_atomicnos)]
    except Exception:
        return np.zeros((M, FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)

    if len(formula_atomicnos) <= 0:
        return np.zeros((M, FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    max_depth = _env_int("FRAG_AUX_MAX_DEPTH", 1)
    max_depth2_bond_pairs = _env_int("FRAG_AUX_MAX_DEPTH2_BOND_PAIRS", 1200)
    max_sources = _env_int("FRAG_AUX_MAX_SOURCES", 100000)

    sources = enumerate_fragment_sources(
        mol,
        formula_atomicnos=formula_atomicnos,
        h_shift_min=h_shift_min,
        h_shift_max=h_shift_max,
        max_depth=max_depth,
        max_depth2_bond_pairs=max_depth2_bond_pairs,
        max_sources=max_sources,
    )

    # if os.environ.get("FRAGMENT_NODE_USE_RECURSIVE_SUBGRAPHS", "0").strip() == "1":
    #     rec_sources = enumerate_recursive_subgraph_sources(
    #         mol,
    #         formula_atomicnos=formula_atomicnos,
    #         h_shift_min=int(h_shift_min),
    #         h_shift_max=int(h_shift_max),
    #         max_depth=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MAX_DEPTH", "3")),
    #         max_branch_per_node=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_BRANCH", "8")),
    #         max_nodes=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MAX_SOURCES", "200000")),
    #         min_heavy_atoms=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MIN_HEAVY", "1")),
    #     )

    #     sources.extend(rec_sources)

    by_formula = defaultdict(list)
    for s in sources:
        by_formula[s['formula_key']].append(s)

    out = np.zeros((M, FRAGMENT_LOCAL_AUX_DIM), dtype=np.float32)
    for i in range(M):
        key = tuple(int(x) for x in formulae_features[i].tolist())
        ss = by_formula.get(key, [])
        if len(ss) == 0:
            continue

        cnt = len(ss)
        cnt_clip = min(cnt, max_source_count_clip)
        h_abs = np.asarray([abs(int(s['h_shift'])) for s in ss], dtype=np.float32)
        brics = np.asarray([float(s['is_brics']) for s in ss], dtype=np.float32)
        depths = np.asarray([float(s['depth']) for s in ss], dtype=np.float32)
        rings = np.asarray([float(s['ring_cut']) for s in ss], dtype=np.float32)
        local = np.stack([s['local_feat'] for s in ss], axis=0).astype(np.float32)

        out[i, 0] = 1.0
        out[i, 1] = np.log1p(cnt)
        out[i, 2] = cnt_clip / float(max_source_count_clip)
        out[i, 3] = float(np.min(h_abs)) / 4.0
        out[i, 4] = float(np.mean(h_abs)) / 4.0
        out[i, 5] = 1.0 if np.any(brics > 0.5) else 0.0
        out[i, 6] = min(float(np.sum(brics)), max_source_count_clip) / float(max_source_count_clip)
        out[i, 7] = float(np.min(depths)) / 4.0
        out[i, 8] = 1.0 if np.any(rings > 0.5) else 0.0
        out[i, 9] = float(np.mean(rings))
        out[i, 10:10 + local.shape[1]] = np.mean(local, axis=0)
        out[i, 10 + local.shape[1]:10 + local.shape[1] * 2] = np.max(local, axis=0)

    return out.astype(np.float32)



def source_record_to_frag_aux(source, max_source_count_clip=16):
    """
    Convert one source record into the same 72-dim frag aux format used by
    aggregate_source_features_for_formulae().

    For a source instance:
      has_source = 1
      source_count = 1
      local_mean = local_feat
      local_max = local_feat
    """
    local = np.asarray(source.get("local_feat", None), dtype=np.float32)
    if local.ndim != 1 or local.size <= 0:
        # Derive local_d from FRAGMENT_LOCAL_AUX_DIM.
        local_d = (int(FRAGMENT_LOCAL_AUX_DIM) - 10) // 2
        local = np.zeros((local_d,), dtype=np.float32)

    local_d = int(local.shape[0])
    D = 10 + local_d * 2
    out = np.zeros((D,), dtype=np.float32)

    h_shift = int(source.get("h_shift", 0))
    depth = float(source.get("depth", 1))
    is_brics = float(source.get("is_brics", 0.0))
    ring_cut = float(source.get("ring_cut", 0.0))

    out[0] = 1.0                                  # has_source
    out[1] = np.log1p(1.0)                         # log_source_count
    out[2] = 1.0 / float(max_source_count_clip)    # source_count_norm
    out[3] = abs(float(h_shift)) / 4.0             # min_abs_h_shift_norm
    out[4] = abs(float(h_shift)) / 4.0             # mean_abs_h_shift_norm
    out[5] = 1.0 if is_brics > 0.5 else 0.0        # has_brics
    out[6] = (1.0 if is_brics > 0.5 else 0.0) / float(max_source_count_clip)
    out[7] = depth / 4.0                           # min_depth_norm
    out[8] = 1.0 if ring_cut > 0.5 else 0.0        # ring_cut_any
    out[9] = 1.0 if ring_cut > 0.5 else 0.0        # ring_cut_frac

    out[10:10 + local_d] = local
    out[10 + local_d:10 + local_d * 2] = local

    # Pad/crop to the global dim, to keep model input stable.
    if D < int(FRAGMENT_LOCAL_AUX_DIM):
        padded = np.zeros((int(FRAGMENT_LOCAL_AUX_DIM),), dtype=np.float32)
        padded[:D] = out
        return padded
    return out[:int(FRAGMENT_LOCAL_AUX_DIM)].astype(np.float32)

def build_source_instance_records_for_formulae(
    mol,
    formulae_features,
    formula_atomicnos,
    h_shift_min=-4,
    h_shift_max=4,
    max_depth=2,
    max_per_formula=3,
    max_total_source_instances=2048,
    max_depth2_bond_pairs=1200,
    max_sources=100000,
):
    """
    Build source-instance records for existing formula candidates.

    Returns a list of records:
      {
        orig_idx: int,              # index of original formula candidate
        frag_aux: np.ndarray[D],    # source-specific 72-d feature
        source_depth: int,
        source_h_shift: int,
        source_is_brics: float,
        source_ring_cut: float,
      }

    Important:
      This function does not create new formulas.
      It only creates source instances for formulas already present in formulae_features.
    """
    formulae_features = np.asarray(formulae_features, dtype=np.int16)
    M = int(formulae_features.shape[0])
    if M <= 0:
        return []

    if formula_atomicnos is None:
        return []

    try:
        formula_atomicnos = [int(x) for x in list(formula_atomicnos)]
    except Exception:
        return []

    # Map formula vector -> first original formula index.
    # If duplicate formula rows exist, keep the first valid one.
    formula_to_idx = {}
    for i in range(M):
        key = tuple(int(x) for x in formulae_features[i].tolist())
        if key not in formula_to_idx:
            formula_to_idx[key] = int(i)

    sources = enumerate_fragment_sources(
        mol,
        formula_atomicnos=formula_atomicnos,
        h_shift_min=h_shift_min,
        h_shift_max=h_shift_max,
        max_depth=max_depth,
        max_depth2_bond_pairs=max_depth2_bond_pairs,
        max_sources=max_sources,
    )

    by_formula = defaultdict(list)
    for s in sources:
        key = tuple(int(x) for x in s.get("formula_key", ()))
        if key in formula_to_idx:
            by_formula[key].append(s)

    def _source_sort_key(s):
        # Higher is better.
        local = np.asarray(s.get("local_feat", []), dtype=np.float32)
        local_stability = float(local[-1]) if local.ndim == 1 and local.size > 0 else 0.0
        is_brics = float(s.get("is_brics", 0.0))
        depth = int(s.get("depth", 9))
        abs_h = abs(int(s.get("h_shift", 0)))
        ring_cut = float(s.get("ring_cut", 0.0))

        return (
            is_brics,                  # prefer BRICS source
            local_stability,            # prefer stable local environment
            -float(abs_h),              # prefer smaller H shift
            -float(depth),              # prefer shallower depth if otherwise tied
            -float(ring_cut),           # prefer non-ring cut if otherwise tied
        )

    records = []
    for key, ss in by_formula.items():
        if len(ss) <= 0:
            continue

        ss = sorted(ss, key=_source_sort_key, reverse=True)
        ss = ss[:max(1, int(max_per_formula))]

        orig_idx = int(formula_to_idx[key])
        for s in ss:
            records.append({
                "orig_idx": orig_idx,
                "frag_aux": source_record_to_frag_aux(s),
                "source_depth": int(s.get("depth", 1)),
                "source_h_shift": int(s.get("h_shift", 0)),
                "source_is_brics": float(s.get("is_brics", 0.0)),
                "source_ring_cut": float(s.get("ring_cut", 0.0)),
            })

            if len(records) >= int(max_total_source_instances):
                return records

    return records

def formula_mass_from_vec(formula_vec, formula_atomicnos):
    """
    Exact isotope mass from formula vector and atomic numbers.
    formula_vec and formula_atomicnos must have the same length.
    """
    pt = Chem.GetPeriodicTable()
    fv = np.asarray(formula_vec, dtype=np.float64).reshape(-1)

    mass = 0.0
    for cnt, z in zip(fv.tolist(), list(formula_atomicnos)):
        c = float(cnt)
        if c <= 0:
            continue
        mass += c * float(pt.GetMostCommonIsotopeMass(int(z)))

    return float(mass)


def _normalize_cut_bond_ids(x):
    if x is None:
        return tuple()
    if isinstance(x, (list, tuple, np.ndarray)):
        return tuple(int(v) for v in list(x))
    try:
        return (int(x),)
    except Exception:
        return tuple()


def _source_to_fragment_node_sort_key(source):
    """
    Higher is better.
    Used only to make cache deterministic and keep stronger sources first.
    """
    local = np.asarray(source.get("local_feat", []), dtype=np.float32)
    local_stability = float(local[-1]) if local.ndim == 1 and local.size > 0 else 0.0

    is_brics = float(source.get("is_brics", 0.0))
    depth = int(source.get("depth", 9))
    abs_h = abs(int(source.get("h_shift", 0)))
    ring_cut = float(source.get("ring_cut", 0.0))
    atom_n = len(source.get("atom_ids", []))

    return (
        is_brics,
        local_stability,
        -float(abs_h),
        -float(depth),
        -float(ring_cut),
        float(atom_n),
    )


def _fragment_node_record_dedup_key(source):
    formula_key = tuple(int(x) for x in source.get("formula_key", ()))
    atom_ids = tuple(sorted(int(x) for x in source.get("atom_ids", [])))
    cut_ids = _normalize_cut_bond_ids(source.get("cut_bond_idx", None))
    h_shift = int(source.get("h_shift", 0))
    return (formula_key, atom_ids, cut_ids, h_shift)


def _fragment_node_record_bin(
    source,
    formula_atomicnos,
    official_bin_width,
    official_max_mz,
    add_proton=True,
    proton_mass=1.007276466812,
):
    try:
        formula_key = tuple(int(x) for x in source.get("formula_key", ()))
        if len(formula_key) != len(formula_atomicnos):
            return -1

        mz = formula_mass_from_vec(formula_key, formula_atomicnos)
        if bool(add_proton):
            mz += float(proton_mass)

        if not np.isfinite(mz) or mz <= 0.0 or mz >= float(official_max_mz):
            return -1

        bw = float(max(1e-6, float(official_bin_width)))
        return int(np.floor(float(mz) / bw + 1e-8))
    except Exception:
        return -1


def _bin_diverse_select_fragment_records(
    records,
    max_n,
    formula_atomicnos,
    official_bin_width,
    official_max_mz,
    add_proton=True,
    proton_mass=1.007276466812,
    max_per_bin=4,
    selected_keys=None,
    bin_counts=None,
):
    """
    Select records with per-bin diversity.

    First pass:
      keep at most max_per_bin records per official bin.

    This prevents recursive candidates from filling 8192 slots with many
    structurally different nodes that all land on the same few m/z bins.
    """
    max_n = int(max_n)
    if max_n <= 0 or len(records) <= 0:
        return [], selected_keys if selected_keys is not None else set(), bin_counts if bin_counts is not None else {}

    if selected_keys is None:
        selected_keys = set()
    if bin_counts is None:
        bin_counts = {}

    selected = []
    records_sorted = sorted(records, key=_source_to_fragment_node_sort_key, reverse=True)

    for s in records_sorted:
        if len(selected) >= max_n:
            break

        k = _fragment_node_record_dedup_key(s)
        if k in selected_keys:
            continue

        b = _fragment_node_record_bin(
            s,
            formula_atomicnos=formula_atomicnos,
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            add_proton=add_proton,
            proton_mass=proton_mass,
        )
        if b < 0:
            continue

        c = int(bin_counts.get(int(b), 0))
        if c >= int(max_per_bin):
            continue

        selected.append(s)
        selected_keys.add(k)
        bin_counts[int(b)] = c + 1

    return selected, selected_keys, bin_counts


def _select_fragment_node_records_balanced(
    records,
    max_nodes,
    formula_atomicnos,
    official_bin_width,
    official_max_mz,
    add_proton=True,
    proton_mass=1.007276466812,
):
    """
    V3C candidate selection.

    Protect base fragments first:
      source_type 1 = single-bond
      source_type 2 = simultaneous two-bond
      source_type 3 = recursive subgraph

    Recursive fragments are useful, but they must not crowd out the base
    candidates that already had better oracle coverage.
    """
    max_nodes = int(max_nodes)
    if max_nodes <= 0:
        return []

    if os.environ.get("FRAGMENT_NODE_BALANCE_SOURCE_TYPES", "0").strip() != "1":
        return sorted(records, key=_source_to_fragment_node_sort_key, reverse=True)[:max_nodes]

    base_records = [
        s for s in records
        if int(float(s.get("source_type", 0))) in (1, 2)
    ]
    recursive_records = [
        s for s in records
        if int(float(s.get("source_type", 0))) == 3
    ]
    other_records = [
        s for s in records
        if int(float(s.get("source_type", 0))) not in (1, 2, 3)
    ]

    base_reserve = int(os.environ.get("FRAGMENT_NODE_BASE_RESERVE", "4096"))
    recursive_reserve = int(os.environ.get("FRAGMENT_NODE_RECURSIVE_RESERVE", "4096"))

    base_max_per_bin = int(os.environ.get("FRAGMENT_NODE_BASE_MAX_PER_BIN", "8"))
    recursive_max_per_bin = int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MAX_PER_BIN", "4"))

    selected = []
    selected_keys = set()
    bin_counts = {}

    # 1) Keep base candidates first.
    base_take = min(base_reserve, max_nodes)
    base_sel, selected_keys, bin_counts = _bin_diverse_select_fragment_records(
        base_records,
        max_n=base_take,
        formula_atomicnos=formula_atomicnos,
        official_bin_width=official_bin_width,
        official_max_mz=official_max_mz,
        add_proton=add_proton,
        proton_mass=proton_mass,
        max_per_bin=base_max_per_bin,
        selected_keys=selected_keys,
        bin_counts=bin_counts,
    )
    selected.extend(base_sel)

    # 2) Add recursive candidates as supplement.
    remaining = max_nodes - len(selected)
    if remaining > 0:
        rec_take = min(recursive_reserve, remaining)
        rec_sel, selected_keys, bin_counts = _bin_diverse_select_fragment_records(
            recursive_records,
            max_n=rec_take,
            formula_atomicnos=formula_atomicnos,
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            add_proton=add_proton,
            proton_mass=proton_mass,
            max_per_bin=recursive_max_per_bin,
            selected_keys=selected_keys,
            bin_counts=bin_counts,
        )
        selected.extend(rec_sel)

    # 3) Fill remaining slots with ordinary score order, but do not duplicate exact structural records.
    if os.environ.get("FRAGMENT_NODE_FILL_REMAINING", "1").strip() == "1":
        if len(selected) < max_nodes:
            all_sorted = sorted(
                list(base_records) + list(recursive_records) + list(other_records),
                key=_source_to_fragment_node_sort_key,
                reverse=True,
            )

            for s in all_sorted:
                if len(selected) >= max_nodes:
                    break

                k = _fragment_node_record_dedup_key(s)
                if k in selected_keys:
                    continue

                b = _fragment_node_record_bin(
                    s,
                    formula_atomicnos=formula_atomicnos,
                    official_bin_width=official_bin_width,
                    official_max_mz=official_max_mz,
                    add_proton=add_proton,
                    proton_mass=proton_mass,
                )
                if b < 0:
                    continue

                selected.append(s)
                selected_keys.add(k)

    return selected[:max_nodes]

def build_fragment_node_candidate_tensors(
    mol,
    formula_atomicnos,
    official_bin_width=0.01,
    official_max_mz=1005.0,
    precursor_mz=None,
    h_shift_min=-4,
    h_shift_max=4,
    max_depth=2,
    max_depth2_bond_pairs=1200,
    max_sources=100000,
    max_nodes=4096,
    add_proton=True,
    proton_mass=1.007276466812,
    filter_precursor=True,
    precursor_tol_da=0.05,
):
    """
    V3A fragment-node candidates.

    Difference from V2:
      V2 candidate = formula + source_instance, but peak/template still copied from formula.
      V3 candidate = one actual fragment source / break instance / h-shift instance.

    Returned tensors are fixed-size [max_nodes, ...] with fragment_node_mask.
    """
    if formula_atomicnos is None:
        formula_atomicnos = []

    try:
        formula_atomicnos = [int(x) for x in list(formula_atomicnos)]
    except Exception:
        formula_atomicnos = []

    if len(formula_atomicnos) <= 0:
        local_d = (int(FRAGMENT_LOCAL_AUX_DIM) - 10) // 2
        return {
            "fragment_node_formula": np.zeros((int(max_nodes), 0), dtype=np.int16),
            "fragment_node_mz": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_official_idx": np.full((int(max_nodes),), -1, dtype=np.int64),
            "fragment_node_intensity": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_local_feat": np.zeros((int(max_nodes), local_d), dtype=np.float32),
            "fragment_node_depth": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_h_shift": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_is_brics": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_ring_cut": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_atom_count": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_cut_count": np.zeros((int(max_nodes),), dtype=np.float32),
            "fragment_node_group_formula_id": np.full((int(max_nodes),), -1, dtype=np.int32),
            "fragment_node_mask": np.zeros((int(max_nodes),), dtype=np.float32),
        }

    try:
        mol.UpdatePropertyCache(strict=False)
    except Exception:
        pass

    max_nodes = int(max_nodes)
    bw = float(max(1e-6, float(official_bin_width)))
    max_mz = float(max(bw, float(official_max_mz)))

    sources = enumerate_fragment_sources(
        mol,
        formula_atomicnos=formula_atomicnos,
        h_shift_min=int(h_shift_min),
        h_shift_max=int(h_shift_max),
        max_depth=int(max_depth),
        max_depth2_bond_pairs=int(max_depth2_bond_pairs),
        max_sources=int(max_sources),
    )

    # ------------------------------------------------------------------
    # V3B: recursive connected-subgraph fragment candidates.
    #
    # Important:
    # This must be inside build_fragment_node_candidate_tensors(), not only
    # aggregate_source_features_for_formulae(), because fragment_node_* cache
    # fields are built from this local `sources` list.
    # ------------------------------------------------------------------
    if os.environ.get("FRAGMENT_NODE_USE_RECURSIVE_SUBGRAPHS", "0").strip() == "1":
        rec_sources = enumerate_recursive_subgraph_sources(
            mol,
            formula_atomicnos=formula_atomicnos,
            h_shift_min=int(h_shift_min),
            h_shift_max=int(h_shift_max),
            max_depth=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MAX_DEPTH", "3")),
            max_branch_per_node=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_BRANCH", "8")),
            max_nodes=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MAX_SOURCES", "200000")),
            min_heavy_atoms=int(os.environ.get("FRAGMENT_NODE_RECURSIVE_MIN_HEAVY", "1")),
        )

        if os.environ.get("DEBUG_FRAGMENT_NODE_RECURSIVE", "0").strip() == "1":
            print(
                "[FRAGMENT_NODE_RECURSIVE]",
                "base_sources=", len(sources),
                "rec_sources=", len(rec_sources),
                "total_sources=", len(sources) + len(rec_sources),
                flush=True,
            )

        sources.extend(rec_sources)

    # Dedup by actual structural source + formula + h-shift.
    seen = set()
    records = []

    for s in sources:
        formula_key = tuple(int(x) for x in s.get("formula_key", ()))
        atom_ids = tuple(sorted(int(x) for x in s.get("atom_ids", [])))
        cut_ids = _normalize_cut_bond_ids(s.get("cut_bond_idx", None))
        h_shift = int(s.get("h_shift", 0))

        if len(formula_key) != len(formula_atomicnos):
            continue
        if len(atom_ids) <= 0:
            continue

        key = (formula_key, atom_ids, cut_ids, h_shift)
        if key in seen:
            continue
        seen.add(key)

        mz = formula_mass_from_vec(formula_key, formula_atomicnos)
        if bool(add_proton):
            mz += float(proton_mass)

        if not np.isfinite(mz) or mz <= 0.0 or mz >= max_mz:
            continue

        if bool(filter_precursor) and precursor_mz is not None:
            try:
                pmz = float(precursor_mz)
                if np.isfinite(pmz) and mz > (pmz + float(precursor_tol_da)):
                    continue
            except Exception:
                pass

        records.append(s)

    records = _select_fragment_node_records_balanced(
        records,
        max_nodes=max_nodes,
        formula_atomicnos=formula_atomicnos,
        official_bin_width=bw,
        official_max_mz=max_mz,
        add_proton=bool(add_proton),
        proton_mass=float(proton_mass),
    )

    local_d = (int(FRAGMENT_LOCAL_AUX_DIM) - 10) // 2

    formula_arr = np.zeros((max_nodes, len(formula_atomicnos)), dtype=np.int16)
    mz_arr = np.zeros((max_nodes,), dtype=np.float32)
    official_idx = np.full((max_nodes,), -1, dtype=np.int64)
    intensity = np.zeros((max_nodes,), dtype=np.float32)

    local_feat = np.zeros((max_nodes, local_d), dtype=np.float32)
    depth = np.zeros((max_nodes,), dtype=np.float32)
    h_shift_arr = np.zeros((max_nodes,), dtype=np.float32)
    is_brics = np.zeros((max_nodes,), dtype=np.float32)
    ring_cut = np.zeros((max_nodes,), dtype=np.float32)
    atom_count = np.zeros((max_nodes,), dtype=np.float32)
    cut_count = np.zeros((max_nodes,), dtype=np.float32)
    source_type = np.zeros((max_nodes,), dtype=np.float32)
    group_formula_id = np.full((max_nodes,), -1, dtype=np.int32)
    mask = np.zeros((max_nodes,), dtype=np.float32)

    formula_to_gid = {}

    for i, s in enumerate(records):
        formula_key = tuple(int(x) for x in s.get("formula_key", ()))
        fv = np.asarray(formula_key, dtype=np.int16).reshape(-1)

        mz = formula_mass_from_vec(fv, formula_atomicnos)
        if bool(add_proton):
            mz += float(proton_mass)

        bin_idx = int(np.floor(float(mz) / bw + 1e-8))
        if bin_idx < 0 or bin_idx >= int(np.floor(max_mz / bw)):
            continue

        if formula_key not in formula_to_gid:
            formula_to_gid[formula_key] = len(formula_to_gid)

        loc = np.asarray(s.get("local_feat", np.zeros((local_d,), dtype=np.float32)), dtype=np.float32).reshape(-1)
        if loc.shape[0] < local_d:
            loc2 = np.zeros((local_d,), dtype=np.float32)
            loc2[:loc.shape[0]] = loc
            loc = loc2
        else:
            loc = loc[:local_d]

        formula_arr[i, :] = fv
        mz_arr[i] = float(mz)
        official_idx[i] = int(bin_idx)
        intensity[i] = 1.0

        local_feat[i] = loc.astype(np.float32)
        depth[i] = float(s.get("depth", 0.0))
        h_shift_arr[i] = float(s.get("h_shift", 0.0))
        is_brics[i] = float(s.get("is_brics", 0.0))
        ring_cut[i] = float(s.get("ring_cut", 0.0))
        atom_count[i] = float(len(s.get("atom_ids", [])))
        cut_count[i] = float(len(_normalize_cut_bond_ids(s.get("cut_bond_idx", None))))
        source_type[i] = float(s.get("source_type", 0.0))
        group_formula_id[i] = int(formula_to_gid[formula_key])
        mask[i] = 1.0

    return {
        "fragment_node_formula": formula_arr,
        "fragment_node_mz": mz_arr,
        "fragment_node_official_idx": official_idx,
        "fragment_node_intensity": intensity,
        "fragment_node_local_feat": local_feat,
        "fragment_node_depth": depth,
        "fragment_node_h_shift": h_shift_arr,
        "fragment_node_is_brics": is_brics,
        "fragment_node_ring_cut": ring_cut,
        "fragment_node_atom_count": atom_count,
        "fragment_node_cut_count": cut_count,
        "fragment_node_group_formula_id": group_formula_id,
        "fragment_node_mask": mask,
        "fragment_node_source_type": source_type,
    }