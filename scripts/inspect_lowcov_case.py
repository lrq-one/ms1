#!/usr/bin/env python3
import os
import argparse
import pickle
import numpy as np
from rdkit import Chem

ATOMICNOS_DEFAULT = [1,6,7,8,9,14,15,16,17,34,35,53]

def as1d(x, dtype):
    if x is None:
        return np.zeros((0,), dtype=dtype)
    try:
        return np.asarray(x, dtype=dtype).reshape(-1)
    except Exception:
        return np.zeros((0,), dtype=dtype)

def get_feat(obj):
    if isinstance(obj, dict) and isinstance(obj.get("features", None), dict):
        return obj["features"]
    return obj if isinstance(obj, dict) else {}

def get_any(obj, key, default=None):
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        meta = obj.get("meta", None)
        if isinstance(meta, dict) and key in meta:
            return meta[key]
    return default

def formula_from_counts(counts, atomicnos):
    pt = Chem.GetPeriodicTable()
    parts = []
    for n, z in zip(counts, atomicnos):
        n = int(round(float(n)))
        if n <= 0:
            continue
        sym = pt.GetElementSymbol(int(z))
        parts.append(sym if n == 1 else f"{sym}{n}")
    return "".join(parts) if parts else "EMPTY"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--bin-width", type=float, default=0.01)
    ap.add_argument("--atomicnos", default=os.environ.get("FORMULA_ATOMICNOS", "1,6,7,8,9,14,15,16,17,34,35,53"))
    ap.add_argument("--mz-window", type=float, default=0.05)
    ap.add_argument("--topn", type=int, default=20)
    args = ap.parse_args()

    atomicnos = [int(x) for x in args.atomicnos.split(",") if x.strip()]
    bw = float(args.bin_width)

    obj = pickle.load(open(args.pkl, "rb"))
    feat = get_feat(obj)

    print("===== META =====")
    for k in [
        "identifier", "smiles", "formula", "inchikey", "adduct",
        "instrument_type", "fragmentation_method",
        "collision_energy", "collision_energy_raw",
        "precursor_mz",
    ]:
        print(k, "=", get_any(obj, k, None))

    mol = get_any(obj, "rdmol", None)
    if mol is not None:
        try:
            print("rdmol_formula =", Chem.rdMolDescriptors.CalcMolFormula(mol))
            print("rdmol_smiles  =", Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True))
            print("rdmol_inchikey=", Chem.MolToInchiKey(mol))
        except Exception as e:
            print("rdmol inspect error:", repr(e))

    true_idx = as1d(feat.get("true_official_idx", None), np.int64)
    true_int = as1d(feat.get("true_official_intensity", None), np.float32)
    n = min(true_idx.size, true_int.size)
    true_idx = true_idx[:n]
    true_int = true_int[:n]
    valid = (true_idx >= 0) & np.isfinite(true_int) & (true_int > 0)
    true_idx = true_idx[valid]
    true_int = true_int[valid]

    order = np.argsort(-true_int)
    print("\n===== TOP TRUE PEAKS =====")
    for rank, j in enumerate(order[:args.topn], 1):
        print(rank, "mz=", true_idx[j] * bw, "bin=", int(true_idx[j]), "int=", float(true_int[j]))

    fmask = as1d(feat.get("formulae_mask", None), np.float32) > 0.5
    ffeat = np.asarray(feat.get("formulae_features", []), dtype=np.float32)
    off_idx = np.asarray(feat.get("formulae_peaks_official_idx", []), dtype=np.int64)
    off_int = np.asarray(
        feat.get("formulae_peaks_official_intensity", feat.get("formulae_peaks_intensity", [])),
        dtype=np.float32,
    )

    print("\n===== CANDIDATE SHAPE =====")
    print("formulae_features", ffeat.shape)
    print("formulae_mask valid", int(fmask.sum()), "/", fmask.size)
    print("off_idx", off_idx.shape)
    print("off_int", off_int.shape)

    support_to_cands = {}
    M = min(off_idx.shape[0], fmask.shape[0], ffeat.shape[0])
    for ci in range(M):
        if not fmask[ci]:
            continue
        row_idx = off_idx[ci].reshape(-1)
        row_int = off_int[ci].reshape(-1) if off_int.ndim >= 2 else np.zeros_like(row_idx, dtype=np.float32)
        for b, val in zip(row_idx, row_int):
            if int(b) < 0:
                continue
            if not np.isfinite(val) or float(val) <= 0:
                continue
            support_to_cands.setdefault(int(b), []).append(ci)

    print("\n===== TOP TRUE PEAK NEIGHBOR SUPPORT =====")
    win_bins = int(np.ceil(float(args.mz_window) / bw))
    for j in order[:args.topn]:
        b0 = int(true_idx[j])
        mz0 = b0 * bw
        print(f"\nTRUE mz={mz0:.4f} bin={b0} int={float(true_int[j]):.4g}")
        found = []
        for b in range(b0 - win_bins, b0 + win_bins + 1):
            cands = support_to_cands.get(b, [])
            if cands:
                found.append((b, len(cands), cands[:5]))
        if not found:
            print("  no candidate support within window")
            continue

        for b, cn, cands in found[:20]:
            print(f"  cand bin={b} mz={b*bw:.4f} n_cands={cn}")
            for ci in cands[:5]:
                counts = ffeat[ci, :len(atomicnos)]
                print("    ci=", ci, "formula=", formula_from_counts(counts, atomicnos))

if __name__ == "__main__":
    main()