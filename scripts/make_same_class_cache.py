import os
import shutil
import pickle
import argparse
import numpy as np

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


def get_mol_from_meta(meta):
    # 1. direct RDKit mol
    for k in ["rdmol", "mol"]:
        if k in meta and meta[k] is not None:
            obj = meta[k]
            if isinstance(obj, Chem.Mol):
                return obj
            try:
                m = Chem.Mol(obj)
                if m is not None:
                    return m
            except Exception:
                pass

    # 2. smiles
    for k in ["smiles", "SMILES", "smi", "canonical_smiles"]:
        if k in meta and meta[k] is not None:
            m = Chem.MolFromSmiles(str(meta[k]))
            if m is not None:
                return m

    # 3. sometimes stored inside features
    features = meta.get("features", {})
    if isinstance(features, dict):
        for k in ["smiles", "SMILES", "smi", "canonical_smiles"]:
            if k in features and features[k] is not None:
                m = Chem.MolFromSmiles(str(features[k]))
                if m is not None:
                    return m

    return None


def atom_set_of_mol(mol):
    return {a.GetSymbol() for a in mol.GetAtoms()}


def ok_by_rule(mol, rule, mw_min, mw_max):
    atom_set = atom_set_of_mol(mol)
    mw = float(Descriptors.MolWt(mol))
    ring = int(rdMolDescriptors.CalcNumRings(mol))
    aromatic = int(rdMolDescriptors.CalcNumAromaticRings(mol))

    if mw < mw_min:
        return False
    if mw_max > 0 and mw >= mw_max:
        return False

    if rule == "all":
        return True

    if rule == "chno":
        return atom_set.issubset({"C", "H", "N", "O"})

    if rule == "chnops":
        return atom_set.issubset({"C", "H", "N", "O", "P", "S"})

    if rule == "no_halogen":
        return not any(x in atom_set for x in ["F", "Cl", "Br", "I"])

    if rule == "halogen":
        return any(x in atom_set for x in ["F", "Cl", "Br", "I"])

    if rule == "aromatic":
        return aromatic > 0

    if rule == "non_aromatic":
        return aromatic == 0

    if rule == "ring":
        return ring > 0

    if rule == "no_ring":
        return ring == 0

    raise ValueError(f"unknown rule: {rule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_cache", required=True)
    ap.add_argument("--dst_cache", required=True)
    ap.add_argument("--rule", default="chno")
    ap.add_argument("--mw_min", type=float, default=150.0)
    ap.add_argument("--mw_max", type=float, default=350.0)
    ap.add_argument("--max_n", type=int, default=2000)
    ap.add_argument("--copy", action="store_true", help="copy files instead of symlink")
    args = ap.parse_args()

    os.makedirs(args.dst_cache, exist_ok=True)

    src_files = []
    for name in os.listdir(args.src_cache):
        if not name.endswith(".pkl"):
            continue
        try:
            idx = int(os.path.splitext(name)[0])
        except Exception:
            continue
        src_files.append((idx, os.path.join(args.src_cache, name)))

    src_files.sort()

    selected = []
    missing_mol = 0
    bad_pkl = 0

    mw_values = []
    atom_sets = {}

    for idx, path in src_files:
        try:
            with open(path, "rb") as f:
                meta = pickle.load(f)
        except Exception:
            bad_pkl += 1
            continue

        mol = get_mol_from_meta(meta)
        if mol is None:
            missing_mol += 1
            continue

        atom_set = ",".join(sorted(atom_set_of_mol(mol)))
        atom_sets[atom_set] = atom_sets.get(atom_set, 0) + 1
        mw_values.append(float(Descriptors.MolWt(mol)))

        if ok_by_rule(mol, args.rule, args.mw_min, args.mw_max):
            selected.append((idx, path))

        if args.max_n > 0 and len(selected) >= args.max_n:
            break

    print("src_cache:", args.src_cache)
    print("dst_cache:", args.dst_cache)
    print("total pkl:", len(src_files))
    print("selected:", len(selected))
    print("missing_mol:", missing_mol)
    print("bad_pkl:", bad_pkl)

    if len(mw_values) > 0:
        arr = np.asarray(mw_values, dtype=float)
        print("mw all: min/mean/max:", float(arr.min()), float(arr.mean()), float(arr.max()))

    print("top atom sets:")
    for k, v in sorted(atom_sets.items(), key=lambda x: -x[1])[:20]:
        print(k, v)

    if len(selected) == 0:
        print("No selected molecules. Check whether cache pkl contains mol/smiles.")
        return

    # clear old pkl links/copies in dst
    for name in os.listdir(args.dst_cache):
        if name.endswith(".pkl") or name.endswith(".err"):
            try:
                os.remove(os.path.join(args.dst_cache, name))
            except Exception:
                pass

    mapping_path = os.path.join(args.dst_cache, "mapping.tsv")
    with open(mapping_path, "w") as mf:
        mf.write("new_idx\told_idx\tsrc_path\n")
        for new_idx, (old_idx, src_path) in enumerate(selected):
            dst_path = os.path.join(args.dst_cache, f"{new_idx}.pkl")
            if args.copy:
                shutil.copy2(src_path, dst_path)
            else:
                os.symlink(os.path.abspath(src_path), dst_path)
            mf.write(f"{new_idx}\t{old_idx}\t{src_path}\n")

    print("saved mapping:", mapping_path)


if __name__ == "__main__":
    main()