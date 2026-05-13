import os
import pickle
import argparse
from pprint import pprint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    files = []
    for name in os.listdir(args.cache):
        if name.endswith(".pkl"):
            try:
                idx = int(os.path.splitext(name)[0])
                files.append((idx, os.path.join(args.cache, name)))
            except Exception:
                pass

    files = sorted(files)[: args.n]
    print("cache:", args.cache)
    print("num pkl:", len([x for x in os.listdir(args.cache) if x.endswith(".pkl")]))
    print("show:", len(files))

    for idx, path in files:
        print("\n" + "=" * 80)
        print("idx:", idx, "path:", path)
        with open(path, "rb") as f:
            meta = pickle.load(f)

        print("top-level keys:")
        pprint(list(meta.keys()))

        for k in [
            "smiles",
            "smi",
            "canonical_smiles",
            "mol",
            "rdmol",
            "mol_id",
            "precursor_mz",
            "precursor_formula",
            "adduct",
            "instrument_type",
            "collision_energy",
        ]:
            if k in meta:
                print(f"{k}:", type(meta[k]), meta[k] if k != "rdmol" else "<rdmol>")

        features = meta.get("features", {})
        print("feature keys sample:")
        pprint(list(features.keys())[:50])

        for k in [
            "possible_formulae",
            "formulae_features",
            "formulae_peaks_official_idx",
            "formulae_peaks_official_intensity",
            "formulae_frag_aux_feat",
        ]:
            if k in features:
                v = features[k]
                shape = getattr(v, "shape", None)
                print("feature", k, "type=", type(v), "shape=", shape)


if __name__ == "__main__":
    main()