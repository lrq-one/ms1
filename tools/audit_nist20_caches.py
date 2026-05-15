import os
import glob
import pickle
import argparse
import numpy as np


def safe_shape(x):
    if x is None:
        return None
    try:
        return tuple(np.asarray(x).shape)
    except Exception:
        try:
            return tuple(x.shape)
        except Exception:
            return None


def inspect_cache(cache_dir, max_files=3):
    files = sorted(glob.glob(os.path.join(cache_dir, "*.pkl")))
    out = {
        "cache_dir": cache_dir,
        "exists": os.path.isdir(cache_dir),
        "pkl_n": len(files),
        "first_files": files[:max_files],
        "ok": False,
        "error": None,
        "samples": [],
    }

    if not os.path.isdir(cache_dir):
        out["error"] = "cache dir does not exist"
        return out

    if len(files) == 0:
        out["error"] = "no .pkl files"
        return out

    for fp in files[:max_files]:
        try:
            with open(fp, "rb") as f:
                meta = pickle.load(f)
            feat = meta.get("features", {})

            sample = {
                "file": os.path.basename(fp),
                "keys_n": len(feat),
                "vect_feat": safe_shape(feat.get("vect_feat")),
                "adj": safe_shape(feat.get("adj")),
                "input_mask": safe_shape(feat.get("input_mask")),
                "formulae_features": safe_shape(feat.get("formulae_features")),
                "formulae_mask": safe_shape(feat.get("formulae_mask")),
                "formulae_peaks": safe_shape(feat.get("formulae_peaks")),
                "formulae_peaks_mass_idx": safe_shape(feat.get("formulae_peaks_mass_idx")),
                "formulae_peaks_intensity": safe_shape(feat.get("formulae_peaks_intensity")),
                "formulae_peaks_official_idx": safe_shape(feat.get("formulae_peaks_official_idx")),
                "formulae_peaks_official_intensity": safe_shape(feat.get("formulae_peaks_official_intensity")),
                "formulae_frag_aux_feat": safe_shape(feat.get("formulae_frag_aux_feat")),
                "teacher_formula_probs": safe_shape(feat.get("teacher_formula_probs")),
                "true_all_official_idx": safe_shape(meta.get("true_all_official_idx")),
                "true_top20_official_idx": safe_shape(meta.get("true_top20_official_idx")),
                "spect": safe_shape(meta.get("spect")),
                "spect_dense": safe_shape(meta.get("spect_dense")),
                "adduct": meta.get("adduct", None),
                "collision_energy": meta.get("collision_energy", None),
                "instrument_type": meta.get("instrument_type", None),
                "precursor_mz": meta.get("precursor_mz", None),
            }

            ff = feat.get("formulae_features")
            if ff is not None:
                arr = np.asarray(ff)
                if arr.ndim >= 2:
                    sample["formula_dim_inferred"] = int(arr.shape[-1])
                else:
                    sample["formula_dim_inferred"] = None
            else:
                sample["formula_dim_inferred"] = None

            vf = feat.get("vect_feat")
            if vf is not None:
                arr = np.asarray(vf)
                if arr.ndim >= 2:
                    sample["atom_feat_dim_inferred"] = int(arr.shape[-1])
                else:
                    sample["atom_feat_dim_inferred"] = None
            else:
                sample["atom_feat_dim_inferred"] = None

            out["samples"].append(sample)
            out["ok"] = True

        except Exception as e:
            out["samples"].append({
                "file": os.path.basename(fp),
                "error": f"{type(e).__name__}: {e}",
            })

    return out


def print_report(rep):
    print("=" * 120)
    print("CACHE:", rep["cache_dir"])
    print("exists:", rep["exists"], "pkl_n:", rep["pkl_n"], "ok:", rep["ok"])
    if rep["error"]:
        print("ERROR:", rep["error"])
        return

    for s in rep["samples"]:
        print("-" * 120)
        for k, v in s.items():
            print(f"{k}: {v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cache_dirs", nargs="+")
    ap.add_argument("--max-files", type=int, default=3)
    args = ap.parse_args()

    for d in args.cache_dirs:
        rep = inspect_cache(d, max_files=args.max_files)
        print_report(rep)


if __name__ == "__main__":
    main()