#!/usr/bin/env python3
import argparse, glob, os, pickle, re, math
import numpy as np

SYM_TO_AN = {"H":1, "C":6, "N":7, "O":8, "F":9, "Na":11, "P":15, "S":16, "Cl":17, "Br":35, "I":53}
MASS = {
    "H": 1.00782503223,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "Br": 78.9183376,
    "I": 126.9044719,
}
ELECTRON = 0.00054858

def parse_formula(formula, atomicnos):
    an_to_i = {int(a): i for i, a in enumerate(atomicnos)}
    counts = np.zeros((len(atomicnos),), dtype=np.int16)
    exact = 0.0
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if sym not in SYM_TO_AN:
            return None, None
        cnt = int(num) if num else 1
        exact += MASS[sym] * cnt
        an = SYM_TO_AN[sym]
        if an not in an_to_i:
            return None, None
        counts[an_to_i[an]] += cnt
    return counts, exact

def extract_peak_formulas_from_msp(msp_path, wanted_ids):
    wanted_ids = set(wanted_ids)
    cur_id = None
    reading = False
    out = {x: [] for x in wanted_ids}
    peak_re = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
    quote_re = re.compile(r'"([^"]*)"')

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.startswith("Name:"):
                cur_id = None
                reading = False
                continue
            if not line.strip():
                reading = False
                continue
            if ":" in line and not reading:
                k, v = line.split(":", 1)
                if k.strip().lower() == "id":
                    try:
                        cur_id = f"nist20:{int(float(v.strip()))}"
                    except Exception:
                        cur_id = None
                if k.strip().lower() == "num peaks":
                    reading = True
                continue
            if reading and cur_id in wanted_ids:
                m = peak_re.match(line)
                if not m:
                    continue
                mz = float(m.group(1))
                inten = float(m.group(2))
                formulas = []
                for q in quote_re.findall(line):
                    for part in q.split(";"):
                        left = part.strip().split("=", 1)[0].strip()
                        if not left or left in ("p", "?"):
                            continue

                        # Remove isotope suffix like C14H25N2O3+i for formula matching;
                        # its m/z should be checked against M+1 isotope peak.
                        left_clean = left
                        if left_clean.endswith("+i"):
                            left_clean = left_clean[:-2]

                        # Accept only normal chemical formula starting with an element token.
                        if not re.match(r"^[A-Z][a-z]?\d*", left_clean):
                            continue

                        # Reject strings that still contain slash/space/comment tokens.
                        if "/" in left_clean or " " in left_clean:
                            continue

                        formulas.append(left_clean)
                out[cur_id].append((mz, inten, formulas, line.strip()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--msp", required=True)
    ap.add_argument("--ids", nargs="+", default=["nist20:47092", "nist20:47083", "nist20:46642", "nist20:43130"])
    ap.add_argument("--atomicnos", default="1,6,7,8,9,15,16,17,35,53,11")
    args = ap.parse_args()

    atomicnos = [int(x) for x in args.atomicnos.split(",") if x.strip()]
    ann = extract_peak_formulas_from_msp(args.msp, args.ids)

    cache_by_id = {}
    for p in glob.glob(os.path.join(args.cache_dir, "*.pkl")):
        with open(p, "rb") as f:
            obj = pickle.load(f)
        sid = str(obj.get("identifier", ""))
        if sid in args.ids:
            cache_by_id[sid] = (p, obj)

    for sid in args.ids:
        if sid not in cache_by_id:
            continue
        p, obj = cache_by_id[sid]
        feat = obj["features"]
        ff = np.asarray(feat["formulae_features"], dtype=np.int16)
        fm = np.asarray(feat["formulae_mask"], dtype=np.float32).reshape(-1) > 0.5
        peaks = np.asarray(feat["formulae_peaks"], dtype=np.float64)

        print("\n====", sid, os.path.basename(p), "pmz", obj.get("precursor_mz"), "====")
        shown = 0
        for mz, inten, formulas, rawline in sorted(ann.get(sid, []), key=lambda x: -x[1]):
            for formula in formulas:
                cc, exact = parse_formula(formula, atomicnos)
                if cc is None:
                    continue
                ids = np.where((ff[:, :len(atomicnos)] == cc[None, :]).all(axis=1) & fm)[0]
                product_ion_exact = exact - ELECTRON
                print("\nRAW:", rawline)
                print("formula", formula, "nist_mz", mz)
                print("neutral_exact", exact, "positive_ion_exact_minus_e", product_ion_exact, "delta_nist_minus_posion", mz - product_ion_exact)
                print("candidate_n", len(ids))
                for cid in ids[:5]:
                    row = peaks[int(cid)]
                    valid = np.isfinite(row[:,0]) & np.isfinite(row[:,1]) & (row[:,1] > 0)
                    rr = row[valid]
                    rr = rr[np.argsort(-rr[:,1])][:8]
                    print("  cid", int(cid), "candidate_peaks", [(float(x), float(y)) for x,y in rr])
                    if rr.shape[0] > 0:
                        print("  nearest_candidate_delta", float(np.min(np.abs(rr[:,0] - mz))))
                shown += 1
                if shown >= 20:
                    break
            if shown >= 20:
                break

if __name__ == "__main__":
    main()
