#!/usr/bin/env python3
import argparse, glob, os, pickle, re, math
import numpy as np

PEAK_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
QUOTE_RE = re.compile(r'"([^"]*)"')
FORMULA_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")

def parse_formula_counts(formula, atomicnos):
    # Minimal periodic map for your element set
    sym_to_an = {"H":1, "C":6, "N":7, "O":8, "F":9, "Na":11, "P":15, "S":16, "Cl":17, "Br":35, "I":53}
    an_to_i = {int(a): i for i, a in enumerate(atomicnos)}
    counts = np.zeros((len(atomicnos),), dtype=np.int16)
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        if sym not in sym_to_an:
            return None
        an = sym_to_an[sym]
        if an not in an_to_i:
            return None
        cnt = int(num) if num else 1
        counts[an_to_i[an]] += cnt
    return counts

def extract_formulas_from_comment(comment):
    out = []
    # comments like C14H25N2O3=p-C6H13NO/1.2ppm;C17H23N3=...
    for part in str(comment).split(";"):
        left = part.strip().split("=", 1)[0].strip()
        left = left.split("/", 1)[0].strip()
        left = left.split(" ", 1)[0].strip()
        if left in ("p", "?", ""):
            continue
        if FORMULA_RE.match(left):
            out.append(left)
    return out

def iter_msp_annotation_records(msp_path):
    cur = None
    reading = False

    def finish(x):
        if x and x.get("id") and x.get("peaks"):
            return x
        return None

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.startswith("Name:"):
                old = finish(cur)
                if old:
                    yield old
                cur = {"peaks": []}
                reading = False
                continue
            if cur is None:
                continue
            s = line.strip()
            if not s:
                reading = False
                continue

            if reading:
                # one peak per line in new export, but also handle ; separated old style
                chunks = s.split(";")
                # Important: if semicolon is inside quote, this is imperfect,
                # but peak itself is at the line start; use full line for first peak.
                m = PEAK_RE.match(s)
                if m:
                    mz = float(m.group(1)); inten = float(m.group(2))
                    formulas = []
                    for q in QUOTE_RE.findall(s):
                        formulas.extend(extract_formulas_from_comment(q))
                    cur["peaks"].append((mz, inten, formulas))
                continue

            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                if k == "id":
                    try:
                        cur["id"] = f"nist20:{int(float(v))}"
                    except Exception:
                        pass
                if k == "num peaks":
                    reading = True
                continue

        old = finish(cur)
        if old:
            yield old

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--msp", required=True)
    ap.add_argument("--atomicnos", default="1,6,7,8,9,15,16,17,35,53,11")
    ap.add_argument("--bin-width", type=float, default=0.01)
    ap.add_argument("--sample-n", type=int, default=200)
    args = ap.parse_args()

    atomicnos = [int(x) for x in args.atomicnos.split(",") if x.strip()]
    msp_map = {r["id"]: r for r in iter_msp_annotation_records(args.msp)}

    paths = sorted(glob.glob(os.path.join(args.cache_dir, "*.pkl")))
    if args.sample_n > 0:
        paths = paths[:args.sample_n]

    rows = []
    for p in paths:
        with open(p, "rb") as f:
            obj = pickle.load(f)
        sid = str(obj.get("identifier", ""))
        rec = msp_map.get(sid)
        if rec is None:
            continue

        feat = obj["features"]
        ff = np.asarray(feat.get("formulae_features", []), dtype=np.int16)
        fmask = np.asarray(feat.get("formulae_mask", []), dtype=np.float32).reshape(-1) > 0.5
        off_idx = np.asarray(feat.get("formulae_peaks_official_idx_agg", feat.get("formulae_peaks_official_idx", [])), dtype=np.int64)

        n = min(ff.shape[0], fmask.shape[0], off_idx.shape[0])
        ff = ff[:n, :len(atomicnos)]
        fmask = fmask[:n]
        off_idx = off_idx[:n]

        cand_keys = {}
        for i in range(n):
            if not fmask[i]:
                continue
            key = tuple(int(x) for x in ff[i].tolist())
            cand_keys.setdefault(key, []).append(i)

        ann_peak_n = 0
        parsed_formula_n = 0
        formula_in_pool_n = 0
        formula_bin_hit_n = 0

        for mz, inten, formulas in rec["peaks"]:
            if not formulas:
                continue
            ann_peak_n += 1
            target_bin = int(math.floor(float(mz) / float(args.bin_width) + 1e-8))

            peak_formula_found = False
            peak_bin_hit = False

            for formula in formulas:
                cc = parse_formula_counts(formula, atomicnos)
                if cc is None:
                    continue
                parsed_formula_n += 1
                key = tuple(int(x) for x in cc.tolist())
                ids = cand_keys.get(key, [])
                if ids:
                    peak_formula_found = True
                    for cid in ids:
                        row = off_idx[int(cid)]
                        if np.any(row == target_bin):
                            peak_bin_hit = True
                            break

            formula_in_pool_n += int(peak_formula_found)
            formula_bin_hit_n += int(peak_bin_hit)

        rows.append({
            "file": os.path.basename(p),
            "sample_id": sid,
            "ann_peak_n": ann_peak_n,
            "parsed_formula_n": parsed_formula_n,
            "formula_in_pool_frac": formula_in_pool_n / max(1, ann_peak_n),
            "formula_bin_hit_frac": formula_bin_hit_n / max(1, ann_peak_n),
        })

    print(f"[ann-support] n_samples={len(rows)}")
    for k in ["ann_peak_n", "parsed_formula_n", "formula_in_pool_frac", "formula_bin_hit_frac"]:
        arr = np.asarray([r[k] for r in rows], dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            print(f"{k}: mean={arr.mean():.4f} p50={np.percentile(arr,50):.4f} p90={np.percentile(arr,90):.4f}")

    print("\nfirst 20 rows:")
    for r in rows[:20]:
        print(r)

if __name__ == "__main__":
    main()
