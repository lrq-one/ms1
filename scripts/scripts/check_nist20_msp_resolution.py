#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import glob
import math
import numpy as np
from collections import Counter, defaultdict

FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def norm(s):
    if s is None:
        return ""
    return str(s).strip().lower()

def parse_peak_line_raw(line):
    """
    Robust raw MSP peak parser.
    Handles:
      91 999
      91.0542 999
      91.0542,999
      91.0542 999; 105.0699 816
      91.0542 999 "comment"
    """
    out = []
    raw = str(line).strip()
    if not raw:
        return out

    # MSP 里有时一行多个 peak，用 ; 分割
    chunks = raw.split(";")
    for ch in chunks:
        nums = FLOAT_RE.findall(ch)
        if len(nums) < 2:
            continue
        try:
            mz = float(nums[0])
            inten = float(nums[1])
        except Exception:
            continue
        if math.isfinite(mz) and math.isfinite(inten) and mz >= 0 and inten > 0:
            out.append((mz, inten))
    return out

def finalize_record(cur):
    if cur is None:
        return None
    peaks = cur.get("_peaks", [])
    if not peaks:
        return None

    out = dict(cur)
    out.pop("_raw_peak_lines", None)
    out["_peaks"] = np.asarray(peaks, dtype=np.float64)
    return out

def iter_msp_records_raw(msp_path, keep_raw_lines=False):
    cur = None
    reading_peaks = False
    peaks_left = None

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            stripped = line.strip()

            if line.startswith("Name:"):
                old = finalize_record(cur)
                if old is not None:
                    yield old
                cur = {
                    "_start_lineno": lineno,
                    "name": line.split(":", 1)[1].strip(),
                    "_peaks": [],
                    "_raw_peak_lines": [],
                }
                reading_peaks = False
                peaks_left = None
                continue

            if cur is None:
                continue

            if not stripped:
                reading_peaks = False
                peaks_left = None
                continue

            if reading_peaks:
                if keep_raw_lines:
                    cur["_raw_peak_lines"].append((lineno, line))
                cur["_peaks"].extend(parse_peak_line_raw(line))
                if peaks_left is not None:
                    peaks_left -= 1
                    if peaks_left <= 0:
                        reading_peaks = False
                        peaks_left = None
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                k = key.strip().lower()
                v = value.strip()
                cur[k] = v
                if k == "num peaks":
                    reading_peaks = True
                    try:
                        peaks_left = int(FLOAT_RE.findall(v)[0])
                    except Exception:
                        peaks_left = None
                continue

            # 容错：有些 MSP 在 Num Peaks 后不严格设置 reading_peaks
            if "num peaks" in cur:
                if keep_raw_lines:
                    cur["_raw_peak_lines"].append((lineno, line))
                cur["_peaks"].extend(parse_peak_line_raw(line))

    old = finalize_record(cur)
    if old is not None:
        yield old

def record_id(rec):
    for k in ["id", "library id", "nistno", "db#", "spectrumid"]:
        if k in rec:
            s = str(rec.get(k, "")).strip()
            if s:
                nums = FLOAT_RE.findall(s)
                if nums:
                    try:
                        return str(int(float(nums[0])))
                    except Exception:
                        return s
                return s
    return ""

def normalize_adduct(s):
    ss = str(s or "").replace(" ", "").upper()
    if ss in ("[M+H]+", "M+H", "M+H+"):
        return "[M+H]+"
    if ss in ("[M+NA]+", "M+NA", "M+NA+"):
        return "[M+Na]+"
    return str(s or "").strip()

def instrument_platform(rec):
    # 你的 reader 里 instrument 是平台，instrument_type 通常是 HCD/CID
    inst = rec.get("instrument", "") or rec.get("instrument type", "") or rec.get("instrument_type", "")
    return str(inst)

def fragmentation_method(rec):
    return rec.get("instrument_type", rec.get("instrument type", ""))

def precursor_type(rec):
    return rec.get("precursor_type", rec.get("precursor type", ""))

def decimal_report(name, mzs):
    mzs = np.asarray(mzs, dtype=np.float64)
    mzs = mzs[np.isfinite(mzs)]
    print(f"\n[{name}] n={mzs.size}")
    if mzs.size == 0:
        return

    frac = np.abs(mzs - np.round(mzs))
    print("integer_ratio_1e-6 =", float(np.mean(frac < 1e-6)))
    print("integer_ratio_1e-4 =", float(np.mean(frac < 1e-4)))
    print("integer_ratio_1e-3 =", float(np.mean(frac < 1e-3)))
    print("integer_ratio_1e-2 =", float(np.mean(frac < 1e-2)))
    print("frac_p50 =", float(np.percentile(frac, 50)))
    print("frac_p90 =", float(np.percentile(frac, 90)))
    print("frac_p99 =", float(np.percentile(frac, 99)))
    nonint = mzs[frac >= 1e-4]
    print("non_integer_count_1e-4 =", int(nonint.size))
    if nonint.size:
        print("first_non_integer_mz =", nonint[:20].tolist())
    print("first_30_mz =", mzs[:30].tolist())

def scan_one_msp(msp_path, ids=None, max_records=0, raw_lines_for_ids=False,
                 filter_orbitrap_hcd_mh=False):
    ids = set(str(x) for x in (ids or []))
    print("\n" + "=" * 100)
    print("MSP:", msp_path)
    print("size_MB:", os.path.getsize(msp_path) / 1024 / 1024)

    all_mz = []
    all_top_mz = []
    subset_mz = []
    subset_top_mz = []
    rec_n = 0
    matched_n = 0
    field_counter = Counter()
    inst_counter = Counter()
    frag_counter = Counter()
    adduct_counter = Counter()

    for rec in iter_msp_records_raw(msp_path, keep_raw_lines=bool(ids)):
        rec_n += 1
        if max_records and rec_n > max_records:
            break

        rid = record_id(rec)
        peaks = rec["_peaks"]
        if peaks.size == 0:
            continue
        mz = peaks[:, 0]
        it = peaks[:, 1]
        valid = np.isfinite(mz) & np.isfinite(it) & (it > 0)
        mz = mz[valid]
        it = it[valid]
        if mz.size == 0:
            continue

        all_mz.extend(mz.tolist())
        order = np.argsort(-it)[:min(20, len(it))]
        all_top_mz.extend(mz[order].tolist())

        inst = instrument_platform(rec)
        frag = fragmentation_method(rec)
        add = normalize_adduct(precursor_type(rec))
        inst_counter[inst] += 1
        frag_counter[frag] += 1
        adduct_counter[add] += 1

        is_subset = (
            ("orbitrap" in norm(inst))
            and ("hcd" in norm(frag))
            and (add == "[M+H]+")
        )
        if is_subset:
            subset_mz.extend(mz.tolist())
            subset_top_mz.extend(mz[order].tolist())

        if rid in ids:
            matched_n += 1
            print("\n" + "-" * 100)
            print("FOUND ID:", rid, "record_start_line:", rec.get("_start_lineno"))
            for k in sorted([x for x in rec.keys() if not x.startswith("_")]):
                if k in ("name", "id", "precursormz", "precursor_mz", "precursor_type", "precursor type",
                         "instrument", "instrument_type", "instrument type", "collision_energy", "formula"):
                    print(f"{k}: {rec.get(k)}")
            print("peak_count:", len(mz))
            print("top20 parsed peaks:")
            for j in order:
                print(f"  mz={float(mz[j]):.8f} intensity={float(it[j]):.4f} frac={abs(float(mz[j])-round(float(mz[j]))):.8g}")

            print("\nraw peak lines after Num Peaks, first 40:")
            raw_lines = rec.get("_raw_peak_lines", [])
            for ln, line in raw_lines[:40]:
                print(f"{ln}: {line}")

        if rec_n % 20000 == 0:
            print("scanned_records:", rec_n, "all_peaks:", len(all_mz))

    print("\nrecords_scanned:", rec_n)
    print("target_ids_found:", matched_n, "/", len(ids))
    print("\nTop instruments:", inst_counter.most_common(10))
    print("Top fragmentation/instrument_type:", frag_counter.most_common(10))
    print("Top precursor/adduct:", adduct_counter.most_common(10))

    decimal_report("RAW all peaks", all_mz)
    decimal_report("RAW top20 peaks", all_top_mz)
    decimal_report("RAW Orbitrap+HCD+[M+H]+ peaks", subset_mz)
    decimal_report("RAW Orbitrap+HCD+[M+H]+ top20 peaks", subset_top_mz)

def find_msp_files(root):
    pats = [
        os.path.join(root, "**", "*.MSP"),
        os.path.join(root, "**", "*.msp"),
        os.path.join(root, "**", "*.msp.txt"),
        os.path.join(root, "**", "*MSMS*"),
        os.path.join(root, "**", "*msms*"),
    ]
    out = []
    for pat in pats:
        out.extend(glob.glob(pat, recursive=True))
    out = sorted(set([p for p in out if os.path.isfile(p)]))
    return out

def quick_scan_many(files, max_records=5000):
    print("\n" + "=" * 100)
    print("QUICK SCAN MSP CANDIDATE FILES")
    for p in files:
        mzs = []
        rec_n = 0
        for rec in iter_msp_records_raw(p, keep_raw_lines=False):
            rec_n += 1
            peaks = rec["_peaks"]
            if peaks.size:
                mzs.extend(peaks[:, 0].tolist())
            if rec_n >= max_records:
                break
        arr = np.asarray(mzs, dtype=np.float64)
        if arr.size == 0:
            print(f"{p} | records={rec_n} peaks=0")
            continue
        frac = np.abs(arr - np.round(arr))
        print(
            f"{p} | sizeMB={os.path.getsize(p)/1024/1024:.1f} "
            f"records={rec_n} peaks={arr.size} "
            f"integer_ratio_1e-4={float(np.mean(frac < 1e-4)):.4f} "
            f"frac_p50={float(np.percentile(frac,50)):.6g} "
            f"frac_p90={float(np.percentile(frac,90)):.6g} "
            f"example={arr[:8].tolist()}"
        )

def compare_with_project_parser(msp_path, ids):
    print("\n" + "=" * 100)
    print("COMPARE WITH rassp.data.nist20_reader._iter_msp_records")
    try:
        from rassp.data.nist20_reader import _iter_msp_records
    except Exception as e:
        print("Cannot import project parser:", repr(e))
        return

    ids = set(str(x) for x in ids)
    found = 0
    for rec in _iter_msp_records(msp_path):
        rid = str(int(rec["id"])) if "id" in rec else ""
        if rid not in ids:
            continue
        found += 1
        spect = np.asarray(rec.get("spect", []), dtype=np.float64)
        if spect.size == 0:
            continue
        spect = spect.reshape(-1, 2)
        mz = spect[:, 0]
        it = spect[:, 1]
        order = np.argsort(-it)[:min(20, len(it))]
        frac = np.abs(mz - np.round(mz))
        print("\nPROJECT PARSER ID:", rid)
        print("integer_ratio_1e-4:", float(np.mean(frac < 1e-4)))
        for j in order:
            print(f"  mz={float(mz[j]):.8f} intensity={float(it[j]):.4f} frac={float(frac[j]):.8g}")
    print("project_parser_target_ids_found:", found, "/", len(ids))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp", default="data/nist_20/hr_nist_msms.MSP")
    ap.add_argument("--root", default="data/nist_20")
    ap.add_argument("--ids", nargs="*", default=["46642", "43130", "47083", "47092"])
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--scan-all-msp", action="store_true")
    args = ap.parse_args()

    files = find_msp_files(args.root)
    print("Found MSP-like files:")
    for p in files:
        print(" ", p, "sizeMB=", os.path.getsize(p)/1024/1024)

    if args.scan_all_msp:
        quick_scan_many(files, max_records=5000)

    if not os.path.isfile(args.msp):
        raise FileNotFoundError(args.msp)

    scan_one_msp(args.msp, ids=args.ids, max_records=args.max_records)
    compare_with_project_parser(args.msp, args.ids)

if __name__ == "__main__":
    main()
