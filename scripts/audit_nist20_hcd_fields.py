#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import re
from collections import Counter, defaultdict

_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def norm_key(s):
    return str(s or "").strip().lower()

def compact(s):
    return re.sub(r"[^a-z0-9]+", "", str(s or "").lower())

def norm_adduct(s):
    raw = str(s or "").strip()
    key = raw.replace(" ", "").upper()
    if key in ("[M+H]+", "M+H", "M+H+"):
        return "[M+H]+"
    if key in ("[M+NA]+", "M+NA", "M+NA+"):
        return "[M+Na]+"
    if key in ("[M+NH4]+", "M+NH4", "M+NH4+"):
        return "[M+NH4]+"
    return raw or "<missing>"

def infer_platform(fields):
    # NIST MSP 里 Instrument 通常更像具体平台，例如 Orbitrap / QTOF / ITFT
    candidates = [
        fields.get("Instrument", ""),
        fields.get("Instrument_type", ""),
    ]
    text = " | ".join(str(x) for x in candidates)
    k = compact(text)
    if "orbitrap" in k or "qexactive" in k or "exactive" in k or "qexactivehf" in k:
        return "Orbitrap"
    if "qtof" in k or "quadrupoletof" in k or "quadtof" in k:
        return "QTOF"
    if "itft" in k or "iontrapft" in k:
        return "ITFT"
    if "fticr" in k:
        return "FTICR"
    if "iontrap" in k:
        return "IonTrap"
    return "<unknown>"

def infer_frag_method(fields):
    # 这里故意扫描多个字段，因为 HCD/CID 在不同 NIST 导出里可能不在同一个字段
    scan_keys = [
        "Instrument_type",
        "Instrument",
        "Collision_energy",
        "Comment",
        "Notes",
        "Spectrum_type",
        "msN_pathway",
    ]
    text = " | ".join(str(fields.get(k, "")) for k in scan_keys)
    low = text.lower()
    k = compact(text)

    # HCD 常见写法
    if (
        "hcd" in low
        or "higher-energy" in low
        or "higher energy" in low
        or "higherenergy" in k
        or "higherenergycollisionaldissociation" in k
        or "higherenergycollisiondissociation" in k
    ):
        return "HCD"

    # CID 常见写法
    if (
        "cid" in low
        or "collision-induced" in low
        or "collision induced" in low
        or "collisioninduceddissociation" in k
    ):
        return "CID"

    if "etd" in low:
        return "ETD"

    # 注意：Q-TOF、Orbitrap 不是碎裂方式，不要返回 Q-TOF
    return "<unknown>"

def parse_ce(raw):
    s = str(raw or "").strip()
    if not s:
        return None
    m = _FLOAT_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None

def iter_msp_blocks(path, limit=0):
    fields = None
    reading_peaks = False

    def emit():
        if fields is None:
            return None
        if "Name" not in fields:
            return None
        return dict(fields)

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        yielded = 0
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.startswith("Name:"):
                old = emit()
                if old is not None:
                    yield old
                    yielded += 1
                    if limit > 0 and yielded >= limit:
                        return
                fields = {"Name": line.split(":", 1)[1].strip()}
                reading_peaks = False
                continue

            if fields is None:
                continue

            if not line.strip():
                reading_peaks = False
                continue

            if reading_peaks:
                continue

            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                # Synon 可能出现多次，这里只为了统计 metadata，重复字段保留第一个即可
                if k not in fields:
                    fields[k] = v
                else:
                    fields[k] = str(fields[k]) + " || " + v
                if k.lower() == "num peaks":
                    reading_peaks = True

        old = emit()
        if old is not None:
            yield old

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-csv", default="nist20_hcd_candidates.csv")
    args = ap.parse_args()

    total = 0
    adduct_counter = Counter()
    platform_counter = Counter()
    frag_counter = Counter()
    instrument_raw_counter = Counter()
    instrument_type_raw_counter = Counter()
    ce_raw_counter = Counter()

    combo_counter = Counter()
    examples = []
    hcd_examples = []
    orbitrap_mh_no_hcd_examples = []

    for rec in iter_msp_blocks(args.msp, limit=args.limit):
        total += 1

        adduct = norm_adduct(rec.get("Precursor_type", ""))
        platform = infer_platform(rec)
        frag = infer_frag_method(rec)
        precursor_mz = parse_ce(rec.get("PrecursorMZ", ""))
        ce = parse_ce(rec.get("Collision_energy", ""))

        adduct_counter[adduct] += 1
        platform_counter[platform] += 1
        frag_counter[frag] += 1
        instrument_raw_counter[str(rec.get("Instrument", "<missing>"))] += 1
        instrument_type_raw_counter[str(rec.get("Instrument_type", "<missing>"))] += 1
        ce_raw_counter[str(rec.get("Collision_energy", "<missing>"))] += 1

        is_mh = adduct == "[M+H]+"
        is_orbitrap = platform == "Orbitrap"
        is_hcd = frag == "HCD"
        mz_ok = precursor_mz is not None and precursor_mz <= 1500.0
        ce_ok = ce is not None

        combo = (
            "mh" if is_mh else "not_mh",
            "orbitrap" if is_orbitrap else "not_orbitrap",
            "hcd" if is_hcd else "not_hcd",
            "mz_ok" if mz_ok else "mz_bad",
            "ce_ok" if ce_ok else "ce_bad",
        )
        combo_counter[combo] += 1

        if is_hcd and len(hcd_examples) < 30:
            hcd_examples.append(rec)

        if is_orbitrap and is_mh and (not is_hcd) and len(orbitrap_mh_no_hcd_examples) < 30:
            orbitrap_mh_no_hcd_examples.append(rec)

        if is_mh and is_orbitrap and is_hcd and mz_ok and ce_ok:
            if len(examples) < 1000:
                examples.append({
                    "ID": rec.get("ID", ""),
                    "Name": rec.get("Name", ""),
                    "Precursor_type": rec.get("Precursor_type", ""),
                    "PrecursorMZ": rec.get("PrecursorMZ", ""),
                    "Collision_energy": rec.get("Collision_energy", ""),
                    "Instrument": rec.get("Instrument", ""),
                    "Instrument_type": rec.get("Instrument_type", ""),
                    "Comment": rec.get("Comment", ""),
                    "Notes": rec.get("Notes", ""),
                    "Spectrum_type": rec.get("Spectrum_type", ""),
                    "Formula": rec.get("Formula", ""),
                    "InChIKey": rec.get("InChIKey", ""),
                })

    print("===== BASIC =====")
    print("total_blocks:", total)

    print("\n===== adduct top 30 =====")
    for k, v in adduct_counter.most_common(30):
        print(k, v)

    print("\n===== inferred platform =====")
    for k, v in platform_counter.most_common(30):
        print(k, v)

    print("\n===== inferred fragmentation method =====")
    for k, v in frag_counter.most_common(30):
        print(k, v)

    print("\n===== key combo counts =====")
    for k, v in combo_counter.most_common(30):
        print(k, v)

    print("\n===== Instrument_type raw top 50 =====")
    for k, v in instrument_type_raw_counter.most_common(50):
        print(repr(k), v)

    print("\n===== Instrument raw top 50 =====")
    for k, v in instrument_raw_counter.most_common(50):
        print(repr(k), v)

    print("\n===== Collision_energy raw top 50 =====")
    for k, v in ce_raw_counter.most_common(50):
        print(repr(k), v)

    print("\n===== HCD examples =====")
    for r in hcd_examples[:10]:
        print({
            "ID": r.get("ID"),
            "Instrument": r.get("Instrument"),
            "Instrument_type": r.get("Instrument_type"),
            "Collision_energy": r.get("Collision_energy"),
            "Comment": r.get("Comment"),
            "Notes": r.get("Notes"),
            "Precursor_type": r.get("Precursor_type"),
        })

    print("\n===== Orbitrap + [M+H]+ but no HCD examples =====")
    for r in orbitrap_mh_no_hcd_examples[:10]:
        print({
            "ID": r.get("ID"),
            "Instrument": r.get("Instrument"),
            "Instrument_type": r.get("Instrument_type"),
            "Collision_energy": r.get("Collision_energy"),
            "Comment": r.get("Comment"),
            "Notes": r.get("Notes"),
            "Precursor_type": r.get("Precursor_type"),
        })

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "ID", "Name", "Precursor_type", "PrecursorMZ", "Collision_energy",
            "Instrument", "Instrument_type", "Comment", "Notes", "Spectrum_type",
            "Formula", "InChIKey",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in examples:
            w.writerow(r)

    print("\nwritten clean candidates csv:", args.out_csv)
    print("clean_candidate_example_n:", len(examples))

if __name__ == "__main__":
    main()