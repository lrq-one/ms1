#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import glob
import math
import tempfile
from collections import Counter
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.info")
FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
PEAK_PAIR_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)
QUOTED_RE = re.compile(r'"[^"]*"')
ID_FILE_RE = re.compile(r"^ID(\d+)\.MOL$", re.IGNORECASE)
LIBID_RE = re.compile(r"Library\s*ID\s*=\s*(\d+)", re.IGNORECASE)

def parse_peak_line_safe(line):
    """
    安全解析 MSP peak 行。
    防止把注释里的 C17H23、14/14、ppm 等数字解析成假峰。
    """
    raw = str(line).strip()
    if not raw:
        return []

    no_comment = QUOTED_RE.sub("", raw)
    out = []
    for token in no_comment.split(";"):
        token = token.strip()
        if not token:
            continue
        m = PEAK_PAIR_RE.match(token)
        if m is None:
            continue
        mz = float(m.group(1))
        inten = float(m.group(2))
        if math.isfinite(mz) and math.isfinite(inten) and mz >= 0 and inten > 0:
            out.append((mz, inten))
    return out

def iter_msp_records(msp_path, keep_raw_for_ids=None, max_records=0):
    keep_raw_for_ids = set(str(x) for x in (keep_raw_for_ids or []))
    cur = None
    reading = False

    def finalize(x):
        if not x or not x.get("peaks"):
            return None
        x["peaks"] = np.asarray(x["peaks"], dtype=np.float64)
        return x

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.rstrip("\n\r")
            s = line.strip()

            if line.startswith("Name:"):
                old = finalize(cur)
                if old is not None:
                    yield old
                cur = {
                    "start_line": lineno,
                    "meta": {"name": line.split(":", 1)[1].strip()},
                    "peaks": [],
                    "raw_peak_lines": [],
                }
                reading = False
                continue

            if cur is None:
                continue

            if not s:
                reading = False
                continue

            if reading:
                rid = get_msp_id_from_meta(cur["meta"])
                if rid in keep_raw_for_ids:
                    cur["raw_peak_lines"].append((lineno, line))
                cur["peaks"].extend(parse_peak_line_safe(line))
                continue

            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                cur["meta"][k] = v
                if k == "num peaks":
                    reading = True
                continue

        old = finalize(cur)
        if old is not None:
            yield old

def get_msp_id_from_meta(meta):
    v = meta.get("id", "")
    try:
        return str(int(float(v)))
    except Exception:
        return str(v).strip()

def get_float_meta(meta, keys):
    for k in keys:
        if k in meta:
            try:
                return float(FLOAT_RE.findall(str(meta[k]))[0])
            except Exception:
                pass
    return float("nan")

def norm_formula(s):
    return str(s or "").strip()

def formula_from_mol(mol):
    try:
        return rdMolDescriptors.CalcMolFormula(mol)
    except Exception:
        return ""

def inchikey_from_mol(mol):
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""

def exact_mass_from_mol(mol):
    try:
        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return float("nan")

def parse_mol_header(path):
    lines = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for _ in range(4):
                lines.append(f.readline().rstrip("\n\r"))
    except Exception:
        pass
    text = "\n".join(lines)
    m = LIBID_RE.search(text)
    libid = None
    if m:
        libid = str(int(m.group(1)))
    return libid, lines

def load_mol_index(mol_path):
    """
    支持两种情况：
    1. mol_path 是目录：里面有 IDxxxxx.MOL
    2. mol_path 是单个 SDF/MOL-like 文件：尝试用 SDMolSupplier 读取
    """
    out = {}

    if os.path.isdir(mol_path):
        files = sorted(glob.glob(os.path.join(mol_path, "*.MOL")) + glob.glob(os.path.join(mol_path, "*.mol")))
        for p in files:
            name = os.path.basename(p)
            mid = None
            m = ID_FILE_RE.match(name)
            if m:
                mid = str(int(m.group(1)))
            if mid is None:
                libid, _ = parse_mol_header(p)
                mid = libid
            if mid is None:
                continue
            mol = Chem.MolFromMolFile(p, sanitize=False, removeHs=False)
            if mol is None:
                out[mid] = {"path": p, "mol": None, "parse_ok": False}
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass
            out[mid] = {"path": p, "mol": mol, "parse_ok": mol is not None}
        return out

    if os.path.isfile(mol_path):
        # 尝试当作 SDF/multi-mol 文件读取
        suppl = Chem.SDMolSupplier(mol_path, sanitize=False, removeHs=False)
        for i, mol in enumerate(suppl):
            if mol is None:
                continue
            mid = None
            for prop in ["ID", "Id", "id", "Library ID", "NISTNO", "NIST No"]:
                if mol.HasProp(prop):
                    try:
                        mid = str(int(float(mol.GetProp(prop))))
                        break
                    except Exception:
                        mid = mol.GetProp(prop).strip()
                        break

            # 如果属性没有 ID，就退而求其次用 _Name 里的数字
            if mid is None and mol.HasProp("_Name"):
                nums = re.findall(r"\d+", mol.GetProp("_Name"))
                if nums:
                    mid = str(int(nums[-1]))

            if mid is None:
                mid = f"__idx_{i}"

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                try:
                    mol.UpdatePropertyCache(strict=False)
                except Exception:
                    pass

            out[mid] = {"path": f"{mol_path}#{i}", "mol": mol, "parse_ok": True}

        return out

    raise FileNotFoundError(mol_path)

def report_msp_precision(records):
    mzs = []
    top_mzs = []
    for rec in records:
        peaks = rec["peaks"]
        if peaks.size == 0:
            continue
        mz = peaks[:, 0]
        inten = peaks[:, 1]
        valid = np.isfinite(mz) & np.isfinite(inten) & (inten > 0)
        mz = mz[valid]
        inten = inten[valid]
        if mz.size == 0:
            continue
        mzs.extend(mz.tolist())
        order = np.argsort(-inten)[:min(20, len(inten))]
        top_mzs.extend(mz[order].tolist())

    def one(name, arr):
        arr = np.asarray(arr, dtype=np.float64)
        frac = np.abs(arr - np.round(arr))
        print(f"\n[{name}] n={arr.size}")
        if arr.size == 0:
            return
        print("integer_ratio_1e-4 =", float(np.mean(frac < 1e-4)))
        print("integer_ratio_1e-3 =", float(np.mean(frac < 1e-3)))
        print("frac_p50 =", float(np.percentile(frac, 50)))
        print("frac_p90 =", float(np.percentile(frac, 90)))
        print("non_integer_count_1e-4 =", int(np.sum(frac >= 1e-4)))
        print("first_20_mz =", arr[:20].tolist())

    one("MSP all product peaks", mzs)
    one("MSP top20 product peaks", top_mzs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp", required=True)
    ap.add_argument("--mol", required=True)
    ap.add_argument("--ids", nargs="*", default=["43130", "46642", "47083", "47092"])
    ap.add_argument("--max-records", type=int, default=100000)
    args = ap.parse_args()

    ids = set(str(int(float(x))) for x in args.ids)

    print("=" * 100)
    print("MSP:", args.msp)
    print("MOL:", args.mol)

    records = []
    found = {}
    for i, rec in enumerate(iter_msp_records(args.msp, keep_raw_for_ids=ids), start=1):
        records.append(rec)
        rid = get_msp_id_from_meta(rec["meta"])
        if rid in ids:
            found[rid] = rec
        if args.max_records and i >= args.max_records and len(found) >= len(ids):
            break
        if args.max_records and i >= args.max_records and not ids:
            break

    report_msp_precision(records)

    print("\n" + "=" * 100)
    print("Loading MOL index...")
    mol_index = load_mol_index(args.mol)
    print("MOL indexed n =", len(mol_index))
    print("MOL first keys =", list(mol_index.keys())[:20])

    print("\n" + "=" * 100)
    print("Per-ID pair check")
    for rid in sorted(ids):
        rec = found.get(rid)
        mrow = mol_index.get(rid)

        print("\n" + "-" * 100)
        print("ID:", rid)
        if rec is None:
            print("MSP: NOT FOUND")
            continue
        meta = rec["meta"]
        peaks = rec["peaks"]
        print("MSP name:", meta.get("name"))
        print("MSP formula:", meta.get("formula"))
        print("MSP precursor:", meta.get("precursormz"))
        print("MSP exactmass:", meta.get("exactmass"))
        print("MSP instrument:", meta.get("instrument"), "|", meta.get("instrument_type"))
        print("MSP peak_n:", peaks.shape[0])

        print("raw peak lines first 8:")
        for ln, line in rec.get("raw_peak_lines", [])[:8]:
            print(f"  {ln}: {line}")

        if peaks.size:
            mz = peaks[:, 0]
            inten = peaks[:, 1]
            order = np.argsort(-inten)[:min(10, len(inten))]
            print("parsed top peaks:")
            for j in order:
                print(f"  mz={mz[j]:.6f} int={inten[j]:.2f}")

        if mrow is None:
            print("MOL: NOT FOUND for this ID")
            continue

        mol = mrow["mol"]
        print("MOL path:", mrow["path"])
        print("MOL parse_ok:", mrow["parse_ok"])
        if mol is None:
            continue

        mf = formula_from_mol(mol)
        mi = inchikey_from_mol(mol)
        mm = exact_mass_from_mol(mol)
        msp_formula = norm_formula(meta.get("formula"))
        msp_exact = get_float_meta(meta, ["exactmass", "exact mass"])

        print("MOL formula:", mf)
        print("MOL exact_mass:", mm)
        print("MOL inchikey:", mi)

        print("formula_match:", mf == msp_formula)
        if np.isfinite(msp_exact) and np.isfinite(mm):
            print("exact_mass_delta:", mm - msp_exact)

if __name__ == "__main__":
    main()
