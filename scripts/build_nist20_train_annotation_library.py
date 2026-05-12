#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict

PEAK_RE = re.compile(r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
QUOTE_RE = re.compile(r'"([^"]*)"')
FORM_RE = re.compile(r"^[A-Z][A-Za-z0-9]*$")

def _norm_formula(s):
    s = str(s or "").strip()
    if not s:
        return ""
    if s in ("p", "?", "p+i"):
        return ""
    if s.endswith("+i"):
        s = s[:-2]
    if "/" in s or " " in s:
        return ""
    if not re.match(r"^[A-Z][a-z]?\d*", s):
        return ""
    return s

def _parse_formula_counts(formula):
    out = defaultdict(int)
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", str(formula)):
        out[sym] += int(num) if num else 1
    return dict(out)

def _counts_to_formula(counts):
    counts = {k: int(v) for k, v in counts.items() if int(v) > 0}
    if not counts:
        return ""
    order = []
    if "C" in counts:
        order.append("C")
    if "H" in counts:
        order.append("H")
    order += sorted([k for k in counts if k not in ("C", "H")])
    s = ""
    for k in order:
        s += k
        if counts[k] != 1:
            s += str(counts[k])
    return s

def _formula_sub(parent, child):
    pc = _parse_formula_counts(parent)
    cc = _parse_formula_counts(child)
    out = {}
    for k in set(pc) | set(cc):
        v = int(pc.get(k, 0)) - int(cc.get(k, 0))
        if v < 0:
            return ""
        if v > 0:
            out[k] = v
    return _counts_to_formula(out)

def _split_name_for_key(key, seed=42, ratios=(0.8, 0.1, 0.1)):
    raw = (str(key) + f"|{int(seed)}").encode("utf-8")
    h = int(hashlib.sha1(raw).hexdigest()[:12], 16)
    x = (h % 1000000) / 1000000.0
    if x < ratios[0]:
        return "train"
    if x < ratios[0] + ratios[1]:
        return "val"
    return "test"

def _iter_msp_records(msp_path):
    cur = None
    reading = False

    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")
            if line.startswith("Name:"):
                if cur is not None:
                    yield cur
                cur = {"peaks": []}
                reading = False
                continue

            if cur is None:
                continue

            if not line.strip():
                reading = False
                continue

            if ":" in line and not reading:
                k, v = line.split(":", 1)
                k = k.strip().lower()
                v = v.strip()
                cur[k] = v
                if k == "num peaks":
                    reading = True
                continue

            if reading:
                m = PEAK_RE.match(line)
                if not m:
                    continue
                mz = float(m.group(1))
                inten = float(m.group(2))
                formulas = []
                for q in QUOTE_RE.findall(line):
                    for part in q.split(";"):
                        left = part.strip().split("=", 1)[0].strip()
                        nf = _norm_formula(left)
                        if nf:
                            formulas.append(nf)
                cur["peaks"].append((mz, inten, formulas))

    if cur is not None:
        yield cur

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--msp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--split-ratios", default="0.8,0.1,0.1")
    ap.add_argument("--top-product", type=int, default=10000)
    ap.add_argument("--top-loss", type=int, default=10000)
    args = ap.parse_args()

    ratios = tuple(float(x) for x in str(args.split_ratios).split(","))
    product_int = defaultdict(float)
    product_cnt = defaultdict(int)
    loss_int = defaultdict(float)
    loss_cnt = defaultdict(int)

    train_records = 0
    ann_peak_n = 0

    for rec in _iter_msp_records(args.msp):
        identifier = rec.get("id", "")
        inchikey = str(rec.get("inchikey", "") or "").strip().upper()
        split_key = f"inchikey:{inchikey}" if inchikey else f"id:{identifier}"
        split = _split_name_for_key(split_key, seed=args.split_seed, ratios=ratios)
        if split != "train":
            continue

        parent_formula = rec.get("formula", "")
        if not parent_formula:
            continue

        train_records += 1

        for mz, inten, formulas in rec.get("peaks", []):
            if not formulas:
                continue
            ann_peak_n += 1
            w = max(0.0, float(inten)) / max(1, len(formulas))

            for f in formulas:
                product_int[f] += w
                product_cnt[f] += 1

                lf = _formula_sub(parent_formula, f)
                if lf:
                    loss_int[lf] += w
                    loss_cnt[lf] += 1

    def _top_payload(score_d, cnt_d, topn):
        items = sorted(score_d.items(), key=lambda kv: -kv[1])[:int(topn)]
        return {
            k: {
                "intensity": float(v),
                "count": int(cnt_d.get(k, 0)),
                "log_intensity": float(math.log1p(v)),
            }
            for k, v in items
        }

    payload = {
        "source_msp": args.msp,
        "split": "train",
        "split_seed": int(args.split_seed),
        "split_ratios": args.split_ratios,
        "train_records": int(train_records),
        "annotated_peak_n": int(ann_peak_n),
        "product": _top_payload(product_int, product_cnt, args.top_product),
        "neutral_loss": _top_payload(loss_int, loss_cnt, args.top_loss),
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("wrote", args.out)
    print("train_records", train_records)
    print("annotated_peak_n", ann_peak_n)
    print("product_n", len(payload["product"]))
    print("neutral_loss_n", len(payload["neutral_loss"]))

if __name__ == "__main__":
    main()