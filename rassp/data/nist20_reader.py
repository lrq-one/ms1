import hashlib
import os
import re
from collections import Counter

import numpy as np
from rdkit import Chem

from rassp.msutil.collision_energy import parse_collision_energy_to_ev


_ID_RE = re.compile(r"^ID(\d+)\.MOL$", re.IGNORECASE)
_CAS_MOL_RE = re.compile(r"CAS\s*rn\s*=\s*([0-9]+)", re.IGNORECASE)
_LIBID_MOL_RE = re.compile(r"Library\s*ID\s*=\s*(\d+)", re.IGNORECASE)
_FLOAT_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_PEAK_PAIR_RE = re.compile(
    r"^\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\b"
)
_QUOTED_RE = re.compile(r'"[^"]*"')

def _to_optional_str(v):
    if v is None:
        return None
    s = str(v).strip()
    return s if s else None


def _to_optional_float(v):
    if v is None:
        return None
    try:
        if not np.isfinite(float(v)):
            return None
        return float(v)
    except Exception:
        pass
    s = str(v).strip()
    if not s:
        return None
    m = _FLOAT_RE.search(s)
    if m is None:
        return None
    try:
        out = float(m.group(0))
        return out if np.isfinite(out) else None
    except Exception:
        return None


def _norm_name(v):
    s = _to_optional_str(v)
    if s is None:
        return None
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _norm_inchikey(v):
    s = _to_optional_str(v)
    if s is None:
        return None
    return s.strip().upper()


def _normalize_adduct(v):
    s = _to_optional_str(v)
    if s is None:
        return None
    key = s.replace(" ", "").upper()
    if key in ("[M+H]+", "M+H", "M+H+"):
        return "[M+H]+"
    if key in ("[M+NA]+", "M+NA", "M+NA+"):
        return "[M+Na]+"
    if key in ("[M+NH4]+", "M+NH4", "M+NH4+"):
        return "[M+NH4]+"
    return s


def _normalize_instrument(v):
    s = _to_optional_str(v)
    if s is None:
        return None
    key = s.lower().replace(" ", "").replace("-", "").replace("_", "")
    if "orbitrap" in key:
        return "Orbitrap"
    if "qtof" in key or "quadrupoletof" in key:
        return "QTOF"
    if "itft" in key:
        return "ITFT"
    return s

def _parse_nce_value(raw):
    s = _to_optional_str(raw)
    if s is None:
        return None

    low = s.lower()
    m = re.search(r"nce\s*=\s*([-+]?\d*\.?\d+)", low)
    if m is not None:
        try:
            return float(m.group(1))
        except Exception:
            return None

    # 对 HCD / Orbitrap 的纯数字，先按 NCE 处理
    m = _FLOAT_RE.search(s)
    if m is not None:
        try:
            return float(m.group(0))
        except Exception:
            return None

    return None

def _normalize_fragmentation_method(v):
    s = _to_optional_str(v)
    if s is None:
        return None

    low = s.lower()
    key = s.lower().replace(" ", "").replace("-", "").replace("_", "")

    # 真正的碎裂方式
    if (
        "hcd" in low
        or "higher-energy" in low
        or "higher energy" in low
        or "higherenergy" in key
        or "higherenergycollisionaldissociation" in key
        or "higherenergycollisiondissociation" in key
    ):
        return "HCD"

    if (
        "cid" in low
        or "collision-induced" in low
        or "collision induced" in low
        or "collisioninduceddissociation" in key
    ):
        return "CID"

    if "etd" in low:
        return "ETD"

    # 平台 / analyzer，不是碎裂方式
    if (
        "qtof" in key
        or "quadtof" in key
        or "quadrupoletof" in key
        or key in {"tof", "orbitrap", "qexactive", "exactive", "itft", "fticr"}
        or "iontrap" in key
    ):
        return None

    return None

def _parse_peak_line(line):
    """
    Safely parse MSP peak lines.

    Supports:
      375.3005 80.12 "C23H39N2O2=p-H2O/-0.3ppm 12/12"
      375   80; 393 999;

    Important:
      Quoted annotations may contain semicolons and formulas like C17H23N3.
      Do not parse quoted annotation numbers as fake peaks.
    """
    out = []
    raw = str(line).strip()
    if not raw:
        return out

    # Remove quoted annotation first, otherwise "C17H23..." can become fake mz=17,int=23.
    no_comment = _QUOTED_RE.sub("", raw)

    for token in no_comment.split(";"):
        tt = token.strip()
        if not tt:
            continue

        # Only accept a peak pair at the beginning of each token.
        m = _PEAK_PAIR_RE.match(tt)
        if m is None:
            continue

        try:
            mz = float(m.group(1))
            intensity = float(m.group(2))
        except Exception:
            continue

        if not np.isfinite(mz) or not np.isfinite(intensity):
            continue
        if mz < 0 or intensity <= 0:
            continue

        out.append((mz, intensity))

    return out
def _spect_to_csv_columns(spect):
    mzs = []
    intensities = []
    for mz, intensity in spect:
        mzs.append(str(float(mz)))
        intensities.append(str(float(intensity)))
    return ",".join(mzs), ",".join(intensities)


def _parse_mol_header(path):
    name = None
    source = None
    ann = None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            name = f.readline().rstrip("\n\r")
            source = f.readline().rstrip("\n\r")
            ann = f.readline().rstrip("\n\r")
    except Exception:
        return {
            "name_mol": None,
            "source_mol": None,
            "casno_mol": None,
            "library_id": None,
        }

    casno = None
    library_id = None
    if ann:
        m_cas = _CAS_MOL_RE.search(ann)
        if m_cas is not None:
            casno = m_cas.group(1)
        m_lib = _LIBID_MOL_RE.search(ann)
        if m_lib is not None:
            try:
                library_id = int(m_lib.group(1))
            except Exception:
                library_id = None

    return {
        "name_mol": _to_optional_str(name),
        "source_mol": _to_optional_str(source),
        "casno_mol": _to_optional_str(casno),
        "library_id": library_id,
    }


def _load_mol_index(mol_dir):
    if not os.path.isdir(mol_dir):
        raise FileNotFoundError(mol_dir)

    out = {}
    for de in os.scandir(mol_dir):
        if not de.is_file():
            continue
        m = _ID_RE.match(de.name)
        if m is None:
            continue
        mol_id = int(m.group(1))
        hdr = _parse_mol_header(de.path)
        hdr["path"] = de.path
        if hdr.get("library_id", None) is None:
            hdr["library_id"] = int(mol_id)
        out[int(mol_id)] = hdr
    return out


def _load_mol_record(mol_path):
    mol = Chem.MolFromMolFile(mol_path, sanitize=False, removeHs=False)
    if mol is None:
        return None, None, None

    smiles = None
    inchikey = None
    try:
        mol_key = Chem.Mol(mol)
        Chem.SanitizeMol(mol_key)
        smiles = Chem.MolToSmiles(mol_key, canonical=True, isomericSmiles=True)
        inchikey = Chem.MolToInchiKey(mol_key)
        return mol_key, _to_optional_str(smiles), _to_optional_str(inchikey)
    except Exception:
        try:
            mol.UpdatePropertyCache(strict=False)
        except Exception:
            pass
        try:
            smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            smiles = None
    return mol, _to_optional_str(smiles), _to_optional_str(inchikey)


def _iter_msp_records(msp_path):
    if not os.path.isfile(msp_path):
        raise FileNotFoundError(msp_path)

    def _finalize(cur):
        if cur is None:
            return None
        rec_id = _to_optional_float(cur.get("id", None))
        if rec_id is None:
            return None

        peaks = cur.get("_peaks", [])
        if not peaks:
            return None

        spect = np.asarray(peaks, dtype=np.float32)
        if spect.ndim != 2 or spect.shape[1] != 2 or spect.shape[0] <= 0:
            return None

        out = {
            "id": int(rec_id),
            "name": _to_optional_str(cur.get("name", None)),
            "precursor_type": _to_optional_str(cur.get("precursor_type", None)),
            "precursor_mz": _to_optional_float(cur.get("precursormz", None)),
            "instrument_type": _to_optional_str(cur.get("instrument_type", None)),
            "instrument": _to_optional_str(cur.get("instrument", None)),
            "collision_energy_raw": _to_optional_str(cur.get("collision_energy", None)),
            "collision_energy": _to_optional_float(cur.get("collision_energy", None)),
            "ion_mode": _to_optional_str(cur.get("ion_mode", None)),
            "formula": _to_optional_str(cur.get("formula", None)),
            "inchikey": _to_optional_str(cur.get("inchikey", None)),
            "casno": _to_optional_str(cur.get("casno", None)),
            "nistno": _to_optional_str(cur.get("nistno", None)),
            "spect": spect,
        }
        return out

    cur = None
    reading_peaks = False
    with open(msp_path, "r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.rstrip("\n\r")

            if line.startswith("Name:"):
                old = _finalize(cur)
                if old is not None:
                    yield old
                cur = {"name": line.split(":", 1)[1].strip(), "_peaks": []}
                reading_peaks = False
                continue

            if cur is None:
                continue

            stripped = line.strip()
            if not stripped:
                reading_peaks = False
                continue

            if reading_peaks:
                cur["_peaks"].extend(_parse_peak_line(stripped))
                continue

            if ":" in line:
                key, value = line.split(":", 1)
                key_norm = key.strip().lower()
                val = value.strip()
                cur[key_norm] = val
                if key_norm == "num peaks":
                    reading_peaks = True
                continue

            if "num peaks" in cur:
                cur["_peaks"].extend(_parse_peak_line(stripped))

    old = _finalize(cur)
    if old is not None:
        yield old


def load_nist20_records(
    mol_dir,
    msp_path,
    split=None,
    seed=42,
    split_ratios="0.8,0.1,0.1",
    limit=0,
    require_ce=0,
    allowed_adducts=None,
    allowed_instruments=None,
    allowed_fragmentation_methods=None,
    max_precursor_mz=0.0,
    progress_every=5000,
):
    mol_index = _load_mol_index(mol_dir)

    split_norm = None if split is None else str(split).strip().lower()
    if split_norm is not None and split_norm not in ("train", "val", "test"):
        raise ValueError(f"invalid split: {split}")

    allowed_adducts_norm = set()
    for x in _parse_csv_list(allowed_adducts):
        xx = _normalize_adduct(x)
        if xx is not None:
            allowed_adducts_norm.add(xx)

    allowed_instruments_norm = set()
    for x in _parse_csv_list(allowed_instruments):
        xx = _normalize_instrument(x)
        if xx is not None:
            allowed_instruments_norm.add(xx)

    allowed_frag_methods_norm = set()
    for x in _parse_csv_list(allowed_fragmentation_methods):
        xx = _normalize_fragmentation_method(x)
        if xx is not None:
            allowed_frag_methods_norm.add(xx)

    records = []
    stats = Counter()
    for msp in _iter_msp_records(msp_path):
        stats["msp_total"] += 1
        mol_id = int(msp["id"])

        # Use cheap MSP metadata for split assignment before loading MOL.
        msp_inchikey = _norm_inchikey(msp.get("inchikey", None))
        group_key = f"inchikey:{msp_inchikey}" if msp_inchikey else f"source_id:{mol_id}"
        rec_split = _split_name_for_group_key(
            group_key=group_key,
            seed=seed,
            split_ratios=split_ratios,
        )
        if split_norm is not None and rec_split != split_norm:
            stats["split_skip"] += 1
            continue

        # Apply MSP-level filters before expensive MOL parsing.
        adduct = _normalize_adduct(msp.get("precursor_type", None))
        # instrument 是平台，例如 Orbitrap/QTOF
        instrument_platform = _normalize_instrument(msp.get("instrument", None))

        # instrument_type 通常是碎裂方式，例如 HCD/ITFT/CID
        fragmentation_method = _normalize_fragmentation_method(msp.get("instrument_type", None))

        # 极少数记录可能把平台写在 instrument_type 里，做一个兜底。
        if instrument_platform is None:
            instrument_platform = _normalize_instrument(msp.get("instrument_type", None))
        precursor_mz = _to_optional_float(msp.get("precursor_mz", None))
        ce_raw = msp.get("collision_energy_raw", None)

        ce_nce = _parse_nce_value(ce_raw)

        ce_ev, ce_raw_str, ce_type_ev, ce_ok_ev = parse_collision_energy_to_ev(
            ce_raw,
            precursor_mz=precursor_mz,
            instrument=instrument_platform,
            charge=1,
            numeric_as_nce_for_orbitrap=(os.environ.get("NIST20_NUMERIC_CE_AS_NCE", "1") == "1"),
        )

        if os.environ.get("NIST20_USE_NCE_AS_MODEL_CE", "1") == "1":
            collision_energy = ce_nce
            ce_type = "NCE_raw" if ce_nce is not None else "missing"
            ce_ok = int(ce_nce is not None)
        else:
            collision_energy = ce_ev
            ce_type = ce_type_ev
            ce_ok = ce_ok_ev

        if int(require_ce) == 1 and collision_energy is None:
            stats["ce_missing"] += 1
            continue

        if allowed_adducts_norm:
            if adduct is None or adduct not in allowed_adducts_norm:
                stats["adduct_not_allowed"] += 1
                continue

        if allowed_instruments_norm:
            if instrument_platform is None or instrument_platform not in allowed_instruments_norm:
                stats["instrument_not_allowed"] += 1
                continue

        if allowed_frag_methods_norm:
            if fragmentation_method is None or fragmentation_method not in allowed_frag_methods_norm:
                stats["fragmentation_method_not_allowed"] += 1
                continue

        if float(max_precursor_mz) > 0:
            if precursor_mz is None or precursor_mz > float(max_precursor_mz):
                stats["precursor_mz_invalid"] += 1
                continue

        spect = np.asarray(msp.get("spect", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if spect.ndim != 2 or spect.shape[1] != 2 or spect.shape[0] <= 0:
            stats["empty_spectrum"] += 1
            continue

        # Load MOL only when record passes split + metadata checks.
        mol_meta = mol_index.get(mol_id, None)
        if mol_meta is None:
            stats["join_missing_mol"] += 1
            continue

        mol, smiles, inchikey = _load_mol_record(mol_meta["path"])
        if mol is None:
            stats["mol_parse_failed"] += 1
            continue

        cas_msp = _to_optional_str(msp.get("casno", None))
        cas_mol = _to_optional_str(mol_meta.get("casno_mol", None))
        if cas_msp is not None and cas_mol is not None and cas_msp != cas_mol:
            stats["hard_cas_mismatch"] += 1
            continue

        msp_name = _to_optional_str(msp.get("name", None))
        mol_name = _to_optional_str(mol_meta.get("name_mol", None))
        soft_name_match = None
        if msp_name is not None and mol_name is not None:
            soft_name_match = int(_norm_name(msp_name) == _norm_name(mol_name))

        soft_inchikey_match = None
        if msp_inchikey is not None and inchikey is not None:
            soft_inchikey_match = int(_norm_inchikey(inchikey) == msp_inchikey)

        record = {
            "identifier": f"nist20:{mol_id}",
            "rdmol": mol,
            "smiles": smiles,
            "inchikey": inchikey,
            "spect": spect,
            "adduct": adduct,
            "precursor_mz": precursor_mz,
            "precursor_formula": _to_optional_str(msp.get("formula", None)),
            "instrument_type": instrument_platform,
            "fragmentation_method": fragmentation_method,
            "source_id": int(mol_id),
            "casno": cas_msp if cas_msp is not None else cas_mol,
            "name_msp": msp_name,
            "name_mol": mol_name,
            "msp_inchikey": msp_inchikey,
            "soft_name_match": soft_name_match,
            "soft_inchikey_match": soft_inchikey_match,
            "nistno": _to_optional_str(msp.get("nistno", None)),
            "collision_energy": collision_energy,
            "collision_energy_raw": ce_raw_str,
            "collision_energy_type": ce_type,
            "collision_energy_parse_ok": int(ce_ok),
            "collision_energy_nce": ce_nce,
            "collision_energy_ev": ce_ev,
        }
        records.append(record)
        stats["joined_ok"] += 1
        if soft_name_match == 0:
            stats["soft_name_mismatch"] += 1
        if soft_inchikey_match == 0:
            stats["soft_inchikey_mismatch"] += 1

        if int(progress_every) > 0 and int(stats["msp_total"]) % int(progress_every) == 0:
            print(
                f"[nist20_reader] scanned_msp={stats['msp_total']} kept={stats['joined_ok']} "
                f"split={split_norm} limit={int(limit)}",
                flush=True,
            )

        if int(limit) > 0 and len(records) >= int(limit):
            print(
                f"[nist20_reader] early stop: kept={len(records)} reached limit={int(limit)} "
                f"for split={split_norm}",
                flush=True,
            )
            break

    print(f"[nist20_reader] summary split={split_norm} stats={dict(stats)}", flush=True)
    return records


def _parse_split_ratios(raw):
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        vals = [float(raw[0]), float(raw[1]), float(raw[2])]
    else:
        parts = [x.strip() for x in str(raw).split(",")]
        if len(parts) != 3:
            raise ValueError(f"split-ratios must contain 3 comma-separated values, got: {raw}")
        vals = [float(parts[0]), float(parts[1]), float(parts[2])]

    if any(v < 0 for v in vals):
        raise ValueError(f"split-ratios cannot be negative, got: {vals}")
    s = float(sum(vals))
    if s <= 0:
        raise ValueError(f"split-ratios sum must be > 0, got: {vals}")
    return [vals[0] / s, vals[1] / s, vals[2] / s]


def _parse_csv_list(raw):
    if raw is None:
        return []
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _stable_hash_u01(key, seed):
    digest = hashlib.sha1(f"{int(seed)}|{key}".encode("utf-8")).digest()
    x = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return float(x) / float(2**64)


def _split_name_for_group_key(group_key, seed=42, split_ratios="0.8,0.1,0.1"):
    train_r, val_r, test_r = _parse_split_ratios(split_ratios)
    th_train = train_r
    th_val = train_r + val_r
    h = _stable_hash_u01(group_key, seed)
    if h < th_train:
        return "train"
    if h < th_val:
        return "val"
    return "test"


def _choose_group_key(rec):
    ik = _norm_inchikey(rec.get("inchikey", None))
    if ik:
        return f"inchikey:{ik}"

    smi = _to_optional_str(rec.get("smiles", None))
    if smi:
        return f"smiles:{smi}"

    sid = rec.get("source_id", None)
    return f"source_id:{sid}"


def split_nist20_records(records, split, seed=42, split_ratios="0.8,0.1,0.1"):
    split = str(split).strip().lower()
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be one of train/val/test, got: {split}")

    train_r, val_r, test_r = _parse_split_ratios(split_ratios)
    th_train = train_r
    th_val = train_r + val_r

    group_to_split = {}
    for rec in records:
        gk = _choose_group_key(rec)
        if gk not in group_to_split:
            h = _stable_hash_u01(gk, seed)
            if h < th_train:
                group_to_split[gk] = "train"
            elif h < th_val:
                group_to_split[gk] = "val"
            else:
                group_to_split[gk] = "test"

    out = []
    for rec in records:
        gk = _choose_group_key(rec)
        if group_to_split.get(gk, "train") == split:
            out.append(rec)
    return out


def records_to_dataframes(records, split):
    df_rows = []
    tsv_rows = []
    for r in records:
        spect = np.asarray(r.get("spect", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        if spect.ndim != 2 or spect.shape[1] != 2:
            spect = spect.reshape(-1, 2)
        mzs_str, ints_str = _spect_to_csv_columns(spect)
        df_rows.append(
            {
                "rdmol": r.get("rdmol", None),
                "spect": spect,
                "smiles": r.get("smiles", None),
                "inchi_key": r.get("inchikey", None),
                "mol_id": r.get("source_id", None),
            }
        )
        tsv_rows.append(
            {
                "identifier": r.get("identifier", None),
                "mzs": mzs_str,
                "intensities": ints_str,
                "smiles": r.get("smiles", None),
                "inchikey": r.get("inchikey", None),
                "fold": str(split),
                "collision_energy": r.get("collision_energy", None),
                "adduct": r.get("adduct", None),
                "precursor_mz": r.get("precursor_mz", None),
                "precursor_formula": r.get("precursor_formula", None),
                "instrument_type": r.get("instrument_type", None),
                "fragmentation_method": r.get("fragmentation_method", None),
                "collision_energy_raw": r.get("collision_energy_raw", None),
                "collision_energy_type": r.get("collision_energy_type", None),
                "collision_energy_nce": r.get("collision_energy_nce", None),
                "collision_energy_ev": r.get("collision_energy_ev", None),
                "collision_energy_parse_ok": r.get("collision_energy_parse_ok", None),
            }
        )
    return df_rows, tsv_rows
