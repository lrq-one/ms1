# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Utility script for data preparation, cache building, model evaluation, or diagnostics around the main pipeline.

import argparse
import json
import os
import pickle
import re
import sqlite3
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from rassp.featurize import msutil
from rassp import dataset, netutil
from rassp.model import formulaenets
from train_ms_subsetnet import prepare_batch_cpu, move_batch_to_device

ADDUCT_VOCAB = {
    "[M+H]+": 1,
    "[M+Na]+": 2,
    "[M+K]+": 3,
    "[M-H]-": 4,
    "[M+NH4]+": 5,
}

MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNK__"

INSTRUMENT_VOCAB = {
    "orbitrap": 1,
    "qtof": 2,
    "tof": 3,
    "iontrap": 4,
    "fticr": 5,
    "triplequad": 6,
}


# Function overview: _load_vocab_from_json handles a specific workflow step in this module.
def _load_vocab_from_json(env_var_name, fallback_vocab, field_name):
    path = os.environ.get(env_var_name, "").strip()
    if not path or (not os.path.exists(path)):
        return dict(fallback_vocab)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and field_name in obj and isinstance(obj[field_name], dict):
            raw = obj[field_name]
        elif isinstance(obj, dict):
            raw = obj
        else:
            return dict(fallback_vocab)
        out = {}
        for k, v in raw.items():
            try:
                idx = int(v)
            except Exception:
                continue
            if idx >= 0:
                out[str(k)] = idx
        return out if len(out) > 0 else dict(fallback_vocab)
    except Exception:
        return dict(fallback_vocab)


# Function overview: _vocab_uses_separate_unknown handles a specific workflow step in this module.
def _vocab_uses_separate_unknown(vocab):
    return isinstance(vocab, dict) and MISSING_TOKEN in vocab and UNKNOWN_TOKEN in vocab


# Function overview: _vocab_missing_index handles a specific workflow step in this module.
def _vocab_missing_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(MISSING_TOKEN, 0))
        except Exception:
            return 0
    return 0


# Function overview: _vocab_unknown_index handles a specific workflow step in this module.
def _vocab_unknown_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(UNKNOWN_TOKEN, 1))
        except Exception:
            return 1
    return 0


# Function overview: _instrument_norm_key handles a specific workflow step in this module.
def _instrument_norm_key(val):
    return str(val).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


# Function overview: _build_normalized_instrument_vocab handles a specific workflow step in this module.
def _build_normalized_instrument_vocab(vocab):
    out = {}
    if not isinstance(vocab, dict):
        return out
    for k, v in vocab.items():
        if str(k) in (MISSING_TOKEN, UNKNOWN_TOKEN):
            continue
        try:
            idx = int(v)
        except Exception:
            continue
        if idx <= 0:
            continue
        nk = _instrument_norm_key(k)
        if nk and nk not in out:
            out[nk] = idx
    return out


ADDUCT_VOCAB = _load_vocab_from_json("ADDUCT_VOCAB_PATH", ADDUCT_VOCAB, "adduct_vocab")
INSTRUMENT_VOCAB = _load_vocab_from_json("INSTRUMENT_VOCAB_PATH", INSTRUMENT_VOCAB, "instrument_vocab")
MS_LEVEL_VOCAB = _load_vocab_from_json("MS_LEVEL_VOCAB_PATH", {}, "ms_level_vocab")
INSTRUMENT_VOCAB_NORM = _build_normalized_instrument_vocab(INSTRUMENT_VOCAB)


# Function overview: encode_adduct_batch handles a specific workflow step in this module.
def encode_adduct_batch(values):
    missing_idx = _vocab_missing_index(ADDUCT_VOCAB)
    unknown_idx = _vocab_unknown_index(ADDUCT_VOCAB)
    out = []
    for val in values:
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            out.append(int(val))
            continue
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
            else:
                out.append(int(val))
            continue
        out.append(ADDUCT_VOCAB.get(str(val).strip(), unknown_idx))
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: _normalize_instrument_key handles a specific workflow step in this module.
def _normalize_instrument_key(val):
    return _instrument_norm_key(val)


# Function overview: encode_instrument_batch handles a specific workflow step in this module.
def encode_instrument_batch(values):
    missing_idx = _vocab_missing_index(INSTRUMENT_VOCAB)
    unknown_idx = _vocab_unknown_index(INSTRUMENT_VOCAB)
    out = []
    for val in values:
        if val is None or (isinstance(val, str) and not val.strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            out.append(int(val))
            continue
        if isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
            else:
                out.append(int(val))
            continue
        key = _normalize_instrument_key(val)
        idx = INSTRUMENT_VOCAB_NORM.get(key, unknown_idx)
        if idx == unknown_idx:
            if "orbitrap" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("orbitrap", unknown_idx)
            elif "qtof" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("qtof", unknown_idx)
            elif key == "tof":
                idx = INSTRUMENT_VOCAB_NORM.get("tof", unknown_idx)
            elif "iontrap" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("iontrap", unknown_idx)
            elif "fticr" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("fticr", unknown_idx)
            elif "triplequad" in key or "triplequadrupole" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("triplequad", unknown_idx)
        out.append(idx)
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: encode_ms_level_batch handles a specific workflow step in this module.
def encode_ms_level_batch(values):
    out = []
    has_vocab = len(MS_LEVEL_VOCAB) > 0
    missing_idx = _vocab_missing_index(MS_LEVEL_VOCAB)
    unknown_idx = _vocab_unknown_index(MS_LEVEL_VOCAB)
    for val in values:
        if val is None or (isinstance(val, str) and not str(val).strip()):
            out.append(missing_idx)
            continue
        if isinstance(val, (int, np.integer)):
            raw = int(val)
        elif isinstance(val, (float, np.floating)):
            if np.isnan(val):
                out.append(missing_idx)
                continue
            raw = int(val)
        else:
            sval = str(val).strip().lower()
            if not sval:
                out.append(missing_idx)
                continue
            else:
                try:
                    raw = int(float(sval))
                except Exception:
                    m = re.search(r"\d+", sval)
                    if m is None:
                        out.append(unknown_idx if has_vocab else 0)
                        continue
                    raw = int(m.group(0))
        out.append(int(MS_LEVEL_VOCAB.get(str(raw), unknown_idx)) if has_vocab else raw)
    return torch.as_tensor(out, dtype=torch.long)


# Function overview: to_dense_binned_spectrum handles a specific workflow step in this module.
def to_dense_binned_spectrum(s, spect_bin_obj):
    if isinstance(s, torch.Tensor):
        s = s.detach().cpu().numpy()

    if isinstance(s, np.ndarray) and s.ndim == 1 and s.shape[0] == 1024:
        return torch.as_tensor(s, dtype=torch.float32)

    if isinstance(s, (list, tuple)) and len(s) == 1024 and not isinstance(s[0], (list, tuple, np.ndarray, torch.Tensor)):
        return torch.as_tensor(s, dtype=torch.float32)

    if isinstance(s, np.ndarray) and s.dtype == object and s.ndim == 1:
        try:
            peaks = np.stack([np.asarray(x, dtype=np.float32) for x in s], axis=0)
        except Exception:
            peaks = np.asarray(s, dtype=np.float32)
    else:
        peaks = np.asarray(s, dtype=np.float32)

    if peaks.size == 0:
        return torch.zeros(1024, dtype=torch.float32)

    if peaks.ndim != 2 or peaks.shape[1] != 2:
        try:
            peaks = peaks.reshape(-1, 2)
        except Exception:
            return torch.zeros(1024, dtype=torch.float32)

    _, _, spect_out = spect_bin_obj.histogram(peaks[:, 0], peaks[:, 1])
    return torch.as_tensor(spect_out, dtype=torch.float32)


# Function overview: my_collate_fn handles a specific workflow step in this module.
def my_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        out[key] = [d[key] for d in batch]
    return out


# Function overview: count_cache_pickles handles a specific workflow step in this module.
def count_cache_pickles(cache_dir):
    if not cache_dir or (not os.path.isdir(cache_dir)):
        return 0
    n = 0
    for name in os.listdir(cache_dir):
        if name.endswith(".pkl"):
            n += 1
    return n


# Function overview: create_sqlite handles a specific workflow step in this module.
def create_sqlite(out_sqlite):
    os.makedirs(os.path.dirname(out_sqlite), exist_ok=True)
    conn = sqlite3.connect(out_sqlite)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS spect")
    cur.execute(
        """
        CREATE TABLE spect (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mol_id TEXT,
            phase TEXT,
            source_idx INTEGER,
            n_peaks INTEGER,
            spect BLOB
        )
        """
    )
    conn.commit()
    return conn


# Function overview: dense_to_sparse_peaks handles a specific workflow step in this module.
def dense_to_sparse_peaks(pred_vec, bin_centers, min_prob=1e-6, normalize=True):
    p = pred_vec.astype(np.float32)
    if normalize:
        denom = float(np.sum(p))
        if denom > 1e-12:
            p = p / denom
    nz = p > float(min_prob)
    if not np.any(nz):
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([bin_centers[nz].astype(np.float32), p[nz].astype(np.float32)], axis=-1)


# Function overview: exact_to_sparse_peaks handles a specific workflow step in this module.
def exact_to_sparse_peaks(peaks, min_prob=1e-6, normalize=True):
    if peaks is None:
        return np.zeros((0, 2), dtype=np.float32)

    if torch.is_tensor(peaks):
        arr = peaks.detach().cpu().numpy()
    else:
        arr = np.asarray(peaks)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if arr.dtype == object:
        try:
            arr = np.stack([np.asarray(x, dtype=np.float32) for x in arr], axis=0)
        except Exception:
            arr = np.asarray(arr, dtype=np.float32)

    try:
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
    except Exception:
        return np.zeros((0, 2), dtype=np.float32)

    valid = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1]) & (arr[:, 1] > float(min_prob))
    arr = arr[valid]
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    if normalize:
        denom = float(np.sum(arr[:, 1]))
        if denom > 1e-12:
            arr = arr.copy()
            arr[:, 1] = arr[:, 1] / denom

    return arr.astype(np.float32)


# Function overview: parse_args handles a specific workflow step in this module.
def parse_args():
    parser = argparse.ArgumentParser(description="Forward evaluate MassSpecGym model and save predictions to sqlite")
    parser.add_argument("--dataset", default="data/massspecgym/massspecgym_val.parquet")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--meta", default="checkpoints/best_model.meta")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.model")
    parser.add_argument("--out", default="forward.preds/massspecgym_best_model.spect.sqlite")
    parser.add_argument("--phase", default="test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--min-prob", type=float, default=1e-6)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


# Function overview: main handles a specific workflow step in this module.
def main():
    # Stage 1: parse args and load model metadata/checkpoint artifacts.
    args = parse_args()

    if not os.path.exists(args.meta):
        raise FileNotFoundError(f"meta file not found: {args.meta}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"checkpoint file not found: {args.checkpoint}")

    with open(args.meta, "rb") as f:
        meta = pickle.load(f)

    spect_bin_config = meta["spectrum_bin_config"]
    featurize_config = meta["featurize_config"]

    cache_dir = args.cache_dir.strip()
    if not cache_dir:
        cache_dir = os.environ.get("FEAT_CACHE_DIR", "").strip()
    if not cache_dir:
        cache_dir = args.dataset + ".cache"

    # Stage 2: load dataset, preferring pre-built cache when available.
    prev_cache = os.environ.get("FEAT_CACHE_DIR")
    try:
        if cache_dir and count_cache_pickles(cache_dir) > 0:
            os.environ["FEAT_CACHE_DIR"] = cache_dir
        elif "FEAT_CACHE_DIR" in os.environ:
            del os.environ["FEAT_CACHE_DIR"]

        spect_bin = msutil.binutils.create_spectrum_bins(**spect_bin_config)
        ds = dataset.ParquetDataset(args.dataset, spect_bin, featurize_config, {})
    finally:
        if prev_cache is None:
            if "FEAT_CACHE_DIR" in os.environ:
                del os.environ["FEAT_CACHE_DIR"]
        else:
            os.environ["FEAT_CACHE_DIR"] = prev_cache

    if args.max_samples > 0 and args.max_samples < len(ds):
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(ds), size=args.max_samples, replace=False).tolist()
        ds = torch.utils.data.Subset(ds, idx)

    if len(ds) == 0:
        raise RuntimeError("dataset is empty after filtering")


    # Stage 3: reconstruct model (module object or state_dict path).
    ckpt_obj = torch.load(args.checkpoint, map_location="cpu")

    if not isinstance(ckpt_obj, torch.nn.Module):
        raise RuntimeError(
            f"--checkpoint must be a full .model checkpoint, but got object type: {type(ckpt_obj)}. "
            f"Please pass best_model.model instead of best_model.state"
        )

    model = ckpt_obj
    if hasattr(model, "module"):
        model = model.module

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = model.to(device)
    model.eval()

    print(f"[forward] model_type={type(model).__name__}")
    print(f"[forward] model_file={getattr(formulaenets, '__file__', 'unknown')}")
    print(f"[forward] backend={os.environ.get('FORMULA_PROJ_IMPL', 'scatter').strip().lower()}")
    print(f"[forward] device={device}")

    # Stage 4: create dataloader + sqlite sink for streaming predictions.
    dl_kwargs = dict(
        dataset=ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=my_collate_fn,
    )
    if args.num_workers > 0:
        dl_kwargs["persistent_workers"] = True
        dl_kwargs["prefetch_factor"] = args.prefetch_factor

    dl = torch.utils.data.DataLoader(**dl_kwargs)

    conn = create_sqlite(args.out)
    cur = conn.cursor()

    bin_centers = spect_bin.get_bin_centers().astype(np.float32)
    normalize_pred = not args.no_normalize

    inserted = 0
    t0 = time.time()

    # Stage 5: forward pass batches and persist sparse peak arrays into sqlite.
    with torch.no_grad():
        pbar = tqdm(dl, desc="Forward eval")
        for step_i, raw_batch in enumerate(pbar, start=1):
            if args.max_steps > 0 and step_i > args.max_steps:
                break

            processed_batch = prepare_batch_cpu(raw_batch, spect_bin)
            batch = move_batch_to_device(processed_batch, device)
            res = model(**batch)
            pred = res["spect"] if isinstance(res, dict) and "spect" in res else res
            pred_np = pred.detach().cpu().numpy()
            pred_exact_batch = res.get("pred_exact_peaks", None) if isinstance(res, dict) else None

            mol_id_list = raw_batch.get("mol_id", None)
            if mol_id_list is None:
                mol_id_list = raw_batch.get("input_idx", list(range(inserted, inserted + pred_np.shape[0])))
            if torch.is_tensor(mol_id_list):
                mol_id_list = mol_id_list.detach().cpu().numpy().tolist()
            idx_list = raw_batch.get("input_idx", list(range(inserted, inserted + pred_np.shape[0])))
            if torch.is_tensor(idx_list):
                idx_list = idx_list.detach().cpu().numpy().tolist()

            rows = []
            for i in range(pred_np.shape[0]):
                exact_peaks_i = None
                if isinstance(pred_exact_batch, (list, tuple)) and i < len(pred_exact_batch):
                    exact_peaks_i = pred_exact_batch[i]
                elif torch.is_tensor(pred_exact_batch) and pred_exact_batch.dim() >= 3 and i < int(pred_exact_batch.shape[0]):
                    exact_peaks_i = pred_exact_batch[i]
                elif isinstance(pred_exact_batch, np.ndarray) and pred_exact_batch.ndim >= 3 and i < int(pred_exact_batch.shape[0]):
                    exact_peaks_i = pred_exact_batch[i]

                if exact_peaks_i is not None:
                    peaks = exact_to_sparse_peaks(exact_peaks_i, min_prob=args.min_prob, normalize=normalize_pred)
                else:
                    peaks = dense_to_sparse_peaks(pred_np[i], bin_centers, min_prob=args.min_prob, normalize=normalize_pred)
                mol_id_val = mol_id_list[i]
                rows.append((
                    str(mol_id_val),
                    args.phase,
                    int(idx_list[i]),
                    int(peaks.shape[0]),
                    sqlite3.Binary(pickle.dumps(peaks, protocol=pickle.HIGHEST_PROTOCOL)),
                ))

            cur.executemany(
                "INSERT INTO spect (mol_id, phase, source_idx, n_peaks, spect) VALUES (?, ?, ?, ?, ?)",
                rows,
            )
            conn.commit()

            inserted += len(rows)
            pbar.set_postfix(inserted=inserted)

    conn.close()

    # Stage 6: finalize and report throughput summary.
    dt = time.time() - t0
    print(f"[forward] done: inserted={inserted}, elapsed={dt:.2f}s, out={args.out}")


if __name__ == "__main__":
    main()
