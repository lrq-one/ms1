import json
import os
import re

import numpy as np
import torch


def my_collate_fn(batch):
    batch_dict = {}
    for key in batch[0].keys():
        batch_dict[key] = [d[key] for d in batch]
    return batch_dict


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


def _load_vocab_from_json(env_var_name, fallback_vocab, field_name):
    path = os.environ.get(env_var_name, "").strip()
    if not path:
        return dict(fallback_vocab)
    if not os.path.exists(path):
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


def _vocab_uses_separate_unknown(vocab):
    return isinstance(vocab, dict) and MISSING_TOKEN in vocab and UNKNOWN_TOKEN in vocab


def _vocab_missing_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(MISSING_TOKEN, 0))
        except Exception:
            return 0
    return 0


def _vocab_unknown_index(vocab):
    if _vocab_uses_separate_unknown(vocab):
        try:
            return int(vocab.get(UNKNOWN_TOKEN, 1))
        except Exception:
            return 1
    return 0


def _instrument_norm_key(val):
    return str(val).strip().lower().replace(" ", "").replace("-", "").replace("_", "")


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
        sval = str(val).strip()
        out.append(ADDUCT_VOCAB.get(sval, unknown_idx))
    return torch.as_tensor(out, dtype=torch.long)


def _normalize_instrument_key(val):
    return _instrument_norm_key(val)


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
            elif "tof" == key:
                idx = INSTRUMENT_VOCAB_NORM.get("tof", unknown_idx)
            elif "iontrap" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("iontrap", unknown_idx)
            elif "fticr" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("fticr", unknown_idx)
            elif "triplequad" in key or "triplequadrupole" in key:
                idx = INSTRUMENT_VOCAB_NORM.get("triplequad", unknown_idx)
        out.append(idx)
    return torch.as_tensor(out, dtype=torch.long)


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
            try:
                raw = int(float(sval))
            except Exception:
                m = re.search(r"\d+", sval)
                if m is None:
                    out.append(unknown_idx if has_vocab else 0)
                    continue
                raw = int(m.group(0))
        if has_vocab:
            out.append(int(MS_LEVEL_VOCAB.get(str(raw), unknown_idx)))
        else:
            out.append(raw)
    return torch.as_tensor(out, dtype=torch.long)


def _stack_list_to_tensor(values, target_dtype=None):
    if values is None or len(values) == 0:
        return values

    first = values[0]
    try:
        if torch.is_tensor(first):
            t = torch.stack([v if torch.is_tensor(v) else torch.as_tensor(v) for v in values])
            return t.to(dtype=target_dtype) if target_dtype is not None else t

        arr = np.asarray(values)
        if arr.dtype != object:
            t = torch.as_tensor(arr)
            return t.to(dtype=target_dtype) if target_dtype is not None else t
    except Exception:
        pass

    try:
        elems = [torch.as_tensor(v) for v in values]
        t = torch.stack(elems)
        return t.to(dtype=target_dtype) if target_dtype is not None else t
    except Exception:
        return values


def prepare_batch_cpu(raw_batch, spect_bin):
    try:
        prefix_topk = int(os.environ.get("CANDIDATE_PREFIX_TOPK", "0"))
    except Exception:
        prefix_topk = 0

    processed_batch = {}
    for k, v in raw_batch.items():
        if k == "spect":
            dense = _stack_list_to_tensor(v, target_dtype=torch.float32)
            if torch.is_tensor(dense) and dense.dim() == 2 and dense.shape[1] == 1024:
                processed_batch[k] = dense
            else:
                spect_list = [to_dense_binned_spectrum(s, spect_bin) for s in v]
                processed_batch[k] = torch.stack(spect_list)

        elif k == "spect_raw":
            processed_batch[k] = v

        elif k in ["formulae_feats", "formula_feats"]:
            processed_batch[k] = _stack_list_to_tensor(v, target_dtype=torch.long)

        elif k in [
            "true_official_idx",
            "true_official_intensity",
            "true_top20_official_idx",
            "true_top20_official_intensity",
            "true_all_official_idx",
            "true_all_official_intensity",
        ]:
            processed_batch[k] = v

        elif k in [
            "vect_feat", "adj", "input_mask", "mol_feat", "ce", "precursor_mz",
            "ce_missing", "adduct_missing", "instrument_missing", "ms_level_missing",
            "precursor_mz_missing", "formulae_mask", "teacher_formula_probs",
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            "true_precursor_intensity",
            "true_precursor_prob_in_all",
            "true_precursor_present",
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            "true_precursor_bin",
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in ["formulae_n_raw", "formulae_n_kept", "formulae_truncated"]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v
        elif k in ["formulae_active_mask", "formulae_prior_score"]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in ["formulae_source_flag", "formulae_break_depth", "formulae_ring_cut_flag"]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k == "adduct":
            processed_batch[k] = encode_adduct_batch(v)

        elif k == "instrument_type":
            processed_batch[k] = encode_instrument_batch(v)

        elif k == "ms_level":
            processed_batch[k] = encode_ms_level_batch(v)

        elif k in [
            "formulae_features", "formulae_peaks", "formulae_peaks_mass_idx", "formulae_peaks_intensity",
            "formulae_peaks_official_idx", "formulae_peaks_official_intensity",
            "formulae_peaks_official_idx_agg", "formulae_peaks_official_intensity_agg",
            "formulae_aux_feat", "formulae_frag_aux_feat", "formulae_instance_is_source",
            "formulae_instance_group_id", "formulae_instance_depth", "formulae_instance_h_shift",
            "vert_element_oh",
            "fragment_node_mz", "fragment_node_intensity", "fragment_node_local_feat",
            "fragment_node_depth", "fragment_node_h_shift", "fragment_node_is_brics",
            "fragment_node_ring_cut", "fragment_node_atom_count", "fragment_node_cut_count",
            "fragment_node_source_type", "fragment_node_mask", "fragment_node_label",
            "fragment_node_true_intensity", "fragment_node_true_intensity_share",
            "fragment_node_bin_dup_count", "fragment_node_label_top20",
        ]:
            if k in ["formulae_features"]:
                stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            elif ("idx" in k) or ("mass_idx" in k):
                stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            else:
                stacked = _stack_list_to_tensor(v, target_dtype=torch.float32)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        elif k in [
            "fragment_node_formula", "fragment_node_official_idx", "fragment_node_group_formula_id", "fragment_node_n_valid",
        ]:
            stacked = _stack_list_to_tensor(v, target_dtype=torch.long)
            processed_batch[k] = stacked if torch.is_tensor(stacked) else v

        else:
            processed_batch[k] = v

    if prefix_topk > 0 and torch.is_tensor(processed_batch.get("formulae_features", None)):
        cand_n = int(processed_batch["formulae_features"].shape[1])
        keep_n = max(1, min(int(prefix_topk), cand_n))

        for kk in [
            "formulae_features",
            "formulae_peaks",
            "formulae_peaks_mass_idx",
            "formulae_peaks_intensity",
            "formulae_peaks_official_idx",
            "formulae_peaks_official_intensity",
            "formulae_peaks_official_idx_agg",
            "formulae_peaks_official_intensity_agg",
            "formulae_aux_feat",
            "formulae_frag_aux_feat",
            "formulae_active_mask",
            "formulae_prior_score",
            "formulae_source_flag",
            "formulae_break_depth",
            "formulae_ring_cut_flag",
            "formulae_mask",
            "teacher_formula_probs",
            "formulae_instance_is_source",
            "formulae_instance_group_id",
            "formulae_instance_depth",
            "formulae_instance_h_shift",
        ]:
            vv = processed_batch.get(kk, None)
            if torch.is_tensor(vv) and vv.dim() >= 2 and int(vv.shape[1]) >= keep_n:
                processed_batch[kk] = vv[:, :keep_n, ...]

        n_kept = processed_batch.get("formulae_n_kept", None)
        if torch.is_tensor(n_kept):
            processed_batch["formulae_n_kept"] = torch.clamp(n_kept.long(), max=int(keep_n))

        n_raw = processed_batch.get("formulae_n_raw", None)
        if torch.is_tensor(n_raw):
            trunc = (n_raw.long() > int(keep_n)).long()
            if torch.is_tensor(processed_batch.get("formulae_truncated", None)):
                processed_batch["formulae_truncated"] = torch.maximum(
                    processed_batch["formulae_truncated"].long(),
                    trunc,
                )
            else:
                processed_batch["formulae_truncated"] = trunc

    for kk in [
        "formulae_peaks_mass_idx",
        "formulae_peaks_intensity",
        "formulae_peaks_official_idx",
        "formulae_peaks_official_intensity",
        "formulae_peaks_official_idx_agg",
        "formulae_peaks_official_intensity_agg",
        "true_precursor_bin",
        "true_precursor_intensity",
        "true_precursor_prob_in_all",
        "true_precursor_present",
    ]:
        vv = processed_batch.get(kk, None)
        if isinstance(vv, list):
            try:
                arr = np.asarray(vv)
                if ("idx" in kk) or ("mass_idx" in kk) or kk in ["true_precursor_bin"]:
                    processed_batch[kk] = torch.as_tensor(arr, dtype=torch.long)
                else:
                    processed_batch[kk] = torch.as_tensor(arr, dtype=torch.float32)
            except Exception:
                pass
    return processed_batch


def move_batch_to_device(processed_batch, device):
    batch = {}
    for k, v in processed_batch.items():
        if torch.is_tensor(v):
            batch[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and torch.is_tensor(v[0]):
            batch[k] = [i.to(device, non_blocking=True) for i in v]
        else:
            batch[k] = v
    return batch
