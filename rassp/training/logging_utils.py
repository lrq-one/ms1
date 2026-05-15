import math

import numpy as np


def safe_mean(values, default=0.0):
    if values is None:
        return default
    vals = []
    for value in values:
        try:
            fv = float(value)
        except Exception:
            continue
        if math.isfinite(fv):
            vals.append(fv)
    if len(vals) == 0:
        return default
    return float(np.mean(vals))


def format_metric_line(prefix, metrics):
    keep_keys = [
        "train_loss",
        "train_official_spectral",
        "train_official_dense_false",
        "train_selector_loss",
        "train_selector_bce",
        "train_selector_recall_bce",
        "train_selector_kl",
        "train_selector_pairwise",
        "train_selector_utility",
        "train_false_support",
        "val_loss",
        "val_official_cos_no_precursor",
        "val_official_js_no_precursor",
        "val_false_support",
        "val_model_topk_oracle_cos@8",
        "val_model_topk_oracle_false_support@8",
        "val_selected_true_hit_mass@8",
        "val_selected_false_mass@8",
        "val_model_topk_oracle_cos@16",
        "val_model_topk_oracle_false_support@16",
        "val_selected_true_hit_mass@16",
        "val_selected_false_mass@16",
        "val_model_topk_oracle_cos@32",
        "val_model_topk_oracle_false_support@32",
        "val_selected_true_hit_mass@32",
        "val_selected_false_mass@32",
        "val_model_topk_oracle_cos@64",
        "val_model_topk_oracle_false_support@64",
        "val_selected_true_hit_mass@64",
        "val_selected_false_mass@64",
        "val_model_topk_oracle_cos@128",
        "val_model_topk_oracle_false_support@128",
        "val_selected_true_hit_mass@128",
        "val_selected_false_mass@128",
        "val_teacher_oracle_cos",
        "val_teacher_oracle_false_support",
        "train_selector_hard_topk_ce",
        "train_selector_hard_topk_margin",
        "train_selector_hard_topk_bce",
        "train_selector_teacher_topk_recall",
        "train_selector_teacher_topk_precision",
        "val_overlap_teacher_selected_true_hit_mass@8",
        "val_overlap_teacher_selected_false_mass@8",
        "val_overlap_teacher_selected_n",
        "val_setcover_teacher_selected_true_hit_mass@8",
        "val_setcover_teacher_selected_false_mass@8",
        "val_setcover_teacher_selected_n",
    ]

    parts = [prefix]
    if not isinstance(metrics, dict):
        return str(prefix)

    for key in keep_keys:
        if key not in metrics:
            continue
        value = metrics[key]
        try:
            parts.append(f"{key}={float(value):.4f}")
        except Exception:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


class MetricAccumulator:
    def __init__(self):
        self.data = {}

    def add(self, key, value):
        if value is None:
            return
        try:
            value = float(value)
        except Exception:
            return
        if not math.isfinite(value):
            return
        self.data.setdefault(key, []).append(value)

    def add_dict(self, values):
        if not isinstance(values, dict):
            return
        for key, value in values.items():
            self.add(key, value)

    def mean_dict(self):
        return {key: safe_mean(values) for key, values in self.data.items()}
