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
    parts = [prefix]
    if not isinstance(metrics, dict):
        return str(prefix)
    for key, value in metrics.items():
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
