import os

import torch


def save_checkpoint(path, model, optimizer=None, scaler=None, epoch=None, metrics=None, extra=None):
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    obj = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics or {},
        "extra": extra or {},
    }

    if optimizer is not None:
        obj["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None:
        try:
            obj["scaler_state_dict"] = scaler.state_dict()
        except Exception:
            pass

    torch.save(obj, path)


def is_better_metric(current, best, mode="max"):
    if current is None:
        return False
    if best is None:
        return True

    try:
        current = float(current)
        best = float(best)
    except Exception:
        return False

    if mode == "min":
        return current < best
    return current > best
