import math
import torch

from rassp.training.runtime_selector_targets import (
    build_selector_teacher_dist_from_official_overlap,
    build_selector_teacher_dist_setcover,
)
from rassp.training.selector_metrics import (
    select_model_topk_indices,
    compute_selected_support_metrics,
)


@torch.no_grad()
def compute_teacher_audit_pack(
    selector_logits,
    batch,
    selector_cfg,
    metric_cfg,
):
    out = {}

    if selector_logits is None or not torch.is_tensor(selector_logits):
        return out

    formulae_mask = batch.get("formulae_mask", None)
    if formulae_mask is None or not torch.is_tensor(formulae_mask):
        return out

    device = selector_logits.device
    B, M = selector_logits.shape[:2]

    fm = formulae_mask.float().to(device)
    if fm.dim() > 2:
        fm = fm.reshape(fm.shape[0], -1)
    fm = fm[:B, :M]

    official_bin_width = float(metric_cfg.get("bin_width", 0.01))
    official_max_mz = float(metric_cfg.get("max_mz", 1005.0))
    official_bin_n = int(math.floor(official_max_mz / official_bin_width)) + 1

    k = int(getattr(selector_cfg, "model_topk_eval", 64))
    use_group_unique = bool(getattr(selector_cfg, "use_group_unique_model", False))

    teacher_scores = {}

    overlap_dist = build_selector_teacher_dist_from_official_overlap(
        batch=batch,
        formulae_mask=fm,
        official_bin_n=official_bin_n,
    )
    if torch.is_tensor(overlap_dist):
        teacher_scores["overlap_teacher"] = overlap_dist

    setcover_dist = build_selector_teacher_dist_setcover(
        batch=batch,
        formulae_mask=fm,
        official_bin_n=official_bin_n,
    )
    if torch.is_tensor(setcover_dist):
        teacher_scores["setcover_teacher"] = setcover_dist

    for name, score in teacher_scores.items():
        score = score.to(device=device).float()[:B, :M]

        topk_idx = select_model_topk_indices(
            selector_logits=score,
            batch=batch,
            k=k,
            use_coverage=False,
            use_group_unique=use_group_unique,
        )

        metrics = compute_selected_support_metrics(topk_idx, batch)

        for mk, mv in metrics.items():
            out[f"{name}_{mk}@{k}"] = mv

        out[f"{name}_selected_n"] = float(
            (score > 0).float().sum(dim=1).mean().detach().cpu().item()
        )

    return out