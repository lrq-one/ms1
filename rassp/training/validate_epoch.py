import torch
from tqdm import tqdm

from rassp.training.batch_utils import move_batch_to_device, prepare_batch_cpu
from rassp.training.logging_utils import MetricAccumulator
from rassp.training.official_metrics import compute_batch_official_metrics
from rassp.training.selector_metrics import (
    compute_selector_eval_pack,
    compute_selected_support_metrics,
    select_model_topk_indices,
)
from rassp.training.selector_losses import build_selector_utility_tensors

def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    device,
    spect_bin,
    spect_bin_centers,
    epoch,
    run_cfg=None,
    selector_cfg=None,
    metric_cfg=None,
):
    model.eval()
    acc = MetricAccumulator()
    max_steps = int(_cfg_value(run_cfg, "max_val_steps", 0) or 0)
    metric_cfg = metric_cfg or {}

    for step, raw_batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Val]")):
        if max_steps > 0 and step >= max_steps:
            break

        processed = prepare_batch_cpu(raw_batch, spect_bin)
        batch = move_batch_to_device(processed, device)
        res = model(**batch)

        pred_spect = res.get("spect", None) if isinstance(res, dict) else None
        if torch.is_tensor(pred_spect):
            official = compute_batch_official_metrics(
                raw_batch=batch,
                pred_spect=pred_spect,
                spect_bin_centers=spect_bin_centers,
                metric_cfg=metric_cfg,
                pred_exact_peaks=res.get("pred_exact_peaks", None),
                debug_ctx=None,
            )
            if isinstance(official, dict):
                for key, value in official.items():
                    if isinstance(value, list):
                        for item in value:
                            acc.add(key, item)
                    else:
                        acc.add(key, value)

        selector_logits = res.get("selector_logits", None) if isinstance(res, dict) else None
        if torch.is_tensor(selector_logits):
            topk_idx = select_model_topk_indices(
                selector_logits=selector_logits,
                batch=batch,
                k=int(_cfg_value(selector_cfg, "model_topk_eval", 64)),
                use_coverage=bool(_cfg_value(selector_cfg, "use_coverage_aware_topk", False)),
                use_group_unique=bool(_cfg_value(selector_cfg, "use_group_unique_model", False)),
            )
            selector_metrics = compute_selected_support_metrics(topk_idx, batch)
            suffix = int(_cfg_value(selector_cfg, "model_topk_eval", 64))
            acc.add_dict({f"{key}@{suffix}": value for key, value in selector_metrics.items()})
            utility, utility_dist, valid_mask, util_stats = build_selector_utility_tensors(
                selector_logits,
                batch,
            )

            if utility is not None and valid_mask is not None:
                utility_topk_idx = select_model_topk_indices(
                    selector_logits=utility,
                    batch=batch,
                    k=int(_cfg_value(selector_cfg, "model_topk_eval", 64)),
                    use_coverage=False,
                    use_group_unique=bool(_cfg_value(selector_cfg, "use_group_unique_model", False)),
                )

                utility_metrics = compute_selected_support_metrics(utility_topk_idx, batch)
                suffix = int(_cfg_value(selector_cfg, "model_topk_eval", 64))

                acc.add_dict({
                    f"utility_topk_{key}@{suffix}": value
                    for key, value in utility_metrics.items()
                })

                if isinstance(util_stats, dict):
                    for k, v in util_stats.items():
                        if torch.is_tensor(v):
                            acc.add(f"val_{k}", v.detach().item())
    return acc.mean_dict()
