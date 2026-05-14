import torch
from tqdm import tqdm
import os
from rassp.training.teacher_audit import compute_teacher_audit_pack
from rassp.training.batch_utils import move_batch_to_device, prepare_batch_cpu
from rassp.training.logging_utils import MetricAccumulator
from rassp.training.official_metrics import compute_batch_official_metrics
from rassp.training.selector_metrics import (
    build_mask_from_topk_indices,
    compute_candidate_support_stats,
    compute_selector_eval_pack,
    compute_selected_support_metrics,
    select_model_topk_indices,
)

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

        pred_spect = None
        if isinstance(res, dict):
            pred_spect = res.get("spect_out_official", None)
            if not torch.is_tensor(pred_spect):
                pred_spect = res.get("spect", None)
        if torch.is_tensor(pred_spect):
            official = compute_batch_official_metrics(
                raw_batch=batch,
                pred_spect=pred_spect,
                spect_bin_centers=spect_bin_centers,
                metric_cfg=metric_cfg,
                pred_exact_peaks=None,
                debug_ctx={
                    "enabled": os.environ.get("DEBUG_EVAL_SUPPORT", "0") == "1",
                    "epoch": epoch,
                    "batch": step,
                    "printed": 0,
                    "max_samples": int(os.environ.get("DEBUG_EVAL_SUPPORT_N", "2")),
                },
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
            suffix = int(_cfg_value(selector_cfg, "model_topk_eval", 64))
            topk_idx = select_model_topk_indices(
                selector_logits=selector_logits,
                batch=batch,
                k=suffix,
                use_coverage=False,
                use_group_unique=bool(_cfg_value(selector_cfg, "use_group_unique_model", False)),
            )
            selector_metrics = compute_selected_support_metrics(topk_idx, batch)
            acc.add_dict({f"{key}@{suffix}": value for key, value in selector_metrics.items()})

            topk_mask = build_mask_from_topk_indices(
                topk_idx,
                selector_logits,
                formulae_mask=batch.get("formulae_mask", None),
            )
            support_stats = compute_candidate_support_stats(
                batch,
                topk_mask,
                official_bin_width=float(metric_cfg.get("bin_width", 0.01)),
                official_max_mz=float(metric_cfg.get("max_mz", 1005.0)),
            )
            if isinstance(support_stats, dict):
                if "official_cos" in support_stats:
                    acc.add(f"model_topk_oracle_cos@{suffix}", support_stats["official_cos"])
                if "false_support" in support_stats:
                    acc.add(f"model_topk_oracle_false_support@{suffix}", support_stats["false_support"])

        teacher_probs = batch.get("teacher_formula_probs", None)
        if torch.is_tensor(teacher_probs):
            teacher_stats = compute_candidate_support_stats(
                batch,
                teacher_probs,
                official_bin_width=float(metric_cfg.get("bin_width", 0.01)),
                official_max_mz=float(metric_cfg.get("max_mz", 1005.0)),
            )
            if isinstance(teacher_stats, dict):
                if "official_cos" in teacher_stats:
                    acc.add("teacher_oracle_cos", teacher_stats["official_cos"])
                if "false_support" in teacher_stats:
                    acc.add("teacher_oracle_false_support", teacher_stats["false_support"])
        if os.environ.get("ENABLE_TEACHER_AUDIT", "0") == "1":
            audit_metrics = compute_teacher_audit_pack(
                selector_logits=selector_logits,
                batch=batch,
                selector_cfg=selector_cfg,
                metric_cfg=metric_cfg,
            )
            acc.add_dict(audit_metrics)
    return acc.mean_dict()
