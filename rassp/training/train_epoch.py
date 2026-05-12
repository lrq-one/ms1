import torch
import torch.nn.functional as F
from tqdm import tqdm

from rassp.training.batch_utils import move_batch_to_device, prepare_batch_cpu
from rassp.training.formula_targets import (
    apply_teacher_topk_to_target,
    compute_formula_target_probs_from_batch,
)
from rassp.training.logging_utils import MetricAccumulator
from rassp.training.loss_utils import compute_precursor_loss_from_batch, masked_prob_kl
from rassp.training.runtime_selector_targets import (
    build_candidate_local_quality_target,
    build_selector_teacher_dist_from_official_overlap,
    build_selector_teacher_dist_setcover,
)

from rassp.training.selector_losses import (
    compute_selector_false_support_loss,
    compute_selector_utility_target_loss,
    build_selector_utility_tensors,
)

from rassp.training.train_loss_components import (
    selector_pairwise_utility_loss,
)
def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    spect_bin,
    epoch,
    run_cfg=None,
    selector_cfg=None,
    loss_cfg=None,
    metric_cfg=None,
):
    model.train()
    acc = MetricAccumulator()
    max_steps = int(_cfg_value(run_cfg, "max_train_steps", 0) or 0)
    metric_cfg = metric_cfg or {}

    for step, raw_batch in enumerate(tqdm(loader, desc=f"Epoch {epoch} [Train]")):
        if max_steps > 0 and step >= max_steps:
            break

        processed = prepare_batch_cpu(raw_batch, spect_bin)
        batch = move_batch_to_device(processed, device)
        optimizer.zero_grad(set_to_none=True)

        autocast_enabled = bool(torch.cuda.is_available())
        with torch.cuda.amp.autocast(enabled=autocast_enabled):
            res = model(**batch)
            base = res["spect"] if isinstance(res, dict) and torch.is_tensor(res.get("spect", None)) else None
            loss = base.sum() * 0.0 if torch.is_tensor(base) else torch.zeros((), device=device)

            selector_logits = res.get("selector_logits", None) if isinstance(res, dict) else None
            formulae_mask = batch.get("formulae_mask", None)
            if torch.is_tensor(selector_logits) and torch.is_tensor(formulae_mask):
                target_probs = compute_formula_target_probs_from_batch(
                    batch,
                    bin_width=float(metric_cfg.get("bin_width", 0.1)),
                    max_mz=float(metric_cfg.get("max_mz", 1005.0)),
                    support_topk=int(_cfg_value(selector_cfg, "target_support_topk", 64)),
                )
                if torch.is_tensor(target_probs):
                    pred_prob = torch.softmax(
                        selector_logits.masked_fill(formulae_mask <= 0, -1e9),
                        dim=-1,
                    )
                    selector_kl = masked_prob_kl(pred_prob, target_probs, formulae_mask)
                    loss = loss + float(_cfg_value(loss_cfg, "selector_kl_weight", 0.45)) * selector_kl
                    acc.add("selector_kl", selector_kl.detach().item())

                if float(_cfg_value(loss_cfg, "false_support_weight", 0.0)) > 0:
                    fs_loss = compute_selector_false_support_loss(
                        selector_logits,
                        batch,
                        topk=int(_cfg_value(selector_cfg, "selector_topk", 64)),
                    )
                    loss = loss + float(_cfg_value(loss_cfg, "false_support_weight", 0.0)) * fs_loss
                    acc.add("false_support", fs_loss.detach().item())

                if float(_cfg_value(loss_cfg, "selector_utility_weight", 0.0)) > 0:
                    util_loss = compute_selector_utility_target_loss(selector_logits, batch)
                    loss = loss + float(_cfg_value(loss_cfg, "selector_utility_weight", 0.0)) * util_loss
                    acc.add("selector_utility", util_loss.detach().item())
                pairwise_w = float(_cfg_value(loss_cfg, "selector_pairwise_weight", 0.0))
                if pairwise_w > 0:
                    utility, utility_dist, valid_mask, util_stats = build_selector_utility_tensors(
                        selector_logits,
                        batch,
                    )

                    if utility is not None and valid_mask is not None:
                        pairwise_loss = selector_pairwise_utility_loss(
                            selector_logits=selector_logits,
                            utility=utility,
                            valid_mask=valid_mask,
                            high_q=0.80,
                            low_q=0.40,
                            margin=0.2,
                            max_pairs=2048,
                        )

                        if pairwise_loss is not None:
                            loss = loss + pairwise_w * pairwise_loss
                            acc.add("selector_pairwise", pairwise_loss.detach().item())

                        if isinstance(util_stats, dict):
                            for k, v in util_stats.items():
                                if torch.is_tensor(v):
                                    acc.add(k, v.detach().item())
            precursor_loss = compute_precursor_loss_from_batch(batch, res)
            if precursor_loss is not None and float(_cfg_value(loss_cfg, "precursor_weight", 0.0)) > 0:
                loss = loss + float(_cfg_value(loss_cfg, "precursor_weight", 0.0)) * precursor_loss
                acc.add("precursor", precursor_loss.detach().item())

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        acc.add("loss", loss.detach().item())

    return acc.mean_dict()
