import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from rassp.training.batch_utils import move_batch_to_device, prepare_batch_cpu
from rassp.training.formula_targets import (
    apply_teacher_topk_to_target,
    compute_formula_target_probs_from_batch,
    build_true_official_dense_from_cached_sparse_batch,
)
from rassp.training.spectrum_targets import build_true_official_dense_from_raw
from rassp.training.logging_utils import MetricAccumulator
from rassp.training.loss_utils import compute_precursor_loss_from_batch, masked_prob_kl
from rassp.training.runtime_selector_targets import (
    build_candidate_local_quality_target,
    build_selector_teacher_dist_from_official_overlap,
    build_selector_teacher_dist_setcover,
)
from rassp.training.selector_losses import (
    compute_selector_false_support_loss,
    compute_selector_soft_false_support_loss,
    compute_selector_utility_target_loss,
)
from rassp.training.train_loss_components import (
    selector_pairwise_utility_loss,
    compute_official_dense_spectral_loss,
    _false_support_mass_loss_dense,
)


def _cfg_value(cfg, name, default):
    return getattr(cfg, name, default) if cfg is not None else default

def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return float(default)
def _build_true_dense_like_pred(batch, pred_spect, metric_cfg=None):
    """
    Build true dense official spectrum with the same shape as pred_spect.

    pred_spect: [B, bin_n]
    return: true_dense [B, bin_n] or None
    """
    metric_cfg = metric_cfg or {}

    if not torch.is_tensor(pred_spect):
        return None
    if pred_spect.dim() != 2:
        return None

    batch_n = int(pred_spect.shape[0])
    bin_n = int(pred_spect.shape[1])
    device = pred_spect.device

    if batch_n <= 0 or bin_n <= 0:
        return None

    true_dense, used_cache = build_true_official_dense_from_cached_sparse_batch(
        batch=batch,
        batch_n=batch_n,
        device=device,
        official_bin_n=bin_n,
    )

    need_raw_fallback = (
        (not used_cache)
        or (not torch.is_tensor(true_dense))
        or float(true_dense.sum().detach().item()) <= 0.0
    )

    if need_raw_fallback:
        official_bin_width = float(metric_cfg.get("bin_width", 0.01))
        official_max_mz = float(metric_cfg.get("max_mz", 1005.0))

        true_dense = build_true_official_dense_from_raw(
            spect_raw_list=batch.get("spect_raw", None),
            precursor_mz=batch.get("precursor_mz", None),
            official_bin_width=official_bin_width,
            official_max_mz=official_max_mz,
            exclude_precursor=True,
            batch_n=batch_n,
            device=device,
        )

    if not torch.is_tensor(true_dense):
        return None

    true_dense = true_dense[:batch_n, :bin_n].to(
        device=device,
        dtype=pred_spect.dtype,
    )

    true_dense = torch.where(
        torch.isfinite(true_dense),
        true_dense,
        torch.zeros_like(true_dense),
    ).clamp_min(0.0)

    if float(true_dense.sum().detach().item()) <= 0.0:
        return None

    return true_dense
def _as_2d(x):
    if not torch.is_tensor(x):
        return None
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() > 2:
        return x.reshape(x.shape[0], -1)
    return x


def _align_2d(*items):
    tensors = [_as_2d(x) for x in items]
    if any(x is None for x in tensors):
        return None
    batch_n = min(int(x.shape[0]) for x in tensors)
    item_n = min(int(x.shape[1]) for x in tensors)
    if batch_n <= 0 or item_n <= 0:
        return None
    return tuple(x[:batch_n, :item_n] for x in tensors)


def _masked_selector_bce(selector_logits, target_mask, formulae_mask=None, pos_weight=5.0):
    if not (torch.is_tensor(selector_logits) and torch.is_tensor(target_mask)):
        return None
    if formulae_mask is None:
        formulae_mask = torch.ones_like(selector_logits)

    aligned = _align_2d(selector_logits, target_mask, formulae_mask)
    if aligned is None:
        return None
    logits, target, mask = aligned
    logits = logits.float()
    target = (target > 0.5).float().to(device=logits.device, dtype=logits.dtype)
    mask = (mask > 0.5).float().to(device=logits.device, dtype=logits.dtype)

    if float(mask.sum().detach().item()) <= 0.0:
        return logits.sum() * 0.0

    raw = F.binary_cross_entropy_with_logits(
        logits.masked_fill(mask <= 0.5, 0.0),
        target,
        reduction="none",
        pos_weight=logits.new_tensor(float(pos_weight)),
    )
    return (raw * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_selector_recall_bce(
    selector_logits,
    target_mask,
    formulae_mask=None,
    pos_part=0.80,
    neg_part=0.20,
):
    if not (torch.is_tensor(selector_logits) and torch.is_tensor(target_mask)):
        return None
    if formulae_mask is None:
        formulae_mask = torch.ones_like(selector_logits)

    aligned = _align_2d(selector_logits, target_mask, formulae_mask)
    if aligned is None:
        return None
    logits, target, mask = aligned
    logits = logits.float()
    target = (target > 0.5).float().to(device=logits.device, dtype=logits.dtype)
    valid = (mask > 0.5).to(device=logits.device)

    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    pos = valid & (target > 0.5)
    neg = valid & (target <= 0.5)

    zero = logits.sum() * 0.0
    pos_loss = raw[pos].mean() if bool(pos.any().item()) else zero
    neg_loss = raw[neg].mean() if bool(neg.any().item()) else zero
    return float(pos_part) * pos_loss + float(neg_part) * neg_loss


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
            # ------------------------------------------------------------
            # Stage 2: official dense spectral loss
            # ------------------------------------------------------------
            pred_spect = None
            if isinstance(res, dict):
                pred_spect = res.get("spect_out_official", None)
                if not torch.is_tensor(pred_spect):
                    pred_spect = res.get("spect", None)

            if torch.is_tensor(pred_spect):
                official_w = float(_cfg_value(loss_cfg, "official_spectral_weight", 0.0))
                if official_w > 0.0:
                    true_dense = _build_true_dense_like_pred(
                        batch,
                        pred_spect,
                        metric_cfg=metric_cfg,
                    )

                    if torch.is_tensor(true_dense):
                        official_kl_w = _env_float("OFFICIAL_DENSE_KL_WEIGHT", 0.05)

                        official_loss = compute_official_dense_spectral_loss(
                            pred_spect,
                            true_dense,
                            kl_weight=official_kl_w,
                        )

                        if torch.is_tensor(official_loss):
                            loss = loss + official_w * official_loss
                            acc.add("official_spectral", official_loss.detach().item())

                        false_dense_w = _env_float("OFFICIAL_DENSE_FALSE_WEIGHT", 0.0)
                        if false_dense_w > 0.0:
                            dense_false = _false_support_mass_loss_dense(
                                pred_spect,
                                true_dense,
                            )
                            if torch.is_tensor(dense_false):
                                loss = loss + false_dense_w * dense_false
                                acc.add("official_dense_false", dense_false.detach().item())
            selector_logits = res.get("selector_logits", None) if isinstance(res, dict) else None
            formulae_mask = batch.get("formulae_mask", None)
            selector_loss_total = None
            if torch.is_tensor(selector_logits) and torch.is_tensor(formulae_mask):
                official_bin_width = float(metric_cfg.get("bin_width", 0.01))
                official_max_mz = float(metric_cfg.get("max_mz", 1005.0))
                official_bin_n = int(official_max_mz / max(official_bin_width, 1e-8)) + 1

                target_probs = compute_formula_target_probs_from_batch(
                    batch,
                    official_bin_width=official_bin_width,
                    official_max_mz=official_max_mz,
                    support_topk=int(_cfg_value(selector_cfg, "target_support_topk", 64)),
                )
                teacher_dist = build_selector_teacher_dist_setcover(
                    batch=batch,
                    formulae_mask=formulae_mask,
                    official_bin_n=official_bin_n,
                )
                if not torch.is_tensor(teacher_dist):
                    teacher_dist = target_probs

                aligned = _align_2d(selector_logits, teacher_dist, formulae_mask)
                if aligned is not None:
                    logits_use, target_use, mask_use = aligned
                    logits_use = logits_use.float()
                    target_use = target_use.to(device=logits_use.device, dtype=logits_use.dtype).clamp_min(0.0)
                    mask_use = (mask_use > 0.5).to(device=logits_use.device)

                    target_use = target_use * mask_use.float()
                    target_sum = target_use.sum(dim=-1, keepdim=True)
                    target_use = target_use / target_sum.clamp_min(1e-12)
                    target_mask = (target_use > 0).float()

                    selector_loss_total = logits_use.sum() * 0.0

                    bce_w = float(_cfg_value(loss_cfg, "selector_bce_weight", 0.0))
                    if bce_w > 0:
                        bce_loss = _masked_selector_bce(
                            logits_use,
                            target_mask,
                            formulae_mask=mask_use,
                            pos_weight=float(_cfg_value(loss_cfg, "selector_pos_weight", 5.0)),
                        )
                        if torch.is_tensor(bce_loss):
                            selector_loss_total = selector_loss_total + bce_w * bce_loss
                            acc.add("selector_bce", bce_loss.detach().item())

                    recall_w = float(_cfg_value(loss_cfg, "selector_recall_bce_weight", 0.0))
                    if recall_w > 0:
                        try:
                            recall_topk = int(os.environ.get("SELECTOR_RECALL_TARGET_TOPK", "128"))
                        except Exception:
                            recall_topk = 128
                        _, recall_mask = apply_teacher_topk_to_target(
                            target_use,
                            formulae_mask=mask_use.float(),
                            topk=recall_topk,
                        )
                        if not torch.is_tensor(recall_mask):
                            recall_mask = target_mask
                        recall_bce_loss = _masked_selector_recall_bce(
                            logits_use,
                            recall_mask,
                            formulae_mask=mask_use,
                            pos_part=0.80,
                            neg_part=0.20,
                        )
                        if torch.is_tensor(recall_bce_loss):
                            selector_loss_total = selector_loss_total + recall_w * recall_bce_loss
                            acc.add("selector_recall_bce", recall_bce_loss.detach().item())

                    kl_w = float(_cfg_value(loss_cfg, "selector_kl_weight", 0.0))
                    if kl_w > 0:
                        pred_prob = torch.softmax(
                            logits_use.masked_fill(mask_use <= 0, -1e9),
                            dim=-1,
                        )
                        selector_kl = masked_prob_kl(pred_prob, target_use, mask_use.float())
                        selector_loss_total = selector_loss_total + kl_w * selector_kl
                        acc.add("selector_kl", selector_kl.detach().item())

                    pairwise_w = float(_cfg_value(loss_cfg, "selector_pairwise_weight", 0.0))
                    if pairwise_w > 0:
                        pairwise_loss = selector_pairwise_utility_loss(
                            logits_use,
                            target_use,
                            valid_mask=mask_use,
                        )
                        if torch.is_tensor(pairwise_loss):
                            selector_loss_total = selector_loss_total + pairwise_w * pairwise_loss
                            acc.add("selector_pairwise", pairwise_loss.detach().item())

                    utility_w = float(_cfg_value(loss_cfg, "selector_utility_weight", 0.0))
                    if utility_w > 0:
                        util_loss = compute_selector_utility_target_loss(logits_use, batch)
                        if torch.is_tensor(util_loss):
                            selector_loss_total = selector_loss_total + utility_w * util_loss
                            acc.add("selector_utility", util_loss.detach().item())

                    if selector_loss_total is not None:
                        loss = loss + float(_cfg_value(loss_cfg, "selector_weight", 1.0)) * selector_loss_total
                        acc.add("selector_loss", selector_loss_total.detach().item())

                false_total = None
                hard_false_w = float(_cfg_value(loss_cfg, "false_support_weight", 0.0))
                if hard_false_w > 0:
                    fs_loss = compute_selector_false_support_loss(
                        selector_logits,
                        batch,
                        topk=int(_cfg_value(selector_cfg, "selector_topk", 64)),
                    )
                    if torch.is_tensor(fs_loss):
                        loss = loss + hard_false_w * fs_loss
                        false_total = fs_loss if false_total is None else false_total + fs_loss

                soft_false_w = float(_cfg_value(loss_cfg, "soft_false_support_weight", 0.0))
                if soft_false_w > 0:
                    soft_fs_loss = compute_selector_soft_false_support_loss(selector_logits, batch)
                    if torch.is_tensor(soft_fs_loss):
                        loss = loss + soft_false_w * soft_fs_loss
                        false_total = soft_fs_loss if false_total is None else false_total + soft_fs_loss

                if torch.is_tensor(false_total):
                    acc.add("false_support", false_total.detach().item())

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
