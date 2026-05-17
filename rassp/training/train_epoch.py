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

def _selector_hard_topk_teacher_loss(
    selector_logits,
    teacher_dist,
    formulae_mask,
    teacher_topk=8,
    hard_neg_topk=64,
    ce_temp=1.0,
    margin=1.0,
    eps=1e-12,
):
    """
    Force selector topK to match runtime teacher topK.

    Why:
      KL over 4096 candidates can decrease while final top8 remains poor.
      This loss directly optimizes:
        1) teacher topK CE
        2) teacher-positive vs hard-negative margin
        3) balanced BCE on teacher topK positives and hard negatives
    """
    if not (
        torch.is_tensor(selector_logits)
        and torch.is_tensor(teacher_dist)
        and torch.is_tensor(formulae_mask)
    ):
        return None, {}

    aligned = _align_2d(selector_logits, teacher_dist, formulae_mask)
    if aligned is None:
        return None, {}

    logits, target, mask = aligned
    logits = logits.float()
    target = target.to(device=logits.device, dtype=logits.dtype).clamp_min(0.0)
    mask = (mask > 0.5).to(device=logits.device)

    B, M = logits.shape
    device = logits.device

    ce_losses = []
    margin_losses = []
    bce_losses = []
    recall_values = []
    precision_values = []

    for b in range(B):
        valid = mask[b]
        if int(valid.sum().detach().item()) <= 1:
            continue

        t = target[b] * valid.float()
        pos_available = (t > eps) & valid
        pos_n = int(pos_available.sum().detach().item())
        if pos_n <= 0:
            continue

        k_pos = max(1, min(int(teacher_topk), pos_n))
        k_neg = max(1, min(int(hard_neg_topk), int(valid.sum().detach().item()) - k_pos))
        if k_neg <= 0:
            continue

        pos_scores = t.masked_fill(~pos_available, -1e9)
        pos_idx = torch.topk(pos_scores, k=k_pos, dim=0).indices

        neg_valid = valid.clone()
        neg_valid[pos_idx] = False

        neg_scores = logits[b].detach().masked_fill(~neg_valid, -1e9)
        neg_idx = torch.topk(neg_scores, k=k_neg, dim=0).indices

        # CE over all valid candidates, but only teacher topK receives target mass.
        q = torch.zeros((M,), dtype=logits.dtype, device=device)
        q[pos_idx] = t[pos_idx]
        q = q / q.sum().clamp_min(eps)

        logp = F.log_softmax(
            logits[b].masked_fill(~valid, -1e9) / max(float(ce_temp), 1e-6),
            dim=0,
        )
        ce_losses.append(-(q[pos_idx].detach() * logp[pos_idx]).sum())

        # Margin: every teacher positive should beat hard negatives.
        pos_log = logits[b, pos_idx]
        neg_log = logits[b, neg_idx]
        diff = pos_log[:, None] - neg_log[None, :]
        margin_losses.append(F.softplus(float(margin) - diff).mean())

        # Balanced BCE on teacher topK positives + hard negatives only.
        use_idx = torch.cat([pos_idx, neg_idx], dim=0)
        y = torch.cat(
            [
                torch.ones_like(pos_idx, dtype=logits.dtype, device=device),
                torch.zeros_like(neg_idx, dtype=logits.dtype, device=device),
            ],
            dim=0,
        )
        bce_losses.append(
            F.binary_cross_entropy_with_logits(
                logits[b, use_idx],
                y,
                reduction="mean",
            )
        )

        # Monitoring: overlap between model topK and teacher topK.
        model_top = torch.topk(
            logits[b].detach().masked_fill(~valid, -1e9),
            k=k_pos,
            dim=0,
        ).indices

        pos_set = set(int(x) for x in pos_idx.detach().cpu().tolist())
        model_set = set(int(x) for x in model_top.detach().cpu().tolist())
        hit = len(pos_set & model_set)

        recall_values.append(hit / max(1, len(pos_set)))
        precision_values.append(hit / max(1, len(model_set)))

    if len(ce_losses) == 0:
        return logits.sum() * 0.0, {
            "selector_teacher_topk_recall": 0.0,
            "selector_teacher_topk_precision": 0.0,
        }

    ce = torch.stack(ce_losses).mean()
    mg = torch.stack(margin_losses).mean()
    bce = torch.stack(bce_losses).mean()

    try:
        ce_w = float(os.environ.get("SELECTOR_HARD_TOPK_CE_WEIGHT", "1.0"))
    except Exception:
        ce_w = 1.0
    try:
        margin_w = float(os.environ.get("SELECTOR_TOPK_MARGIN_WEIGHT", "0.5"))
    except Exception:
        margin_w = 0.5
    try:
        bce_w = float(os.environ.get("SELECTOR_TOPK_BCE_WEIGHT", "0.5"))
    except Exception:
        bce_w = 0.5

    total = float(ce_w) * ce + float(margin_w) * mg + float(bce_w) * bce

    return total, {
        "selector_hard_topk_ce": float(ce.detach().cpu().item()),
        "selector_hard_topk_margin": float(mg.detach().cpu().item()),
        "selector_hard_topk_bce": float(bce.detach().cpu().item()),
        "selector_teacher_topk_recall": float(sum(recall_values) / max(1, len(recall_values))),
        "selector_teacher_topk_precision": float(sum(precision_values) / max(1, len(precision_values))),
    }
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
            if os.environ.get("DEBUG_OFFICIAL_LOSS", "0") == "1" and step == 0:
                print(
                    "[DEBUG_OFFICIAL_LOSS/pre]",
                    "res_keys=", sorted(list(res.keys())) if isinstance(res, dict) else None,
                    "has_spect=", torch.is_tensor(res.get("spect", None)) if isinstance(res, dict) else False,
                    "spect_shape=", tuple(res["spect"].shape) if isinstance(res, dict) and torch.is_tensor(res.get("spect", None)) else None,
                    "has_spect_out_official=", torch.is_tensor(res.get("spect_out_official", None)) if isinstance(res, dict) else False,
                    "spect_out_official_shape=", tuple(res["spect_out_official"].shape) if isinstance(res, dict) and torch.is_tensor(res.get("spect_out_official", None)) else None,
                    "pred_spect_shape=", tuple(pred_spect.shape) if torch.is_tensor(pred_spect) else None,
                    "official_w=", float(_cfg_value(loss_cfg, "official_spectral_weight", 0.0)),
                    "batch_has_true_all_idx=", torch.is_tensor(batch.get("true_all_official_idx", None)),
                    "batch_has_true_idx=", torch.is_tensor(batch.get("true_official_idx", None)),
                    "batch_has_spect_raw=", batch.get("spect_raw", None) is not None,
                    flush=True,
                )
            if torch.is_tensor(pred_spect):
                official_w = float(_cfg_value(loss_cfg, "official_spectral_weight", 0.0))
                if official_w > 0.0:
                    true_dense = _build_true_dense_like_pred(
                        batch,
                        pred_spect,
                        metric_cfg=metric_cfg,
                    )
                    if os.environ.get("DEBUG_OFFICIAL_LOSS", "0") == "1" and step == 0:
                        print(
                            "[DEBUG_OFFICIAL_LOSS/target]",
                            "true_dense_is_tensor=", torch.is_tensor(true_dense),
                            "true_dense_shape=", tuple(true_dense.shape) if torch.is_tensor(true_dense) else None,
                            "true_dense_sum=", float(true_dense.detach().float().sum().cpu().item()) if torch.is_tensor(true_dense) else None,
                            "pred_sum=", float(pred_spect.detach().float().sum().cpu().item()) if torch.is_tensor(pred_spect) else None,
                            "pred_min=", float(pred_spect.detach().float().min().cpu().item()) if torch.is_tensor(pred_spect) else None,
                            "pred_max=", float(pred_spect.detach().float().max().cpu().item()) if torch.is_tensor(pred_spect) else None,
                            flush=True,
                        )
                    if torch.is_tensor(true_dense):
                        official_kl_w = _env_float("OFFICIAL_DENSE_KL_WEIGHT", 0.05)

                        official_loss = compute_official_dense_spectral_loss(
                            pred_spect,
                            true_dense,
                            kl_weight=official_kl_w,
                        )
                        if os.environ.get("DEBUG_OFFICIAL_LOSS", "0") == "1" and step == 0:
                            print(
                                "[DEBUG_OFFICIAL_LOSS/loss]",
                                "official_loss_is_tensor=", torch.is_tensor(official_loss),
                                "official_loss=", float(official_loss.detach().cpu().item()) if torch.is_tensor(official_loss) else None,
                                "official_kl_w=", official_kl_w,
                                flush=True,
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

                # ------------------------------------------------------------
                # Selector runtime teacher mode
                #
                # quality:
                #   Stage1 推荐。使用 build_candidate_local_quality_target()
                #   给 local selector 一个更可学的 pointwise/local target。
                #
                # overlap:
                #   使用 official overlap teacher，比 setcover 更局部。
                #
                # setcover:
                #   保留旧路径。只建议 teacher audit 或后续 reranker/setwise 阶段使用。
                #
                # cached:
                #   fallback 到 compute_formula_target_probs_from_batch()
                # ------------------------------------------------------------
                runtime_teacher_mode = os.environ.get(
                    "SELECTOR_RUNTIME_TEACHER_MODE",
                    "quality",
                ).strip().lower()

                target_mask_source = None
                pairwise_utility = None
                valid_mask_for_loss = formulae_mask.float()
                local_extra = {}

                if runtime_teacher_mode in ("quality", "local", "local_quality"):
                    quality, pos_label, valid_mask_for_loss, local_extra = build_candidate_local_quality_target(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )

                    # KL / hard-topk 使用更平滑的 utility_dist；
                    # 如果 utility_dist 不存在，则 fallback 到 quality。
                    teacher_dist = local_extra.get("utility_dist", None)
                    if not torch.is_tensor(teacher_dist):
                        teacher_dist = quality

                    # BCE / recall BCE 使用明确的 local positive label。
                    target_mask_source = pos_label

                    # pairwise 应该吃连续 utility，而不是 sparse teacher_dist。
                    pairwise_utility = local_extra.get("utility", quality)

                elif runtime_teacher_mode == "overlap":
                    teacher_dist = build_selector_teacher_dist_from_official_overlap(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )

                    # 仍然构造一次 local target，主要为了拿 valid_mask 和 utility。
                    quality, pos_label, valid_mask_for_loss, local_extra = build_candidate_local_quality_target(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )

                    target_mask_source = (teacher_dist > 1e-12).float()
                    pairwise_utility = local_extra.get("utility", quality)

                elif runtime_teacher_mode == "setcover":
                    teacher_dist = build_selector_teacher_dist_setcover(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )

                    target_mask_source = (teacher_dist > 1e-12).float()

                    # setcover 没有信号的行直接从 selector loss 里跳过。
                    if torch.is_tensor(teacher_dist):
                        row_has_signal = (teacher_dist.sum(dim=-1, keepdim=True) > 1e-12).float()
                        valid_mask_for_loss = formulae_mask.float() * row_has_signal
                    else:
                        valid_mask_for_loss = formulae_mask.float()

                    pairwise_utility = None

                elif runtime_teacher_mode in ("cached", "cache", "target_probs"):
                    teacher_dist = target_probs
                    target_mask_source = (teacher_dist > 1e-12).float()
                    row_has_signal = (teacher_dist.sum(dim=-1, keepdim=True) > 1e-12).float()
                    valid_mask_for_loss = formulae_mask.float() * row_has_signal
                    pairwise_utility = None

                else:
                    # 安全 fallback：未知 mode 不要崩，退回 quality。
                    quality, pos_label, valid_mask_for_loss, local_extra = build_candidate_local_quality_target(
                        batch=batch,
                        formulae_mask=formulae_mask,
                        official_bin_n=official_bin_n,
                    )
                    teacher_dist = local_extra.get("utility_dist", quality)
                    target_mask_source = pos_label
                    pairwise_utility = local_extra.get("utility", quality)

                if not torch.is_tensor(teacher_dist):
                    teacher_dist = target_probs
                if not torch.is_tensor(target_mask_source):
                    target_mask_source = (teacher_dist > 1e-12).float()
                if not torch.is_tensor(valid_mask_for_loss):
                    valid_mask_for_loss = formulae_mask.float()

                aligned = _align_2d(
                    selector_logits,
                    teacher_dist,
                    valid_mask_for_loss,
                    target_mask_source,
                )
                if aligned is not None:
                    logits_use, target_use, mask_use, target_mask_source_use = aligned
                    logits_use = logits_use.float()
                    target_use = target_use.to(device=logits_use.device, dtype=logits_use.dtype).clamp_min(0.0)
                    mask_use = (mask_use > 0.5).to(device=logits_use.device)

                    target_use = target_use * mask_use.float()
                    target_sum = target_use.sum(dim=-1, keepdim=True)
                    target_use = target_use / target_sum.clamp_min(1e-12)
                    target_mask = (
                        (target_mask_source_use > 0.5).float()
                        * mask_use.float()
                    )
                    # -----------------------------
                    # target diagnostics
                    # -----------------------------
                    try:
                        acc.add(
                            "selector_target_pos_rate",
                            float(
                                (
                                    target_mask.sum()
                                    / mask_use.float().sum().clamp_min(1.0)
                                ).detach().cpu().item()
                            ),
                        )
                        acc.add(
                            "selector_target_valid_row_rate",
                            float(
                                (
                                    (mask_use.float().sum(dim=-1) > 0).float().mean()
                                ).detach().cpu().item()
                            ),
                        )
                        acc.add(
                            "selector_teacher_nnz",
                            float(
                                (
                                    (target_use > 1e-12).float().sum(dim=-1).mean()
                                ).detach().cpu().item()
                            ),
                        )
                    except Exception:
                        pass

                    if isinstance(local_extra, dict):
                        if torch.is_tensor(local_extra.get("false_mass", None)):
                            acc.add(
                                "selector_local_false_mass_mean",
                                float(local_extra["false_mass"].detach().float().mean().cpu().item()),
                            )
                        if torch.is_tensor(local_extra.get("true_hit_mass", None)):
                            acc.add(
                                "selector_local_true_hit_mass_mean",
                                float(local_extra["true_hit_mass"].detach().float().mean().cpu().item()),
                            )
                        if torch.is_tensor(local_extra.get("clean_pos_label", None)):
                            acc.add(
                                "selector_clean_pos_rate",
                                float(local_extra["clean_pos_label"].detach().float().mean().cpu().item()),
                            )
                        if torch.is_tensor(local_extra.get("teacher_pos_label", None)):
                            acc.add(
                                "selector_teacher_pos_rate",
                                float(local_extra["teacher_pos_label"].detach().float().mean().cpu().item()),
                            )
                    selector_loss_total = logits_use.sum() * 0.0
                    hard_topk_w = float(os.environ.get("SELECTOR_HARD_TOPK_LOSS_WEIGHT", "0.0"))
                    if hard_topk_w > 0:
                        try:
                            teacher_topk = int(os.environ.get("SELECTOR_TEACHER_TOPK", "8"))
                        except Exception:
                            teacher_topk = 8
                        try:
                            hard_neg_topk = int(os.environ.get("SELECTOR_HARD_NEG_TOPK", "64"))
                        except Exception:
                            hard_neg_topk = 64
                        try:
                            ce_temp = float(os.environ.get("SELECTOR_HARD_TOPK_TEMP", "1.0"))
                        except Exception:
                            ce_temp = 1.0
                        try:
                            margin = float(os.environ.get("SELECTOR_TOPK_MARGIN", "1.0"))
                        except Exception:
                            margin = 1.0

                        hard_loss, hard_metrics = _selector_hard_topk_teacher_loss(
                            logits_use,
                            target_use,
                            mask_use,
                            teacher_topk=teacher_topk,
                            hard_neg_topk=hard_neg_topk,
                            ce_temp=ce_temp,
                            margin=margin,
                        )

                        if torch.is_tensor(hard_loss):
                            selector_loss_total = selector_loss_total + hard_topk_w * hard_loss

                        if isinstance(hard_metrics, dict):
                            for hk, hv in hard_metrics.items():
                                acc.add(hk, hv)
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
                        else:
                            # Critical fix:
                            # apply_teacher_topk_to_target(topk=128) may include zero-prob candidates
                            # when teacher has fewer than 128 positives.
                            # Those zero-prob candidates must NOT become recall BCE positives.
                            recall_mask = recall_mask.to(device=target_use.device, dtype=target_use.dtype)
                            recall_mask = recall_mask * (target_use > 1e-12).float()
                            recall_mask = recall_mask * mask_use.float()

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
                        pairwise_target = target_use

                        if torch.is_tensor(pairwise_utility):
                            pair_aligned = _align_2d(
                                logits_use,
                                pairwise_utility,
                                mask_use.float(),
                            )
                            if pair_aligned is not None:
                                _, pairwise_target, pairwise_mask = pair_aligned
                                pairwise_target = pairwise_target.to(
                                    device=logits_use.device,
                                    dtype=logits_use.dtype,
                                )
                                pairwise_mask = pairwise_mask > 0.5
                            else:
                                pairwise_mask = mask_use
                        else:
                            pairwise_mask = mask_use

                        pairwise_loss = selector_pairwise_utility_loss(
                            logits_use,
                            pairwise_target,
                            valid_mask=pairwise_mask,
                        )
                        if torch.is_tensor(pairwise_loss):
                            selector_loss_total = selector_loss_total + pairwise_w * pairwise_loss
                            acc.add("selector_pairwise", pairwise_loss.detach().item())

                    utility_w = float(_cfg_value(loss_cfg, "selector_utility_weight", 0.0))
                    if utility_w > 0:
                        util_loss = None

                        utility_source = os.environ.get(
                            "SELECTOR_UTILITY_SOURCE",
                            "local",
                        ).strip().lower()

                        if utility_source in ("local", "local_extra", "quality"):
                            util_target = None
                            if isinstance(local_extra, dict):
                                util_target = local_extra.get("utility_dist", None)

                            if torch.is_tensor(util_target):
                                util_aligned = _align_2d(
                                    logits_use,
                                    util_target,
                                    mask_use.float(),
                                )

                                if util_aligned is not None:
                                    u_logits, u_target, u_mask = util_aligned
                                    u_logits = u_logits.float()
                                    u_target = u_target.to(
                                        device=u_logits.device,
                                        dtype=u_logits.dtype,
                                    ).clamp_min(0.0)
                                    u_mask = (u_mask > 0.5).to(device=u_logits.device)

                                    u_target = u_target * u_mask.float()
                                    u_sum = u_target.sum(dim=-1, keepdim=True)
                                    valid_rows = u_sum.squeeze(-1) > 1e-12

                                    if bool(valid_rows.any().item()):
                                        u_target = u_target / u_sum.clamp_min(1e-12)

                                        neg_fill = -1e4 if u_logits.dtype in (
                                            torch.float16,
                                            torch.bfloat16,
                                        ) else -1e9

                                        log_prob = F.log_softmax(
                                            u_logits.masked_fill(~u_mask, neg_fill),
                                            dim=-1,
                                        )

                                        per_row = -(u_target.detach() * log_prob).sum(dim=-1)
                                        util_loss = per_row[valid_rows].mean()
                                    else:
                                        util_loss = logits_use.sum() * 0.0

                        elif utility_source in ("legacy", "old", "batch"):
                            util_loss = compute_selector_utility_target_loss(logits_use, batch)

                        else:
                            util_loss = logits_use.sum() * 0.0

                        if torch.is_tensor(util_loss):
                            selector_loss_total = selector_loss_total + utility_w * util_loss
                            acc.add("selector_utility", util_loss.detach().item())

                    if selector_loss_total is not None:
                        loss = loss + float(_cfg_value(loss_cfg, "selector_weight", 1.0)) * selector_loss_total
                        acc.add("selector_loss", selector_loss_total.detach().item())
                    # ------------------------------------------------------------
                    # Setwise reranker loss
                    #
                    # Base selector is still trained by local-quality / overlap target.
                    # Reranker should learn setcover-style target inside the selector pool.
                    # This is the missing bridge between local selector and set-cover teacher.
                    # ------------------------------------------------------------
                    rerank_w = _env_float("RERANK_LOSS_WEIGHT", 0.0)
                    if rerank_w > 0.0 and isinstance(res, dict):
                        rerank_logits = res.get("formulae_scores_raw", None)
                        pool_idx = res.get("selector_pool_idx", None)

                        if torch.is_tensor(rerank_logits) and torch.is_tensor(pool_idx):
                            rerank_teacher_mode = os.environ.get(
                                "RERANK_TEACHER_MODE",
                                "setcover",
                            ).strip().lower()

                            if rerank_teacher_mode == "setcover":
                                rerank_teacher = build_selector_teacher_dist_setcover(
                                    batch=batch,
                                    formulae_mask=formulae_mask,
                                    official_bin_n=official_bin_n,
                                )
                            elif rerank_teacher_mode == "overlap":
                                rerank_teacher = build_selector_teacher_dist_from_official_overlap(
                                    batch=batch,
                                    formulae_mask=formulae_mask,
                                    official_bin_n=official_bin_n,
                                )
                            else:
                                rerank_teacher = teacher_dist

                            r_aligned = _align_2d(
                                rerank_logits,
                                rerank_teacher,
                                formulae_mask.float(),
                            )

                            if r_aligned is not None:
                                r_logits, r_target, r_mask_base = r_aligned
                                r_logits = r_logits.float()
                                r_target = r_target.to(
                                    device=r_logits.device,
                                    dtype=r_logits.dtype,
                                ).clamp_min(0.0)
                                r_mask_base = (r_mask_base > 0.5).to(device=r_logits.device)

                                B_r, M_r = r_logits.shape

                                # Only train reranker on candidates inside selector pool.
                                r_pool_mask = torch.zeros_like(r_mask_base, dtype=torch.bool)
                                pi = pool_idx.to(device=r_logits.device, dtype=torch.long)

                                if pi.dim() == 1:
                                    pi = pi.unsqueeze(0)
                                elif pi.dim() > 2:
                                    pi = pi.reshape(pi.shape[0], -1)

                                use_b = min(B_r, int(pi.shape[0]))
                                use_k = int(pi.shape[1]) if pi.dim() == 2 else 0

                                if use_b > 0 and use_k > 0:
                                    pi_use = pi[:use_b, :].clamp(min=0, max=M_r - 1)
                                    r_pool_mask[:use_b].scatter_(1, pi_use, True)

                                r_mask = r_mask_base & r_pool_mask

                                r_target = r_target * r_mask.float()
                                r_sum = r_target.sum(dim=-1, keepdim=True)
                                r_valid_rows = r_sum.squeeze(-1) > 1e-12

                                if bool(r_valid_rows.any().item()):
                                    r_target = r_target / r_sum.clamp_min(1e-12)

                                    try:
                                        r_teacher_topk = int(os.environ.get("RERANK_TEACHER_TOPK", "8"))
                                    except Exception:
                                        r_teacher_topk = 8
                                    try:
                                        r_hard_neg_topk = int(os.environ.get("RERANK_HARD_NEG_TOPK", "64"))
                                    except Exception:
                                        r_hard_neg_topk = 64
                                    try:
                                        r_temp = float(os.environ.get("RERANK_HARD_TOPK_TEMP", "1.0"))
                                    except Exception:
                                        r_temp = 1.0
                                    try:
                                        r_margin = float(os.environ.get("RERANK_TOPK_MARGIN", "0.5"))
                                    except Exception:
                                        r_margin = 0.5

                                    rerank_loss, rerank_metrics = _selector_hard_topk_teacher_loss(
                                        r_logits,
                                        r_target,
                                        r_mask,
                                        teacher_topk=r_teacher_topk,
                                        hard_neg_topk=r_hard_neg_topk,
                                        ce_temp=r_temp,
                                        margin=r_margin,
                                    )

                                    if torch.is_tensor(rerank_loss):
                                        loss = loss + float(rerank_w) * rerank_loss
                                        acc.add("rerank_selector_loss", rerank_loss.detach().item())

                                    if isinstance(rerank_metrics, dict):
                                        for rk, rv in rerank_metrics.items():
                                            acc.add(f"rerank_{rk}", rv)

                                    acc.add(
                                        "rerank_active_row_rate",
                                        float(r_valid_rows.float().mean().detach().cpu().item()),
                                    )
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
