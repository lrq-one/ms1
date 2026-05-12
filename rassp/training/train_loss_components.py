import os

import numpy as np
import torch
import torch.nn.functional as F

from rassp.model.model_utils import neg_mask_fill_value as _neg_mask_fill_value


def _masked_candidate_kl(scores_tensor, target_probs, formulae_mask=None):
    if (not torch.is_tensor(scores_tensor)) or (not torch.is_tensor(target_probs)):
        return None

    use_b = min(int(scores_tensor.shape[0]), int(target_probs.shape[0]))
    use_m = min(int(scores_tensor.shape[1]), int(target_probs.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None

    scores_use = scores_tensor[:use_b, :use_m]
    target_use = target_probs[:use_b, :use_m].to(device=scores_use.device, dtype=scores_use.dtype)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(use_b, int(fm.shape[0]))
        use_m = min(use_m, int(fm.shape[1]))
        scores_use = scores_use[:use_b, :use_m]
        target_use = target_use[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        scores_use = scores_use.masked_fill(fm <= 0, _neg_mask_fill_value(scores_use))

    valid_rows = torch.isfinite(target_use).all(dim=-1) & (target_use.sum(dim=-1) > 0)
    if not bool(valid_rows.any().item()):
        return scores_use.sum() * 0.0

    per_row_kl = torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(scores_use, dim=-1),
        target_use,
        reduction='none',
    ).sum(dim=-1)
    return per_row_kl[valid_rows].mean()

def _group_masked_candidate_kl(scores_tensor, target_probs, formulae_mask=None, group_id=None, eps=1e-8):
    """
    Group-aware KL for V2 source-instance candidates.

    Instead of:
      KL(target_instance || softmax(instance_scores))

    Use:
      group_score[g] = max score over instances with the same formula group
      target_group[g] = sum target mass over instances with the same formula group
      KL(target_group || softmax(group_score))

    This removes duplicate-source multiplicity pressure from rerank training.
    """
    if (
        not torch.is_tensor(scores_tensor)
        or not torch.is_tensor(target_probs)
        or not torch.is_tensor(group_id)
    ):
        return None

    scores = scores_tensor.float()
    target = target_probs.float()
    gid = group_id.long().to(device=scores.device)

    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if target.dim() == 1:
        target = target.unsqueeze(0)
    elif target.dim() > 2:
        target = target.reshape(target.shape[0], -1)

    if gid.dim() == 1:
        gid = gid.unsqueeze(0)
    elif gid.dim() > 2:
        gid = gid.reshape(gid.shape[0], -1)

    B = min(int(scores.shape[0]), int(target.shape[0]), int(gid.shape[0]))
    M = min(int(scores.shape[1]), int(target.shape[1]), int(gid.shape[1]))
    if B <= 0 or M <= 0:
        return None

    scores = scores[:B, :M]
    target = target[:B, :M].to(device=scores.device, dtype=scores.dtype)
    gid = gid[:B, :M].to(device=scores.device, dtype=torch.long)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(B, int(fm.shape[0]))
        use_m = min(M, int(fm.shape[1]))

        valid = torch.zeros((B, M), dtype=torch.bool, device=scores.device)
        valid[:use_b, :use_m] = fm[:use_b, :use_m].to(device=scores.device) > 0.5
    else:
        valid = torch.ones((B, M), dtype=torch.bool, device=scores.device)

    valid = valid & torch.isfinite(scores) & torch.isfinite(target)
    target = torch.where(torch.isfinite(target), target, torch.zeros_like(target)).clamp_min(0.0)

    losses = []
    neg = _neg_mask_fill_value(scores)

    for bi in range(B):
        valid_b = valid[bi]
        if not bool(valid_b.any().item()):
            continue

        idx_valid = torch.nonzero(valid_b, as_tuple=False).reshape(-1)
        s_b = scores[bi, idx_valid]
        t_b = target[bi, idx_valid].clamp_min(0.0)
        gid_b = gid[bi, idx_valid]

        if float(t_b.sum().detach().item()) <= float(eps):
            continue

        # Critical: group ids are original ids and may be > M after rerank pruning.
        # Do NOT clamp. Remap per row.
        uniq_gid, inv = torch.unique(gid_b, sorted=False, return_inverse=True)
        group_n = int(uniq_gid.shape[0])
        if group_n <= 0:
            continue

        group_scores = s_b.new_full((group_n,), neg)
        group_scores.scatter_reduce_(
            dim=0,
            index=inv,
            src=s_b,
            reduce="amax",
            include_self=True,
        )

        group_target = s_b.new_zeros((group_n,))
        group_target.scatter_add_(0, inv, t_b)
        group_target = group_target / group_target.sum().clamp_min(float(eps))

        logp = torch.log_softmax(group_scores, dim=0)
        kl = group_target * (group_target.clamp_min(float(eps)).log() - logp)
        losses.append(kl.sum())

    if len(losses) <= 0:
        return scores.sum() * 0.0

    return torch.stack(losses).mean()

def _masked_formula_entropy_loss(scores_tensor, formulae_mask=None):
    """
    Encourage formula probability distribution to be less diffuse.

    This is softer than hard FORMULA_RENDER_TOPK.
    It penalizes high entropy among valid candidates.
    """
    if not torch.is_tensor(scores_tensor):
        return None

    scores = scores_tensor.float()
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(int(scores.shape[0]), int(fm.shape[0]))
        use_m = min(int(scores.shape[1]), int(fm.shape[1]))
        scores = scores[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        valid = fm > 0.5
        scores = scores.masked_fill(~valid, _neg_mask_fill_value(scores))
        valid_n = valid.float().sum(dim=-1).clamp_min(2.0)
    else:
        valid_n = torch.full(
            (scores.shape[0],),
            float(max(2, int(scores.shape[1]))),
            dtype=scores.dtype,
            device=scores.device,
        )

    prob = torch.softmax(scores, dim=-1)
    log_prob = torch.log(prob.clamp_min(1e-12))
    ent = -(prob * log_prob).sum(dim=-1)

    # normalize to [0,1] roughly
    ent = ent / torch.log(valid_n).clamp_min(1e-12)

    valid_rows = torch.isfinite(ent)
    if not bool(valid_rows.any().item()):
        return scores.sum() * 0.0

    return ent[valid_rows].mean()
def _renormalize_target_probs(target_probs, formulae_mask=None):
    if not torch.is_tensor(target_probs):
        return None

    tp = target_probs.float()
    if tp.dim() == 1:
        tp = tp.unsqueeze(0)
    elif tp.dim() > 2:
        tp = tp.reshape(tp.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(int(tp.shape[0]), int(fm.shape[0]))
        use_m = min(int(tp.shape[1]), int(fm.shape[1]))
        tp = tp[:use_b, :use_m]
        fm = (fm[:use_b, :use_m] > 0.5).float()
    else:
        fm = torch.ones_like(tp)

    tp = torch.where(torch.isfinite(tp), tp, torch.zeros_like(tp))
    tp = tp.clamp_min(0.0) * fm
    row_sum = tp.sum(dim=-1, keepdim=True)

    fallback = fm / fm.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    tp = torch.where(row_sum > 1e-12, tp / row_sum.clamp_min(1e-12), fallback)
    return tp


def _normalize_dense_prob(x):
    x = x.float().clamp_min(0.0)
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _sqrt_cosine_loss_dense(pred, true):
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)
    pred_s = torch.sqrt(pred.clamp_min(1e-12))
    true_s = torch.sqrt(true.clamp_min(1e-12))
    num = (pred_s * true_s).sum(dim=-1)
    den = torch.norm(pred_s, dim=-1) * torch.norm(true_s, dim=-1)
    cos = num / den.clamp_min(1e-12)
    return (1.0 - cos).mean()


def _cosine_loss_dense(pred, true):
    """
    Dense cosine loss aligned with official_cos_no_precursor.

    Note:
    cosine is scale-invariant, so per-row normalization does not change
    vector direction; it only stabilizes numerical scale.
    """
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)

    num = (pred * true).sum(dim=-1)
    den = torch.norm(pred, dim=-1) * torch.norm(true, dim=-1)
    cos = num / den.clamp_min(1e-12)
    return (1.0 - cos).mean()

def _dense_kl_loss(pred, true):
    pred = _normalize_dense_prob(pred)
    true = _normalize_dense_prob(true)
    return F.kl_div(
        torch.log(pred.clamp_min(1e-12)),
        true,
        reduction='batchmean',
    )


def _false_support_mass_loss_dense(pred_official, true_official, true_eps=1e-12):
    """
    Penalize predicted intensity mass assigned to bins outside true support.

    This directly targets the current failure:
      val_pred_int_on_true is low,
      false predicted support is huge.
    """
    if (not torch.is_tensor(pred_official)) or (not torch.is_tensor(true_official)):
        return None

    use_b = min(int(pred_official.shape[0]), int(true_official.shape[0]))
    use_m = min(int(pred_official.shape[1]), int(true_official.shape[1]))
    if use_b <= 0 or use_m <= 0:
        return None

    pred = pred_official[:use_b, :use_m].float().clamp_min(0.0)
    true = true_official[:use_b, :use_m].float().clamp_min(0.0)

    pred = pred / pred.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    true_support = (true > float(true_eps)).float()

    false_mass = (pred * (1.0 - true_support)).sum(dim=-1)

    valid_rows = torch.isfinite(false_mass)
    if not bool(valid_rows.any().item()):
        return pred.sum() * 0.0

    return false_mass[valid_rows].mean()


def compute_official_dense_spectral_loss(pred_official, true_official, kl_weight=0.2):
    """
    Official dense spectral loss.

    OFFICIAL_SPECTRAL_LOSS_MODE:
      - cos: optimize standard cosine, aligned with official_cos_no_precursor
      - sqrt: old behavior, sqrt-cosine
      - mix: weighted mix of standard cosine and sqrt-cosine
    """
    mode = os.environ.get("OFFICIAL_SPECTRAL_LOSS_MODE", "cos").strip().lower()

    loss_std_cos = _cosine_loss_dense(pred_official, true_official)
    loss_sqrt_cos = _sqrt_cosine_loss_dense(pred_official, true_official)

    if mode in ("sqrt", "sqrt_cos", "sqrtcos"):
        loss_base = loss_sqrt_cos
    elif mode in ("mix", "mixed"):
        try:
            std_w = float(os.environ.get("OFFICIAL_SPECTRAL_STD_COS_WEIGHT", "0.8"))
        except Exception:
            std_w = 0.8
        std_w = float(np.clip(std_w, 0.0, 1.0))
        loss_base = std_w * loss_std_cos + (1.0 - std_w) * loss_sqrt_cos
    else:
        loss_base = loss_std_cos

    if float(kl_weight) > 0:
        loss_kl = _dense_kl_loss(pred_official, true_official)
        return loss_base + float(kl_weight) * loss_kl

    return loss_base


def _get_selector_logits_from_res(res_dict):
    """
    Return formula selector logits used by selector losses and topK selection.

    Default behavior:
      use ordinary formula-level selector logits.

    Optional behavior:
      if USE_FN_FORMULA_LOGITS_AS_SELECTOR=1, blend or replace with
      fn_based_formula_logits.  This must be explicit because fragment-node
      mapping can cover only a subset of formula candidates.
    """
    if not isinstance(res_dict, dict):
        return None

    selector_logits = res_dict.get('selector_logits', None)
    if not torch.is_tensor(selector_logits):
        selector_logits = res_dict.get('formulae_scores_raw', None)
    if not torch.is_tensor(selector_logits):
        selector_logits = res_dict.get('formulae_scores_train', None)

    fn_mapped_logits = res_dict.get('fn_based_formula_logits', None)

    use_fn = os.environ.get("USE_FN_FORMULA_LOGITS_AS_SELECTOR", "0") == "1"
    if use_fn and torch.is_tensor(fn_mapped_logits):
        if not torch.is_tensor(selector_logits):
            return fn_mapped_logits

        sel = selector_logits
        fn = fn_mapped_logits.to(device=sel.device, dtype=sel.dtype)

        if sel.dim() == 1:
            sel = sel.unsqueeze(0)
        elif sel.dim() > 2:
            sel = sel.reshape(sel.shape[0], -1)

        if fn.dim() == 1:
            fn = fn.unsqueeze(0)
        elif fn.dim() > 2:
            fn = fn.reshape(fn.shape[0], -1)

        use_b = min(int(sel.shape[0]), int(fn.shape[0]))
        use_m = min(int(sel.shape[1]), int(fn.shape[1]))
        if use_b <= 0 or use_m <= 0:
            return selector_logits

        sel_use = sel[:use_b, :use_m]
        fn_use = fn[:use_b, :use_m]

        # fn_based_formula_logits uses a very negative fill value for
        # formulae without fragment-node mapping.  Those positions must
        # fall back to ordinary selector logits; otherwise valid formulae
        # are silently killed.
        neg = float(_neg_mask_fill_value(fn_use))
        if abs(neg) < 1e20:
            mapped = torch.isfinite(fn_use) & (fn_use > neg * 0.5)
        else:
            mapped = torch.isfinite(fn_use) & (fn_use > -1e20)

        fn_safe = torch.where(mapped, fn_use, sel_use)

        mode = os.environ.get("FN_SELECTOR_MODE", "blend").strip().lower()
        if mode == "replace":
            fused = fn_safe
        else:
            try:
                alpha = float(os.environ.get("FN_SELECTOR_BLEND_ALPHA", "0.2"))
            except Exception:
                alpha = 0.2
            alpha = float(np.clip(alpha, 0.0, 1.0))
            fused = (1.0 - alpha) * sel_use + alpha * fn_safe

        out = sel.clone()
        out[:use_b, :use_m] = fused
        return out

    return selector_logits if torch.is_tensor(selector_logits) else None


def _get_reranker_scores_from_res(res_dict):
    if not isinstance(res_dict, dict):
        return None
    formulae_scores = res_dict.get('formulae_logits', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    formulae_scores = res_dict.get('formulae_scores_train', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    formulae_scores = res_dict.get('formulae_scores_raw', None)
    if torch.is_tensor(formulae_scores):
        return formulae_scores
    return res_dict.get('formulae_scores', None)


def _get_active_candidate_mask_from_batch(batch, formulae_mask=None):
    """
    Read active candidate mask from batch.

    Preferred:
      batch['formulae_active_mask'] if available.

    Fallback:
      formulae_aux_feat struct-aux active column.

    Important:
      formulae_aux_feat layout is usually:
        first 15 dims = build_formulae_aux_feat(...)
        last 7 dims  = [
            any_source,
            struct_source,
            common_loss_source,
            break_depth_norm,
            ring_cut,
            prior_score_z,
            active_mask,
        ]

      Therefore active_mask is column 21 when aux dim is 22,
      not column 6.
    """
    active = batch.get('formulae_active_mask', None)

    if not torch.is_tensor(active):
        aux = batch.get('formulae_aux_feat', None)
        if torch.is_tensor(aux) and aux.dim() >= 3:
            raw_col = os.environ.get("FORMULAE_AUX_ACTIVE_COL", "").strip()

            if raw_col:
                try:
                    active_col = int(raw_col)
                except Exception:
                    active_col = None
            elif int(aux.shape[-1]) >= 22:
                # 15-D formula aux + 7-D struct aux; active is the last struct field.
                active_col = 21
            elif int(aux.shape[-1]) == 7:
                # Struct aux only.
                active_col = 6
            else:
                active_col = None

            if active_col is not None and int(aux.shape[-1]) > active_col:
                active = aux[..., active_col]

    if not torch.is_tensor(active):
        return None

    active = active.float()
    if active.dim() > 2:
        active = active.reshape(active.shape[0], -1)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)

        use_b = min(int(active.shape[0]), int(fm.shape[0]))
        use_m = min(int(active.shape[1]), int(fm.shape[1]))

        active = active[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        active = active * (fm > 0.5).float()

    return (active > 0.5).float()

def _apply_selector_aux_logit_bias(selector_logits, batch):
    """
    Add label-free cache prior bias to selector logits.

    Important:
      formulae_aux_feat layout is usually:
        first 15 dims = formula aux
        last 7 dims  = struct aux:
          0 any_source
          1 struct_source
          2 common_loss_source
          3 break_depth_norm
          4 ring_cut
          5 prior_score_z
          6 active_mask

    So for 22-D aux, struct aux starts at -7, not column 0.
    """
    if os.environ.get("USE_SELECTOR_AUX_LOGIT_BIAS", "0") != "1":
        return selector_logits

    if not torch.is_tensor(selector_logits):
        return selector_logits

    aux = batch.get('formulae_aux_feat', None)
    if not torch.is_tensor(aux) or aux.dim() < 3:
        return selector_logits

    logits = selector_logits
    aux = aux.to(device=logits.device, dtype=logits.dtype)

    use_b = min(int(logits.shape[0]), int(aux.shape[0]))
    use_m = min(int(logits.shape[1]), int(aux.shape[1]))
    logits = logits[:use_b, :use_m]
    aux = aux[:use_b, :use_m]

    d = int(aux.shape[-1])

    # Correct struct-aux extraction.
    if d >= 22:
        saux = aux[..., -7:]
    elif d == 7:
        saux = aux
    else:
        return selector_logits

    def _w(name, default):
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return float(default)

    any_source = saux[..., 0]
    struct_source = saux[..., 1]
    common_loss_source = saux[..., 2]
    break_depth_norm = saux[..., 3]
    ring_cut = saux[..., 4]
    prior_score_z = saux[..., 5].clamp(-3.0, 3.0)
    active_mask = saux[..., 6]

    bias = torch.zeros_like(logits)

    # Soft prior only. Do not hard-mask with active.
    bias = bias + _w("SELECTOR_BIAS_ANY_SOURCE", 0.05) * any_source
    bias = bias + _w("SELECTOR_BIAS_STRUCT_SOURCE", 0.10) * struct_source
    bias = bias + _w("SELECTOR_BIAS_COMMON_LOSS", 0.05) * common_loss_source
    bias = bias + _w("SELECTOR_BIAS_PRIOR_Z", 0.20) * prior_score_z
    bias = bias + _w("SELECTOR_BIAS_ACTIVE", 0.35) * active_mask

    # ring cut / depth are not always reliable; keep weak.
    bias = bias + _w("SELECTOR_BIAS_RING_CUT", -0.03) * ring_cut
    bias = bias + _w("SELECTOR_BIAS_BREAK_DEPTH", -0.02) * break_depth_norm

    out = logits + bias

    fm = batch.get('formulae_mask', None)
    if torch.is_tensor(fm):
        fm = fm.float()
        if fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:use_b, :use_m].to(device=out.device)
        out = out.masked_fill(fm <= 0.5, _neg_mask_fill_value(out))

    return out

