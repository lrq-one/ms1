import os
import torch

from .model_utils import neg_mask_fill_value as _neg_mask_fill_value


def plain_topk(logits, k, mask=None):
    if not torch.is_tensor(logits):
        return None

    scores = logits
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if torch.is_tensor(mask):
        fm = mask.float()
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(int(scores.shape[0]), int(fm.shape[0]))
        use_m = min(int(scores.shape[1]), int(fm.shape[1]))
        scores = scores[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        scores = scores.masked_fill(fm <= 0, _neg_mask_fill_value(scores))

    kk = max(1, min(int(k), int(scores.shape[1])))
    return torch.topk(scores, k=kk, dim=-1).indices


def group_unique_topk(logits, group_ids, k, mask=None):
    if not torch.is_tensor(logits):
        return None

    scores = logits
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    B, M = int(scores.shape[0]), int(scores.shape[1])
    device = scores.device

    if torch.is_tensor(mask):
        fm = mask.float()
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        use_b = min(B, int(fm.shape[0]))
        use_m = min(M, int(fm.shape[1]))
        scores = scores[:use_b, :use_m]
        fm = fm[:use_b, :use_m]
        scores = scores.masked_fill(fm <= 0, _neg_mask_fill_value(scores))

    if torch.is_tensor(group_ids):
        gid = group_ids.to(device=device, dtype=torch.long)
        if gid.dim() == 1:
            gid = gid.unsqueeze(0)
        elif gid.dim() > 2:
            gid = gid.reshape(gid.shape[0], -1)
        use_b = min(B, int(gid.shape[0]))
        use_m = min(M, int(gid.shape[1]))
        gid_full = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1).clone()
        gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
        gid = gid_full
    else:
        gid = torch.arange(M, device=device, dtype=torch.long).view(1, -1).expand(B, -1)

    kk = max(1, min(int(k), M))
    out = []

    for b in range(B):
        order = torch.argsort(scores[b], descending=True)
        used = set()
        chosen = []

        for oi in order.detach().cpu().tolist():
            idx = int(oi)
            g = int(gid[b, idx].detach().cpu().item())
            if g in used:
                continue
            used.add(g)
            chosen.append(idx)
            if len(chosen) >= kk:
                break

        if len(chosen) < kk:
            for oi in order.detach().cpu().tolist():
                idx = int(oi)
                if idx in chosen:
                    continue
                chosen.append(idx)
                if len(chosen) >= kk:
                    break

        out.append(torch.as_tensor(chosen[:kk], dtype=torch.long, device=device))

    return torch.stack(out, dim=0)


def coverage_aware_topk(
    logits,
    peak_idx,
    peak_int,
    formulae_mask=None,
    group_id=None,
    k=64,
    duplicate_penalty=0.35,
    novelty_bonus=0.10,
    eps=1e-8,
):
    """
    Fast coverage-aware topK.

    Key idea:
      1. Pre-filter by selector logits (topP, e.g. 256).
      2. Greedy coverage within topP only.
      3. Vectorized overlap/novelty per step.
      4. Avoid Python loops over all 4096 candidates.

    This function does NOT use true spectrum, so it is safe for real inference.
    """

    if not torch.is_tensor(logits):
        return None

    scores = logits
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    # fallback: no peak info -> plain topK
    if not (torch.is_tensor(peak_idx) and torch.is_tensor(peak_int)):
        return plain_topk(scores, k=k, mask=formulae_mask)

    if peak_idx.dim() == 2:
        peak_idx = peak_idx.unsqueeze(0)
    elif peak_idx.dim() > 3:
        peak_idx = peak_idx.reshape(peak_idx.shape[0], peak_idx.shape[1], -1)

    if peak_int.dim() == 2:
        peak_int = peak_int.unsqueeze(0)
    elif peak_int.dim() > 3:
        peak_int = peak_int.reshape(peak_int.shape[0], peak_int.shape[1], -1)

    if peak_idx.dim() != 3 or peak_int.dim() != 3:
        return plain_topk(scores, k=k, mask=formulae_mask)

    B = min(int(scores.shape[0]), int(peak_idx.shape[0]), int(peak_int.shape[0]))
    M = min(int(scores.shape[1]), int(peak_idx.shape[1]), int(peak_int.shape[1]))

    scores = scores[:B, :M]
    peak_idx = peak_idx[:B, :M].to(device=scores.device, dtype=torch.long)
    peak_int = peak_int[:B, :M].to(device=scores.device, dtype=torch.float32)

    if torch.is_tensor(formulae_mask):
        fm = formulae_mask.float()
        if fm.dim() == 1:
            fm = fm.unsqueeze(0)
        elif fm.dim() > 2:
            fm = fm.reshape(fm.shape[0], -1)
        fm = fm[:B, :M].to(device=scores.device)
    else:
        fm = torch.ones((B, M), dtype=torch.float32, device=scores.device)

    if torch.is_tensor(group_id):
        gid = group_id.to(device=scores.device, dtype=torch.long)
        if gid.dim() == 1:
            gid = gid.unsqueeze(0)
        elif gid.dim() > 2:
            gid = gid.reshape(gid.shape[0], -1)

        gid_full = torch.arange(M, device=scores.device, dtype=torch.long).view(1, -1).expand(B, -1).clone()
        use_b = min(B, int(gid.shape[0]))
        use_m = min(M, int(gid.shape[1]))
        gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
        gid = gid_full
    else:
        gid = None

    try:
        prefilter_k = int(os.environ.get("COVERAGE_PREFILTER_TOPK", "256"))
    except Exception:
        prefilter_k = 256

    try:
        use_group_block = os.environ.get("COVERAGE_BLOCK_SAME_GROUP", "1") == "1"
    except Exception:
        use_group_block = True

    kk = max(1, min(int(k), M))
    pp = max(kk, min(int(prefilter_k), M))
    neg = _neg_mask_fill_value(scores)

    selected_all = []

    for b in range(B):
        score_b = scores[b].clone()
        valid_cand = fm[b] > 0.5
        score_b = score_b.masked_fill(~valid_cand, neg)

        # prefilter topP
        pref_idx = torch.topk(score_b, k=pp, dim=0).indices

        idx = peak_idx[b, pref_idx]
        inten = peak_int[b, pref_idx]

        valid_peak = (
            (idx >= 0)
            & torch.isfinite(inten)
            & (inten > 0)
        )

        if not bool(valid_peak.any().item()):
            selected_all.append(pref_idx[:kk])
            continue

        max_bin = int(idx[valid_peak].max().detach().item()) + 1
        max_bin = max(1, max_bin)

        idx_safe = idx.clamp(0, max_bin - 1)
        inten = torch.where(valid_peak, inten.clamp_min(0.0), torch.zeros_like(inten))

        inten = inten / inten.sum(dim=-1, keepdim=True).clamp_min(float(eps))

        base_raw = score_b[pref_idx]

        valid_pref = torch.isfinite(base_raw) & (base_raw > neg * 0.5)

        if bool(valid_pref.any().item()):
            vals = base_raw[valid_pref]
            center = vals.median()
            scale = vals.std().clamp_min(1e-6)
            base = ((base_raw - center) / scale).clamp(-5.0, 5.0)
            base = torch.where(valid_pref, base, torch.full_like(base, -5.0))
        else:
            base = torch.zeros_like(base_raw)

        try:
            base_weight = float(os.environ.get("COVERAGE_BASE_WEIGHT", "0.20"))
        except Exception:
            base_weight = 0.20

        picked = torch.zeros((pp,), dtype=torch.bool, device=scores.device)
        selected_pos = []

        covered = torch.zeros((max_bin,), dtype=torch.float32, device=scores.device)

        group_block = None
        if gid is not None and use_group_block:
            gid_pref = gid[b, pref_idx]
            group_block = torch.zeros((pp,), dtype=torch.bool, device=scores.device)
        else:
            gid_pref = None

        for _step in range(kk):
            available = ~picked
            if group_block is not None:
                available = available & (~group_block)

            if not bool(available.any().item()):
                available = ~picked
                if not bool(available.any().item()):
                    break

            covered_at = covered[idx_safe] * valid_peak.float()

            overlap = (covered_at * inten).sum(dim=-1)
            novelty = ((1.0 - covered_at).clamp_min(0.0) * inten).sum(dim=-1)

            adjusted = (
                float(base_weight) * base
                - float(duplicate_penalty) * overlap
                + float(novelty_bonus) * novelty
            )

            adjusted = adjusted.masked_fill(~available, neg)

            pick_pos = torch.argmax(adjusted)
            if bool(picked[pick_pos].item()):
                break

            selected_pos.append(pick_pos)
            picked[pick_pos] = True

            if group_block is not None:
                g = gid_pref[pick_pos]
                group_block = group_block | (gid_pref == g)

            valid_sel = valid_peak[pick_pos]
            if bool(valid_sel.any().item()):
                bins_sel = idx_safe[pick_pos][valid_sel]
                vals_sel = inten[pick_pos][valid_sel]
                vals_sel = vals_sel / vals_sel.max().clamp_min(float(eps))

                covered.scatter_reduce_(
                    0,
                    bins_sel,
                    vals_sel,
                    reduce="amax",
                    include_self=True,
                )

        if len(selected_pos) == 0:
            selected = pref_idx[:kk]
        else:
            selected_pos = torch.stack(selected_pos).long()
            selected = pref_idx[selected_pos]

            if int(selected.shape[0]) < kk:
                need = kk - int(selected.shape[0])
                rest = pref_idx[~picked][:need]
                if int(rest.shape[0]) < need:
                    pad = pref_idx[:need - int(rest.shape[0])]
                    rest = torch.cat([rest, pad], dim=0)
                selected = torch.cat([selected, rest], dim=0)

        if os.environ.get("DEBUG_COVERAGE_TOPK", "0") == "1":
            if not hasattr(coverage_aware_topk, "_printed_debug"):
                plain = pref_idx[:kk]
                sel_tmp = selected[:kk]

                common = torch.isin(sel_tmp.detach().cpu(), plain.detach().cpu()).float().mean().item()

                print(
                    "[COVERAGE_TOPK_DEBUG]",
                    "B=", B,
                    "M=", M,
                    "k=", kk,
                    "prefilter=", pp,
                    "base_raw_min=", float(base_raw.min().detach().cpu().item()),
                    "base_raw_max=", float(base_raw.max().detach().cpu().item()),
                    "base_norm_min=", float(base.min().detach().cpu().item()),
                    "base_norm_max=", float(base.max().detach().cpu().item()),
                    "base_weight=", float(base_weight),
                    "dup_penalty=", float(duplicate_penalty),
                    "novelty_bonus=", float(novelty_bonus),
                    "selected_plain_overlap_ratio=", float(common),
                    flush=True,
                )
                coverage_aware_topk._printed_debug = True

        selected_all.append(selected[:kk])

    return torch.stack(selected_all, dim=0)
