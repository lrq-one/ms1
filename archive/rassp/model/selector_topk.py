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
    if not torch.is_tensor(logits):
        return None

    scores = logits
    if scores.dim() == 1:
        scores = scores.unsqueeze(0)
    elif scores.dim() > 2:
        scores = scores.reshape(scores.shape[0], -1)

    if not (torch.is_tensor(peak_idx) and torch.is_tensor(peak_int)):
        masked = scores
        if torch.is_tensor(formulae_mask):
            fm = formulae_mask.float()
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            elif fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)
            use_b = min(int(scores.shape[0]), int(fm.shape[0]))
            use_m = min(int(scores.shape[1]), int(fm.shape[1]))
            masked = masked[:use_b, :use_m]
            fm = fm[:use_b, :use_m]
            masked = masked.masked_fill(fm <= 0, _neg_mask_fill_value(masked))
        kk = max(1, min(int(k), int(masked.shape[1])))
        return torch.topk(masked, k=kk, dim=-1).indices

    if peak_idx.dim() == 2:
        peak_idx = peak_idx.unsqueeze(0)
    elif peak_idx.dim() > 3:
        peak_idx = peak_idx.reshape(peak_idx.shape[0], peak_idx.shape[1], -1)

    if peak_int.dim() == 2:
        peak_int = peak_int.unsqueeze(0)
    elif peak_int.dim() > 3:
        peak_int = peak_int.reshape(peak_int.shape[0], peak_int.shape[1], -1)

    if peak_idx.dim() != 3 or peak_int.dim() != 3:
        masked = scores
        if torch.is_tensor(formulae_mask):
            fm = formulae_mask.float()
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            elif fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)
            use_b = min(int(scores.shape[0]), int(fm.shape[0]))
            use_m = min(int(scores.shape[1]), int(fm.shape[1]))
            masked = masked[:use_b, :use_m]
            fm = fm[:use_b, :use_m]
            masked = masked.masked_fill(fm <= 0, _neg_mask_fill_value(masked))
        kk = max(1, min(int(k), int(masked.shape[1])))
        return torch.topk(masked, k=kk, dim=-1).indices

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
        use_b = min(B, int(gid.shape[0]))
        use_m = min(M, int(gid.shape[1]))
        gid_full = torch.arange(M, device=scores.device, dtype=torch.long).view(1, -1).expand(B, -1).clone()
        gid_full[:use_b, :use_m] = gid[:use_b, :use_m].clamp_min(0)
        gid = gid_full
    else:
        gid = None

    kk = max(1, min(int(k), M))
    selected_all = []
    neg = _neg_mask_fill_value(scores)

    for b in range(B):
        score = scores[b].clone()
        valid_cand = fm[b] > 0.5
        score = score.masked_fill(~valid_cand, neg)

        idx_b = peak_idx[b]
        int_b = peak_int[b]
        valid_peak = (idx_b >= 0) & torch.isfinite(int_b) & (int_b > 0)

        if bool(valid_peak.any().item()):
            max_bin = int(idx_b[valid_peak].max().item()) + 1
        else:
            max_bin = 1

        covered = torch.zeros((max_bin,), dtype=torch.float32, device=scores.device)

        selected = []
        allow_repeat_groups = False
        group_block = None
        if gid is not None:
            group_block = torch.zeros((M,), dtype=torch.bool, device=scores.device)

        for _ in range(kk):
            adjusted = score.clone()

            if len(selected) > 0:
                adjusted[torch.as_tensor(selected, device=scores.device, dtype=torch.long)] = neg

            if group_block is not None and not allow_repeat_groups:
                adjusted = adjusted.masked_fill(group_block, neg)

            overlap_list = []
            novelty_list = []

            for m in range(M):
                if not bool(valid_cand[m].item()):
                    overlap_list.append(score.new_tensor(1.0))
                    novelty_list.append(score.new_tensor(0.0))
                    continue

                if group_block is not None and (not allow_repeat_groups) and bool(group_block[m].item()):
                    overlap_list.append(score.new_tensor(1.0))
                    novelty_list.append(score.new_tensor(0.0))
                    continue

                v = valid_peak[m] & (idx_b[m] < max_bin)
                if not bool(v.any().item()):
                    overlap_list.append(score.new_tensor(1.0))
                    novelty_list.append(score.new_tensor(0.0))
                    continue

                pidx_v = idx_b[m][v]
                pint_v = int_b[m][v].clamp_min(0.0)
                total = pint_v.sum().clamp_min(float(eps))

                already = covered[pidx_v].clamp(0.0, 1.0)
                overlap = (already * pint_v).sum() / total
                novelty = ((1.0 - already) * pint_v).sum() / total

                overlap_list.append(overlap)
                novelty_list.append(novelty)

            overlap = torch.stack(overlap_list)
            novelty = torch.stack(novelty_list)

            adjusted = adjusted - float(duplicate_penalty) * overlap + float(novelty_bonus) * novelty

            if not bool(torch.isfinite(adjusted).any().item()):
                if group_block is not None and not allow_repeat_groups:
                    allow_repeat_groups = True
                    continue
                break

            pick = int(torch.argmax(adjusted).item())
            selected.append(pick)

            if group_block is not None and not allow_repeat_groups:
                gid_b = int(gid[b, pick].item())
                group_block = group_block | (gid[b] == gid_b)

            v_pick = valid_peak[pick] & (idx_b[pick] < max_bin)
            if bool(v_pick.any().item()):
                pidx_v = idx_b[pick][v_pick]
                pint_v = int_b[pick][v_pick].clamp_min(0.0)
                pint_v = pint_v / pint_v.max().clamp_min(float(eps))
                covered[pidx_v] = torch.maximum(covered[pidx_v], pint_v)

        if len(selected) < kk:
            pad_n = kk - len(selected)
            selected.extend([0] * pad_n)

        selected_all.append(torch.as_tensor(selected[:kk], device=scores.device, dtype=torch.long))

    return torch.stack(selected_all, dim=0)
