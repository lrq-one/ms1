import torch
import torch.nn.functional as F


def masked_prob_kl(pred_prob, target_q, mask, eps=1e-8):
    """KL(target_q || pred_prob), only for rows where target_q has nonzero mass."""
    pred_prob = pred_prob.clamp_min(eps)
    target_q = target_q.float().clamp_min(0.0)
    mask = (mask > 0.5).float()

    target_q = target_q * mask
    target_sum = target_q.sum(dim=-1, keepdim=True)

    valid = target_sum.squeeze(-1) > eps
    target_q = target_q / target_sum.clamp_min(eps)

    kl = target_q * (target_q.clamp_min(eps).log() - pred_prob.log())
    kl = kl.sum(dim=-1)

    if valid.any():
        return kl[valid].mean()

    return pred_prob.new_tensor(0.0)


def compute_precursor_loss_from_batch(batch, res):
    """
    Precursor stability loss.
    Uses:
      res['precursor_logit']                 [B]
      batch['true_precursor_prob_in_all']    [B, 1] or [B]
    """
    if not isinstance(res, dict):
        return None

    precursor_logit = res.get("precursor_logit", None)
    precursor_target = batch.get("true_precursor_prob_in_all", None)

    if not (torch.is_tensor(precursor_logit) and torch.is_tensor(precursor_target)):
        return None

    logit = precursor_logit.float().reshape(-1)
    target = precursor_target.float().reshape(-1).to(device=logit.device)

    use_n = min(int(logit.shape[0]), int(target.shape[0]))
    if use_n <= 0:
        return None

    logit = logit[:use_n]
    target = target[:use_n].clamp(0.0, 1.0)

    valid = torch.isfinite(target) & torch.isfinite(logit)
    if not bool(valid.any().item()):
        return logit.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        logit[valid],
        target[valid],
        reduction="mean",
    )
