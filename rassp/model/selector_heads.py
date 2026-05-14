import torch
import torch.nn as nn


def _safe_neg_value(x):
    if torch.is_tensor(x) and torch.is_floating_point(x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return -1e4
        try:
            return float(torch.finfo(x.dtype).min)
        except Exception:
            return -1e9
    return -1e9


class PeakSupportEncoder(nn.Module):
    """
    Encode candidate-generated peak support.

    Input:
        peak_feat: [B, M, P, 6]
    Output:
        support_h: [B, M, D]
    """

    def __init__(self, peak_feat_dim=6, hidden_dim=128, out_dim=256):
        super().__init__()

        self.peak_mlp = nn.Sequential(
            nn.LayerNorm(peak_feat_dim),
            nn.Linear(peak_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, peak_feat):
        if peak_feat is None or not torch.is_tensor(peak_feat):
            return None

        valid = peak_feat[..., -1:] > 0.5

        h = self.peak_mlp(peak_feat)

        h_masked = h * valid.float()
        mean_h = h_masked.sum(dim=2) / valid.float().sum(dim=2).clamp_min(1.0)

        h_for_max = h.masked_fill(~valid.expand_as(h), -1e4)
        max_h = h_for_max.max(dim=2).values

        support_h = torch.cat([mean_h, max_h], dim=-1)
        support_h = self.out_proj(support_h)

        return support_h


class LocalUtilitySelectorHead(nn.Module):
    """
    Local candidate utility scorer.
    No full candidate self-attention.
    Use this as the base selector before fast coverage/set-cover topK.
    """

    def __init__(
        self,
        hidden_dim,
        peak_feat_dim=6,
        frag_aux_dim=0,
        dropout=0.0,
    ):
        super().__init__()

        self.hidden_dim = int(hidden_dim)
        self.frag_aux_dim = int(frag_aux_dim)

        self.support_encoder = PeakSupportEncoder(
            peak_feat_dim=peak_feat_dim,
            hidden_dim=max(64, self.hidden_dim // 4),
            out_dim=self.hidden_dim,
        )

        if self.frag_aux_dim > 0:
            self.frag_proj = nn.Sequential(
                nn.LayerNorm(self.frag_aux_dim),
                nn.Linear(self.frag_aux_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
        else:
            self.frag_proj = None

        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim * 3),
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )

        self.logit_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        candidate_h,
        formulae_mask=None,
        peak_feat=None,
        frag_aux=None,
    ):
        h = candidate_h

        if formulae_mask is None:
            formulae_mask = torch.ones(
                h.shape[:2],
                dtype=torch.float32,
                device=h.device,
            )
        else:
            formulae_mask = formulae_mask.to(device=h.device, dtype=torch.float32)

        support_h = self.support_encoder(peak_feat)
        if support_h is None:
            support_h = torch.zeros_like(h)
        else:
            support_h = support_h.to(device=h.device, dtype=h.dtype)

        if self.frag_proj is not None and torch.is_tensor(frag_aux):
            frag_aux = frag_aux.to(device=h.device, dtype=h.dtype)
            if frag_aux.dim() == 2:
                frag_aux = frag_aux.unsqueeze(0)

            if frag_aux.shape[0] == h.shape[0] and frag_aux.shape[1] == h.shape[1]:
                frag_aux = frag_aux[..., : self.frag_aux_dim]
                frag_h = self.frag_proj(frag_aux)
            else:
                frag_h = torch.zeros_like(h)
        else:
            frag_h = torch.zeros_like(h)

        x = torch.cat([h, support_h, frag_h], dim=-1)
        logits = self.out(x).squeeze(-1)

        scale = self.logit_scale.clamp(0.5, 8.0)
        logits = logits * scale

        logits = logits.masked_fill(formulae_mask <= 0, _safe_neg_value(logits))
        return logits
