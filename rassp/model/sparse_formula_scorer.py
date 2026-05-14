import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .nets import *
from .model_utils import neg_mask_fill_value as _neg_mask_fill_value
from .selector_heads import SupportAwareSelectorHead, SelectorRankHeadV2
from .formulaenets_legacy import (
    StructuredOneHot,
    project_formula_probs_to_spectrum_dense,
    project_formula_probs_to_exact_sparse,
)
from .peak_features import PeakFeatureMixin


# Class overview: MolAttentionGRUNewSparse is the simplified single-path formula scorer.
class MolAttentionGRUNewSparse(PeakFeatureMixin, nn.Module):
    def __init__(
        self,
        g_feat_in,
        spect_bin,
        embedding_key_size=64,
        formula_oh_sizes=[20, 20, 20, 20, 20],
        formula_oh_accum=True,
        formula_oh_normalize=False,
        internal_d=512,
        prob_softmax=True,
        g_embed_train=True,
        g_embed_bias=True,
        formula_embed_train=True,
        formula_embed_bias=True,
        gru_layer_n=1,
        linear_layer_n=2,
        ce_emb_dim=32,
        adduct_emb_dim=16,
        adduct_vocab_size=32,
        instrument_emb_dim=16,
        instrument_vocab_size=32,
        ms_level_emb_dim=8,
        ms_level_vocab_size=8,
        formulae_aux_dim=0,
        fragment_local_aux_dim=0,
        use_msms_conditioning=False,
        score_cond_concat=True,
        normalize_1_output=False,
        candidate_attn_temperature=0.5,
        pred_exact_topk=0,
        pred_exact_min_prob=0.0,
    ):
        super(MolAttentionGRUNewSparse, self).__init__()

        self.spect_bin = spect_bin
        self.g_feat_in = int(g_feat_in)
        self.prob_softmax = bool(prob_softmax)
        self.normalize_1_output = bool(normalize_1_output)
        self.pred_exact_topk = max(0, int(pred_exact_topk))
        self.pred_exact_min_prob = max(0.0, float(pred_exact_min_prob))
        try:
            self.render_topk_formula = int(os.environ.get("FORMULA_RENDER_TOPK", "0"))
            self.render_topk_train = os.environ.get("FORMULA_RENDER_TOPK_TRAIN", "0") == "1"
        except Exception:
            self.render_topk_formula = 0

        try:
            self.render_min_formula_prob = float(os.environ.get("FORMULA_RENDER_MIN_PROB", "0.0"))
        except Exception:
            self.render_min_formula_prob = 0.0
        self.mainline_signature = 'setwise_peak_clean_v3'

        self.use_msms_conditioning = bool(use_msms_conditioning)
        self.score_cond_concat = bool(score_cond_concat)
        self.formulae_aux_dim = max(0, int(formulae_aux_dim))
        try:
            self.fragment_local_aux_dim = max(0, int(fragment_local_aux_dim))
        except Exception:
            self.fragment_local_aux_dim = 0
        try:
            self.fragment_local_aux_scale = float(os.environ.get("FRAGMENT_LOCAL_AUX_SCALE", "1.0"))
        except Exception:
            self.fragment_local_aux_scale = 1.0
        self.gate_fragment_local_aux = os.environ.get("GATE_FRAGMENT_LOCAL_AUX", "1") == "1"
        self.peak_support_feat_dim = 8
        self.peak_support_proj_dim = 16
        self.peak_support_proj = nn.Sequential(
            nn.LayerNorm(self.peak_support_feat_dim),
            nn.Linear(self.peak_support_feat_dim, self.peak_support_proj_dim),
            nn.ReLU(),
            nn.Linear(self.peak_support_proj_dim, self.peak_support_proj_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(self.g_feat_in)
        self.embed_g_feat = nn.Linear(self.g_feat_in, int(embedding_key_size), bias=g_embed_bias)

        formula_encoding_n = int(np.sum(formula_oh_sizes))
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, int(embedding_key_size), bias=formula_embed_bias)
        self.formula_oh_normalize = bool(formula_oh_normalize)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False

        self.candidate_attn_temperature = max(float(candidate_attn_temperature), 1e-6)
        self.formula_q = nn.Linear(int(embedding_key_size), int(embedding_key_size))
        self.vert_k = nn.Linear(int(embedding_key_size), int(embedding_key_size))
        self.vert_v = nn.Linear(self.g_feat_in, self.g_feat_in)

        cond_feature_d = 0
        if self.use_msms_conditioning and self.score_cond_concat:
            cond_feature_d += int(ce_emb_dim)  # CE
            cond_feature_d += int(ce_emb_dim)  # precursor
            cond_feature_d += int(adduct_emb_dim)
            cond_feature_d += int(instrument_emb_dim)
            cond_feature_d += int(ms_level_emb_dim)
        self.cond_feature_d = int(cond_feature_d)

        base_in_d = (
            int(embedding_key_size)
            + self.g_feat_in
            + self.formulae_aux_dim
            + self.cond_feature_d
            + self.peak_support_proj_dim
        )
        self.base_score_norm = nn.LayerNorm(base_in_d)
        self.base_score_l1 = nn.Linear(base_in_d, int(internal_d))
        self.base_score_l2 = nn.Linear(int(internal_d), int(internal_d))

        if self.fragment_local_aux_dim > 0:
            self.frag_aux_proj = nn.Sequential(
                nn.LayerNorm(self.fragment_local_aux_dim),
                nn.Linear(self.fragment_local_aux_dim, int(internal_d)),
                nn.ReLU(),
                nn.Linear(int(internal_d), int(internal_d)),
            )
        else:
            self.frag_aux_proj = None

        self.align_to_base_proj = nn.Linear(4, int(internal_d))

        try:
            selector_dropout = float(os.environ.get("SELECTOR_HEAD_DROPOUT", "0.0"))
        except Exception:
            selector_dropout = 0.0

        use_selector_head_v2 = os.environ.get("USE_SELECTOR_HEAD_V2", "0") == "1"

        if use_selector_head_v2:
            self.selector_head = SelectorRankHeadV2(
                hidden_dim=int(internal_d),
                peak_feat_dim=6,
                frag_aux_dim=self.fragment_local_aux_dim,
                num_heads=4,
                dropout=selector_dropout,
            )
        else:
            self.selector_head = SupportAwareSelectorHead(
                hidden_dim=int(internal_d),
                peak_feat_dim=6,
                frag_aux_dim=self.fragment_local_aux_dim,
                num_heads=4,
                dropout=selector_dropout,
            )

        # New: set-wise scorer
        self.set_score_norm = nn.LayerNorm(int(internal_d) * 4 + 2)
        self.set_score_l1 = nn.Linear(int(internal_d) * 4 + 2, int(internal_d))
        self.set_score_l2 = nn.Linear(int(internal_d), int(internal_d))
        self.set_score_out = nn.Linear(int(internal_d), 1)

        oos_in_d = int(self.g_feat_in) + int(self.peak_support_proj_dim) + int(self.peak_support_proj_dim) + 4
        self.oos_norm = nn.LayerNorm(oos_in_d)
        self.oos_head = nn.Sequential(
            nn.Linear(oos_in_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.peak_feat_dim = 6
        self.peak_embed_dim = 64

        self.peak_feat_norm = nn.LayerNorm(self.peak_feat_dim)
        self.peak_feat_proj = nn.Sequential(
            nn.Linear(self.peak_feat_dim, self.peak_embed_dim),
            nn.ReLU(),
            nn.Linear(self.peak_embed_dim, self.peak_embed_dim),
            nn.ReLU(),
        )

        peak_head_in = int(internal_d) + self.peak_embed_dim
        if self.use_msms_conditioning and self.score_cond_concat:
            peak_head_in += self.cond_feature_d

        self.peak_score_norm = nn.LayerNorm(peak_head_in)
        self.peak_score_mlp = nn.Sequential(
            nn.Linear(peak_head_in, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.adduct_emb = None
        self.instrument_emb = None
        self.ms_level_emb = None
        self.ce_proj = None
        self.precursor_proj = None

        if self.use_msms_conditioning:
            self.ce_proj = nn.Sequential(
                nn.Linear(1, int(ce_emb_dim)),
                nn.ReLU(),
                nn.Linear(int(ce_emb_dim), int(ce_emb_dim)),
            )
            self.precursor_proj = nn.Sequential(
                nn.Linear(1, int(ce_emb_dim)),
                nn.ReLU(),
                nn.Linear(int(ce_emb_dim), int(ce_emb_dim)),
            )
            self.adduct_emb = nn.Embedding(int(adduct_vocab_size), int(adduct_emb_dim))
            self.instrument_emb = nn.Embedding(int(instrument_vocab_size), int(instrument_emb_dim))
            self.ms_level_emb = nn.Embedding(int(ms_level_vocab_size), int(ms_level_emb_dim))

        # Precursor head must live outside use_msms_conditioning.
        self.use_precursor_head = os.environ.get("USE_PRECURSOR_HEAD", "1") == "1"

        precursor_in_d = self.g_feat_in
        if self.use_msms_conditioning and self.score_cond_concat:
            precursor_in_d += self.cond_feature_d

        self.precursor_head = nn.Sequential(
            nn.LayerNorm(precursor_in_d),
            nn.Linear(precursor_in_d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.use_fragment_node_candidates = os.environ.get("USE_FRAGMENT_NODE_CANDIDATES", "0") == "1"
        if self.use_fragment_node_candidates:
            fn_head_in_d = self.g_feat_in + 31 + 6
            if self.use_msms_conditioning and self.score_cond_concat:
                fn_head_in_d += self.cond_feature_d
            self.fn_selector = nn.Sequential(
                nn.LayerNorm(fn_head_in_d),
                nn.Linear(fn_head_in_d, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
            )
        else:
            self.fn_selector = None

    def _build_candidate_true_alignment_feat(
        self,
        batch_n,
        formula_n,
        device,
        formulae_mask,
        off_idx,
        off_int,
        true_idx,
        true_val,
        official_bin_n,
        official_bin_width,
    ):
        """
        Build per-candidate alignment features between candidate peaks and true spectrum.
        Returns [B, M, 4] tensor.
        """
        del official_bin_width
        feat_dim = 4
        out = torch.zeros((batch_n, formula_n, feat_dim), dtype=torch.float32, device=device)

        if off_idx is None or off_int is None or true_idx is None or true_val is None:
            return out
        if not torch.is_tensor(off_idx) or not torch.is_tensor(off_int):
            return out

        true_dense = torch.zeros((batch_n, official_bin_n), dtype=torch.float32, device=device)

        if isinstance(true_idx, (list, tuple)) and isinstance(true_val, (list, tuple)):
            use_b = min(batch_n, len(true_idx), len(true_val))
            for b in range(use_b):
                try:
                    idx = true_idx[b]
                    val = true_val[b]
                except Exception:
                    continue
                if idx is None or val is None:
                    continue
                if not torch.is_tensor(idx):
                    idx = torch.as_tensor(idx)
                if not torch.is_tensor(val):
                    val = torch.as_tensor(val)
                idx = idx.to(device=device, dtype=torch.long).reshape(-1)
                val = val.to(device=device, dtype=torch.float32).reshape(-1)
                if idx.numel() == 0 or val.numel() == 0:
                    continue
                use_n = min(int(idx.shape[0]), int(val.shape[0]))
                idx = idx[:use_n]
                val = val[:use_n]
                valid = (
                    (idx >= 0)
                    & (idx < int(official_bin_n))
                    & torch.isfinite(val)
                    & (val > 0)
                )
                if not bool(valid.any().item()):
                    continue
                idx_v = idx[valid].clamp(0, max(0, int(official_bin_n) - 1))
                val_v = val[valid]
                true_dense[b].scatter_add_(0, idx_v, val_v)

        elif torch.is_tensor(true_idx) and torch.is_tensor(true_val):
            idx_t = true_idx
            val_t = true_val
            if idx_t.dim() == 1:
                idx_t = idx_t.unsqueeze(0)
            if val_t.dim() == 1:
                val_t = val_t.unsqueeze(0)

            use_b = min(batch_n, int(idx_t.shape[0]), int(val_t.shape[0]))
            for b in range(use_b):
                idx = idx_t[b].to(device=device, dtype=torch.long).reshape(-1)
                val = val_t[b].to(device=device, dtype=torch.float32).reshape(-1)
                if idx.numel() == 0 or val.numel() == 0:
                    continue
                use_n = min(int(idx.shape[0]), int(val.shape[0]))
                idx = idx[:use_n]
                val = val[:use_n]
                valid = (
                    (idx >= 0)
                    & (idx < int(official_bin_n))
                    & torch.isfinite(val)
                    & (val > 0)
                )
                if not bool(valid.any().item()):
                    continue
                idx_v = idx[valid].clamp(0, max(0, int(official_bin_n) - 1))
                val_v = val[valid]
                true_dense[b].scatter_add_(0, idx_v, val_v)
        else:
            return out

        true_support = true_dense > 1e-8
        true_norm = torch.norm(true_dense, dim=1).clamp_min(1e-8)

        off_idx = off_idx[:batch_n, :formula_n]
        off_int = off_int[:batch_n, :formula_n]

        for b in range(batch_n):
            idx_b = off_idx[b].long()
            int_b = off_int[b].float()
            valid_peak = (
                (idx_b >= 0)
                & (idx_b < int(official_bin_n))
                & torch.isfinite(int_b)
                & (int_b > 0)
            )
            if not bool(valid_peak.any().item()):
                continue

            idx_safe = idx_b.clamp(0, max(0, int(official_bin_n) - 1))
            hit = true_support[b][idx_safe].float() * valid_peak.float()
            overlap_count = hit.sum(dim=-1)
            int_overlap = (int_b * hit).sum(dim=-1)

            dot = (int_b * true_dense[b][idx_safe] * valid_peak.float()).sum(dim=-1)
            cand_norm = torch.sqrt((int_b * int_b * valid_peak.float()).sum(dim=-1)).clamp_min(1e-8)
            cos_sim = dot / (cand_norm * true_norm[b] + 1e-8)

            cand_total_int = (int_b * valid_peak.float()).sum(dim=-1).clamp_min(1e-8)
            int_precision = int_overlap / cand_total_int

            feat = torch.stack(
                [
                    overlap_count / 32.0,
                    int_overlap,
                    cos_sim,
                    int_precision,
                ],
                dim=-1,
            )
            out[b, :formula_n] = feat

        if torch.is_tensor(formulae_mask):
            fm = formulae_mask.float()
            if fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)
            use_b = min(batch_n, int(fm.shape[0]))
            use_m = min(formula_n, int(fm.shape[1]))
            out[:use_b, :use_m] = out[:use_b, :use_m] * fm[:use_b, :use_m].unsqueeze(-1)

        return out

    def _sparsify_formula_probs_for_render(self, formulae_probs, formulae_mask=None):
        """
        Make rendering sparse without changing the scoring heads.

        This directly addresses false dense template spectra caused by mixing too many candidates.
        """
        if self.training and not self.render_topk_train:
            return formulae_probs

        p = formulae_probs

        if formulae_mask is not None:
            p = p * formulae_mask.float()

        keep = torch.ones_like(p, dtype=torch.bool)

        if self.render_min_formula_prob > 0:
            keep = keep & (p >= float(self.render_min_formula_prob))

        topk = int(self.render_topk_formula)
        if topk > 0 and p.shape[1] > topk:
            vals, idx = torch.topk(p, k=topk, dim=1)
            topk_mask = torch.zeros_like(p, dtype=torch.bool)
            topk_mask.scatter_(1, idx, True)
            keep = keep & topk_mask

        p_sparse = p * keep.float()

        denom = p_sparse.sum(dim=1, keepdim=True)

        # If a row is emptied by min_prob/topk, fall back to masked p.
        empty = denom <= 1e-8
        if torch.any(empty):
            p_fallback = p
            fallback_denom = p_fallback.sum(dim=1, keepdim=True).clamp_min(1e-8)
            p_fallback = p_fallback / fallback_denom
            p_sparse = torch.where(empty, p_fallback, p_sparse)
            denom = p_sparse.sum(dim=1, keepdim=True)

        p_sparse = p_sparse / denom.clamp_min(1e-8)
        return p_sparse

    def _selector_logits_to_formula_probs(
        self,
        selector_logits,
        formulae_mask,
        kwargs,
        batch_n,
        formula_n,
        device,
    ):
        """
        FraGNNet-lite path:
        Convert selector logits into a probability distribution over all valid candidates.
        """
        if not torch.is_tensor(selector_logits):
            return None

        logits = selector_logits.float()

        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        elif logits.dim() > 2:
            logits = logits.reshape(logits.shape[0], -1)

        use_b = min(int(logits.shape[0]), int(batch_n))
        use_m = min(int(logits.shape[1]), int(formula_n))
        logits = logits[:use_b, :use_m]

        if torch.is_tensor(formulae_mask):
            fm = formulae_mask.float()
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            elif fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)

            mask = torch.zeros_like(logits)
            mb = min(use_b, int(fm.shape[0]))
            mm = min(use_m, int(fm.shape[1]))
            mask[:mb, :mm] = fm[:mb, :mm].to(device=logits.device, dtype=logits.dtype)
        else:
            mask = torch.ones_like(logits)

        valid = mask > 0.5
        logits = logits.masked_fill(~valid, _neg_mask_fill_value(logits))

        try:
            temp = float(os.environ.get("SELECTOR_PROB_TEMP", "1.0"))
        except Exception:
            temp = 1.0
        temp = max(temp, 1e-6)
        logits = logits / temp

        group_id = self._get_group_ids_for_candidates(
            kwargs=kwargs,
            batch_n=use_b,
            formula_n=use_m,
            device=logits.device,
        )

        if torch.is_tensor(group_id) and os.environ.get("USE_GROUPMAX_PROB", "1") == "1":
            probs = self._groupmax_softmax_to_instance_probs(
                scores=logits,
                formulae_mask=mask,
                group_id=group_id,
            )
        else:
            probs = torch.softmax(logits, dim=1)
            probs = probs * valid.float()
            probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        probs = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
        probs = probs.clamp_min(0.0)
        probs = probs * valid.float()
        probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)

        return probs

    def _get_group_ids_for_candidates(self, kwargs, batch_n, formula_n, device):
        """
        Return [B, M] group ids.

        Important:
        In rerank stage, formula_n may be only selector_topk, e.g. 256,
        but group ids are original formula ids, e.g. 0..4095.
        Therefore we must NOT clamp group ids to formula_n - 1.
        The grouping function will remap them row-wise.
        """
        group_id = kwargs.get("formulae_instance_group_id", None)
        if group_id is None:
            return None

        if not torch.is_tensor(group_id):
            try:
                group_id = torch.as_tensor(group_id)
            except Exception:
                return None

        group_id = group_id.to(device=device, dtype=torch.long)

        if group_id.dim() == 1:
            group_id = group_id.unsqueeze(0)
        elif group_id.dim() > 2:
            group_id = group_id.reshape(group_id.shape[0], -1)

        if group_id.shape[0] < batch_n:
            pad = torch.zeros(
                (batch_n - int(group_id.shape[0]), int(group_id.shape[1])),
                dtype=torch.long,
                device=device,
            )
            group_id = torch.cat([group_id, pad], dim=0)

        group_id = group_id[:batch_n]

        if group_id.shape[1] < formula_n:
            # Missing columns should be unique padding groups, not all group 0.
            start = int(group_id.shape[1])
            pad = torch.arange(
                start,
                formula_n,
                dtype=torch.long,
                device=device,
            ).view(1, -1).expand(batch_n, -1)
            group_id = torch.cat([group_id, pad], dim=1)

        group_id = group_id[:, :formula_n]
        group_id = group_id.clamp(min=0)
        return group_id

    def _groupmax_softmax_to_instance_probs(self, scores, formulae_mask, group_id):
        """
        Convert instance scores [B, M] into group-aware instance probabilities [B, M].

        Correct behavior:
          - source-instance candidates can compete inside the same formula group;
          - one formula group gets one probability mass;
          - only the best instance(s) in that group receive that mass;
          - original group ids may be large, especially after rerank pruning, so remap row-wise.
        """
        if (not torch.is_tensor(scores)) or (not torch.is_tensor(group_id)):
            return None

        if scores.dim() == 1:
            scores = scores.unsqueeze(0)
        elif scores.dim() > 2:
            scores = scores.reshape(scores.shape[0], -1)

        if group_id.dim() == 1:
            group_id = group_id.unsqueeze(0)
        elif group_id.dim() > 2:
            group_id = group_id.reshape(group_id.shape[0], -1)

        B = min(int(scores.shape[0]), int(group_id.shape[0]))
        M = min(int(scores.shape[1]), int(group_id.shape[1]))
        if B <= 0 or M <= 0:
            return None

        scores = scores[:B, :M]
        gid = group_id[:B, :M].to(device=scores.device, dtype=torch.long)

        if formulae_mask is None:
            valid = torch.ones((B, M), dtype=torch.bool, device=scores.device)
        else:
            fm = formulae_mask.to(device=scores.device)
            if fm.dim() == 1:
                fm = fm.unsqueeze(0)
            elif fm.dim() > 2:
                fm = fm.reshape(fm.shape[0], -1)
            use_b = min(B, int(fm.shape[0]))
            use_m = min(M, int(fm.shape[1]))
            valid = torch.zeros((B, M), dtype=torch.bool, device=scores.device)
            valid[:use_b, :use_m] = fm[:use_b, :use_m] > 0.5

        valid = valid & torch.isfinite(scores)

        out = torch.zeros_like(scores)
        neg = _neg_mask_fill_value(scores)

        for bi in range(B):
            valid_b = valid[bi]
            if not bool(valid_b.any().item()):
                continue

            idx_valid = torch.nonzero(valid_b, as_tuple=False).reshape(-1)
            s_b = scores[bi, idx_valid]
            gid_b = gid[bi, idx_valid]

            # Remap arbitrary original group ids to compact ids.
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

            group_probs = torch.softmax(group_scores, dim=0)

            best_score_for_instance = group_scores[inv]
            winner = s_b >= (best_score_for_instance - 1e-6)

            winner_count = s_b.new_zeros((group_n,))
            winner_count.scatter_add_(0, inv, winner.float())
            winner_count_for_instance = winner_count[inv].clamp_min(1.0)

            out_b = torch.where(
                winner,
                group_probs[inv] / winner_count_for_instance,
                torch.zeros_like(s_b),
            )

            out[bi, idx_valid] = out_b

        out = out * valid.float()
        out = out / out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return out

    def forward(
        self,
        vert_feat_in,
        vert_mask_in,
        possible_formulae,
        formulae_peaks_mass_idx,
        formulae_peaks_intensity,
        **kwargs,
    ):
        batch_n = int(vert_feat_in.shape[0])
        formula_n = int(possible_formulae.shape[1])
        device = vert_feat_in.device

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_n = int(masked_vert_feat.shape[1])

        formulae_mask = self._to_formulae_mask(kwargs.get('formulae_mask', None), batch_n, formula_n, device)
        formula_topk_orig_idx = kwargs.get('formula_topk_orig_idx', None)
        if formula_topk_orig_idx is not None and (not torch.is_tensor(formula_topk_orig_idx)):
            formula_topk_orig_idx = torch.as_tensor(formula_topk_orig_idx)
        if torch.is_tensor(formula_topk_orig_idx):
            formula_topk_orig_idx = formula_topk_orig_idx.to(device=device, dtype=torch.long)
        # ---- robust coercion for main projection path ----
        formulae_peaks_mass_idx = self._coerce_peak_tensor(
            formulae_peaks_mass_idx,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        formulae_peaks_intensity = self._coerce_peak_tensor(
            formulae_peaks_intensity,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        if formulae_peaks_mass_idx is None or formulae_peaks_intensity is None:
            formulae_peaks_mass_idx = torch.full(
                (batch_n, formula_n, 1),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            formulae_peaks_intensity = torch.zeros(
                (batch_n, formula_n, 1),
                dtype=torch.float32,
                device=device,
            )

        formulae_peaks_exact = self._coerce_exact_peaks_tensor(
            kwargs.get('formulae_peaks', None),
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
        )
        pf_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        pf_oh_flat = self.formula_to_oh(pf_flat)
        possible_formulae_oh = pf_oh_flat.reshape(possible_formulae.shape[0], possible_formulae.shape[1], -1)
        if self.formula_oh_normalize:
            possible_formulae_oh = possible_formulae_oh / (possible_formulae_oh.sum(axis=2).unsqueeze(2) + 1e-8)

        atom_mask = vert_mask_in > 0
        vert_encoded = self.embed_g_feat(masked_vert_feat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae_oh)

        q = self.formula_q(formulae_encoded)
        k = self.vert_k(vert_encoded)
        v = self.vert_v(masked_vert_feat.float())
        attn_scale = max(float(np.sqrt(max(1, int(q.shape[-1])))), 1e-6)
        attn_logits = torch.einsum('bmd,bad->bam', q, k) / attn_scale
        attn_logits = attn_logits / self.candidate_attn_temperature
        attn_logits = attn_logits.masked_fill(~atom_mask.unsqueeze(-1), torch.finfo(attn_logits.dtype).min)
        weighting = torch.softmax(attn_logits, dim=1)
        weighting = weighting * atom_mask.unsqueeze(-1).to(weighting.dtype)
        weighting = weighting / (weighting.sum(dim=1, keepdim=True) + 1e-8)
        vert_att_reduce = torch.bmm(weighting.transpose(1, 2), v)

        cond_embed = None
        if self.use_msms_conditioning:
            cond_parts = []
            ce_2d = self._to_2d_float(kwargs.get('ce', None), batch_n, device)
            precursor_2d = self._to_2d_float(kwargs.get('precursor_mz', None), batch_n, device)
            if self.ce_proj is not None:
                cond_parts.append(self.ce_proj(ce_2d))
            if self.precursor_proj is not None:
                cond_parts.append(self.precursor_proj(precursor_2d))
            if self.adduct_emb is not None:
                adduct_idx = self._to_vocab_index(kwargs.get('adduct', None), batch_n, device, self.adduct_emb.num_embeddings)
                cond_parts.append(self.adduct_emb(adduct_idx))
            if self.instrument_emb is not None:
                ins_idx = self._to_vocab_index(kwargs.get('instrument_type', None), batch_n, device, self.instrument_emb.num_embeddings)
                cond_parts.append(self.instrument_emb(ins_idx))
            if self.ms_level_emb is not None:
                ms_idx = self._to_vocab_index(kwargs.get('ms_level', None), batch_n, device, self.ms_level_emb.num_embeddings)
                cond_parts.append(self.ms_level_emb(ms_idx))
            if len(cond_parts) > 0:
                cond_embed = torch.cat(cond_parts, dim=-1)

        formulae_aux_feat = kwargs.get('formulae_aux_feat', None)
        if formulae_aux_feat is None:
            aux = torch.zeros((batch_n, formula_n, self.formulae_aux_dim), dtype=torch.float32, device=device)
        else:
            if not torch.is_tensor(formulae_aux_feat):
                formulae_aux_feat = torch.as_tensor(formulae_aux_feat)
            aux = formulae_aux_feat.to(device=device, dtype=torch.float32)
            if aux.dim() == 1:
                aux = aux.view(1, 1, -1)
            elif aux.dim() == 2:
                aux = aux.unsqueeze(0)
            elif aux.dim() > 3:
                aux = aux.reshape(aux.shape[0], aux.shape[1], -1)

            if aux.dim() != 3:
                aux = torch.zeros((batch_n, formula_n, self.formulae_aux_dim), dtype=torch.float32, device=device)
            else:
                cur_b, cur_m, cur_d = int(aux.shape[0]), int(aux.shape[1]), int(aux.shape[2])
                if cur_b < batch_n:
                    pad = torch.zeros((batch_n - cur_b, cur_m, cur_d), dtype=aux.dtype, device=device)
                    aux = torch.cat([aux, pad], dim=0)
                aux = aux[:batch_n]

                if int(aux.shape[1]) < formula_n:
                    pad = torch.zeros((batch_n, formula_n - int(aux.shape[1]), int(aux.shape[2])), dtype=aux.dtype, device=device)
                    aux = torch.cat([aux, pad], dim=1)
                aux = aux[:, :formula_n, :]

                if self.formulae_aux_dim <= 0:
                    aux = aux[:, :, :0]
                elif int(aux.shape[2]) < self.formulae_aux_dim:
                    pad = torch.zeros((batch_n, formula_n, self.formulae_aux_dim - int(aux.shape[2])), dtype=aux.dtype, device=device)
                    aux = torch.cat([aux, pad], dim=-1)
                else:
                    aux = aux[:, :, :self.formulae_aux_dim]

        off_idx = kwargs.get('formulae_peaks_official_idx', None)
        off_int = kwargs.get('formulae_peaks_official_intensity', None)
        if off_idx is None:
            off_idx = kwargs.get('formulae_peaks_official_idx_agg', None)
        if off_int is None:
            off_int = kwargs.get('formulae_peaks_official_intensity_agg', None)

        off_idx = self._coerce_peak_tensor(
            off_idx,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        off_int = self._coerce_peak_tensor(
            off_int,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        official_bin_width = float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01'))
        official_bin_n = int(np.floor(float(os.environ.get('OFFICIAL_MAX_MZ', '1005.0')) / official_bin_width)) + 1

        candidate_peak_feat_raw = self._build_candidate_only_peak_summary(
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            formulae_mask=formulae_mask,
            off_idx=off_idx,
            off_int=off_int,
            precursor_mz=kwargs.get('precursor_mz', None),
            official_bin_width=official_bin_width,
            official_bin_n=official_bin_n,
        )
        candidate_peak_feat = self.peak_support_proj(candidate_peak_feat_raw)
        true_idx = kwargs.get('true_official_idx', None)
        true_val = kwargs.get('true_official_intensity', None)
        if true_idx is None:
            true_idx = kwargs.get('true_all_official_idx', None)
            true_val = kwargs.get('true_all_official_intensity', None)
        allow_target_alignment = os.environ.get("MODEL_ALLOW_TARGET_ALIGNMENT_FEAT", "0") == "1"
        align_feat = None
        if allow_target_alignment:
            if self.training:
                raise RuntimeError(
                    "MODEL_ALLOW_TARGET_ALIGNMENT_FEAT=1 is forbidden during training: target leakage risk."
                )
            align_feat = self._build_candidate_true_alignment_feat(
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                formulae_mask=formulae_mask,
                off_idx=off_idx,
                off_int=off_int,
                true_idx=true_idx,
                true_val=true_val,
                official_bin_n=official_bin_n,
                official_bin_width=official_bin_width,
            )

        frag_aux = kwargs.get('formulae_frag_aux_feat', None)
        frag_aux_t = self._coerce_frag_aux(
            frag_aux,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
        )

        base_parts = [formulae_encoded, vert_att_reduce, candidate_peak_feat]
        if self.formulae_aux_dim > 0:
            base_parts.append(aux)
        if self.use_msms_conditioning and self.score_cond_concat and torch.is_tensor(cond_embed) and int(cond_embed.shape[-1]) > 0:
            cond_expand = cond_embed.unsqueeze(1).expand(batch_n, formula_n, int(cond_embed.shape[-1]))
            base_parts.append(cond_expand)

        base_in = torch.cat(base_parts, dim=-1)
        base_in = self.base_score_norm(base_in)
        base_h = F.relu(self.base_score_l1(base_in))
        base_h = F.relu(self.base_score_l2(base_h))

        if align_feat is not None:
            base_h = base_h + self.align_to_base_proj(align_feat)

        frag_has_source = None
        if self.frag_aux_proj is not None and torch.is_tensor(frag_aux_t):
            frag_has_source = (frag_aux_t[..., 0:1] > 0.5).float()
            frag_h = self.frag_aux_proj(frag_aux_t)
            if self.gate_fragment_local_aux:
                frag_h = frag_h * frag_has_source
            if torch.is_tensor(formulae_mask):
                frag_h = frag_h * formulae_mask.unsqueeze(-1).float()
            base_h = base_h + float(self.fragment_local_aux_scale) * frag_h

        selector_peak_idx = kwargs.get('formulae_peaks_official_idx', None)
        selector_peak_int = kwargs.get('formulae_peaks_official_intensity', None)

        selector_peak_idx = self._coerce_peak_tensor(
            selector_peak_idx,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        selector_peak_int = self._coerce_peak_tensor(
            selector_peak_int,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        if selector_peak_idx is None or selector_peak_int is None:
            selector_peak_idx = self._coerce_peak_tensor(
                formulae_peaks_mass_idx,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.long,
                fill_value=-1,
            )
            selector_peak_int = self._coerce_peak_tensor(
                formulae_peaks_intensity,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.float32,
                fill_value=0.0,
            )
            selector_peak_bin_width = 1.0
        else:
            selector_peak_bin_width = float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01'))

        if selector_peak_idx is None or selector_peak_int is None:
            selector_peak_idx = torch.full(
                (batch_n, formula_n, 1),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            selector_peak_int = torch.zeros(
                (batch_n, formula_n, 1),
                dtype=torch.float32,
                device=device,
            )
            selector_peak_bin_width = 1.0

        selector_peak_n = int(selector_peak_idx.shape[-1])
        selector_peak_feat = self._build_peak_features(
            formulae_peaks_mass_idx=selector_peak_idx,
            formulae_peaks_intensity=selector_peak_int,
            precursor_mz=kwargs.get('precursor_mz', None),
            batch_n=batch_n,
            formula_n=formula_n,
            peak_n=selector_peak_n,
            device=device,
            official_bin_width=selector_peak_bin_width,
        )

        # debug diagnostics for explanation matching
        matched_subset_per_sample = torch.zeros((batch_n,), dtype=torch.float32, device=device)
        matched_formula_per_sample = torch.zeros((batch_n,), dtype=torch.float32, device=device)
        invalid_formula = formulae_mask <= 0
        selector_logits = self.selector_head(
            base_h,
            formulae_mask=formulae_mask,
            peak_feat=selector_peak_feat,
            frag_aux=frag_aux_t,
        )
        selector_logits = selector_logits.masked_fill(
            invalid_formula,
            torch.finfo(selector_logits.dtype).min,
        )

        selector_logits_for_feat = selector_logits
        selector_probs_for_feat = torch.softmax(selector_logits, dim=-1)

        selector_logit_feat = selector_logits_for_feat.unsqueeze(-1)
        selector_prob_feat = selector_probs_for_feat.unsqueeze(-1)

        atom_denom = atom_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)
        mol_pool = masked_vert_feat.sum(dim=1) / atom_denom

        # ------------------------------------------------------------
        # V3C: Fragment-node selector forward pass.
        # ------------------------------------------------------------
        fragment_node_logits = None
        fn_based_formula_logits = None

        if self.use_fragment_node_candidates and "fragment_node_mask" in kwargs:
            fn_mask = kwargs["fragment_node_mask"].float()
            fn_local = kwargs["fragment_node_local_feat"].float()

            fn_meta = torch.stack(
                [
                    kwargs["fragment_node_mz"].float() / 1500.0,
                    kwargs["fragment_node_depth"].float() / 4.0,
                    kwargs["fragment_node_h_shift"].float() / 4.0,
                    kwargs["fragment_node_is_brics"].float(),
                    kwargs["fragment_node_ring_cut"].float(),
                    kwargs["fragment_node_source_type"].float() / 3.0,
                ],
                dim=-1,
            )

            mol_pool_exp = mol_pool.unsqueeze(1).expand(-1, fn_mask.shape[1], -1)

            fn_parts = [mol_pool_exp, fn_local, fn_meta]
            if self.use_msms_conditioning and self.score_cond_concat and cond_embed is not None:
                cond_exp = cond_embed.unsqueeze(1).expand(-1, fn_mask.shape[1], -1)
                fn_parts.append(cond_exp)

            fn_in = torch.cat(fn_parts, dim=-1)
            fragment_node_logits = self.fn_selector(fn_in).squeeze(-1)

            fragment_node_logits = fragment_node_logits.masked_fill(
                fn_mask <= 0.5,
                _neg_mask_fill_value(fragment_node_logits),
            )

            # Safe max-pooling bridge to formula logits.
            B = fragment_node_logits.shape[0]
            M = possible_formulae.shape[1]

            # Initialize with very negative values to filter formulae with no fragments.
            fn_based_formula_logits = torch.full(
                (B, M), _neg_mask_fill_value(fragment_node_logits),
                device=device, dtype=torch.float32
            )

            if "fragment_node_group_formula_id" in kwargs:
                group_id = kwargs["fragment_node_group_formula_id"].long()
                valid_mapping = (group_id >= 0) & (group_id < M) & (fn_mask > 0.5)

                for b in range(B):
                    v_idx = valid_mapping[b]
                    if v_idx.any():
                        g_b = group_id[b, v_idx]
                        l_b = fragment_node_logits[b, v_idx]

                        # Sort by score so scatter keeps the highest per formula.
                        order = torch.argsort(l_b)
                        g_b_sorted = g_b[order]
                        l_b_sorted = l_b[order]

                        fn_based_formula_logits[b].scatter_(0, g_b_sorted, l_b_sorted)

        precursor_logit = None
        precursor_prob = None
        if self.use_precursor_head and self.precursor_head is not None:
            precursor_parts = [mol_pool]
            if self.use_msms_conditioning and self.score_cond_concat and torch.is_tensor(cond_embed):
                precursor_parts.append(cond_embed)
            precursor_in = torch.cat(precursor_parts, dim=-1)
            precursor_logit = self.precursor_head(precursor_in).squeeze(-1)
            precursor_prob = torch.sigmoid(precursor_logit)

        # ------------------------------------------------------------
        # Fast pass-1 selector path.
        # Used by train_ms_subsetnet.py for full-candidate selector only.
        # ------------------------------------------------------------
        if bool(kwargs.get("selector_only_forward", False)):
            spect_bin_n = int(self.spect_bin.get_num_bins())

            official_bin_width = float(os.environ.get("OFFICIAL_BIN_WIDTH", "0.01"))
            official_max_mz = float(os.environ.get("OFFICIAL_MAX_MZ", "1005.0"))
            official_bin_n = int(np.floor(official_max_mz / official_bin_width)) + 1

            zero_spect = selector_logits.new_zeros((batch_n, spect_bin_n))
            zero_official = selector_logits.new_zeros((batch_n, official_bin_n))

            if precursor_logit is None:
                precursor_logit = selector_logits.new_zeros((batch_n,))
                precursor_prob = torch.sigmoid(precursor_logit)

            if os.environ.get("USE_GROUP_AWARE_SELECTOR_PROBS", "0") == "1":
                group_id = self._get_group_ids_for_candidates(
                    kwargs,
                    batch_n=batch_n,
                    formula_n=formula_n,
                    device=device,
                )
                selector_probs_group = self._groupmax_softmax_to_instance_probs(
                    selector_logits,
                    formulae_mask,
                    group_id,
                )
                if torch.is_tensor(selector_probs_group):
                    selector_probs = selector_probs_group
                else:
                    selector_probs = torch.softmax(selector_logits, dim=-1)
                    selector_probs = selector_probs * formulae_mask
                    selector_probs = selector_probs / selector_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            else:
                selector_probs = torch.softmax(selector_logits, dim=-1)
                selector_probs = selector_probs * formulae_mask
                selector_probs = selector_probs / selector_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            return {
                "spect_out": zero_spect,
                "spect_out_coarse": zero_spect,
                "spect_out_official": zero_official,

                # Keep these keys so trainer helper functions do not need special cases.
                "formulae_probs": selector_probs,
                "formulae_probs_render": selector_probs,
                "formulae_scores": selector_logits,
                "formulae_scores_raw": selector_logits,
                "formulae_scores_raw_setwise": selector_logits,
                "formulae_scores_train": selector_logits,

                "selector_logits": selector_logits,
                "precursor_logit": precursor_logit,
                "precursor_prob": precursor_prob,
                "fragment_node_logits": fragment_node_logits,
                "fn_based_formula_logits": fn_based_formula_logits,
            }

        use_selector_prob_spectrum = os.environ.get("USE_SELECTOR_PROB_SPECTRUM", "0") == "1"
        selector_prob_formulae_probs = None

        if use_selector_prob_spectrum:
            selector_prob_formulae_probs = self._selector_logits_to_formula_probs(
                selector_logits=selector_logits,
                formulae_mask=formulae_mask,
                kwargs=kwargs,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
            )
        mask3 = formulae_mask.unsqueeze(-1)
        mask3_sum = mask3.sum(dim=1, keepdim=True).clamp_min(1.0)

        candidate_peak_summary = (candidate_peak_feat * mask3).sum(dim=1) / mask3_sum.squeeze(1)
        candidate_raw_summary = (candidate_peak_feat_raw * mask3).sum(dim=1) / mask3_sum.squeeze(1)
        coverage_summary = candidate_raw_summary[:, :4]

        topk_for_oos = max(1, min(64, formula_n))
        topk_idx = torch.topk(selector_logits, k=topk_for_oos, dim=-1).indices
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, int(candidate_peak_feat.shape[-1]))
        topk_candidate_peak = torch.gather(candidate_peak_feat, 1, gather_idx)
        topk_peak_summary = topk_candidate_peak.mean(dim=1)

        oos_feat = torch.cat([mol_pool, candidate_peak_summary, topk_peak_summary, coverage_summary], dim=-1)
        oos_feat = self.oos_norm(oos_feat)
        oos_logit = self.oos_head(oos_feat).squeeze(-1)

        selector_probs = torch.softmax(selector_logits, dim=-1)
        selector_probs = selector_probs * formulae_mask
        selector_probs = selector_probs / selector_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)

        raw_rerank_topk = os.environ.get("RERANK_TOPK", "").strip()
        try:
            if raw_rerank_topk:
                selector_topk = int(raw_rerank_topk)
            else:
                selector_topk = int(os.environ.get("SELECTOR_TOPK", "256"))
        except Exception:
            selector_topk = 256
        pool_k = max(1, min(int(selector_topk), int(formula_n)))
        pool_idx = torch.topk(selector_logits, k=pool_k, dim=1).indices

        pool_hidden = self._gather_candidates_3d(base_h, pool_idx)
        pool_selector_logits = torch.gather(selector_logits, 1, pool_idx)

        if os.environ.get("RERANK_DETACH_SELECTOR", "0") == "1":
            pool_hidden_for_rerank = pool_hidden.detach()
        else:
            pool_hidden_for_rerank = pool_hidden

        pool_mean = pool_hidden_for_rerank.mean(dim=1, keepdim=True).expand_as(pool_hidden_for_rerank)
        pool_max = pool_hidden_for_rerank.max(dim=1, keepdim=True).values.expand_as(pool_hidden_for_rerank)

        rank_pos = torch.arange(pool_k, device=device).float()
        rank_pos = rank_pos.view(1, pool_k, 1).expand(batch_n, pool_k, 1) / max(1.0, float(pool_k - 1))

        selector_logit_feat = pool_selector_logits.unsqueeze(-1)

        set_in = torch.cat(
            [
                pool_hidden_for_rerank,
                pool_mean,
                pool_max,
                pool_hidden_for_rerank - pool_mean,
                selector_logit_feat,
                rank_pos,
            ],
            dim=-1,
        )

        set_in = self.set_score_norm(set_in)
        score_h = F.relu(self.set_score_l1(set_in))
        score_h = F.relu(self.set_score_l2(score_h))
        raw_rerank_delta_pool = self.set_score_out(score_h).squeeze(-1)

        try:
            rerank_loss_weight_env = float(os.environ.get("RERANK_LOSS_WEIGHT", "0.0"))
        except Exception:
            rerank_loss_weight_env = 0.0

        use_rerank_delta = (
            os.environ.get("USE_RERANK_DELTA", "0") == "1"
            or rerank_loss_weight_env > 0.0
        )

        if use_rerank_delta:
            rerank_delta_pool = raw_rerank_delta_pool
        else:
            rerank_delta_pool = torch.zeros_like(pool_selector_logits)

        rerank_delta_pool_mean = rerank_delta_pool.mean().detach()
        rerank_delta_pool_std = rerank_delta_pool.std().detach()
        selector_logits_pool_std = pool_selector_logits.std().detach()

        final_pool_logits = pool_selector_logits + rerank_delta_pool

        neg_fill = _neg_mask_fill_value(selector_logits)
        final_logits = selector_logits.new_full(selector_logits.shape, neg_fill)
        final_logits.scatter_(1, pool_idx, final_pool_logits)

        rerank_logits_full = selector_logits.new_full(selector_logits.shape, neg_fill)
        rerank_logits_full.scatter_(1, pool_idx, rerank_delta_pool)

        formulae_scores_raw = final_logits
        formulae_scores_raw_setwise = rerank_logits_full
        base_score_raw = final_logits

        # Peak head must use non-aggregated official peaks to preserve
        # the original per-candidate peak order. Aggregated peaks remain
        # available for candidate-level summary features above, but not for
        # peak reweighting.
        peak_idx_for_head = kwargs.get('formulae_peaks_official_idx', None)
        peak_int_for_head = kwargs.get('formulae_peaks_official_intensity', None)

        peak_idx_for_head = self._coerce_peak_tensor(
            peak_idx_for_head,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        peak_int_for_head = self._coerce_peak_tensor(
            peak_int_for_head,
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        if peak_idx_for_head is None or peak_int_for_head is None:
            peak_idx_for_head = self._coerce_peak_tensor(
                formulae_peaks_mass_idx,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.long,
                fill_value=-1,
            )
            peak_int_for_head = self._coerce_peak_tensor(
                formulae_peaks_intensity,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.float32,
                fill_value=0.0,
            )
            peak_bin_width = 1.0
        else:
            peak_bin_width = float(os.environ.get('OFFICIAL_BIN_WIDTH', '0.01'))

        # Final fallback if even the second attempt fails.
        if peak_idx_for_head is None or peak_int_for_head is None:
            peak_idx_for_head = torch.full(
                (batch_n, formula_n, 1),
                fill_value=-1,
                dtype=torch.long,
                device=device,
            )
            peak_int_for_head = torch.zeros(
                (batch_n, formula_n, 1),
                dtype=torch.float32,
                device=device,
            )
            peak_bin_width = 1.0

        peak_n = int(peak_idx_for_head.shape[-1])

        peak_feat = self._build_peak_features(
            formulae_peaks_mass_idx=peak_idx_for_head,
            formulae_peaks_intensity=peak_int_for_head,
            precursor_mz=kwargs.get('precursor_mz', None),
            batch_n=batch_n,
            formula_n=formula_n,
            peak_n=peak_n,
            device=device,
            official_bin_width=peak_bin_width,
        )

        peak_feat = self.peak_feat_norm(peak_feat)
        peak_feat_emb = self.peak_feat_proj(peak_feat)

        base_h_expand = base_h.unsqueeze(2).expand(batch_n, formula_n, peak_n, base_h.shape[-1])

        peak_parts = [base_h_expand, peak_feat_emb]

        if self.use_msms_conditioning and self.score_cond_concat and torch.is_tensor(cond_embed) and int(cond_embed.shape[-1]) > 0:
            cond_expand_peak = cond_embed.unsqueeze(1).unsqueeze(2).expand(
                batch_n, formula_n, peak_n, int(cond_embed.shape[-1])
            )
            peak_parts.append(cond_expand_peak)

        peak_in = torch.cat(peak_parts, dim=-1)
        peak_in = self.peak_score_norm(peak_in)
        peak_reweight_logits = self.peak_score_mlp(peak_in).squeeze(-1)

        peak_valid = (
            (peak_idx_for_head >= 0)
            & torch.isfinite(peak_int_for_head)
            & (peak_int_for_head > 0)
        )

        peak_reweight_logits = peak_reweight_logits.masked_fill(
            ~peak_valid,
            torch.finfo(peak_reweight_logits.dtype).min,
        )

        peak_reweight_probs = torch.softmax(peak_reweight_logits, dim=-1)
        peak_intensity_for_render = formulae_peaks_intensity

        use_peak_reweight_in_prob = (
            os.environ.get("USE_PEAK_REWEIGHT_IN_PROB_SPECTRUM", "0") == "1"
        )

        if use_selector_prob_spectrum and use_peak_reweight_in_prob and torch.is_tensor(peak_reweight_probs):
            # 推荐第一版用 redistribute，不用 sigmoid 直接压强度
            # 这样只是重新分配每个 candidate 内部峰强度，不改变该 candidate 总强度
            use_b = min(int(peak_reweight_probs.shape[0]), int(formulae_peaks_intensity.shape[0]))
            use_m = min(int(peak_reweight_probs.shape[1]), int(formulae_peaks_intensity.shape[1]))
            use_k = min(int(peak_reweight_probs.shape[2]), int(formulae_peaks_intensity.shape[2]))

            orig_int = formulae_peaks_intensity[:use_b, :use_m, :use_k].float()
            pr = peak_reweight_probs[:use_b, :use_m, :use_k].float()

            valid_peak = (
                (formulae_peaks_mass_idx[:use_b, :use_m, :use_k] >= 0)
                & torch.isfinite(orig_int)
                & (orig_int > 0)
            )

            pr = pr * valid_peak.float()
            pr = pr / pr.sum(dim=-1, keepdim=True).clamp_min(1e-8)

            orig_total = (orig_int * valid_peak.float()).sum(dim=-1, keepdim=True)

            new_int = pr * orig_total

            peak_intensity_for_render = formulae_peaks_intensity.clone()
            peak_intensity_for_render[:use_b, :use_m, :use_k] = new_int
        peak_reweight_probs = torch.where(
            peak_valid,
            peak_reweight_probs,
            torch.zeros_like(peak_reweight_probs),
        )

        template_int = peak_int_for_head.float().clamp_min(0.0)
        refined_peak_intensity = template_int * peak_reweight_probs
        refined_peak_intensity = refined_peak_intensity / refined_peak_intensity.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        # ------------------------------------------------------------
        # FraGNNet-lite final probability-spectrum branch.
        #
        # IMPORTANT:
        # This branch must be placed AFTER peak_reweight_logits /
        # peak_reweight_probs are computed, otherwise peak head cannot
        # participate in spectrum rendering.
        # ------------------------------------------------------------
        if use_selector_prob_spectrum and torch.is_tensor(selector_prob_formulae_probs):
            formulae_probs = selector_prob_formulae_probs
            formulae_probs_render = self._sparsify_formula_probs_for_render(
                formulae_probs,
                formulae_mask=formulae_mask,
            )

            spect_bin_n = self.spect_bin.get_num_bins()

            # ----- coarse spectrum render -----
            coarse_int_for_prob = formulae_peaks_intensity

            if (
                use_peak_reweight_in_prob
                and torch.is_tensor(refined_peak_intensity)
                and torch.is_tensor(formulae_peaks_intensity)
                and refined_peak_intensity.shape == formulae_peaks_intensity.shape
            ):
                coarse_int_for_prob = refined_peak_intensity
            elif (
                use_peak_reweight_in_prob
                and torch.is_tensor(peak_intensity_for_render)
                and torch.is_tensor(formulae_peaks_intensity)
                and peak_intensity_for_render.shape == formulae_peaks_intensity.shape
            ):
                coarse_int_for_prob = peak_intensity_for_render

            spect_out = project_formula_probs_to_spectrum_dense(
                formulae_probs_render,
                formulae_peaks_mass_idx,
                coarse_int_for_prob,
                spect_bin_n,
                formulae_mask=formulae_mask,
                mass_shift_probs=None,
                mass_shift_offsets=None,
            )

            # ----- official spectrum render -----
            # Peak head was computed from peak_idx_for_head / peak_int_for_head,
            # so for official render this is the most aligned pair.
            official_idx_for_prob = peak_idx_for_head
            official_int_for_prob = peak_int_for_head

            if (
                use_peak_reweight_in_prob
                and torch.is_tensor(refined_peak_intensity)
                and torch.is_tensor(peak_int_for_head)
                and refined_peak_intensity.shape == peak_int_for_head.shape
            ):
                official_int_for_prob = refined_peak_intensity

            spect_out_official = project_formula_probs_to_spectrum_dense(
                formulae_probs_render,
                official_idx_for_prob,
                official_int_for_prob,
                official_bin_n,
                formulae_mask=formulae_mask,
                mass_shift_probs=None,
                mass_shift_offsets=None,
            )

            spect_out_official_formula = spect_out_official

            if os.environ.get('OFFICIAL_EXCLUDE_PRECURSOR', '1') == '1':
                spect_out_official = self._zero_precursor_bin_dense(
                    spect_out_official,
                    kwargs.get('precursor_mz', None),
                    official_bin_width,
                )

            if self.normalize_1_output:
                spect_out = spect_out / (
                    torch.sum(torch.abs(spect_out), dim=1, keepdim=True) + 1e-6
                )
                spect_out_official = spect_out_official / (
                    torch.sum(torch.abs(spect_out_official), dim=1, keepdim=True) + 1e-6
                )

            # Optional exact sparse peaks. Keep it simple in first version.
            pred_exact_peaks = None

            if not hasattr(self, "_printed_prob_peak_fusion_debug"):
                with torch.no_grad():
                    print(
                        "[PROB_PEAK_FUSION_DEBUG]",
                        "use_selector_prob_spectrum=", int(use_selector_prob_spectrum),
                        "use_peak_reweight=", int(use_peak_reweight_in_prob),
                        "has_peak_logits=", int(torch.is_tensor(peak_reweight_logits)),
                        "prob_shape=", tuple(formulae_probs.shape),
                        "prob_nonzero_first=",
                        int((formulae_probs[0] > 1e-8).sum().detach().cpu().item()),
                        "raw_int_sum_first=",
                        float(formulae_peaks_intensity[0].sum().detach().cpu().item()),
                        "coarse_render_int_sum_first=",
                        float(coarse_int_for_prob[0].sum().detach().cpu().item()),
                        "official_render_int_sum_first=",
                        float(official_int_for_prob[0].sum().detach().cpu().item()),
                        "spect_sum_first=",
                        float(spect_out[0].sum().detach().cpu().item()),
                        "official_spect_sum_first=",
                        float(spect_out_official[0].sum().detach().cpu().item()),
                    )
                self._printed_prob_peak_fusion_debug = True

            out = {
                'spect_out': spect_out,
                'spect_out_coarse': spect_out,
                'spect_out_official': spect_out_official,

                'formulae_probs': formulae_probs,
                'formulae_probs_render': formulae_probs_render,

                # In prob-spectrum mode, selector logits are the formula scores.
                'formulae_logits': selector_logits,
                'formulae_scores': selector_logits,
                'formulae_scores_raw': selector_logits,
                'formulae_scores_raw_setwise': formulae_scores_raw_setwise,
                'formulae_scores_train': selector_logits,

                'selector_logits': selector_logits,
                'selector_probs': selector_probs,
                'selector_pool_idx': pool_idx,

                # Keep rerank diagnostics, even if not used for probability spectrum.
                'rerank_delta_pool': rerank_delta_pool,
                'raw_rerank_delta_pool': raw_rerank_delta_pool.detach(),
                'rerank_logits_pool': final_pool_logits,
                'use_rerank_delta': torch.tensor(
                    1.0 if use_rerank_delta else 0.0,
                    device=device,
                    dtype=torch.float32,
                ),
                'rerank_delta_pool_mean': rerank_delta_pool_mean,
                'rerank_delta_pool_std': rerank_delta_pool_std,
                'selector_logits_pool_std': selector_logits_pool_std,
                'base_score_raw': selector_logits,

                'formulae_encoded': formulae_encoded,
                'vert_att_reduce': vert_att_reduce,
                'cond_embed': cond_embed,

                'pred_exact_peaks': pred_exact_peaks,

                'oos_logit': oos_logit,
                'oos_prob': torch.sigmoid(oos_logit),

                'candidate_peak_feat': candidate_peak_feat,
                'candidate_peak_feat_raw': candidate_peak_feat_raw,

                'peak_reweight_logits': peak_reweight_logits,
                'peak_reweight_probs': peak_reweight_probs,
                'refined_peak_intensity': refined_peak_intensity,
                'refined_peak_intensity_official': refined_peak_intensity,
                'peak_idx_for_head': peak_idx_for_head,
                'peak_int_for_head': peak_int_for_head,
                'spect_out_official_formula': spect_out_official_formula,

                'precursor_logit': precursor_logit,
                'precursor_prob': precursor_prob,

                'fragment_node_logits': fragment_node_logits,
                'fn_based_formula_logits': fn_based_formula_logits,
            }

            if torch.is_tensor(frag_has_source):
                out['frag_aux_has_source_ratio'] = frag_has_source.mean().detach()
            else:
                out['frag_aux_has_source_ratio'] = torch.tensor(
                    0.0,
                    device=device,
                    dtype=torch.float32,
                )

            out['frag_aux_scale'] = torch.tensor(
                float(self.fragment_local_aux_scale),
                device=device,
                dtype=torch.float32,
            )

            return out
        # New: logit sharpening controls.
        try:
            score_temperature = float(os.environ.get('FORMULA_SCORE_TEMPERATURE', '1.0'))
        except Exception:
            score_temperature = 1.0
        if (not np.isfinite(score_temperature)) or score_temperature <= 0:
            score_temperature = 1.0

        try:
            score_scale = float(os.environ.get('FORMULA_SCORE_SCALE', '1.0'))
        except Exception:
            score_scale = 1.0
        if (not np.isfinite(score_scale)) or score_scale <= 0:
            score_scale = 1.0

        formulae_scores_train = formulae_scores_raw * float(score_scale)
        formulae_scores_train = formulae_scores_train.masked_fill(
            invalid_formula,
            torch.finfo(formulae_scores_raw.dtype).min,
        )
        formulae_scores_train = formulae_scores_train / float(score_temperature)

        # Re-mask invalid candidates after adding bias.
        formulae_scores_train = formulae_scores_train.masked_fill(
            invalid_formula,
            torch.finfo(formulae_scores_raw.dtype).min,
        )

        if self.prob_softmax:
            formulae_scores = formulae_scores_train
            formulae_probs = torch.softmax(formulae_scores_train, dim=-1)
        else:
            # Legacy sigmoid path keeps raw-score interpretation.
            formulae_scores = formulae_scores_raw * formulae_mask
            formulae_probs = torch.sigmoid(formulae_scores)

        formulae_probs = formulae_probs * formulae_mask

        use_refined_peak_render = os.environ.get('USE_REFINED_PEAK_RENDER', '0') == '1'

        # For pure teacher KL / scorer diagnostics, avoid untrained peak head in render.
        try:
            peak_aux_w_env = float(os.environ.get('PEAK_AUX_LOSS_WEIGHT', '0.0'))
        except Exception:
            peak_aux_w_env = 0.2
        if peak_aux_w_env <= 0.0:
            use_refined_peak_render = False

        spect_bin_n = self.spect_bin.get_num_bins()

        coarse_int_for_render = formulae_peaks_intensity
        if (
            use_refined_peak_render
            and torch.is_tensor(refined_peak_intensity)
            and torch.is_tensor(formulae_peaks_intensity)
            and refined_peak_intensity.shape == formulae_peaks_intensity.shape
        ):
            coarse_int_for_render = refined_peak_intensity

        try:
            peak_render_topk = int(os.environ.get('PEAK_RENDER_TOPK', '0'))
        except Exception:
            peak_render_topk = 0
        if peak_render_topk > 0 and torch.is_tensor(coarse_int_for_render):
            if coarse_int_for_render.shape[-1] > peak_render_topk:
                _, topk_idx = torch.topk(coarse_int_for_render, k=peak_render_topk, dim=-1)
                mask = coarse_int_for_render.new_zeros(coarse_int_for_render.shape).scatter_(-1, topk_idx, 1.0)
                coarse_int_for_render = coarse_int_for_render * mask

        # ------------------------------------------------------------
        # V2B: group-aware render probabilities.
        # ------------------------------------------------------------
        if os.environ.get("USE_GROUP_AWARE_RENDER_PROBS", "0") == "1":
            group_id = self._get_group_ids_for_candidates(
                kwargs,
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
            )

            formulae_probs_group_render = self._groupmax_softmax_to_instance_probs(
                formulae_scores,
                formulae_mask,
                group_id,
            )

            if torch.is_tensor(formulae_probs_group_render):
                formulae_probs_render_base = formulae_probs_group_render
            else:
                formulae_probs_render_base = formulae_probs
        else:
            formulae_probs_render_base = formulae_probs

        # Critical: always apply render sparsification after choosing
        # the base probability distribution. This keeps eval-time
        # FORMULA_RENDER_TOPK behavior consistent for both normal and
        # group-aware render probabilities.
        formulae_probs_render = self._sparsify_formula_probs_for_render(
            formulae_probs_render_base,
            formulae_mask=formulae_mask,
        )

        if os.environ.get("DEBUG_RENDER_TOPK", "0") == "1":
            try:
                max_print = int(os.environ.get("DEBUG_RENDER_TOPK_MAX_PRINT", "20"))
            except Exception:
                max_print = 20
            cnt = int(getattr(self, "_debug_render_topk_count", 0))
            if cnt < max_print:
                try:
                    render_n = (formulae_probs_render > 1e-12).float().sum(dim=-1).mean()
                    base_n = (formulae_probs_render_base > 1e-12).float().sum(dim=-1).mean()
                    print(
                        f"[DEBUG_RENDER_TOPK] training={self.training} "
                        f"render_topk={int(self.render_topk_formula)} "
                        f"render_topk_train={int(bool(self.render_topk_train))} "
                        f"base_n_mean={float(base_n.detach().cpu().item()):.2f} "
                        f"render_n_mean={float(render_n.detach().cpu().item()):.2f}",
                        flush=True,
                    )
                except Exception as e:
                    print(f"[DEBUG_RENDER_TOPK] failed: {type(e).__name__}: {e}", flush=True)
                setattr(self, "_debug_render_topk_count", cnt + 1)

        spect_out = project_formula_probs_to_spectrum_dense(
            formulae_probs_render,
            formulae_peaks_mass_idx,
            coarse_int_for_render,
            spect_bin_n,
            formulae_mask=formulae_mask,
            mass_shift_probs=None,
            mass_shift_offsets=None,
        )

        official_idx_for_render = self._coerce_peak_tensor(
            kwargs.get('formulae_peaks_official_idx', None),
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.long,
            fill_value=-1,
        )
        official_int_template = self._coerce_peak_tensor(
            kwargs.get('formulae_peaks_official_intensity', None),
            batch_n=batch_n,
            formula_n=formula_n,
            device=device,
            dtype=torch.float32,
            fill_value=0.0,
        )

        if (
            official_idx_for_render is None
            or official_int_template is None
            or int(official_idx_for_render.shape[0]) != int(official_int_template.shape[0])
            or int(official_idx_for_render.shape[1]) != int(official_int_template.shape[1])
            or int(official_idx_for_render.shape[2]) != int(official_int_template.shape[2])
        ):
            official_idx_for_render = self._coerce_peak_tensor(
                kwargs.get('formulae_peaks_official_idx_agg', None),
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.long,
                fill_value=-1,
            )
            official_int_template = self._coerce_peak_tensor(
                kwargs.get('formulae_peaks_official_intensity_agg', None),
                batch_n=batch_n,
                formula_n=formula_n,
                device=device,
                dtype=torch.float32,
                fill_value=0.0,
            )

        if (
            official_idx_for_render is None
            or official_int_template is None
            or int(official_idx_for_render.shape[0]) != int(official_int_template.shape[0])
            or int(official_idx_for_render.shape[1]) != int(official_int_template.shape[1])
            or int(official_idx_for_render.shape[2]) != int(official_int_template.shape[2])
        ):
            official_idx_for_render = formulae_peaks_mass_idx
            official_int_template = formulae_peaks_intensity

        official_int_for_render = official_int_template
        if (
            use_refined_peak_render
            and torch.is_tensor(refined_peak_intensity)
            and torch.is_tensor(official_int_template)
            and refined_peak_intensity.shape == official_int_template.shape
        ):
            official_int_for_render = refined_peak_intensity

        if peak_render_topk > 0 and torch.is_tensor(official_int_for_render):
            if official_int_for_render.shape[-1] > peak_render_topk:
                _, topk_idx = torch.topk(official_int_for_render, k=peak_render_topk, dim=-1)
                mask = official_int_for_render.new_zeros(official_int_for_render.shape).scatter_(-1, topk_idx, 1.0)
                official_int_for_render = official_int_for_render * mask

        spect_out_official = project_formula_probs_to_spectrum_dense(
            formulae_probs_render,
            official_idx_for_render,
            official_int_for_render,
            official_bin_n,
            formulae_mask=formulae_mask,
            mass_shift_probs=None,
            mass_shift_offsets=None,
        )

        spect_out_official_formula = spect_out_official

        if os.environ.get('OFFICIAL_EXCLUDE_PRECURSOR', '1') == '1':
            spect_out_official = self._zero_precursor_bin_dense(
                spect_out_official,
                kwargs.get('precursor_mz', None),
                official_bin_width,
            )

        pred_exact_peaks = None
        if formulae_peaks_exact is not None:
            exact_topk = int(self.pred_exact_topk)
            exact_min_prob = float(self.pred_exact_min_prob)

            env_exact_topk = os.environ.get('PRED_EXACT_TOPK_FORMULA', '').strip()
            env_exact_min_prob = os.environ.get('PRED_EXACT_MIN_FORMULA_PROB', '').strip()
            if env_exact_topk:
                try:
                    exact_topk = max(0, int(env_exact_topk))
                except Exception:
                    pass
            if env_exact_min_prob:
                try:
                    exact_min_prob = max(0.0, float(env_exact_min_prob))
                except Exception:
                    pass

            formulae_peaks_for_render = formulae_peaks_exact
            if (
                use_refined_peak_render
                and torch.is_tensor(refined_peak_intensity)
                and int(formulae_peaks_exact.shape[0]) == int(refined_peak_intensity.shape[0])
                and int(formulae_peaks_exact.shape[1]) == int(refined_peak_intensity.shape[1])
                and int(formulae_peaks_exact.shape[2]) == int(refined_peak_intensity.shape[2])
            ):
                formulae_peaks_for_render = formulae_peaks_exact.clone()
                formulae_peaks_for_render[..., 1] = refined_peak_intensity

            pred_exact_peaks = project_formula_probs_to_exact_sparse(
                formulae_probs=formulae_probs_render,
                formulae_peaks=formulae_peaks_for_render,
                formulae_mask=formulae_mask,
                min_formula_prob=exact_min_prob,
                topk_formula=exact_topk,
                ranking_scores=None,
            )

        if self.normalize_1_output:
            spect_out = spect_out / (torch.sum(torch.abs(spect_out), dim=1, keepdim=True) + 1e-6)
            spect_out_official = spect_out_official / (torch.sum(torch.abs(spect_out_official), dim=1, keepdim=True) + 1e-6)
        refined_peak_intensity_official = refined_peak_intensity
        out = {
            'spect_out': spect_out,
            'spect_out_coarse': spect_out,
            'spect_out_official': spect_out_official,
            'formulae_probs': formulae_probs,
            'formulae_probs_render': formulae_probs_render,
            'formulae_logits': formulae_scores,
            'formulae_scores': formulae_scores,
            'formulae_scores_raw': formulae_scores_raw,
            'formulae_scores_raw_setwise': formulae_scores_raw_setwise,
            'formulae_scores_train': formulae_scores_train,
            'selector_logits': selector_logits,
            'selector_probs': selector_probs,
            'selector_pool_idx': pool_idx,
            'rerank_delta_pool': rerank_delta_pool,
            'raw_rerank_delta_pool': raw_rerank_delta_pool.detach(),
            'rerank_logits_pool': final_pool_logits,
            'use_rerank_delta': torch.tensor(
                1.0 if use_rerank_delta else 0.0,
                device=device,
                dtype=torch.float32,
            ),
            'rerank_delta_pool_mean': rerank_delta_pool_mean,
            'rerank_delta_pool_std': rerank_delta_pool_std,
            'selector_logits_pool_std': selector_logits_pool_std,
            'base_score_raw': base_score_raw,
            'formulae_encoded': formulae_encoded,
            'vert_att_reduce': vert_att_reduce,
            'cond_embed': cond_embed,
            'pred_exact_peaks': pred_exact_peaks,
            'oos_logit': oos_logit,
            'oos_prob': torch.sigmoid(oos_logit),
            'candidate_peak_feat': candidate_peak_feat,
            'candidate_peak_feat_raw': candidate_peak_feat_raw,
            'peak_reweight_logits': peak_reweight_logits,
            'peak_reweight_probs': peak_reweight_probs,
            'refined_peak_intensity': refined_peak_intensity,
            'refined_peak_intensity_official': refined_peak_intensity_official,
            'peak_idx_for_head': peak_idx_for_head,
            'peak_int_for_head': peak_int_for_head,
            'spect_out_official_formula': spect_out_official_formula,
            'precursor_logit': precursor_logit,
            'precursor_prob': precursor_prob,
            'fragment_node_logits': fragment_node_logits,
            'fn_based_formula_logits': fn_based_formula_logits,
        }
        if torch.is_tensor(frag_has_source):
            out['frag_aux_has_source_ratio'] = frag_has_source.mean().detach()
        else:
            out['frag_aux_has_source_ratio'] = torch.tensor(0.0, device=device, dtype=torch.float32)
        out['frag_aux_scale'] = torch.tensor(
            float(self.fragment_local_aux_scale),
            device=device,
            dtype=torch.float32,
        )
        return out


__all__ = [
    'MolAttentionGRUNewSparse',
]
