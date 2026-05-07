# File overview: Simplified mainline model definitions for current training pipeline.
# Purpose: Keep only core formula scorer and projection APIs on the main path.

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .int_embedder import get_embedder
from .nets import *
from .formulaenets_legacy import (
    StructuredOneHot,
    project_formula_probs_to_spectrum_dense,
    project_formula_probs_to_exact_sparse,
)

def _neg_mask_fill_value(x):
    if torch.is_tensor(x) and torch.is_floating_point(x):
        if x.dtype in (torch.float16, torch.bfloat16):
            return -1e4
        try:
            return float(torch.finfo(x.dtype).min)
        except Exception:
            return -1e9
    return -1e9

# Class overview: GraphVertSpect wraps graph encoder and formula-spectrum head.
class GraphVertSpect(nn.Module):
    def __init__(
        self,
        g_feature_n,
        spect_bin,
        g_feature_out_n=None,
        int_d=None,
        layer_n=None,
        resnet=True,
        gml_class='GraphMatLayers',
        gml_config={},
        init_noise=1e-5,
        init_bias=0.0,
        agg_func=None,
        GS=1,
        spect_out_class='',
        spect_out_config={},
        spect_mode='dense',
        input_norm='batch',
        inner_norm=None,
        default_render_width=0.1,
        default_mass_max=512,
    ):
        super(GraphVertSpect, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        self.gml = eval(gml_class)(
            g_feature_n,
            g_feature_out_n,
            resnet=resnet,
            noise=init_noise,
            agg_func=parse_agg_func(agg_func),
            norm=inner_norm,
            GS=GS,
            **gml_config,
        )

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.spect_out = eval(spect_out_class)(
            g_feat_in=g_feature_out_n[-1],
            spect_bin=spect_bin,
            **spect_out_config,
        )

        self.spect_mode = spect_mode
        self.default_render_width = default_render_width
        self.default_mass_max = default_mass_max
        self.pos = 0

    def forward(
        self,
        adj,
        vect_feat,
        input_mask,
        input_idx=None,
        adj_oh=None,
        return_g_features=False,
        mol_feat=None,
        formulae_features=None,
        formulae_peaks_mass_idx=None,
        formulae_peaks_intensity=None,
        vert_element_oh=None,
        formula_frag_count=None,
        **kwargs,
    ):
        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, vect_feat, input_mask)

        g_features = self.gml(adj, vect_feat, input_mask)
        if return_g_features:
            return g_features

        g_squeeze = g_features.squeeze(1)
        pred_dense_spect_dict = self.spect_out(
            g_squeeze,
            input_mask,
            formulae_features,
            formulae_peaks_mass_idx,
            formulae_peaks_intensity,
            mol_feat=mol_feat,
            vert_element_oh=vert_element_oh,
            adj_oh=adj_oh,
            formula_frag_count=formula_frag_count,
            **kwargs,
        )

        pred_dense_spect = pred_dense_spect_dict['spect_out']
        out = {'spect': pred_dense_spect, 'masses': None, 'probs': None}
        for k, v in pred_dense_spect_dict.items():
            if k != 'spect_out':
                out[k] = v
        self.pos += 1
        return out


# Class overview: MolAttentionGRUNewSparse is the simplified single-path formula scorer.
class MolAttentionGRUNewSparse(nn.Module):
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

        self.selector_head = nn.Sequential(
            nn.LayerNorm(int(internal_d)),
            nn.Linear(int(internal_d), int(internal_d)),
            nn.ReLU(),
            nn.Linear(int(internal_d), 1),
        )
        
        # 新增：set-wise scorer
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

        # -------- precursor head：也必须放在 use_msms_conditioning 外面 --------
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
    def _build_peak_features(
            self,
            formulae_peaks_mass_idx,
            formulae_peaks_intensity,
            precursor_mz,
            batch_n,
            formula_n,
            peak_n,
            device,
            official_bin_width,
        ):
            mz = (formulae_peaks_mass_idx.float() + 0.5) * float(official_bin_width)
            inten = formulae_peaks_intensity.float()

            if precursor_mz is None:
                precursor = torch.ones((batch_n, 1, 1), dtype=torch.float32, device=device)
            else:
                precursor = precursor_mz
                if not torch.is_tensor(precursor):
                    precursor = torch.as_tensor(precursor)
                precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
                if precursor.shape[0] < batch_n:
                    pad = torch.ones((batch_n - precursor.shape[0],), dtype=torch.float32, device=device)
                    precursor = torch.cat([precursor, pad], dim=0)
                precursor = precursor[:batch_n].view(batch_n, 1, 1)

            mz_rel = mz / precursor.clamp_min(1e-6)
            gap_rel = (precursor - mz).clamp_min(0.0) / precursor.clamp_min(1e-6)
            inten_log = torch.log1p(inten.clamp_min(0.0))
            inten_raw = inten
            mz_raw = mz / 1500.0
            valid = ((formulae_peaks_mass_idx >= 0) & torch.isfinite(inten) & (inten > 0)).float()

            peak_feat = torch.stack(
                [
                    mz_raw,
                    mz_rel,
                    gap_rel,
                    inten_raw,
                    inten_log,
                    valid,
                ],
                dim=-1,
            )
            return peak_feat

    def _zero_precursor_bin_dense(self, dense_spect, precursor_mz, bin_width):
        if (not torch.is_tensor(dense_spect)) or precursor_mz is None:
            return dense_spect

        out = dense_spect.clone()
        if not torch.is_tensor(precursor_mz):
            precursor_mz = torch.as_tensor(precursor_mz)

        pmz = precursor_mz.to(device=out.device, dtype=torch.float32).reshape(-1)
        if pmz.shape[0] < int(out.shape[0]):
            pad = torch.zeros((int(out.shape[0]) - int(pmz.shape[0]),), dtype=torch.float32, device=out.device)
            pmz = torch.cat([pmz, pad], dim=0)
        pmz = pmz[: int(out.shape[0])]

        bin_idx = torch.floor(pmz / float(bin_width) + 1e-8).long()
        valid = torch.isfinite(pmz) & (bin_idx >= 0) & (bin_idx < int(out.shape[1]))
        if bool(valid.any().item()):
            row_idx = torch.arange(int(out.shape[0]), device=out.device)
            out[row_idx[valid], bin_idx[valid]] = 0.0
        return out

    def _to_2d_float(self, x, batch_n, device):
        if x is None:
            return torch.zeros((batch_n, 1), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device)
        if x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() == 1:
            x = x.unsqueeze(-1)
        else:
            x = x.view(x.shape[0], -1)[:, :1]
        if x.shape[0] != batch_n:
            flat = x.reshape(-1)
            if flat.shape[0] < batch_n:
                pad_n = batch_n - flat.shape[0]
                flat = torch.cat([flat, torch.zeros((pad_n,), dtype=flat.dtype, device=device)], dim=0)
            x = flat[:batch_n].view(batch_n, 1)
        return x.float()

    def _to_vocab_index(self, x, batch_n, device, vocab_size):
        if x is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device)
        if x.dim() == 0:
            x = x.view(1)
        elif x.dim() > 1:
            x = torch.argmax(x, dim=-1)
        x = x.long().reshape(-1)
        if x.shape[0] < batch_n:
            x = torch.cat([x, torch.zeros((batch_n - x.shape[0],), dtype=torch.long, device=device)], dim=0)
        x = x[:batch_n]
        return x % max(1, int(vocab_size))

    def _to_formulae_mask(self, x, batch_n, formula_n, device):
        if x is None:
            return torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float32)
        target = batch_n * formula_n
        flat = x.reshape(-1)
        if flat.shape[0] < target:
            flat = torch.cat([flat, torch.ones((target - flat.shape[0],), dtype=torch.float32, device=device)], dim=0)
        mask = flat[:target].view(batch_n, formula_n)
        mask = (mask > 0.5).float()
        has_valid = (mask.sum(dim=1, keepdim=True) > 0)
        if not torch.all(has_valid):
            mask = torch.where(has_valid, mask, torch.ones_like(mask))
        return mask
    def _coerce_peak_tensor(self, x, batch_n, formula_n, device, dtype, fill_value):
        """
        Convert list / numpy / tensor -> [B, M, K] tensor.
        Return None if conversion fails.
        """
        if x is None:
            return None

        try:
            if torch.is_tensor(x):
                t = x.to(device=device, dtype=dtype)
            else:
                arr = np.asarray(x)
                t = torch.as_tensor(arr, device=device, dtype=dtype)
        except Exception:
            return None

        if t.dim() == 1:
            return None
        elif t.dim() == 2:
            # t = t.unsqueeze(0)
            # 改为: 记录 warning 并创建一个 3D 零张量，避免错误广播
            print(f"[WARNING] _coerce_peak_tensor received 2D input ({t.shape}), filling zeros.")
            t = torch.zeros(batch_n, formula_n, t.shape[-1], dtype=dtype, device=device)
        elif t.dim() > 3:
            t = t.reshape(t.shape[0], t.shape[1], -1)

        if t.dim() != 3:
            return None

        cur_b = int(t.shape[0])
        cur_m = int(t.shape[1])
        cur_k = int(t.shape[2])

        if cur_b < batch_n:
            pad = torch.full(
                (batch_n - cur_b, cur_m, cur_k),
                fill_value=fill_value,
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=0)
        t = t[:batch_n]

        if int(t.shape[1]) < formula_n:
            pad = torch.full(
                (batch_n, formula_n - int(t.shape[1]), int(t.shape[2])),
                fill_value=fill_value,
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=1)
        t = t[:, :formula_n, :]

        return t
    def _coerce_exact_peaks_tensor(self, x, batch_n, formula_n, device):
        """
        Convert list / numpy / tensor -> [B, M, K, 2] float tensor.
        Return None if conversion fails.
        """
        if x is None:
            return None

        try:
            if torch.is_tensor(x):
                t = x.to(device=device, dtype=torch.float32)
            else:
                arr = np.asarray(x)
                t = torch.as_tensor(arr, device=device, dtype=torch.float32)
        except Exception:
            return None

        if t.dim() == 3 and int(t.shape[-1]) >= 2:
            t = t.unsqueeze(0)
        elif t.dim() > 4:
            try:
                t = t.reshape(t.shape[0], t.shape[1], -1, t.shape[-1])
            except Exception:
                return None

        if t.dim() != 4 or int(t.shape[-1]) < 2:
            return None

        cur_b = int(t.shape[0])
        cur_m = int(t.shape[1])
        cur_k = int(t.shape[2])

        if cur_b < batch_n:
            pad = torch.zeros(
                (batch_n - cur_b, cur_m, cur_k, int(t.shape[-1])),
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=0)
        t = t[:batch_n]

        if int(t.shape[1]) < formula_n:
            pad = torch.zeros(
                (batch_n, formula_n - int(t.shape[1]), int(t.shape[2]), int(t.shape[3])),
                dtype=t.dtype,
                device=device,
            )
            t = torch.cat([t, pad], dim=1)
        t = t[:, :formula_n, :, :2]

        return t
    def _build_candidate_only_peak_summary(
            self,
            batch_n,
            formula_n,
            device,
            formulae_mask,
            off_idx,
            off_int,
            precursor_mz,
            official_bin_width,
            official_bin_n,
        ):
            feat_dim = 8
            out = torch.zeros((batch_n, formula_n, feat_dim), dtype=torch.float32, device=device)

            if (off_idx is None) or (off_int is None):
                return out
            if (not torch.is_tensor(off_idx)) or (not torch.is_tensor(off_int)):
                return out

            off_idx = off_idx.to(device=device, dtype=torch.long)
            off_int = off_int.to(device=device, dtype=torch.float32)

            if off_idx.dim() == 2:
                off_idx = off_idx.unsqueeze(0)
            if off_int.dim() == 2:
                off_int = off_int.unsqueeze(0)

            use_b = min(batch_n, int(off_idx.shape[0]), int(off_int.shape[0]))
            use_m = min(formula_n, int(off_idx.shape[1]), int(off_int.shape[1]))
            use_k = min(int(off_idx.shape[2]), int(off_int.shape[2]))
            if use_b <= 0 or use_m <= 0 or use_k <= 0:
                return out

            off_idx = off_idx[:use_b, :use_m, :use_k]
            off_int = off_int[:use_b, :use_m, :use_k]

            valid = (
                (off_idx >= 0)
                & (off_idx < int(official_bin_n))
                & torch.isfinite(off_int)
                & (off_int > 0)
            )

            cand_peak_n = valid.float().sum(dim=-1)
            cand_int_sum = (off_int * valid.float()).sum(dim=-1)
            cand_int_max = torch.where(valid, off_int, torch.zeros_like(off_int)).max(dim=-1).values

            cand_prob = off_int * valid.float()
            cand_prob = cand_prob / cand_prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            cand_entropy = -(cand_prob * torch.log(cand_prob.clamp_min(1e-8))).sum(dim=-1)

            mz = (off_idx.float() + 0.5) * float(official_bin_width)
            mz = torch.where(valid, mz, torch.zeros_like(mz))

            mz_mean = mz.sum(dim=-1) / cand_peak_n.clamp_min(1.0)

            mz_centered = torch.where(valid, mz - mz_mean.unsqueeze(-1), torch.zeros_like(mz))
            mz_std = torch.sqrt((mz_centered ** 2).sum(dim=-1) / cand_peak_n.clamp_min(1.0))

            if precursor_mz is None:
                precursor = torch.zeros((use_b, 1), dtype=torch.float32, device=device)
            else:
                precursor = precursor_mz
                if not torch.is_tensor(precursor):
                    precursor = torch.as_tensor(precursor)
                precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
                if precursor.shape[0] < use_b:
                    pad = torch.zeros((use_b - precursor.shape[0],), dtype=torch.float32, device=device)
                    precursor = torch.cat([precursor, pad], dim=0)
                precursor = precursor[:use_b].view(use_b, 1)

            rel_gap = (precursor - mz_mean).clamp_min(0.0) / precursor.clamp_min(1e-6)
            rel_center = mz_mean / precursor.clamp_min(1e-6)

            top1_peak = torch.max(cand_prob, dim=-1).values

            feat = torch.stack(
                [
                    cand_peak_n / 32.0,
                    cand_int_sum,
                    cand_int_max,
                    cand_entropy / np.log(max(2, use_k)),
                    mz_mean / 1500.0,
                    mz_std / 300.0,
                    rel_center.squeeze(-1),
                    rel_gap.squeeze(-1),
                ],
                dim=-1,
            )

            if torch.is_tensor(formulae_mask):
                feat = feat * formulae_mask[:use_b, :use_m].unsqueeze(-1)

            out[:use_b, :use_m] = feat
            return out

    def _build_subset_peak_summary(
            self,
            subset_peaks_mass_idx,
            subset_peaks_intensity,
            precursor_mz,
            batch_n,
            subset_n,
            device,
            official_bin_width,
        ):
            feat_dim = 8
            out = torch.zeros((batch_n, subset_n, feat_dim), dtype=torch.float32, device=device)

            subset_peaks_mass_idx = self._coerce_peak_tensor(
                subset_peaks_mass_idx,
                batch_n=batch_n,
                formula_n=subset_n,
                device=device,
                dtype=torch.long,
                fill_value=-1,
            )
            subset_peaks_intensity = self._coerce_peak_tensor(
                subset_peaks_intensity,
                batch_n=batch_n,
                formula_n=subset_n,
                device=device,
                dtype=torch.float32,
                fill_value=0.0,
            )

            if subset_peaks_mass_idx is None or subset_peaks_intensity is None:
                return out

            use_b = min(batch_n, int(subset_peaks_mass_idx.shape[0]), int(subset_peaks_intensity.shape[0]))
            use_s = min(subset_n, int(subset_peaks_mass_idx.shape[1]), int(subset_peaks_intensity.shape[1]))
            use_k = min(int(subset_peaks_mass_idx.shape[2]), int(subset_peaks_intensity.shape[2]))
            if use_b <= 0 or use_s <= 0 or use_k <= 0:
                return out

            subset_peaks_mass_idx = subset_peaks_mass_idx[:use_b, :use_s, :use_k]
            subset_peaks_intensity = subset_peaks_intensity[:use_b, :use_s, :use_k]

            valid = (
                (subset_peaks_mass_idx >= 0)
                & torch.isfinite(subset_peaks_intensity)
                & (subset_peaks_intensity > 0)
            )

            peak_n = valid.float().sum(dim=-1)
            int_sum = (subset_peaks_intensity * valid.float()).sum(dim=-1)
            int_max = torch.where(valid, subset_peaks_intensity, torch.zeros_like(subset_peaks_intensity)).max(dim=-1).values

            prob = subset_peaks_intensity * valid.float()
            prob = prob / prob.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(dim=-1)

            mz = (subset_peaks_mass_idx.float() + 0.5) * float(official_bin_width)
            mz = torch.where(valid, mz, torch.zeros_like(mz))
            mz_mean = mz.sum(dim=-1) / peak_n.clamp_min(1.0)

            mz_centered = torch.where(valid, mz - mz_mean.unsqueeze(-1), torch.zeros_like(mz))
            mz_std = torch.sqrt((mz_centered ** 2).sum(dim=-1) / peak_n.clamp_min(1.0))

            if precursor_mz is None:
                precursor = torch.zeros((use_b, 1), dtype=torch.float32, device=device)
            else:
                precursor = precursor_mz
                if not torch.is_tensor(precursor):
                    precursor = torch.as_tensor(precursor)
                precursor = precursor.to(device=device, dtype=torch.float32).reshape(-1)
                if precursor.shape[0] < use_b:
                    pad = torch.zeros((use_b - precursor.shape[0],), dtype=torch.float32, device=device)
                    precursor = torch.cat([precursor, pad], dim=0)
                precursor = precursor[:use_b].view(use_b, 1)

            rel_gap = (precursor - mz_mean).clamp_min(0.0) / precursor.clamp_min(1e-6)
            rel_center = mz_mean / precursor.clamp_min(1e-6)

            feat = torch.stack(
                [
                    peak_n / 32.0,
                    int_sum,
                    int_max,
                    entropy / np.log(max(2, use_k)),
                    mz_mean / 1500.0,
                    mz_std / 300.0,
                    rel_center.squeeze(-1),
                    rel_gap.squeeze(-1),
                ],
                dim=-1,
            )
            out[:use_b, :use_s] = feat
            return out

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

        # 如果某一行被 min_prob/topk 弄成全 0，就回退到原始 masked p，避免空谱
        empty = denom <= 1e-8
        if torch.any(empty):
            p_fallback = p
            fallback_denom = p_fallback.sum(dim=1, keepdim=True).clamp_min(1e-8)
            p_fallback = p_fallback / fallback_denom
            p_sparse = torch.where(empty, p_fallback, p_sparse)
            denom = p_sparse.sum(dim=1, keepdim=True)

        p_sparse = p_sparse / denom.clamp_min(1e-8)
        return p_sparse

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
            # Missing columns should be unique-ish padding groups, not all group 0.
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

            # Critical: remap arbitrary original group ids to compact ids.
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

        

        frag_aux = kwargs.get('formulae_frag_aux_feat', None)
        frag_aux_t = None
        if self.frag_aux_proj is not None:
            if frag_aux is None:
                frag_aux_t = torch.zeros((batch_n, formula_n, self.fragment_local_aux_dim), dtype=torch.float32, device=device)
            else:
                if not torch.is_tensor(frag_aux):
                    frag_aux = torch.as_tensor(frag_aux)
                frag_aux_t = frag_aux.to(device=device, dtype=torch.float32)
                if frag_aux_t.dim() == 1:
                    frag_aux_t = frag_aux_t.view(1, 1, -1)
                elif frag_aux_t.dim() == 2:
                    frag_aux_t = frag_aux_t.unsqueeze(0)
                elif frag_aux_t.dim() > 3:
                    frag_aux_t = frag_aux_t.reshape(frag_aux_t.shape[0], frag_aux_t.shape[1], -1)

                if frag_aux_t.dim() != 3:
                    frag_aux_t = torch.zeros((batch_n, formula_n, self.fragment_local_aux_dim), dtype=torch.float32, device=device)
                else:
                    if int(frag_aux_t.shape[0]) < batch_n:
                        pad = torch.zeros((batch_n - int(frag_aux_t.shape[0]), int(frag_aux_t.shape[1]), int(frag_aux_t.shape[2])), dtype=frag_aux_t.dtype, device=device)
                        frag_aux_t = torch.cat([frag_aux_t, pad], dim=0)
                    frag_aux_t = frag_aux_t[:batch_n]

                    if int(frag_aux_t.shape[1]) < formula_n:
                        pad = torch.zeros((batch_n, formula_n - int(frag_aux_t.shape[1]), int(frag_aux_t.shape[2])), dtype=frag_aux_t.dtype, device=device)
                        frag_aux_t = torch.cat([frag_aux_t, pad], dim=1)
                    frag_aux_t = frag_aux_t[:, :formula_n, :]

                    if int(frag_aux_t.shape[2]) < self.fragment_local_aux_dim:
                        pad = torch.zeros((batch_n, formula_n, self.fragment_local_aux_dim - int(frag_aux_t.shape[2])), dtype=frag_aux_t.dtype, device=device)
                        frag_aux_t = torch.cat([frag_aux_t, pad], dim=-1)
                    frag_aux_t = frag_aux_t[:, :, :self.fragment_local_aux_dim]

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

        # ---- 后融合：fragment 辅助特征（保持原逻辑） ----
        if self.frag_aux_proj is not None and torch.is_tensor(frag_aux_t):
            base_h = base_h + self.frag_aux_proj(frag_aux_t)

        # debug diagnostics for explanation matching
        matched_subset_per_sample = torch.zeros((batch_n,), dtype=torch.float32, device=device)
        matched_formula_per_sample = torch.zeros((batch_n,), dtype=torch.float32, device=device)
        invalid_formula = formulae_mask <= 0
        selector_logits = self.selector_head(base_h).squeeze(-1)
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
        # V3C: Fragment-Node Selector 前向传播
        # ------------------------------------------------------------
        fragment_node_logits = None
        fn_based_formula_logits = None  # 🌟 必须在这里初始化

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

            # ======== 🌟 纯 PyTorch 实现的安全 Max-Pooling 桥梁 ========
            B = fragment_node_logits.shape[0]
            M = possible_formulae.shape[1]
            
            # 初始化为极小值，过滤掉没有碎片的 Fallback 公式
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
                        
                        # 巧妙利用 argsort 和 scatter_ 实现安全的 Max-Pooling
                        # 按分数升序排列，这样 scatter_ 最后写入的一定是该公式下所有碎片的“最高分”！
                        order = torch.argsort(l_b)
                        g_b_sorted = g_b[order]
                        l_b_sorted = l_b[order]
                        
                        fn_based_formula_logits[b].scatter_(0, g_b_sorted, l_b_sorted)
            # =========================================================

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
        # Fast PASS-1 selector path.
        # Used by train_ms_subsetnet.py for full-candidate selector only.
        # It avoids expensive peak head + spectrum projection over 8192 candidates.
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

        # -------- set-wise context scorer (v2b: loo + selector features) --------
        sum_h = (base_h * mask3).sum(dim=1, keepdim=True)
        cnt_h = mask3.sum(dim=1, keepdim=True)

        loo_ctx = (sum_h - base_h) / (cnt_h - 1.0).clamp_min(1.0)
        single_mask = (cnt_h <= 1.5).expand_as(base_h)
        loo_ctx = torch.where(single_mask, torch.zeros_like(base_h), loo_ctx)

        set_delta = base_h - loo_ctx
        set_prod = base_h * loo_ctx

        score_in = torch.cat(
            [base_h, loo_ctx, set_delta, set_prod, selector_logit_feat, selector_prob_feat],
            dim=-1,
        )

        score_in = self.set_score_norm(score_in)
        score_h = F.relu(self.set_score_l1(score_in))
        score_h = F.relu(self.set_score_l2(score_h))
        formulae_scores_raw = self.set_score_out(score_h).squeeze(-1)

        formulae_scores_raw_setwise = formulae_scores_raw


        base_score_raw = formulae_scores_raw

        # peak head must use non-aggregated official peaks to preserve the
        # original per-candidate peak order. Aggregated peaks remain available
        # for candidate-level summary features above, but not for peak reweighting.
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

        # 最后兜底：如果连 fallback 都失败，就构一个全空 peak tensor，避免直接崩
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

        # new: logit sharpening controls
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

        # 做纯 teacher KL / scorer 诊断时，默认不要让未训练的 peak head 改写渲染
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
        #
        # Source-instance candidates are useful for scoring,
        # but rendering must still obey FORMULA_RENDER_TOPK /
        # FORMULA_RENDER_MIN_PROB.  The previous logic returned
        # group-aware probabilities directly and therefore bypassed
        # _sparsify_formula_probs_for_render(), causing very large
        # predicted support sets when USE_GROUP_AWARE_RENDER_PROBS=1.
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
        # the base probability distribution.  This keeps eval-time
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
            'formulae_scores': formulae_scores,
            'formulae_scores_raw': formulae_scores_raw,
            'formulae_scores_raw_setwise': formulae_scores_raw_setwise,
            'formulae_scores_train': formulae_scores_train,
            'selector_logits': selector_logits,
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
            'fn_based_formula_logits': fn_based_formula_logits,  # 🌟 必须加这一行，让外面的训练脚本能拿到！
        }
        return out


__all__ = [
    'StructuredOneHot',
    'project_formula_probs_to_spectrum_dense',
    'project_formula_probs_to_exact_sparse',
    'GraphVertSpect',
    'MolAttentionGRUNewSparse',
]

