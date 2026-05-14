import os

import torch.nn as nn

from .model_utils import TARGET_LEAKAGE_KEYS as _TARGET_LEAKAGE_KEYS
from .nets import *
from .sparse_formula_scorer import MolAttentionGRUNewSparse


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
        allow_target_alignment = os.environ.get("MODEL_ALLOW_TARGET_ALIGNMENT_FEAT", "0") == "1"
        strict_kwarg_whitelist = os.environ.get("STRICT_MODEL_KWARG_WHITELIST", "1") == "1"

        safe_kwargs = {}
        for kk, vv in kwargs.items():
            if (not allow_target_alignment) and kk in _TARGET_LEAKAGE_KEYS:
                if os.environ.get("DEBUG_TARGET_LEAKAGE_CHECK", "0") == "1":
                    print(f"[TARGET_LEAKAGE_BLOCKED] key={kk}", flush=True)
                continue
            safe_kwargs[kk] = vv

        if strict_kwarg_whitelist:
            allowed_model_keys = {
                'ce',
                'adduct',
                'instrument_type',
                'ms_level',
                'precursor_mz',
                'ce_missing',
                'adduct_missing',
                'instrument_missing',
                'ms_level_missing',
                'precursor_mz_missing',
                'formulae_mask',
                'formulae_aux_feat',
                'formulae_frag_aux_feat',
                'formulae_active_mask',
                'formulae_prior_score',
                'formulae_source_flag',
                'formulae_break_depth',
                'formulae_ring_cut_flag',
                'formulae_peaks_official_idx',
                'formulae_peaks_official_intensity',
                'formulae_peaks_official_idx_agg',
                'formulae_peaks_official_intensity_agg',
                'formulae_instance_is_source',
                'formulae_instance_group_id',
                'formulae_instance_depth',
                'formulae_instance_h_shift',
                'fragment_node_mz',
                'fragment_node_intensity',
                'fragment_node_local_feat',
                'fragment_node_depth',
                'fragment_node_h_shift',
                'fragment_node_is_brics',
                'fragment_node_ring_cut',
                'fragment_node_atom_count',
                'fragment_node_cut_count',
                'fragment_node_source_type',
                'fragment_node_mask',
                'fragment_node_formula',
                'fragment_node_official_idx',
                'fragment_node_group_formula_id',
                'fragment_node_n_valid',
                'fragment_node_bin_dup_count',
                'formulae_peaks',
                'formula_topk_orig_idx',
                'selector_only_forward',
            }

            if allow_target_alignment:
                allowed_model_keys = allowed_model_keys | set(_TARGET_LEAKAGE_KEYS)

            safe_kwargs = {kk: vv for kk, vv in safe_kwargs.items() if kk in allowed_model_keys}
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
            **safe_kwargs,
        )

        pred_dense_spect = pred_dense_spect_dict['spect_out']
