# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Model architecture and loss definitions used by the RASSP/MS prediction pipeline.

"""
Nets that have an explicit representation of the formulae
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time
import os

from .nets import *

# Function overview: create_mass_matrix_oh handles a specific workflow step in this module.
def create_mass_matrix_oh(masses, SPECT_BIN_N, mass_intensities=None):
    """
    Create a one-hot mass matrix from a list of formula masses

    input: 
    masses = BATCH_N x POSSIBLE_FORMULAE_N (integers)
    spect_bin_N : max number of spectral bins
    
    mass_intensities: array with same shape as masses, if None then we just use all 1s
    
    
    output:
    BATCH_N x POSSIBLE_FORMAULE_N x SPECT_BIN_N : 1-hot encoding of masses 
    """
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape

    #print(masses.shape, masses.dtype, mass_intensities.shape, mass_intensities.dtype)
    out = torch.zeros((BATCH_N, POSSIBLE_FORMULAE_N, SPECT_BIN_N)).to(masses.device)
    a = out.reshape(-1, SPECT_BIN_N)

    if mass_intensities is None:
        mass_intensities = torch.ones_like(masses).float().to(masses.device)
    
    b = masses.reshape(-1, 1)
    c = mass_intensities.reshape(-1, 1)
    a.scatter_(1, b,  c)

    return a.reshape(out.shape)

# Class overview: StructuredOneHot encapsulates a reusable component in this module.
class StructuredOneHot(nn.Module):
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, oh_sizes, cumulative=False):
        """
      
        """
        super(StructuredOneHot, self).__init__()
        self.oh_sizes = oh_sizes
        self.cumulative = cumulative 
        # Default: keep training logs clean. Set ONEHOT_WARN_OOB=1 to print one-time warnings.
        self.warn_oob = os.environ.get('ONEHOT_WARN_OOB', '0') == '1'
        self._warned_cols = set()
        
        if self.cumulative:
            n = np.sum(oh_sizes)
            accum_mat = np.zeros((n, n), dtype=np.int64)
            offset = 0
            for i, s in enumerate(oh_sizes):
                e = np.tril(np.ones((s, s)))
                accum_mat[offset:offset+s, offset:offset+s, ] = e
                offset += s

            a = torch.Tensor(accum_mat).long()
            self.register_buffer('accum_mat', a)
            
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, data):
        BATCH_N, OH_N = data.shape
        data = data.long()
        oh_list = []
        for i, gs in enumerate(self.oh_sizes):
            col = data[:, i]

            if self.warn_oob and (i not in self._warned_cols):
                with torch.no_grad():
                    oob = torch.logical_or(col < 0, col >= gs)
                    if bool(oob.any().item()):
                        minv = int(col.min().detach().item())
                        maxv = int(col.max().detach().item())
                        print(
                            f"StructuredOneHot: col {i} index out of range "
                            f"(min={minv}, max={maxv}, gs={gs}), clamping to [0,{gs-1}]"
                        )
                        self._warned_cols.add(i)

            col = col.clamp(0, gs - 1)
            oh_list.append(F.one_hot(col, gs))

        oh = torch.cat(oh_list, -1)
        
        if self.cumulative:
            return oh.float() @ self.accum_mat.float()
        else:
            return oh.float()

# Class overview: MolDotFormulaNet encapsulates a reusable component in this module.
class MolDotFormulaNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolDotFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.f_embed_l = nn.Linear(formula_encoding_n, g_feat_in)
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        embedded_formulae = self.f_embed_l(possible_formulae)

        formulae_scores = torch.einsum("ij,ikj->ik", mol_agg, embedded_formulae)
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


# Class overview: MolLinearFormulaNet encapsulates a reusable component in this module.
class MolLinearFormulaNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolLinearFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1)
        combined = torch.cat([possible_formulae, mol_agg_expand], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out



# =========================================================================
# 🏢 [框架外壳] GraphVertSpect (图节点质谱网络)
# 这个类是整个模型的大管家，包裹了图操作和预测头。工作流如下：
# 1. 拿到分子图中每个原子的初始特征 `vect_feat` (比如：这是个碳，连着几条键)。
# 2. 扔进 `self.gml` (GraphMatLayers) 跑图消息传递，让原子互相交流。
# 3. 把变聪明后的原子特征 `g_squeeze` 连同备选化学式、环境信息一起丢给 `self.spect_out` 预测头打分。
# =========================================================================
# Class overview: GraphVertSpect encapsulates a reusable component in this module.
class GraphVertSpect(nn.Module):
    """
    g_feature_n: starting number of input features
    g_feature_out_n: (override)the number of intermediate features
    int_d: number of intermediate features
    layer_n: number of layers to apply


    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self,
                 g_feature_n, spect_bin,
                 g_feature_out_n=None, 
                 int_d = None, layer_n = None, 

                resnet=True, 
                gml_class = 'GraphMatLayers',
                gml_config = {}, 
                init_noise=1e-5,
                init_bias=0.0,
                agg_func=None,
                GS=1,

                spect_out_class='',
                spect_out_config = {},
                spect_mode = 'dense', 
                input_norm='batch',
                
                inner_norm=None,
                default_render_width = 0.1,
                default_mass_max = 512, 
        ):
        
        """

        """
        super( GraphVertSpect, self).__init__()

        if layer_n is not None:
            g_feature_out_n = [int_d] * layer_n

        self.gml = eval(gml_class)(g_feature_n, g_feature_out_n, 
                                   resnet=resnet, noise=init_noise,
                                   agg_func=parse_agg_func(agg_func), 
                                   norm = inner_norm, 
                                   GS=GS,
                                   **gml_config)

        if input_norm == 'batch':
            self.input_norm = MaskedBatchNorm1d(g_feature_n)
        elif input_norm == 'layer':
            self.input_norm = MaskedLayerNorm1d(g_feature_n)
        else:
            self.input_norm = None

        self.spect_out = eval(spect_out_class)(g_feat_in = g_feature_out_n[-1],
                                               spect_bin = spect_bin,
                                               
                                               **spect_out_config)

        self.spect_mode = spect_mode
        self.default_render_width = default_render_width
        self.default_mass_max = default_mass_max
        
        self.pos = 0
        
    
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, adj, vect_feat, input_mask, input_idx=None, adj_oh=None,
                return_g_features = False, 
                mol_feat=None,
                formulae_features = None,
                formulae_peaks_mass_idx = None,
                formulae_peaks_intensity = None, 
                vert_element_oh = None,
                formula_frag_count = None,
                **kwargs):

        G = adj
        
        BATCH_N, MAX_N, F_N = vect_feat.shape

        if self.input_norm is not None:
            vect_feat = apply_masked_1d_norm(self.input_norm, 
                                             vect_feat, 
                                             input_mask)
        
        # we compute a global embedding of the parent molecule
        G_features = self.gml(G, vect_feat, input_mask)
        if return_g_features:
            return G_features

        g_squeeze = G_features.squeeze(1)

        
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

        #pred_masses, pred_probs = dense_to_sparse(pred_dense_spect)
        pred_masses = None
        pred_probs = None
            
        out = {'spect' : pred_dense_spect,
               'masses' : pred_masses,
               'probs' : pred_probs}
        for k, v in pred_dense_spect_dict.items():
            if k != 'spect_out':
                out[k] = v

        self.pos += 1
        return out


    

# Class overview: MolBilinearFormulaNet encapsulates a reusable component in this module.
class MolBilinearFormulaNet(nn.Module):
    """
    ERROR TOO MUCH MEMORY 
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolBilinearFormulaNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)


        self.combine_bilinear = nn.Bilinear(g_feat_in, formula_encoding_n, 512)
        self.combine_norm = nn.LayerNorm(512)
        self.f_combine_l1 = nn.Linear(512, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1).clone()

        combined = self.combine_bilinear(mol_agg_expand, possible_formulae)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out

    
# Class overview: MolLinearFormulaNet2 encapsulates a reusable component in this module.
class MolLinearFormulaNet2(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128, 
                 spect_bin_n = 512):

        
        super( MolLinearFormulaNet2, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)
        self.norm2 = nn.LayerNorm(512)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_X x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        
        if self.mol_agg == 'sum':
            mol_agg = torch.sum(masked_vert_feat, dim=1)
        elif self.mol_agg == 'mean':
            mol_agg = torch.mean(masked_vert_feat, dim=1)
        elif self.mol_agg == 'max':
            mol_agg = goodmax(masked_vert_feat, dim=1)
        mol_agg = self.norm(mol_agg)

        mol_agg_expand =  mol_agg.unsqueeze(1).expand(-1, possible_formulae.shape[1],-1)
        combined = torch.cat([possible_formulae, mol_agg_expand], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


# Class overview: MolAttentionNet encapsulates a reusable component in this module.
class MolAttentionNet(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 mol_agg = 'sum',
                 formula_encoding_n = 128,
                 embedding_key_size = 16,
                 spect_bin_n = 512):

        
        super( MolAttentionNet, self).__init__()

        self.mol_agg = mol_agg

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(512)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, 512)
        self.f_combine_l2 = nn.Linear(512, 512)
        self.f_combine_score = nn.Linear(512, 1)
        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N  integer masses

        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrix = create_mass_matrix_oh(formulae_masses, self.spect_bin_n).to(vert_feat_in.device)
        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


# Class overview: MolAttentionNetOH encapsulates a reusable component in this module.
class MolAttentionNetOH(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 spect_bin_n = 512):

        
        super( MolAttentionNetOH, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


# Class overview: MolAttentionNetOHMultiHead encapsulates a reusable component in this module.
class MolAttentionNetOHMultiHead(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 head_n = 1, 
                 internal_d = 512, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHMultiHead, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.vert_formula_attn = nn.ModuleList([VertFormulaAttn(g_feat_in, formula_encoding_n,
                                                                embedding_key_size) for _ in range(head_n)])
        

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in * head_n)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in*head_n, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.norm2 = nn.LayerNorm(internal_d)
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        # vert_attn_reduce = self.vert_formula_attn(masked_vert_feat,
        #                                           possible_formulae)
        vert_attn_reduce_list = [h(masked_vert_feat,possible_formulae) for h in self.vert_formula_attn]
        
        
        combined = torch.cat([possible_formulae] + vert_attn_reduce_list, -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out




# Class overview: VertFormulaAttn encapsulates a reusable component in this module.
class VertFormulaAttn(nn.Module):
    """
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 formula_encoding_n, 
                 embedding_key_size = 16):

        
        super( VertFormulaAttn, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, masked_vert_feat, 
                possible_formulae):
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(masked_vert_feat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        return vert_att_reduce


# Class overview: MolAttentionNetElementOH encapsulates a reusable component in this module.
class MolAttentionNetElementOH(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetElementOH, self).__init__()


        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.elt_type_lin = nn.Linear(g_feat_in, g_feat_in)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in + g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in + g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae_in, formulae_masses,
                vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        vert_Element_oh: BATCH_N x ATOM_N x possible_elements
        """

        BATCH_N, ATOM_N, V_F = vert_feat_in.shape
        _, _, ELT_N = vert_element_oh.shape
        _, POSSIBLE_FORMULAE_N, ELT_N_2 = possible_formulae_in.shape

        assert ELT_N == ELT_N_2
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae_in.reshape(-1, possible_formulae_in.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae_in.shape[0],
                                                              possible_formulae_in.shape[1], -1)
        
        
        ### reduce vertex features by atom type
        vertf_by_elt = torch.einsum('ijk,ijl->ilk', masked_vert_feat, vert_element_oh)
        #print(vertf_by_elt.shape,  (BATCH_N, ELT_N, V_F))
        assert vertf_by_elt.shape == (BATCH_N, ELT_N, V_F)
        vertf_by_elt_post = F.relu(self.elt_type_lin(vertf_by_elt))
        
        ## FIXME remember formula-in are not one-hot encoded: we should either norm the result or whatever
        elt_weighted_vert_f = torch.einsum("ijk,ilj->ilk", vertf_by_elt_post, possible_formulae_in.float())
        assert elt_weighted_vert_f.shape == (BATCH_N, POSSIBLE_FORMULAE_N, V_F)
        
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce, elt_weighted_vert_f], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out











# Class overview: MolAttentionNetMHA encapsulates a reusable component in this module.
class MolAttentionNetMHA(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 num_heads = 4, 
                 
                 spect_bin_n = 512):

        
        super( MolAttentionNetMHA, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)


        self.mha1 = nn.MultiheadAttention(embed_dim = embedding_key_size,
                                          num_heads = num_heads)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        

        
        # encoded inputs
        formulae_encoded = self.embed_formulae_feat(possible_formulae)
        

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_expanded = formulae_encoded.unsqueeze(1).permute(1, 0, 2) # .expand(-1, vert_encoded.shape[1],-1)
        vert_atom_first = vert_encoded.permute(1, 0, 2)
        attn_output, attn_weights = self.mha1(vert_atom_first,
                                              formulae_expanded, 
                                              vert_atom_first)
        attn_output = attn_output.permute(1, 0, 2)
        
        

        dot_prod = (formulae_encoded.unsqueeze(1) * attn_output.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


# Class overview: MolAttentionNetOHSparse encapsulates a reusable component in this module.
class MolAttentionNetOHSparse(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparse, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


# Function overview: create_mass_matrix_sparse handles a specific workflow step in this module.
def create_mass_matrix_sparse(masses, SPECT_BIN_N, mass_intensities):
    BATCH_N, POSSIBLE_FORMULAE_N = masses.shape

    row_idx_offset = torch.arange(BATCH_N, device=masses.device)\
                     .unsqueeze(1).repeat(1, POSSIBLE_FORMULAE_N).flatten()*SPECT_BIN_N
    row_idx_offset = row_idx_offset.to(masses.device)
    row_idx = row_idx_offset + masses.flatten()

    col_idx = torch.arange(BATCH_N *POSSIBLE_FORMULAE_N, device=masses.device ).to(masses.device)
    idx = torch.stack([row_idx, col_idx], -1).T
    
    mat_val_oh = mass_intensities.flatten()
    
    #mat_val_oh_nz = (mat_val_oh > 1e-4)
    
    sparse_mat = torch.sparse_coo_tensor(idx, mat_val_oh,
                                         #idx[:, mat_val_oh_nz], mat_val_oh[mat_val_oh_nz], 
                                         (SPECT_BIN_N * BATCH_N , 
                                          BATCH_N* POSSIBLE_FORMULAE_N, ), device=masses.device)
    
    return sparse_mat


# Function overview: mat_matrix_sparse_mm handles a specific workflow step in this module.
def mat_matrix_sparse_mm(sparse_matrix, formulae_probs):
    BATCH_N, _ = formulae_probs.shape
    f_flat =formulae_probs.flatten()
    #a = torch.mm(sparse_matrix, f_flat.unsqueeze(1))
    a = torch.sparse.mm(sparse_matrix, f_flat.unsqueeze(1))
    sparse_spect_out = a.reshape(BATCH_N, -1)
    return sparse_spect_out


# Function overview: project_formula_probs_to_spectrum_dense handles a specific workflow step in this module.
def project_formula_probs_to_spectrum_dense(formulae_probs, formulae_peaks_mass_idx,
                                            formulae_peaks_intensity, spect_bin_n,
                                            formulae_mask=None,
                                            mass_shift_probs=None,
                                            mass_shift_offsets=None):
    """
    Vectorized projection from per-formula probabilities to dense spectrum bins.

    formulae_probs: B x M
    formulae_peaks_mass_idx: B x M x K (long)
    formulae_peaks_intensity: B x M x K
    mass_shift_probs: optional B x M x S, per-candidate shift distribution
    mass_shift_offsets: optional S offsets (e.g., [-2,-1,0,1,2])
    return: B x spect_bin_n
    """
    # Core idea:
    # candidate score (formulae_probs) -> weighted candidate peaks -> scatter into output bins.
    # This keeps projection fully differentiable and GPU friendly.
    BATCH_N = formulae_probs.shape[0]

    if formulae_peaks_mass_idx.dtype != torch.long:
        mass_idx = formulae_peaks_mass_idx.long()
    else:
        mass_idx = formulae_peaks_mass_idx

    weighted_peaks = formulae_probs.float().unsqueeze(-1) * formulae_peaks_intensity.float()
    if formulae_mask is not None:
        weighted_peaks = weighted_peaks * formulae_mask.float().unsqueeze(-1)

    # Fast path: no learned mass-shift branch, directly aggregate candidate peaks.
    if mass_shift_probs is None or mass_shift_offsets is None:
        mass_idx_flat = mass_idx.reshape(BATCH_N, -1)
        weighted_flat = weighted_peaks.reshape(BATCH_N, -1)

        valid = (mass_idx_flat >= 0) & (mass_idx_flat < spect_bin_n)
        if not torch.all(valid):
            mass_idx_flat = mass_idx_flat.clamp(0, spect_bin_n - 1)
            weighted_flat = weighted_flat * valid.to(weighted_flat.dtype)

        spect_out = torch.zeros((BATCH_N, spect_bin_n), dtype=weighted_flat.dtype, device=formulae_probs.device)
        spect_out.scatter_add_(1, mass_idx_flat, weighted_flat)
        return spect_out

    # Shift-aware path: distribute each candidate's peak mass by learned offset weights.
    shift_probs = mass_shift_probs.float()
    if shift_probs.dim() != 3:
        # Fallback to baseline projection for malformed shift tensor.
        return project_formula_probs_to_spectrum_dense(
            formulae_probs,
            formulae_peaks_mass_idx,
            formulae_peaks_intensity,
            spect_bin_n,
            formulae_mask=formulae_mask,
        )

    if not torch.is_tensor(mass_shift_offsets):
        shift_offsets = torch.as_tensor(mass_shift_offsets, device=mass_idx.device, dtype=torch.long)
    else:
        shift_offsets = mass_shift_offsets.to(device=mass_idx.device, dtype=torch.long)

    if shift_offsets.dim() != 1:
        shift_offsets = shift_offsets.reshape(-1)

    shift_n = int(min(shift_probs.shape[-1], shift_offsets.shape[0]))
    if shift_n <= 0:
        return project_formula_probs_to_spectrum_dense(
            formulae_probs,
            formulae_peaks_mass_idx,
            formulae_peaks_intensity,
            spect_bin_n,
            formulae_mask=formulae_mask,
        )

    if shift_n != shift_probs.shape[-1]:
        shift_probs = shift_probs[..., :shift_n]
    if shift_n != shift_offsets.shape[0]:
        shift_offsets = shift_offsets[:shift_n]

    # Align formula dimension defensively in case caller provides stale shift logits.
    if shift_probs.shape[1] != mass_idx.shape[1]:
        min_formula_n = min(shift_probs.shape[1], mass_idx.shape[1])
        shift_probs = shift_probs[:, :min_formula_n, :]
        mass_idx = mass_idx[:, :min_formula_n, :]
        weighted_peaks = weighted_peaks[:, :min_formula_n, :]

    valid_base = (mass_idx >= 0) & (mass_idx < spect_bin_n)
    spect_out = torch.zeros((BATCH_N, spect_bin_n), dtype=weighted_peaks.dtype, device=formulae_probs.device)

    # For each discrete shift bucket, scatter-add shifted peak contributions.
    for si in range(shift_n):
        shifted_idx = mass_idx + int(shift_offsets[si].item())
        valid = valid_base & (shifted_idx >= 0) & (shifted_idx < spect_bin_n)
        shifted_idx = shifted_idx.clamp(0, spect_bin_n - 1)

        w = weighted_peaks * shift_probs[:, :, si].unsqueeze(-1)
        w = w * valid.to(w.dtype)

        spect_out.scatter_add_(1, shifted_idx.reshape(BATCH_N, -1), w.reshape(BATCH_N, -1))

    return spect_out


# Function overview: project_formula_probs_to_exact_sparse handles a specific workflow step in this module.
def project_formula_probs_to_exact_sparse(
    formulae_probs,
    formulae_peaks,
    formulae_mask=None,
    min_formula_prob=0.0,
    topk_formula=0,
    ranking_scores=None,
):
    """
    Project per-formula probabilities into exact sparse peaks (mz, intensity).

    formulae_probs: B x M
    formulae_peaks: B x M x K x 2, where [..., 0]=mz and [..., 1]=peak intensity
    formulae_mask: optional B x M
    return: list of length B, each item is N_i x 2 np.float32
    """
    if not torch.is_tensor(formulae_probs):
        formulae_probs = torch.as_tensor(formulae_probs)
    formulae_probs = formulae_probs.float()

    batch_n = int(formulae_probs.shape[0])
    out = []

    if formulae_peaks is None:
        for _ in range(batch_n):
            out.append(np.zeros((0, 2), dtype=np.float32))
        return out

    if not torch.is_tensor(formulae_peaks):
        formulae_peaks = torch.as_tensor(formulae_peaks)
    formulae_peaks = formulae_peaks.to(device=formulae_probs.device, dtype=torch.float32)

    if ranking_scores is not None:
        if not torch.is_tensor(ranking_scores):
            ranking_scores = torch.as_tensor(ranking_scores)
        ranking_scores = ranking_scores.to(device=formulae_probs.device, dtype=torch.float32)
        if ranking_scores.dim() == 1:
            ranking_scores = ranking_scores.unsqueeze(0)

    if formulae_peaks.dim() == 3 and formulae_peaks.shape[-1] == 2:
        formulae_peaks = formulae_peaks.unsqueeze(0)

    if formulae_peaks.dim() != 4 or formulae_peaks.shape[-1] != 2:
        for _ in range(batch_n):
            out.append(np.zeros((0, 2), dtype=np.float32))
        return out

    for b in range(batch_n):
        if b >= formulae_peaks.shape[0]:
            out.append(np.zeros((0, 2), dtype=np.float32))
            continue

        p = formulae_probs[b]
        peaks_b = formulae_peaks[b]
        formula_n = int(min(p.shape[0], peaks_b.shape[0]))
        if formula_n <= 0:
            out.append(np.zeros((0, 2), dtype=np.float32))
            continue

        p = p[:formula_n]
        peaks_b = peaks_b[:formula_n]
        rank_b = None
        if torch.is_tensor(ranking_scores) and b < ranking_scores.shape[0]:
            rank_b = ranking_scores[b].reshape(-1)
            if rank_b.shape[0] < formula_n:
                pad_n = formula_n - rank_b.shape[0]
                rank_b = torch.cat(
                    [
                        rank_b,
                        torch.full((pad_n,), -1e9, dtype=torch.float32, device=p.device),
                    ],
                    dim=0,
                )
            rank_b = rank_b[:formula_n]

        if formulae_mask is not None:
            if torch.is_tensor(formulae_mask):
                if b < formulae_mask.shape[0]:
                    mask_b = formulae_mask[b].to(device=p.device, dtype=torch.float32).reshape(-1)
                else:
                    mask_b = torch.zeros((formula_n,), dtype=torch.float32, device=p.device)
            else:
                mask_t = torch.as_tensor(formulae_mask, device=p.device, dtype=torch.float32)
                if mask_t.dim() >= 2 and b < mask_t.shape[0]:
                    mask_b = mask_t[b].reshape(-1)
                else:
                    mask_b = torch.zeros((formula_n,), dtype=torch.float32, device=p.device)

            if mask_b.shape[0] < formula_n:
                pad_n = formula_n - mask_b.shape[0]
                mask_b = torch.cat([mask_b, torch.zeros((pad_n,), dtype=torch.float32, device=p.device)], dim=0)
            valid_formula = mask_b[:formula_n] > 0
            p = p * mask_b[:formula_n]
            if rank_b is not None:
                rank_b = rank_b.masked_fill(~valid_formula, torch.finfo(rank_b.dtype).min)

        if topk_formula and int(topk_formula) > 0 and formula_n > int(topk_formula):
            topk_n = int(min(int(topk_formula), formula_n))
            rank_base = rank_b if rank_b is not None else p
            _, topi = torch.topk(rank_base, k=topk_n, largest=True, sorted=False)
            keep = torch.zeros_like(p, dtype=torch.bool)
            keep[topi] = True
            if float(min_formula_prob) > 0:
                keep = keep & (p > float(min_formula_prob))
        else:
            keep = p > float(min_formula_prob)

        if keep.sum().item() == 0:
            out.append(np.zeros((0, 2), dtype=np.float32))
            continue

        peaks_keep = peaks_b[keep]
        prob_keep = p[keep].unsqueeze(-1)

        mz = peaks_keep[..., 0].reshape(-1)
        inten = (peaks_keep[..., 1] * prob_keep).reshape(-1)

        valid = torch.isfinite(mz) & torch.isfinite(inten) & (inten > 0)
        mz = mz[valid]
        inten = inten[valid]

        if mz.numel() == 0:
            out.append(np.zeros((0, 2), dtype=np.float32))
        else:
            out.append(torch.stack([mz, inten], dim=-1).detach().cpu().numpy().astype(np.float32))

    return out
    

# Function overview: construct_sparse_mm_from_peak_info handles a specific workflow step in this module.
def construct_sparse_mm_from_peak_info(peaks, spect_bin_n=512):
    assert peaks.shape[-1] == 2
    BATCH_N, SUBSET_N, PEAK_N, _ = peaks.shape
    sparse_mass_matrices = [
        create_mass_matrix_sparse(
            peaks[:, :, i, 0].round().long(),
            spect_bin_n,
            peaks[:, :, i, 1]
        ) for i in range(PEAK_N)
    ]

    sparse_mass_matrix = sparse_mass_matrices[0]
    for m in sparse_mass_matrices[1:]:
        sparse_mass_matrix += m
    
    return sparse_mass_matrix


# Class overview: MolAttentionMultiTransform encapsulates a reusable component in this module.
class MolAttentionMultiTransform(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 attn_transform = 'softmax',
                 formulae_prob_transform = 'softmax',
                 spect_bin_n = 512):

        
        super( MolAttentionMultiTransform, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n

        if formulae_prob_transform == 'softmax':
            self.formulae_prob_transform = nn.Softmax(dim=-1)
        elif formulae_prob_transform == 'sigsoftmax':
            self.formulae_prob_transform = SigSoftmax(dim=-1)
        elif formulae_prob_transform == 'sigmoid':
            self.formulae_prob_transform = nn.Sigmoid()

        if attn_transform == 'softmax':
            self.attn_transform = nn.Softmax(dim=1)
        elif attn_transform == 'sigsoftmax':
            self.attn_transform = SigSoftmax(dim=1)
        elif attn_transform == 'sigmoid':
            self.attn_transform = nn.Sigmoid()




    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses, vert_element_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
        weighting = self.attn_transform(dot_prod)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = self.formulae_prob_transform(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return spect_out





# Class overview: SigSoftmax encapsulates a reusable component in this module.
class SigSoftmax(nn.Module):
    """
    SigSoftmax from the paper - https://arxiv.org/pdf/1805.10829.pdf

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, dim = 0):
        
        super( SigSoftmax, self).__init__()
        self.dim = dim


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, logits):
        
        max_values = torch.max(logits, self.dim, keepdim = True)[0]
        exp_logits_sigmoided = torch.exp(logits - max_values) * torch.sigmoid(logits)
        sum_exp_logits_sigmoided = exp_logits_sigmoided.sum(self.dim, keepdim = True)
        return exp_logits_sigmoided / sum_exp_logits_sigmoided

    

# Class overview: MolAttentionGRU encapsulates a reusable component in this module.
class MolAttentionGRU(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 spect_bin, 
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
    ):

        
        super( MolAttentionGRU, self).__init__()

        self.spect_bin = spect_bin
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize=formula_oh_normalize

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.prob_softmax = prob_softmax

    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity, 
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
            computed global embedding for molecule
        vert_feat_mask : BATCH_N x ATOM_N  
            input mask for valid atoms

        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
            'formulae_features'
        formulae_peaks_mass_idx : BATCH_N x MAX_FORMULAE_N x NUM_MASSES 
         formulae_peaks_intensity, 
        
        
        vert_element_oh: BATCH_N x MAX_ATOM_N=32 x MAX_ELEMENT_N=8
            one-hot encoding of which element corresponds to which vertex
        adj_oh: BATCH_N x N_CHANNELS x MAX_ATOM_N x MAX_ATOM_N
            one-hot adjacency matrix
        formula_frag_count:
            ?
        """

        BATCH_N = vert_feat_in.shape[0]
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        if self.formula_oh_normalize:
            possible_formulae = possible_formulae / possible_formulae.sum(axis=2).unsqueeze(2)
            
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)


        # [步骤3] 通过注意力分数，把整个分子的原子特征浓缩成了特定于该“候选分子式”的隐特征
        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()

        # per batch
        out_y = []
        for batch_i in range(BATCH_N):
            sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(formulae_peaks_mass_idx[batch_i],
                                                                               formulae_peaks_intensity[batch_i], 
                                                                               self.spect_bin.get_num_bins())
            # AMP 下稀疏矩阵不支持 half，强制用 float32 做乘法
            y = sparse_mat_matrix.float() @ formulae_probs[batch_i].float()
            out_y.append(y)
        spect_out = torch.stack(out_y)
                
        # sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
        #                                            self.spect_bin_n,
        #                                            formulae_masses[:, :, i, 1]
        # )\
        #                  .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        # sparse_mass_matrix = sparse_mass_matrices[0]
        # for m in sparse_mass_matrices[1:]:
        #     sparse_mass_matrix += m

        # spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}



# =========================================================================
# 🥇 核心化学式注意力预测头 (Core Formula Attention Head)
# =========================================================================
# 它的原理：
# 1. 它不再尝试在 2D 图上切断原子的边。
# 2. 它拿到了外层传来的 `possible_formulae` (所有依据元素守恒算出的可能分子式)。
# 3. 它将分子图全局特征 (vert_feat_in) 和候选化学式通过 Attention(注意力机制) 融合。
# 4. 根据 MS/MS 下的【碰撞能量】和【加合物种类】等环境信息，为每个化学式存在概率打分。
# Class overview: MolAttentionGRUNewSparse encapsulates a reusable component in this module.
class MolAttentionGRUNewSparse(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 spect_bin, 
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
                 
                 # New MS/MS args
                 ce_emb_dim=32,
                 adduct_emb_dim=16,
                 adduct_vocab_size=32,
                 instrument_emb_dim=16,
                 instrument_vocab_size=0,
                 ms_level_emb_dim=8,
                 ms_level_vocab_size=0,
                 mol_feat_in=0,
                 mol_feat_emb_dim=32,
                 use_msms_conditioning=False,
                 normalize_1_output=False,
                 formula_aux_feat_in=0,
                 use_formula_selector=False,
                 selector_hidden_d=128,
                 selector_topk=0,
                 selector_temperature=1.0,
                 selector_loss_weight=0.0,
                 selector_target_mode='exact_official',
                 selector_official_bin_width=0.01,
                 selector_official_max_mz=1005.0,
                 selector_dropout=0.1,
                 use_formula_ranking=False,
                 ranking_hidden_d=128,
                 ranking_temperature=1.0,
                 ranking_loss_weight=0.0,
                 ranking_target_bin_width=0.1,
                 ranking_target_max_mz=1005.0,
                 ranking_dropout=0.1,
                 use_local_break_head=True,
                 local_break_dim=128,
                 local_break_logit_weight=0.7,
                 local_break_bond_feat_dim=4,
                 candidate_attn_temperature=0.5,
                 use_precursor_hard_mask=True,
                 precursor_hard_mask_tol=1,
                 use_mass_shift_head=False,
                 mass_shift_range=2,
                 mass_shift_hidden_d=128,
                 mass_shift_temperature=1.0,
                 mass_shift_dropout=0.1,
                 use_mass_shift=None,
                 mass_shift_hidden=None,
                 mass_shift_temp=None,
                 pred_exact_topk=0,
                 pred_exact_min_prob=0.0,
    ):
        super(MolAttentionGRUNewSparse, self).__init__()

        self.spect_bin = spect_bin
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize=formula_oh_normalize

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False

        # Keep formula-specific signal in candidate_x by explicitly fusing formula embedding.
        self.candidate_fuse = nn.Sequential(
            nn.Linear(g_feat_in + int(embedding_key_size), g_feat_in),
            nn.GELU(),
            nn.Linear(g_feat_in, g_feat_in),
        )
        self.candidate_attn_temperature = max(float(candidate_attn_temperature), 1e-6)
        self.formula_q = nn.Linear(int(embedding_key_size), int(embedding_key_size))
        self.vert_k = nn.Linear(int(embedding_key_size), int(embedding_key_size))
        self.vert_v = nn.Linear(g_feat_in, g_feat_in)

        # Main score head explicitly consumes formula aux + peak summary so scoring can reason about mass support transfer.
        self.peak_summary_feat_dim = 11
        self.peak_summary_dim = max(8, min(int(g_feat_in), 64))
        self.peak_summary_official_bin_width = max(1e-6, float(selector_official_bin_width))
        peak_summary_hidden_d = max(16, int(internal_d // 2))
        self.peak_summary_mlp = nn.Sequential(
            nn.Linear(self.peak_summary_feat_dim, peak_summary_hidden_d),
            nn.GELU(),
            nn.Linear(peak_summary_hidden_d, self.peak_summary_dim),
            nn.GELU(),
        )

        formula_aux_feat_d = int(max(0, formula_aux_feat_in))
        main_score_input_d = (
            int(g_feat_in)
            + int(embedding_key_size)
            + int(g_feat_in)
            + formula_aux_feat_d
            + int(self.peak_summary_dim)
        )
        self.main_score_mlp = nn.Sequential(
            nn.Linear(main_score_input_d, internal_d),
            nn.GELU(),
            nn.Linear(internal_d, internal_d),
            nn.GELU(),
        )
        self.main_score_out = nn.Linear(internal_d, 1)
        self.score_skip_candidate = nn.Linear(g_feat_in, 1, bias=False)
        self.score_skip_formula = nn.Linear(int(embedding_key_size), 1, bias=False)

        self.use_local_break_head = bool(use_local_break_head)
        self.local_break_dim = max(8, int(local_break_dim))
        self.local_break_logit_weight = float(local_break_logit_weight)
        self.local_break_bond_feat_dim = max(0, int(local_break_bond_feat_dim))

        if self.use_local_break_head:
            edge_local_in_dim = (int(g_feat_in) * 5) + self.local_break_bond_feat_dim
            self.edge_local_mlp = nn.Sequential(
                nn.Linear(edge_local_in_dim, internal_d),
                nn.GELU(),
                nn.Linear(internal_d, self.local_break_dim),
                nn.GELU(),
            )
            self.local_formula_proj = nn.Linear(int(embedding_key_size), self.local_break_dim)
            self.local_edge_proj = nn.Linear(self.local_break_dim, self.local_break_dim)
            self.local_cond_proj = nn.Linear(int(g_feat_in), self.local_break_dim)
            self.local_formula_bias = nn.Linear(int(embedding_key_size), 1, bias=False)
            self.local_edge_bias = nn.Linear(self.local_break_dim, 1, bias=False)
        else:
            self.edge_local_mlp = None
            self.local_formula_proj = None
            self.local_edge_proj = None
            self.local_cond_proj = None
            self.local_formula_bias = None
            self.local_edge_bias = None
            
        
        self.norm2 = nn.LayerNorm(internal_d)
        
        self.use_msms_conditioning = use_msms_conditioning
        self.normalize_1_output = normalize_1_output
        self.g_feat_in = g_feat_in
        self.use_peak_aware_score_input = os.environ.get('USE_PEAK_AWARE_SCORE_INPUT', '1') == '1'
        self.support_os_input_dim = int((self.g_feat_in * 2) + self.peak_summary_feat_dim)
        self.instrument_vocab_size = int(instrument_vocab_size)
        self.ms_level_vocab_size = int(ms_level_vocab_size)
        self.mol_feat_in = int(mol_feat_in)
        self.formula_aux_feat_in = int(formula_aux_feat_in)
        self.use_formula_selector = bool(use_formula_selector)
        self.selector_hidden_d = int(selector_hidden_d)
        self.selector_topk = int(selector_topk)
        self.selector_temperature = max(float(selector_temperature), 1e-6)
        self.selector_loss_weight = float(selector_loss_weight)
        selector_target_mode = str(selector_target_mode).strip().lower()
        if selector_target_mode == '':
            selector_target_mode = 'exact_official'
        self.selector_target_mode = selector_target_mode
        self.selector_official_bin_width = max(1e-6, float(selector_official_bin_width))
        self.selector_official_max_mz = max(self.selector_official_bin_width, float(selector_official_max_mz))
        self.use_formula_ranking = bool(use_formula_ranking)
        self.ranking_hidden_d = int(ranking_hidden_d)
        self.ranking_temperature = max(float(ranking_temperature), 1e-6)
        self.ranking_loss_weight = float(ranking_loss_weight)
        self.ranking_target_bin_width = max(1e-6, float(ranking_target_bin_width))
        self.ranking_target_max_mz = max(self.ranking_target_bin_width, float(ranking_target_max_mz))
        self.use_precursor_hard_mask = bool(use_precursor_hard_mask)
        self.precursor_hard_mask_tol = max(0, int(precursor_hard_mask_tol))
        self.pred_exact_topk = max(0, int(pred_exact_topk))
        self.pred_exact_min_prob = max(0.0, float(pred_exact_min_prob))

        # Alias support: allows train scripts to use shorter mass-shift keys.
        if use_mass_shift is not None:
            use_mass_shift_head = bool(use_mass_shift)
        if mass_shift_hidden is not None:
            mass_shift_hidden_d = int(mass_shift_hidden)
        if mass_shift_temp is not None:
            mass_shift_temperature = float(mass_shift_temp)

        self.use_mass_shift_head = bool(use_mass_shift_head)
        self.mass_shift_range = max(0, int(mass_shift_range))
        self.mass_shift_hidden_d = int(mass_shift_hidden_d)
        self.mass_shift_temperature = max(float(mass_shift_temperature), 1e-6)
        if self.mass_shift_range <= 0:
            self.use_mass_shift_head = False

        if self.use_msms_conditioning:
            self.ce_proj = nn.Sequential(
                nn.Linear(1, ce_emb_dim),
                nn.ReLU(),
                nn.Linear(ce_emb_dim, g_feat_in),
            )
            self.precursor_proj = nn.Sequential(
                nn.Linear(1, ce_emb_dim),
                nn.ReLU(),
                nn.Linear(ce_emb_dim, g_feat_in),
            )
            self.adduct_emb = nn.Embedding(adduct_vocab_size, adduct_emb_dim)
            self.adduct_proj = nn.Linear(adduct_emb_dim, g_feat_in)
            # Missingness-aware conditioning: avoids conflating "missing" with numeric 0.
            self.ce_missing_emb = nn.Embedding(2, g_feat_in)
            self.adduct_missing_emb = nn.Embedding(2, g_feat_in)
            self.precursor_missing_emb = nn.Embedding(2, g_feat_in)
            if self.instrument_vocab_size > 0:
                self.instrument_emb = nn.Embedding(self.instrument_vocab_size, instrument_emb_dim)
                self.instrument_proj = nn.Linear(instrument_emb_dim, g_feat_in)
                self.instrument_missing_emb = nn.Embedding(2, g_feat_in)
            else:
                self.instrument_emb = None
                self.instrument_proj = None
                self.instrument_missing_emb = None

            if self.ms_level_vocab_size > 0:
                self.ms_level_emb = nn.Embedding(self.ms_level_vocab_size, ms_level_emb_dim)
                self.ms_level_proj = nn.Linear(ms_level_emb_dim, g_feat_in)
                self.ms_level_missing_emb = nn.Embedding(2, g_feat_in)
            else:
                self.ms_level_emb = None
                self.ms_level_proj = None
                self.ms_level_missing_emb = None

            if self.mol_feat_in > 0:
                self.mol_feat_proj = nn.Sequential(
                    nn.Linear(self.mol_feat_in, mol_feat_emb_dim),
                    nn.ReLU(),
                    nn.Linear(mol_feat_emb_dim, g_feat_in),
                )
            else:
                self.mol_feat_proj = None

        selector_input_d = self.formula_aux_feat_in + (self.g_feat_in * 2)
        if self.use_formula_selector and self.formula_aux_feat_in > 0:
            self.selector_norm = nn.LayerNorm(selector_input_d)
            self.selector_mlp = nn.Sequential(
                nn.Linear(selector_input_d, self.selector_hidden_d),
                nn.ReLU(),
                nn.Dropout(selector_dropout),
                nn.Linear(self.selector_hidden_d, self.selector_hidden_d),
                nn.ReLU(),
                nn.Dropout(selector_dropout),
                nn.Linear(self.selector_hidden_d, 1),
            )
        else:
            self.selector_norm = None
            self.selector_mlp = None

        if self.use_formula_ranking:
            self.ranking_norm = nn.LayerNorm(selector_input_d)
            self.ranking_mlp = nn.Sequential(
                nn.Linear(selector_input_d, self.ranking_hidden_d),
                nn.ReLU(),
                nn.Dropout(ranking_dropout),
                nn.Linear(self.ranking_hidden_d, self.ranking_hidden_d),
                nn.ReLU(),
                nn.Dropout(ranking_dropout),
                nn.Linear(self.ranking_hidden_d, 1),
            )
        else:
            self.ranking_norm = None
            self.ranking_mlp = None

        self.mass_shift_norm = None
        self.mass_shift_mlp = None
        if self.use_mass_shift_head:
            mass_shift_bins = (self.mass_shift_range * 2) + 1
            offsets = torch.arange(-self.mass_shift_range, self.mass_shift_range + 1, dtype=torch.long)
            self.register_buffer('mass_shift_offsets', offsets)
            self.mass_shift_norm = nn.LayerNorm(selector_input_d)
            self.mass_shift_mlp = nn.Sequential(
                nn.Linear(selector_input_d, self.mass_shift_hidden_d),
                nn.ReLU(),
                nn.Dropout(mass_shift_dropout),
                nn.Linear(self.mass_shift_hidden_d, self.mass_shift_hidden_d),
                nn.ReLU(),
                nn.Dropout(mass_shift_dropout),
                nn.Linear(self.mass_shift_hidden_d, mass_shift_bins),
            )
        else:
            self.mass_shift_offsets = None

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.prob_softmax = prob_softmax

        # Rollback toggles for scorer-ablation experiments.
        self.rollback_legacy_score_head = os.environ.get('ROLLBACK_LEGACY_SCORE_HEAD', '0') == '1'
        self.rollback_plain_attention = os.environ.get('ROLLBACK_PLAIN_ATTENTION', '0') == '1'
        self.rollback_plain_attention_no_fuse = os.environ.get('ROLLBACK_PLAIN_ATTENTION_NO_FUSE', '1') == '1'

        # Simple scorer mode: keep the main score path clean for target-ranking diagnosis.
        self.simple_score_head = os.environ.get('SIMPLE_SCORE_HEAD', '1') == '1'
        self.simple_disable_selector = os.environ.get('SIMPLE_DISABLE_SELECTOR', '1') == '1'
        self.simple_disable_ranking = os.environ.get('SIMPLE_DISABLE_RANKING', '1') == '1'
        self.simple_disable_local_break = os.environ.get('SIMPLE_DISABLE_LOCAL_BREAK', '1') == '1'
        self.simple_disable_mass_shift = os.environ.get('SIMPLE_DISABLE_MASS_SHIFT', '1') == '1'
        self.simple_disable_precursor_hard_mask = os.environ.get('SIMPLE_DISABLE_PRECURSOR_HARD_MASK', '1') == '1'
        self.simple_score_use_cond_bias = os.environ.get('SIMPLE_SCORE_USE_COND_BIAS', '1') == '1'
        self.simple_score_cond_scale = float(os.environ.get('SIMPLE_SCORE_COND_SCALE', '0.05'))

        base_in_d = int(embedding_key_size) + int(g_feat_in)
        self.base_score_norm = nn.LayerNorm(base_in_d)
        self.base_score_l1 = nn.Linear(base_in_d, internal_d)
        self.base_score_l2 = nn.Linear(internal_d, internal_d)
        self.base_score_out = nn.Linear(internal_d, 1)
        self.base_cond_proj = nn.Linear(int(g_feat_in), int(embedding_key_size))


    # Function overview: _to_2d_float handles a specific workflow step in this module.
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
                flat = torch.cat(
                    [flat, torch.zeros((pad_n,), dtype=flat.dtype, device=device)],
                    dim=0,
                )
            x = flat[:batch_n].view(batch_n, 1)
        return x.float()

    # Function overview: _to_vocab_index handles a specific workflow step in this module.
    def _to_vocab_index(self, x, batch_n, device, vocab_size):
        if x is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        if not torch.is_tensor(x):
            try:
                x = torch.as_tensor(x)
            except Exception:
                if isinstance(x, (list, tuple, np.ndarray)):
                    parsed = []
                    for v in x:
                        try:
                            parsed.append(int(v))
                        except Exception:
                            parsed.append(0)
                    x = torch.as_tensor(parsed)
                else:
                    x = torch.zeros((batch_n,), dtype=torch.long)
        x = x.to(device)
        if x.dim() == 0:
            x = x.view(1)
        elif x.dim() > 1:
            x = torch.argmax(x, dim=-1)
        x = x.long().reshape(-1)
        if x.shape[0] < batch_n:
            pad_n = batch_n - x.shape[0]
            x = torch.cat([x, torch.zeros((pad_n,), dtype=torch.long, device=device)], dim=0)
        x = x[:batch_n]
        x = x % max(1, int(vocab_size))
        return x

    # Function overview: _to_adduct_index handles a specific workflow step in this module.
    def _to_adduct_index(self, x, batch_n, device):
        return self._to_vocab_index(x, batch_n, device, self.adduct_emb.num_embeddings)

    # Function overview: _to_instrument_index handles a specific workflow step in this module.
    def _to_instrument_index(self, x, batch_n, device):
        if self.instrument_emb is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        return self._to_vocab_index(x, batch_n, device, self.instrument_emb.num_embeddings)

    # Function overview: _to_ms_level_index handles a specific workflow step in this module.
    def _to_ms_level_index(self, x, batch_n, device):
        if self.ms_level_emb is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        return self._to_vocab_index(x, batch_n, device, self.ms_level_emb.num_embeddings)

    # Function overview: _to_missing_index handles a specific workflow step in this module.
    def _to_missing_index(self, x, batch_n, device):
        if x is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        v = self._to_2d_float(x, batch_n, device).squeeze(-1)
        return (v > 0.5).long()

    # Function overview: _resolve_missing_index handles a specific workflow step in this module.
    def _resolve_missing_index(self, explicit_missing, inferred_missing_bool, batch_n, device):
        if explicit_missing is not None:
            return self._to_missing_index(explicit_missing, batch_n, device)
        if inferred_missing_bool is None:
            return torch.zeros((batch_n,), dtype=torch.long, device=device)
        if not torch.is_tensor(inferred_missing_bool):
            inferred_missing_bool = torch.as_tensor(inferred_missing_bool)
        inferred_missing_bool = inferred_missing_bool.to(device).reshape(-1)
        if inferred_missing_bool.shape[0] < batch_n:
            pad_n = batch_n - inferred_missing_bool.shape[0]
            inferred_missing_bool = torch.cat(
                [inferred_missing_bool, torch.zeros((pad_n,), dtype=inferred_missing_bool.dtype, device=device)],
                dim=0,
            )
        inferred_missing_bool = inferred_missing_bool[:batch_n]
        return (inferred_missing_bool > 0.5).long()

    # Function overview: _to_formulae_mask handles a specific workflow step in this module.
    def _to_formulae_mask(self, x, batch_n, formula_n, device):
        if x is None:
            return torch.ones((batch_n, formula_n), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float32)
        target = batch_n * formula_n
        flat = x.reshape(-1)
        if flat.shape[0] < target:
            pad_n = target - flat.shape[0]
            flat = torch.cat([flat, torch.ones((pad_n,), dtype=torch.float32, device=device)], dim=0)
        mask = flat[:target].view(batch_n, formula_n)
        mask = (mask > 0.5).float()
        # Avoid all-zero rows that would produce NaNs after masked softmax.
        has_valid = (mask.sum(dim=1, keepdim=True) > 0)
        if not torch.all(has_valid):
            mask = torch.where(has_valid, mask, torch.ones_like(mask))
        return mask

    # Function overview: _to_feature_2d handles a specific workflow step in this module.
    def _to_feature_2d(self, x, batch_n, feat_n, device):
        if feat_n <= 0:
            return torch.zeros((batch_n, 0), dtype=torch.float32, device=device)
        if x is None:
            return torch.zeros((batch_n, feat_n), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float32)
        x = x.reshape(x.shape[0], -1) if x.dim() > 1 else x.view(-1, 1)
        if x.shape[1] < feat_n:
            pad = torch.zeros((x.shape[0], feat_n - x.shape[1]), dtype=torch.float32, device=device)
            x = torch.cat([x, pad], dim=1)
        elif x.shape[1] > feat_n:
            x = x[:, :feat_n]
        if x.shape[0] < batch_n:
            pad_rows = torch.zeros((batch_n - x.shape[0], feat_n), dtype=torch.float32, device=device)
            x = torch.cat([x, pad_rows], dim=0)
        return x[:batch_n]

    # Function overview: _to_sparse_peak_tensor converts mixed spectrum input into [N, 2] peak tensor.
    def _to_sparse_peak_tensor(self, x, device):
        if x is None:
            return torch.zeros((0, 2), dtype=torch.float32, device=device)

        if torch.is_tensor(x):
            arr = x.detach().to(device=device, dtype=torch.float32)
        else:
            try:
                arr = torch.as_tensor(x, dtype=torch.float32, device=device)
            except Exception:
                try:
                    arr = torch.as_tensor(np.asarray(x, dtype=np.float32), dtype=torch.float32, device=device)
                except Exception:
                    return torch.zeros((0, 2), dtype=torch.float32, device=device)

        if arr.numel() <= 0:
            return torch.zeros((0, 2), dtype=torch.float32, device=device)

        if arr.dim() == 1:
            # Dense vectors are not exact peak lists.
            return torch.zeros((0, 2), dtype=torch.float32, device=device)

        if arr.shape[-1] != 2:
            try:
                arr = arr.reshape(-1, 2)
            except Exception:
                return torch.zeros((0, 2), dtype=torch.float32, device=device)
        else:
            arr = arr.reshape(-1, 2)

        return arr

    # Function overview: _build_binned_true_dense bins raw exact peaks to configurable dense bins.
    def _build_binned_true_dense(self, spect_raw, batch_n, device, bin_width, max_mz):
        bin_width = float(max(1e-6, bin_width))
        max_mz = float(max(bin_width, max_mz))
        max_bin = int(np.floor(max_mz / bin_width))
        bin_n = max_bin + 1

        dense = torch.zeros((batch_n, bin_n), dtype=torch.float32, device=device)
        has_raw = torch.zeros((batch_n,), dtype=torch.bool, device=device)
        if spect_raw is None:
            return dense, has_raw

        for b in range(batch_n):
            peaks_b = None
            if torch.is_tensor(spect_raw):
                if spect_raw.dim() == 3 and spect_raw.shape[-1] == 2 and b < int(spect_raw.shape[0]):
                    peaks_b = spect_raw[b]
                elif spect_raw.dim() == 2 and spect_raw.shape[-1] == 2 and batch_n == 1:
                    peaks_b = spect_raw
            elif isinstance(spect_raw, (list, tuple)):
                if b < len(spect_raw):
                    peaks_b = spect_raw[b]
            elif isinstance(spect_raw, np.ndarray):
                if spect_raw.ndim == 3 and spect_raw.shape[-1] == 2 and b < int(spect_raw.shape[0]):
                    peaks_b = spect_raw[b]
                elif spect_raw.ndim == 2 and spect_raw.shape[-1] == 2 and batch_n == 1:
                    peaks_b = spect_raw

            peaks_t = self._to_sparse_peak_tensor(peaks_b, device)
            if peaks_t.shape[0] <= 0:
                continue

            mz = peaks_t[:, 0]
            intensity = peaks_t[:, 1]
            valid = torch.isfinite(mz) & torch.isfinite(intensity) & (intensity > 0)
            if not bool(valid.any().item()):
                continue

            mz = mz[valid]
            intensity = intensity[valid]
            idx = torch.floor((mz / bin_width) + 1e-8).long()
            valid_idx = (idx >= 0) & (idx <= max_bin)
            if not bool(valid_idx.any().item()):
                continue

            idx = idx[valid_idx]
            intensity = intensity[valid_idx]
            dense[b].scatter_add_(0, idx, intensity)
            has_raw[b] = True

        return dense, has_raw

    # Function overview: _build_official_true_dense bins raw exact peaks to official 0.01-style dense vectors.
    def _build_official_true_dense(self, spect_raw, batch_n, device):
        return self._build_binned_true_dense(
            spect_raw,
            batch_n,
            device,
            bin_width=self.selector_official_bin_width,
            max_mz=self.selector_official_max_mz,
        )

    # Function overview: _compute_candidate_overlap_exact_binned_from_peaks computes overlap target from exact candidate peaks.
    def _compute_candidate_overlap_exact_binned_from_peaks(
        self,
        spect_raw,
        formulae_peaks,
        formulae_mask,
        batch_n,
        formula_n,
        device,
        bin_width,
        max_mz,
    ):
        if spect_raw is None or formulae_peaks is None:
            return None, None

        if not torch.is_tensor(formulae_peaks):
            try:
                formulae_peaks = torch.as_tensor(formulae_peaks)
            except Exception:
                return None, None

        peaks = formulae_peaks.to(device=device, dtype=torch.float32)
        mask = formulae_mask.float().to(device)
        if peaks.dim() == 3 and peaks.shape[-1] == 2:
            peaks = peaks.unsqueeze(0)
        if peaks.dim() != 4 or peaks.shape[-1] != 2:
            return None, None

        use_b = min(int(batch_n), int(peaks.shape[0]), int(mask.shape[0]))
        use_m = min(int(formula_n), int(peaks.shape[1]), int(mask.shape[1]))
        use_k = int(peaks.shape[2])
        if use_b <= 0 or use_m <= 0 or use_k <= 0:
            return None, None

        true_dense, has_raw = self._build_binned_true_dense(
            spect_raw,
            use_b,
            device,
            bin_width=bin_width,
            max_mz=max_mz,
        )

        peaks = peaks[:use_b, :use_m, :, :]
        mask = mask[:use_b, :use_m]
        mz = peaks[..., 0]
        peak_int = peaks[..., 1]
        bwidth = float(max(1e-6, bin_width))
        bin_n = int(true_dense.shape[-1])

        idx = torch.floor((mz / bwidth) + 1e-8).long()
        valid_peak = (
            torch.isfinite(mz)
            & torch.isfinite(peak_int)
            & (peak_int > 0)
            & (idx >= 0)
            & (idx < bin_n)
        )
        safe_idx = idx.clamp(0, max(0, bin_n - 1))
        true_at_peak = torch.gather(true_dense, 1, safe_idx.reshape(use_b, -1)).reshape(use_b, use_m, use_k)

        candidate_overlap = (
            true_at_peak
            * peak_int
            * valid_peak.to(peak_int.dtype)
        ).sum(dim=-1)
        candidate_overlap = candidate_overlap * mask

        overlap_full = torch.zeros((batch_n, formula_n), dtype=candidate_overlap.dtype, device=device)
        overlap_full[:use_b, :use_m] = candidate_overlap

        has_raw_full = torch.zeros((batch_n,), dtype=torch.bool, device=device)
        has_raw_full[:use_b] = has_raw

        return overlap_full, has_raw_full

    # Function overview: _compute_candidate_overlap_exact_official computes selector targets under official exact bins.
    def _compute_candidate_overlap_exact_official(
        self,
        spect_raw,
        formulae_peaks_official_idx,
        formulae_peaks_intensity,
        formulae_mask,
        batch_n,
        formula_n,
        device,
    ):
        if spect_raw is None or formulae_peaks_official_idx is None:
            return None, None

        if not torch.is_tensor(formulae_peaks_official_idx):
            try:
                formulae_peaks_official_idx = torch.as_tensor(formulae_peaks_official_idx)
            except Exception:
                return None, None

        peak_idx = formulae_peaks_official_idx.to(device=device, dtype=torch.long)
        peak_int = formulae_peaks_intensity.float().to(device)
        mask = formulae_mask.float().to(device)
        if peak_idx.dim() != 3 or peak_int.dim() != 3:
            return None, None

        use_b = min(int(batch_n), int(peak_idx.shape[0]), int(peak_int.shape[0]), int(mask.shape[0]))
        use_m = min(int(formula_n), int(peak_idx.shape[1]), int(peak_int.shape[1]), int(mask.shape[1]))
        use_k = min(int(peak_idx.shape[2]), int(peak_int.shape[2]))
        if use_b <= 0 or use_m <= 0 or use_k <= 0:
            return None, None

        true_dense, has_raw = self._build_official_true_dense(spect_raw, use_b, device)
        if true_dense is None:
            return None, None

        peak_idx = peak_idx[:use_b, :use_m, :use_k]
        peak_int = peak_int[:use_b, :use_m, :use_k]
        mask = mask[:use_b, :use_m]

        official_bin_n = int(true_dense.shape[-1])
        idx_flat = peak_idx.reshape(use_b, -1)
        valid_peak_flat = (idx_flat >= 0) & (idx_flat < official_bin_n)
        safe_idx = idx_flat.clamp(0, max(0, official_bin_n - 1))
        true_at_peak = torch.gather(true_dense, 1, safe_idx).reshape(use_b, use_m, use_k)

        candidate_overlap = (
            true_at_peak
            * peak_int
            * valid_peak_flat.reshape(use_b, use_m, use_k).to(peak_int.dtype)
        ).sum(dim=-1)
        candidate_overlap = candidate_overlap * mask

        overlap_full = torch.zeros((batch_n, formula_n), dtype=candidate_overlap.dtype, device=device)
        overlap_full[:use_b, :use_m] = candidate_overlap

        has_raw_full = torch.zeros((batch_n,), dtype=torch.bool, device=device)
        if has_raw is not None:
            has_raw_full[:use_b] = has_raw

        return overlap_full, has_raw_full

    # Function overview: _to_candidate_feat_3d handles a specific workflow step in this module.
    def _to_candidate_feat_3d(self, x, batch_n, formula_n, feat_n, device):
        if feat_n <= 0:
            return None
        if x is None:
            return torch.zeros((batch_n, formula_n, feat_n), dtype=torch.float32, device=device)
        if not torch.is_tensor(x):
            x = torch.as_tensor(x)
        x = x.to(device=device, dtype=torch.float32)
        if x.dim() == 1:
            x = x.view(1, 1, -1)
        elif x.dim() == 2:
            x = x.unsqueeze(0)
        elif x.dim() > 3:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        if x.shape[-1] < feat_n:
            pad = torch.zeros((x.shape[0], x.shape[1], feat_n - x.shape[-1]), dtype=torch.float32, device=device)
            x = torch.cat([x, pad], dim=-1)
        elif x.shape[-1] > feat_n:
            x = x[..., :feat_n]
        if x.shape[0] < batch_n:
            pad_rows = torch.zeros((batch_n - x.shape[0], x.shape[1], x.shape[2]), dtype=torch.float32, device=device)
            x = torch.cat([x, pad_rows], dim=0)
        if x.shape[1] < formula_n:
            pad_cols = torch.zeros((x.shape[0], formula_n - x.shape[1], x.shape[2]), dtype=torch.float32, device=device)
            x = torch.cat([x, pad_cols], dim=1)
        return x[:batch_n, :formula_n, :feat_n]

    # Function overview: _gather_formula_candidates handles a specific workflow step in this module.
    def _gather_formula_candidates(self, x, topk_idx):
        if x is None or topk_idx is None:
            return x
        if not torch.is_tensor(topk_idx):
            topk_idx = torch.as_tensor(topk_idx, dtype=torch.long)
        else:
            topk_idx = topk_idx.long()
        if topk_idx.dim() == 1:
            topk_idx = topk_idx.unsqueeze(0)
        elif topk_idx.dim() > 2:
            topk_idx = topk_idx.reshape(topk_idx.shape[0], -1)

        if not torch.is_tensor(x):
            x = torch.as_tensor(x, device=topk_idx.device)
        elif x.device != topk_idx.device:
            x = x.to(topk_idx.device)

        if x.dim() < 2:
            return x
        if x.shape[1] <= 0:
            return x[:, :0, ...]

        if x.shape[0] != topk_idx.shape[0]:
            use_b = min(int(x.shape[0]), int(topk_idx.shape[0]))
            if use_b <= 0:
                return x[:0]
            x = x[:use_b]
            topk_idx = topk_idx[:use_b]

        valid = (topk_idx >= 0) & (topk_idx < int(x.shape[1]))
        safe_idx = topk_idx.clamp(0, max(0, int(x.shape[1]) - 1))

        if x.dim() == 2:
            out = torch.gather(x, 1, safe_idx)
            if not torch.all(valid):
                out = torch.where(valid, out, torch.zeros_like(out))
            return out
        if x.dim() == 3:
            gather_idx = safe_idx.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            out = torch.gather(x, 1, gather_idx)
            if not torch.all(valid):
                out = torch.where(valid.unsqueeze(-1), out, torch.zeros_like(out))
            return out
        if x.dim() == 4:
            gather_idx = safe_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[-2], x.shape[-1])
            out = torch.gather(x, 1, gather_idx)
            if not torch.all(valid):
                out = torch.where(valid.unsqueeze(-1).unsqueeze(-1), out, torch.zeros_like(out))
            return out
        return x

    # Function overview: _build_candidate_peak_summary derives compact per-candidate peak descriptors for score conditioning.
    def _build_candidate_peak_summary(self, peak_idx, peak_int, formulae_mask, precursor_mz=None, use_official_idx=False):
        if (not torch.is_tensor(formulae_mask)):
            return None, None

        mask = formulae_mask.float()
        if mask.dim() > 2:
            mask = mask.reshape(mask.shape[0], -1)
        batch_n = int(mask.shape[0])
        formula_n = int(mask.shape[1]) if mask.dim() > 1 else 0
        device = mask.device

        feat_full = torch.zeros(
            (batch_n, formula_n, self.peak_summary_feat_dim),
            dtype=torch.float32,
            device=device,
        )
        pooled_denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        if (
            (not torch.is_tensor(peak_idx))
            or (not torch.is_tensor(peak_int))
            or peak_idx.dim() != 3
            or peak_int.dim() != 3
            or formula_n <= 0
        ):
            pooled = (feat_full * mask.unsqueeze(-1)).sum(dim=1) / pooled_denom
            return feat_full, pooled

        use_b = min(batch_n, int(peak_idx.shape[0]), int(peak_int.shape[0]))
        use_m = min(formula_n, int(peak_idx.shape[1]), int(peak_int.shape[1]))
        use_k = min(int(peak_idx.shape[2]), int(peak_int.shape[2]))
        if use_b <= 0 or use_m <= 0 or use_k <= 0:
            pooled = (feat_full * mask.unsqueeze(-1)).sum(dim=1) / pooled_denom
            return feat_full, pooled

        idx = peak_idx[:use_b, :use_m, :use_k].to(device=device, dtype=torch.long)
        inten = peak_int[:use_b, :use_m, :use_k].to(device=device, dtype=torch.float32)
        mask_use = mask[:use_b, :use_m]

        valid_peak = (
            torch.isfinite(inten)
            & (inten > 0)
            & (idx >= 0)
            & (mask_use > 0).unsqueeze(-1)
        )

        idx_safe = idx.clamp(min=0)
        idx_f = idx_safe.float()
        inten_safe = torch.where(valid_peak, inten.clamp_min(0.0), torch.zeros_like(inten))

        peak_count = valid_peak.float().sum(dim=-1)
        intensity_sum = inten_safe.sum(dim=-1).clamp_min(1e-8)
        mean_idx = (idx_f * inten_safe).sum(dim=-1) / intensity_sum
        var_idx = ((idx_f - mean_idx.unsqueeze(-1)) ** 2 * inten_safe).sum(dim=-1) / intensity_sum
        std_idx = torch.sqrt(var_idx + 1e-8)
        max_int = inten_safe.max(dim=-1).values
        mean_int = intensity_sum / peak_count.clamp_min(1.0)

        topk_n = min(3, use_k)
        top_int, top_pos = torch.topk(inten_safe, k=topk_n, dim=-1)
        top_idx = torch.gather(idx_f, -1, top_pos)
        top1_idx = top_idx[..., 0]
        top1_int = top_int[..., 0]
        topk_idx_mean = top_idx.mean(dim=-1)
        topk_int_mean = top_int.mean(dim=-1)

        fill_neg = torch.full_like(idx_safe, -1)
        sorted_idx = torch.sort(torch.where(valid_peak, idx_safe, fill_neg), dim=-1).values
        valid_sorted = sorted_idx >= 0
        if use_k > 1:
            first_col = valid_sorted[..., :1]
            rest = valid_sorted[..., 1:] & (
                (sorted_idx[..., 1:] != sorted_idx[..., :-1])
                | (~valid_sorted[..., :-1])
            )
            unique_marks = torch.cat([first_col, rest], dim=-1)
        else:
            unique_marks = valid_sorted
        unique_count = unique_marks.sum(dim=-1).float()

        idx_scale = idx_f.amax(dim=-1).clamp_min(1.0)
        mean_idx_norm = mean_idx / idx_scale
        std_idx_norm = std_idx / idx_scale
        top1_idx_norm = top1_idx / idx_scale
        topk_idx_norm = topk_idx_mean / idx_scale
        peak_count_norm = peak_count / max(1.0, float(use_k))
        unique_count_norm = unique_count / max(1.0, float(use_k))

        gap_top1_norm = torch.zeros_like(top1_idx_norm)
        gap_mean_norm = torch.zeros_like(mean_idx_norm)
        if precursor_mz is not None:
            precursor = self._to_2d_float(precursor_mz, use_b, device).squeeze(-1)
            if use_official_idx:
                precursor_idx = precursor / max(self.peak_summary_official_bin_width, 1e-6)
            else:
                try:
                    first_bin = float(getattr(self.spect_bin, 'first_bin_center', 1.0))
                    bin_width = float(getattr(self.spect_bin, 'bin_width', 1.0))
                except Exception:
                    first_bin = 1.0
                    bin_width = 1.0
                precursor_idx = (precursor - first_bin) / max(bin_width, 1e-6)
            precursor_idx = torch.where(torch.isfinite(precursor_idx), precursor_idx, torch.zeros_like(precursor_idx))
            precursor_idx = torch.clamp(precursor_idx, min=0.0)
            precursor_idx = precursor_idx.unsqueeze(-1).expand(-1, use_m)
            gap_top1_norm = torch.abs(top1_idx - precursor_idx) / (precursor_idx.abs() + 1.0)
            gap_mean_norm = torch.abs(mean_idx - precursor_idx) / (precursor_idx.abs() + 1.0)

        feat_use = torch.stack(
            [
                peak_count_norm,
                unique_count_norm,
                mean_idx_norm,
                std_idx_norm,
                max_int,
                mean_int,
                top1_idx_norm,
                top1_int,
                topk_idx_norm,
                topk_int_mean,
                0.5 * (gap_top1_norm + gap_mean_norm),
            ],
            dim=-1,
        )
        feat_use = feat_use * (mask_use > 0).unsqueeze(-1).to(feat_use.dtype)
        feat_full[:use_b, :use_m, :] = feat_use

        pooled = (feat_full * mask.unsqueeze(-1)).sum(dim=1) / pooled_denom
        return feat_full, pooled

    # Function overview: _compute_local_break_logits builds edge-local cleavage evidence logits per formula.
    def _compute_local_break_logits(
        self,
        masked_vert_feat,
        vert_mask_in,
        formulae_encoded,
        possible_formulae_raw,
        cond,
        formulae_mask,
        adj_oh,
        vert_element_oh,
    ):
        if (not self.use_local_break_head) or (self.edge_local_mlp is None):
            return None, None
        if (not torch.is_tensor(masked_vert_feat)) or masked_vert_feat.dim() != 3:
            return None, None
        if (not torch.is_tensor(adj_oh)):
            return None, None

        device = masked_vert_feat.device
        BATCH_N, ATOM_N, FEAT_N = masked_vert_feat.shape
        atom_mask = (vert_mask_in > 0)

        if adj_oh.dim() == 4:
            adj_bool = (adj_oh.abs().sum(dim=1) > 0)
        elif adj_oh.dim() == 3:
            adj_bool = (adj_oh.abs() > 0)
        else:
            return None, None

        if adj_bool.shape[0] != BATCH_N:
            use_b = min(int(adj_bool.shape[0]), int(BATCH_N))
            adj_bool = adj_bool[:use_b]
            atom_mask = atom_mask[:use_b]
            masked_vert_feat = masked_vert_feat[:use_b]
            formulae_encoded = formulae_encoded[:use_b]
            possible_formulae_raw = possible_formulae_raw[:use_b]
            cond = cond[:use_b]
            formulae_mask = formulae_mask[:use_b]
            BATCH_N = int(use_b)
            if BATCH_N <= 0:
                return None, None

        edge_idx_list = []
        max_edge_n = 0
        for bi in range(BATCH_N):
            adj_b = adj_bool[bi].bool()
            valid_b = atom_mask[bi].bool()
            adj_b = adj_b & valid_b.unsqueeze(0) & valid_b.unsqueeze(1)
            adj_b = torch.triu(adj_b, diagonal=1)
            idx_b = torch.nonzero(adj_b, as_tuple=False)
            edge_idx_list.append(idx_b)
            max_edge_n = max(max_edge_n, int(idx_b.shape[0]))

        if max_edge_n <= 0:
            return torch.zeros_like(formulae_mask.float()), {
                'edge_count_mean': 0.0,
                'compat_ratio': 0.0,
                'valid_formula_ratio': 0.0,
            }

        edge_u = torch.zeros((BATCH_N, max_edge_n), dtype=torch.long, device=device)
        edge_v = torch.zeros((BATCH_N, max_edge_n), dtype=torch.long, device=device)
        edge_mask = torch.zeros((BATCH_N, max_edge_n), dtype=torch.bool, device=device)

        bond_feat_pack = None
        if self.local_break_bond_feat_dim > 0:
            bond_feat_pack = torch.zeros(
                (BATCH_N, max_edge_n, self.local_break_bond_feat_dim),
                dtype=torch.float32,
                device=device,
            )

        endpoint_elem_pack = None
        endpoint_elem_dim = 0
        if torch.is_tensor(vert_element_oh) and vert_element_oh.dim() >= 3:
            endpoint_elem_dim = min(int(vert_element_oh.shape[-1]), int(possible_formulae_raw.shape[-1]))
            if endpoint_elem_dim > 0:
                endpoint_elem_pack = torch.zeros(
                    (BATCH_N, max_edge_n, endpoint_elem_dim),
                    dtype=torch.float32,
                    device=device,
                )

        for bi in range(BATCH_N):
            idx_b = edge_idx_list[bi]
            edge_n = int(idx_b.shape[0])
            if edge_n <= 0:
                continue
            ub = idx_b[:, 0].long()
            vb = idx_b[:, 1].long()
            edge_u[bi, :edge_n] = ub
            edge_v[bi, :edge_n] = vb
            edge_mask[bi, :edge_n] = True

            if bond_feat_pack is not None:
                if adj_oh.dim() == 4:
                    bf = adj_oh[bi, :, ub, vb].permute(1, 0).float()
                else:
                    bf = adj_oh[bi, ub, vb].float().unsqueeze(-1)
                if int(bf.shape[-1]) < self.local_break_bond_feat_dim:
                    pad = torch.zeros((edge_n, self.local_break_bond_feat_dim - int(bf.shape[-1])), dtype=torch.float32, device=device)
                    bf = torch.cat([bf, pad], dim=-1)
                elif int(bf.shape[-1]) > self.local_break_bond_feat_dim:
                    bf = bf[:, :self.local_break_bond_feat_dim]
                bond_feat_pack[bi, :edge_n, :] = bf

            if endpoint_elem_pack is not None and endpoint_elem_dim > 0:
                veb = vert_element_oh[bi].float()
                if veb.dim() > 2:
                    veb = veb.reshape(veb.shape[0], -1)
                em = torch.maximum(veb[ub, :endpoint_elem_dim], veb[vb, :endpoint_elem_dim])
                endpoint_elem_pack[bi, :edge_n, :] = em

        gather_u = edge_u.unsqueeze(-1).expand(-1, -1, FEAT_N)
        gather_v = edge_v.unsqueeze(-1).expand(-1, -1, FEAT_N)
        hu = torch.gather(masked_vert_feat.float(), 1, gather_u)
        hv = torch.gather(masked_vert_feat.float(), 1, gather_v)
        hmul = hu * hv
        hdiff = torch.abs(hu - hv)
        hpool = 0.5 * (hu + hv)

        edge_parts = [hu, hv, hmul, hdiff, hpool]
        if bond_feat_pack is not None:
            edge_parts.append(bond_feat_pack)
        edge_input = torch.cat(edge_parts, dim=-1)
        edge_hidden = self.edge_local_mlp(edge_input)

        formula_proj = F.gelu(self.local_formula_proj(formulae_encoded))
        edge_proj = F.gelu(self.local_edge_proj(edge_hidden))
        cond_proj = F.gelu(self.local_cond_proj(cond)).unsqueeze(1)
        edge_proj = edge_proj + cond_proj

        dscale = max(float(np.sqrt(max(1, int(formula_proj.shape[-1])))), 1e-6)
        local_scores = torch.einsum('bmd,bed->bme', formula_proj, edge_proj) / dscale
        local_scores = (
            local_scores
            + self.local_formula_bias(formulae_encoded).squeeze(-1).unsqueeze(-1)
            + self.local_edge_bias(edge_hidden).squeeze(-1).unsqueeze(1)
        )

        valid_pairs = edge_mask.unsqueeze(1) & (formulae_mask > 0).unsqueeze(-1)
        if endpoint_elem_pack is not None and endpoint_elem_dim > 0:
            formula_elem = (possible_formulae_raw[:, :, :endpoint_elem_dim] > 0).float()
            edge_elem = (endpoint_elem_pack[:, :, :endpoint_elem_dim] > 0).float()
            compat_hits = torch.einsum('bmk,bek->bme', formula_elem, edge_elem)
            valid_pairs = valid_pairs & (compat_hits > 0)

        local_scores = local_scores.masked_fill(~valid_pairs, torch.finfo(local_scores.dtype).min)
        local_logits = torch.logsumexp(local_scores, dim=-1)
        valid_formula_rows = valid_pairs.any(dim=-1)
        local_logits = torch.where(valid_formula_rows, local_logits, torch.zeros_like(local_logits))
        local_logits = local_logits * formulae_mask.float()

        stats = {
            'edge_count_mean': float(edge_mask.float().sum(dim=1).mean().item()),
            'compat_ratio': float(valid_pairs.float().mean().item()),
            'valid_formula_ratio': float(valid_formula_rows.float().mean().item()),
        }
        return local_logits, stats

    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity, **kwargs):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
            computed global embedding for molecule
        vert_feat_mask : BATCH_N x ATOM_N  
            input mask for valid atoms

        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
            'formulae_features'
        formulae_peaks_mass_idx : BATCH_N x MAX_FORMULAE_N x NUM_MASSES 
         formulae_peaks_intensity, 
        
        
        vert_element_oh: BATCH_N x MAX_ATOM_N=32 x MAX_ELEMENT_N=8
            one-hot encoding of which element corresponds to which vertex
        adj_oh: BATCH_N x N_CHANNELS x MAX_ATOM_N x MAX_ATOM_N
            one-hot adjacency matrix
        formula_frag_count:
            ?
        """

        # Stage A: sanitize inputs and build masks/features for candidate formulas.
        BATCH_N = vert_feat_in.shape[0]
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        formula_n = possible_formulae.shape[1]
        formulae_mask = self._to_formulae_mask(kwargs.get('formulae_mask', None), BATCH_N, formula_n, vert_feat_in.device)
        formulae_peaks = kwargs.get('formulae_peaks', None)
        formulae_peaks_official_idx = kwargs.get('formulae_peaks_official_idx', None)
        if formulae_peaks_official_idx is not None and (not torch.is_tensor(formulae_peaks_official_idx)):
            try:
                formulae_peaks_official_idx = torch.as_tensor(formulae_peaks_official_idx)
            except Exception:
                formulae_peaks_official_idx = None
        if torch.is_tensor(formulae_peaks_official_idx):
            formulae_peaks_official_idx = formulae_peaks_official_idx.to(vert_feat_in.device)
        formulae_aux_feat = self._to_candidate_feat_3d(
            kwargs.get('formulae_aux_feat', None),
            BATCH_N,
            formula_n,
            self.formula_aux_feat_in,
            vert_feat_in.device,
        )

        # Stage B: encode candidate formulas into model latent space.
        possible_formulae_raw = possible_formulae.float()
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        if self.formula_oh_normalize:
            possible_formulae = possible_formulae / (possible_formulae.sum(axis=2).unsqueeze(2) + 1e-8)
            
        # Stage C: attention from molecule-graph representation to each candidate formula.
        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(masked_vert_feat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        atom_mask = vert_mask_in > 0
        if self.rollback_plain_attention:
            # Near-original attention path: direct dot-product between formula and vertex embeddings.
            dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)
            dot_prod = dot_prod.masked_fill(~atom_mask.unsqueeze(-1), torch.finfo(dot_prod.dtype).min)
            weighting = torch.softmax(dot_prod, dim=1)
            weighting = weighting * atom_mask.unsqueeze(-1).to(weighting.dtype)
            weighting = weighting / (weighting.sum(dim=1, keepdim=True) + 1e-8)
            vert_att_reduce = torch.bmm(weighting.transpose(1, 2), masked_vert_feat.float())
            if self.rollback_plain_attention_no_fuse:
                candidate_x = vert_att_reduce
            else:
                candidate_x_input = torch.cat([vert_att_reduce, formulae_encoded], dim=-1)
                candidate_x = self.candidate_fuse(candidate_x_input)
        else:
            q = self.formula_q(formulae_encoded)
            k = self.vert_k(vert_encoded)
            v = self.vert_v(masked_vert_feat.float())
            attn_scale = max(float(np.sqrt(max(1, int(q.shape[-1])))), 1e-6)
            attn_logits = torch.einsum("bmd,bad->bam", q, k) / attn_scale
            attn_logits = attn_logits / self.candidate_attn_temperature
            attn_logits = attn_logits.masked_fill(~atom_mask.unsqueeze(-1), torch.finfo(attn_logits.dtype).min)

            # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
            weighting = torch.softmax(attn_logits, dim=1)
            weighting = weighting * atom_mask.unsqueeze(-1).to(weighting.dtype)
            weighting = weighting / (weighting.sum(dim=1, keepdim=True) + 1e-8)

            # Equivalent to summing masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2) over atoms,
            # but avoids creating a huge 4D temporary tensor.
            vert_att_reduce = torch.bmm(weighting.transpose(1, 2), v)

            # [步骤3] 通过注意力分数，把整个分子的原子特征浓缩成了特定于该“候选分子式”的隐特征
            # 并显式拼接 formula embedding，避免 candidate 表征在该步骤被过度抹平。
            candidate_x_input = torch.cat([vert_att_reduce, formulae_encoded], dim=-1)
            candidate_x = self.candidate_fuse(candidate_x_input)
        cond = torch.zeros((BATCH_N, self.g_feat_in), dtype=torch.float32, device=vert_feat_in.device)

        atom_mask_f = atom_mask.to(masked_vert_feat.dtype).unsqueeze(-1)
        mol_embed = (masked_vert_feat.float() * atom_mask_f).sum(dim=1) / atom_mask_f.sum(dim=1).clamp_min(1.0)

        if getattr(self, 'use_msms_conditioning', False):
            ce_val = kwargs.get('ce', None)
            adduct_val = kwargs.get('adduct', None)
            precursor_mz_val = kwargs.get('precursor_mz', None)
            instrument_val = kwargs.get('instrument_type', None)
            ms_level_val = kwargs.get('ms_level', None)
            ce_missing_val = kwargs.get('ce_missing', None)
            adduct_missing_val = kwargs.get('adduct_missing', None)
            precursor_missing_val = kwargs.get('precursor_mz_missing', None)
            instrument_missing_val = kwargs.get('instrument_missing', None)
            ms_level_missing_val = kwargs.get('ms_level_missing', None)

            ce_2d = self._to_2d_float(ce_val, BATCH_N, vert_feat_in.device)
            precursor_2d = self._to_2d_float(precursor_mz_val, BATCH_N, vert_feat_in.device)
            adduct_idx = self._to_adduct_index(adduct_val, BATCH_N, vert_feat_in.device)
            instrument_idx = self._to_instrument_index(instrument_val, BATCH_N, vert_feat_in.device)
            ms_level_idx = self._to_ms_level_index(ms_level_val, BATCH_N, vert_feat_in.device)

            ce_inferred_missing = torch.logical_or(~torch.isfinite(ce_2d.squeeze(-1)), ce_2d.squeeze(-1) <= 0.0)
            precursor_inferred_missing = torch.logical_or(~torch.isfinite(precursor_2d.squeeze(-1)), precursor_2d.squeeze(-1) <= 0.0)
            adduct_inferred_missing = adduct_idx == 0
            instrument_inferred_missing = instrument_idx == 0
            ms_level_inferred_missing = ms_level_idx == 0

            ce_missing_idx = self._resolve_missing_index(ce_missing_val, ce_inferred_missing, BATCH_N, vert_feat_in.device)
            adduct_missing_idx = self._resolve_missing_index(adduct_missing_val, adduct_inferred_missing, BATCH_N, vert_feat_in.device)
            precursor_missing_idx = self._resolve_missing_index(precursor_missing_val, precursor_inferred_missing, BATCH_N, vert_feat_in.device)
            instrument_missing_idx = self._resolve_missing_index(instrument_missing_val, instrument_inferred_missing, BATCH_N, vert_feat_in.device)
            ms_level_missing_idx = self._resolve_missing_index(ms_level_missing_val, ms_level_inferred_missing, BATCH_N, vert_feat_in.device)

            ce_present = (1.0 - ce_missing_idx.float()).unsqueeze(1)
            adduct_present = (1.0 - adduct_missing_idx.float()).unsqueeze(1)
            precursor_present = (1.0 - precursor_missing_idx.float()).unsqueeze(1)
            instrument_present = (1.0 - instrument_missing_idx.float()).unsqueeze(1)
            ms_level_present = (1.0 - ms_level_missing_idx.float()).unsqueeze(1)
            ce_missing_mask = ce_missing_idx.float().unsqueeze(1)
            adduct_missing_mask = adduct_missing_idx.float().unsqueeze(1)
            precursor_missing_mask = precursor_missing_idx.float().unsqueeze(1)
            instrument_missing_mask = instrument_missing_idx.float().unsqueeze(1)
            ms_level_missing_mask = ms_level_missing_idx.float().unsqueeze(1)


            # [步骤4] (MS/MS 独有逻辑)
            # 把 碰撞能量(CE)+前体离子质量+加合物类型 提取特征并合并起来
            cond = (
                self.ce_proj(ce_2d) * ce_present
                + self.precursor_proj(precursor_2d) * precursor_present
                + self.adduct_proj(self.adduct_emb(adduct_idx)) * adduct_present
                + self.ce_missing_emb(ce_missing_idx) * ce_missing_mask
                + self.precursor_missing_emb(precursor_missing_idx) * precursor_missing_mask
                + self.adduct_missing_emb(adduct_missing_idx) * adduct_missing_mask
            )
            if self.instrument_emb is not None:
                cond = (
                    cond
                    + self.instrument_proj(self.instrument_emb(instrument_idx)) * instrument_present
                    + self.instrument_missing_emb(instrument_missing_idx) * instrument_missing_mask
                )
            if self.ms_level_emb is not None:
                cond = (
                    cond
                    + self.ms_level_proj(self.ms_level_emb(ms_level_idx)) * ms_level_present
                    + self.ms_level_missing_emb(ms_level_missing_idx) * ms_level_missing_mask
                )
            if self.mol_feat_proj is not None:
                mol_feat_val = kwargs.get('mol_feat', None)
                mol_feat_2d = self._to_feature_2d(mol_feat_val, BATCH_N, self.mol_feat_in, vert_feat_in.device)
                cond = cond + self.mol_feat_proj(mol_feat_2d)

        # Stage D: optional selector head (train-time KL target + optional hard top-k prune).
        simple_mode = bool(self.simple_score_head)
        use_selector_now = bool(self.use_formula_selector and (not (simple_mode and self.simple_disable_selector)))
        use_ranking_now = bool(self.use_formula_ranking and (not (simple_mode and self.simple_disable_ranking)))
        use_local_break_now = bool(self.use_local_break_head and (not (simple_mode and self.simple_disable_local_break)))
        use_precursor_hard_mask_now = bool(self.use_precursor_hard_mask and (not (simple_mode and self.simple_disable_precursor_hard_mask)))
        use_mass_shift_now = bool(self.use_mass_shift_head and (not (simple_mode and self.simple_disable_mass_shift)))

        selector_loss = None
        selector_topk_idx = None
        selector_logits_for_diag = None
        selector_logits_for_exact = None
        selector_logits_raw = None
        ranking_logits_for_diag = None
        ranking_logits_for_exact = None
        ranking_logits_raw = None
        mass_shift_probs = None
        mass_shift_reg = None
        if use_selector_now and self.selector_mlp is not None and formulae_aux_feat is not None:
            # Selector sees the per-candidate graph embedding and the explicit MS/MS condition separately.
            selector_input = torch.cat(
                [formulae_aux_feat, candidate_x, cond.unsqueeze(1).expand(-1, formula_n, -1)],
                dim=-1,
            )
            if self.selector_norm is not None:
                selector_input = self.selector_norm(selector_input)
            selector_logits = self.selector_mlp(selector_input).squeeze(-1) / self.selector_temperature
            selector_logits = selector_logits.masked_fill(formulae_mask <= 0, torch.finfo(selector_logits.dtype).min)
            selector_logits_raw = selector_logits
            selector_logits_for_diag = selector_logits
            selector_logits_for_exact = selector_logits

            true_spect = kwargs.get('true_spect', None)
            selector_loss_enabled = kwargs.get('selector_loss_enabled', None)
            selector_target_mode = str(kwargs.get('selector_target_mode', self.selector_target_mode)).strip().lower()
            if selector_target_mode == '':
                selector_target_mode = 'exact_official'
            if selector_loss_enabled is None:
                selector_loss_enabled = bool(self.selector_loss_weight > 0)
            else:
                selector_loss_enabled = bool(selector_loss_enabled)

            if self.training and selector_loss_enabled and true_spect is not None:
                # selector_loss is KL(pred_selector || overlap_target):
                # candidates whose theoretical peaks overlap true spectrum get larger target mass.
                true_spect = self._to_feature_2d(true_spect, BATCH_N, self.spect_bin.get_num_bins(), vert_feat_in.device)
                peak_mass_idx = formulae_peaks_mass_idx.long()
                valid_peak = (peak_mass_idx >= 0) & (peak_mass_idx < self.spect_bin.get_num_bins())
                peak_mass_idx = peak_mass_idx.clamp(0, self.spect_bin.get_num_bins() - 1)
                true_at_peak_flat = torch.gather(true_spect, 1, peak_mass_idx.reshape(BATCH_N, -1))
                candidate_overlap_coarse = (
                    true_at_peak_flat.reshape(BATCH_N, formula_n, -1)
                    * formulae_peaks_intensity.float()
                    * valid_peak.float()
                ).sum(dim=-1)
                candidate_overlap_coarse = candidate_overlap_coarse * formulae_mask

                candidate_overlap = candidate_overlap_coarse
                if selector_target_mode in ('exact_official', 'official_exact', 'exact'):
                    candidate_overlap_exact, exact_has_raw = self._compute_candidate_overlap_exact_official(
                        spect_raw=kwargs.get('spect_raw', None),
                        formulae_peaks_official_idx=kwargs.get('formulae_peaks_official_idx', None),
                        formulae_peaks_intensity=kwargs.get('formulae_peaks_official_intensity', formulae_peaks_intensity),
                        formulae_mask=formulae_mask,
                        batch_n=BATCH_N,
                        formula_n=formula_n,
                        device=vert_feat_in.device,
                    )
                    if candidate_overlap_exact is not None:
                        if exact_has_raw is not None and (not bool(torch.all(exact_has_raw).item())):
                            candidate_overlap = torch.where(
                                exact_has_raw.unsqueeze(-1),
                                candidate_overlap_exact,
                                candidate_overlap_coarse,
                            )
                        else:
                            candidate_overlap = candidate_overlap_exact
                overlap_sum = candidate_overlap.sum(dim=-1, keepdim=True)
                valid_sum = formulae_mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
                target_probs = torch.where(
                    overlap_sum > 0,
                    candidate_overlap / overlap_sum.clamp_min(1e-8),
                    formulae_mask / valid_sum,
                )
                selector_loss = F.kl_div(F.log_softmax(selector_logits, dim=-1), target_probs, reduction='batchmean')

            if self.selector_topk > 0 and self.selector_topk < formula_n:
                topk = int(min(self.selector_topk, formula_n))
                selector_topk_idx = torch.topk(selector_logits, topk, dim=-1).indices
                possible_formulae = self._gather_formula_candidates(possible_formulae, selector_topk_idx)
                possible_formulae_raw = self._gather_formula_candidates(possible_formulae_raw, selector_topk_idx)
                formulae_encoded = self._gather_formula_candidates(formulae_encoded, selector_topk_idx)
                candidate_x = self._gather_formula_candidates(candidate_x, selector_topk_idx)
                formulae_peaks_mass_idx = self._gather_formula_candidates(formulae_peaks_mass_idx, selector_topk_idx)
                formulae_peaks_intensity = self._gather_formula_candidates(formulae_peaks_intensity, selector_topk_idx)
                if formulae_peaks is not None:
                    formulae_peaks = self._gather_formula_candidates(formulae_peaks, selector_topk_idx)
                if torch.is_tensor(formulae_peaks_official_idx):
                    formulae_peaks_official_idx = self._gather_formula_candidates(formulae_peaks_official_idx, selector_topk_idx)
                formulae_mask = self._gather_formula_candidates(formulae_mask, selector_topk_idx)
                formulae_aux_feat = self._gather_formula_candidates(formulae_aux_feat, selector_topk_idx)
                selector_logits_for_exact = self._gather_formula_candidates(selector_logits_for_exact, selector_topk_idx)
                formula_n = possible_formulae.shape[1]

        # Stage D2: optional independent ranking head for exact-candidate ordering.
        if use_ranking_now and self.ranking_mlp is not None:
            ranking_aux_feat = formulae_aux_feat
            if self.formula_aux_feat_in > 0 and ranking_aux_feat is None:
                ranking_aux_feat = self._to_candidate_feat_3d(
                    None,
                    BATCH_N,
                    formula_n,
                    self.formula_aux_feat_in,
                    vert_feat_in.device,
                )

            ranking_parts = []
            if self.formula_aux_feat_in > 0 and ranking_aux_feat is not None:
                ranking_parts.append(ranking_aux_feat)
            ranking_parts.append(candidate_x)
            ranking_parts.append(cond.unsqueeze(1).expand(-1, formula_n, -1))
            ranking_input = torch.cat(ranking_parts, dim=-1)
            if self.ranking_norm is not None:
                ranking_input = self.ranking_norm(ranking_input)

            ranking_logits = self.ranking_mlp(ranking_input).squeeze(-1) / self.ranking_temperature
            ranking_logits = ranking_logits.masked_fill(formulae_mask <= 0, torch.finfo(ranking_logits.dtype).min)
            ranking_logits_raw = ranking_logits
            ranking_logits_for_diag = ranking_logits
            ranking_logits_for_exact = ranking_logits

            if self.selector_topk > 0 and self.selector_topk < ranking_logits.shape[1] and selector_topk_idx is not None:
                ranking_logits_for_exact = self._gather_formula_candidates(ranking_logits_for_exact, selector_topk_idx)

        # Optional physical constraint for MS/MS: drop candidate peaks above precursor bin (+tol).
        hard_mask = None
        if use_precursor_hard_mask_now:
            precursor_raw = kwargs.get('precursor_mz', None)
            if precursor_raw is not None:
                precursor_2d = self._to_2d_float(precursor_raw, BATCH_N, vert_feat_in.device)
                apply_mask = precursor_2d.squeeze(-1) > 0
                if bool(apply_mask.any().item()):
                    try:
                        first_bin = float(getattr(self.spect_bin, 'first_bin_center', 1.0))
                        bin_width = float(getattr(self.spect_bin, 'bin_width', 1.0))
                    except Exception:
                        first_bin = 1.0
                        bin_width = 1.0

                    spect_bin_n_local = int(self.spect_bin.get_num_bins())
                    precursor_bin = torch.round((precursor_2d.squeeze(-1) - first_bin) / max(bin_width, 1e-8)).long()
                    precursor_bin = torch.clamp(precursor_bin, 0, max(0, spect_bin_n_local - 1))

                    peak_idx = formulae_peaks_mass_idx.long()
                    base_valid = (peak_idx >= 0) & (peak_idx < spect_bin_n_local)
                    precursor_limit = (precursor_bin + int(self.precursor_hard_mask_tol)).view(BATCH_N, 1, 1)
                    peak_ok_masked = base_valid & (peak_idx <= precursor_limit)
                    peak_ok = torch.where(apply_mask.view(BATCH_N, 1, 1), peak_ok_masked, base_valid)
                    hard_mask = peak_ok

                    candidate_has_peak = peak_ok.any(dim=-1)
                    has_any_candidate = candidate_has_peak.any(dim=-1)
                    if bool((~has_any_candidate).any().item()):
                        # Fallback for pathological rows: do not fully zero-out all candidates.
                        fallback_rows = (~has_any_candidate).view(BATCH_N, 1, 1)
                        peak_ok = torch.where(fallback_rows, base_valid, peak_ok)
                        candidate_has_peak = peak_ok.any(dim=-1)

                    formulae_peaks_intensity = formulae_peaks_intensity * peak_ok.to(formulae_peaks_intensity.dtype)
                    formulae_mask = formulae_mask * candidate_has_peak.to(formulae_mask.dtype)

                    has_valid_formula = (formulae_mask.sum(dim=1, keepdim=True) > 0)
                    if not torch.all(has_valid_formula):
                        formulae_mask = torch.where(has_valid_formula, formulae_mask, torch.ones_like(formulae_mask))

        # Stage E: optional mass-shift head to model integer-bin calibration offsets.
        if use_mass_shift_now and self.mass_shift_mlp is not None:
            if formulae_aux_feat is None and self.formula_aux_feat_in > 0:
                formulae_aux_feat = self._to_candidate_feat_3d(
                    None,
                    BATCH_N,
                    formula_n,
                    self.formula_aux_feat_in,
                    vert_feat_in.device,
                )
            shift_input_parts = [candidate_x, cond.unsqueeze(1).expand(-1, formula_n, -1)]
            if self.formula_aux_feat_in > 0 and formulae_aux_feat is not None:
                shift_input_parts = [formulae_aux_feat] + shift_input_parts
            shift_input = torch.cat(shift_input_parts, dim=-1)
            if self.mass_shift_norm is not None:
                shift_input = self.mass_shift_norm(shift_input)

            mass_shift_logits = self.mass_shift_mlp(shift_input) / self.mass_shift_temperature
            mass_shift_logits = mass_shift_logits.masked_fill((formulae_mask <= 0).unsqueeze(-1), torch.finfo(mass_shift_logits.dtype).min)
            mass_shift_probs = F.softmax(mass_shift_logits, dim=-1)
            mass_shift_probs = mass_shift_probs * formulae_mask.unsqueeze(-1)

            # Normalize each valid candidate's shift distribution and zero-out invalid rows.
            shift_norm = mass_shift_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            mass_shift_probs = torch.where(
                (formulae_mask > 0).unsqueeze(-1),
                mass_shift_probs / shift_norm,
                torch.zeros_like(mass_shift_probs),
            )

            offsets_abs = self.mass_shift_offsets.to(mass_shift_probs.dtype).abs().view(1, 1, -1)
            expected_abs_shift = (mass_shift_probs * offsets_abs).sum(dim=-1)
            mass_shift_reg = (expected_abs_shift * formulae_mask).sum() / formulae_mask.sum().clamp_min(1.0)

        candidate_x_std_mean = None
        if torch.is_tensor(candidate_x):
            candidate_mask = (formulae_mask > 0).to(candidate_x.dtype).unsqueeze(-1)
            denom = candidate_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            candidate_mean = (candidate_x * candidate_mask).sum(dim=1, keepdim=True) / denom
            candidate_var = ((candidate_x - candidate_mean) ** 2 * candidate_mask).sum(dim=1, keepdim=True) / denom
            candidate_x_std_mean = torch.sqrt(candidate_var + 1e-8).mean()

        local_formula_logits = None
        local_break_stats = None
        if use_local_break_now:
            local_formula_logits, local_break_stats = self._compute_local_break_logits(
                masked_vert_feat=masked_vert_feat,
                vert_mask_in=vert_mask_in,
                formulae_encoded=formulae_encoded,
                possible_formulae_raw=possible_formulae_raw,
                cond=cond,
                formulae_mask=formulae_mask,
                adj_oh=kwargs.get('adj_oh', None),
                vert_element_oh=kwargs.get('vert_element_oh', None),
            )

        formulae_aux_for_score = None
        if self.formula_aux_feat_in > 0:
            if formulae_aux_feat is None:
                formulae_aux_for_score = self._to_candidate_feat_3d(
                    None,
                    BATCH_N,
                    formula_n,
                    self.formula_aux_feat_in,
                    vert_feat_in.device,
                )
            else:
                formulae_aux_for_score = formulae_aux_feat

        if simple_mode:
            peak_summary_feat = torch.zeros(
                (BATCH_N, formula_n, self.peak_summary_feat_dim),
                dtype=candidate_x.dtype,
                device=vert_feat_in.device,
            )
            candidate_pool_summary = torch.zeros(
                (BATCH_N, self.peak_summary_feat_dim),
                dtype=candidate_x.dtype,
                device=vert_feat_in.device,
            )
            peak_summary_repr = torch.zeros(
                (BATCH_N, formula_n, self.peak_summary_dim),
                dtype=candidate_x.dtype,
                device=vert_feat_in.device,
            )
        else:
            use_official_idx_for_score = torch.is_tensor(formulae_peaks_official_idx)
            peak_idx_for_score = formulae_peaks_official_idx if use_official_idx_for_score else formulae_peaks_mass_idx
            peak_int_for_score = kwargs.get('formulae_peaks_official_intensity', formulae_peaks_intensity) if use_official_idx_for_score else formulae_peaks_intensity
            peak_summary_feat, candidate_pool_summary = self._build_candidate_peak_summary(
                peak_idx=peak_idx_for_score,
                peak_int=peak_int_for_score,
                formulae_mask=formulae_mask,
                precursor_mz=kwargs.get('precursor_mz', None),
                use_official_idx=bool(use_official_idx_for_score),
            )
            peak_summary_repr = self.peak_summary_mlp(peak_summary_feat)

        # Stage F: either current main score head or rollback legacy GRU+MLP score head.
        score_input = candidate_x
        base_in = torch.cat([formulae_encoded, vert_att_reduce], dim=-1)
        base_in = self.base_score_norm(base_in)
        base_h = F.relu(self.base_score_l1(base_in))
        base_h = F.relu(self.base_score_l2(base_h))
        base_score_raw = self.base_score_out(base_h).squeeze(-1)

        if simple_mode:
            formulae_scores_raw = base_score_raw
            if self.use_msms_conditioning and self.simple_score_use_cond_bias:
                cond_latent = torch.tanh(self.base_cond_proj(cond))
                cond_bias = torch.einsum("bmd,bd->bm", formulae_encoded, cond_latent)
                formulae_scores_raw = formulae_scores_raw + (float(self.simple_score_cond_scale) * cond_bias)
            global_formulae_scores_raw = formulae_scores_raw
            score_input = base_in
        elif self.rollback_legacy_score_head:
            x = candidate_x
            pf_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
            x_flat = x.reshape(-1, x.shape[-1])
            for l in self.combine_layers:
                x_flat = l(pf_flat, x_flat)
            x = x_flat.reshape(BATCH_N, formula_n, -1)
            x = F.relu(self.f_combine_l1(x))
            x = self.f_combine_l2(x)
            x = self.norm2(x)
            global_formulae_scores_raw = self.f_combine_score(x).squeeze(2)
            score_input = x
        else:
            cond_expand = cond.unsqueeze(1).expand(-1, formula_n, -1)
            score_parts = [candidate_x, formulae_encoded, cond_expand]
            if self.formula_aux_feat_in > 0:
                if self.use_peak_aware_score_input and formulae_aux_for_score is not None:
                    score_parts.append(formulae_aux_for_score)
                else:
                    score_parts.append(torch.zeros((BATCH_N, formula_n, self.formula_aux_feat_in), dtype=candidate_x.dtype, device=vert_feat_in.device))

            if self.use_peak_aware_score_input:
                score_parts.append(peak_summary_repr)
            else:
                score_parts.append(torch.zeros((BATCH_N, formula_n, self.peak_summary_dim), dtype=candidate_x.dtype, device=vert_feat_in.device))
            score_input = torch.cat(score_parts, dim=-1)
            score_hidden = self.main_score_mlp(score_input)
            global_formulae_scores_raw = (
                self.main_score_out(score_hidden).squeeze(-1)
                + self.score_skip_candidate(candidate_x).squeeze(-1)
                + self.score_skip_formula(formulae_encoded).squeeze(-1)
            )
            formulae_scores_raw = global_formulae_scores_raw

        if torch.is_tensor(local_formula_logits):
            formulae_scores_raw = formulae_scores_raw + (float(self.local_break_logit_weight) * local_formula_logits)

        debug_formulae_branch_stats = None
        if os.environ.get("DEBUG_FORMULAE_BRANCH_STATS", "0") == "1":
            # Branch stats help determine whether per-formula pathways already collapsed before scoring.
            def _masked_formula_tensor_stats(tensor3d, mask2d):
                if (not torch.is_tensor(tensor3d)) or tensor3d.dim() != 3:
                    return None
                t = tensor3d.detach().float()
                m = None
                if torch.is_tensor(mask2d):
                    m = mask2d.detach().float()
                    if m.dim() > 2:
                        m = m.reshape(m.shape[0], -1)
                abs_mean = float(t.abs().mean().item())
                global_std = float(t.std(unbiased=False).item())
                formula_std_vals = []
                for bi in range(int(t.shape[0])):
                    tb = t[bi]
                    if m is not None and bi < int(m.shape[0]):
                        mb = m[bi]
                        if int(mb.shape[0]) != int(tb.shape[0]):
                            use_n = min(int(mb.shape[0]), int(tb.shape[0]))
                            tb = tb[:use_n]
                            mb = mb[:use_n]
                        valid = mb > 0
                        if bool(valid.any().item()):
                            tb = tb[valid]
                    if int(tb.shape[0]) <= 1:
                        formula_std_vals.append(0.0)
                    else:
                        formula_std_vals.append(float(tb.std(dim=0, unbiased=False).mean().item()))
                if len(formula_std_vals) == 0:
                    formula_std_vals = [0.0]
                arr = np.asarray(formula_std_vals, dtype=np.float64)
                return {
                    'abs_mean': abs_mean,
                    'std': global_std,
                    'across_formula_std_mean': float(np.mean(arr)),
                    'across_formula_std_p50': float(np.percentile(arr, 50)),
                }

            debug_formulae_branch_stats = {
                'formula_embed': _masked_formula_tensor_stats(formulae_encoded, formulae_mask),
                'candidate_x': _masked_formula_tensor_stats(candidate_x, formulae_mask),
                'score_input': _masked_formula_tensor_stats(score_input, formulae_mask),
            }

        invalid_formula = formulae_mask <= 0
        if self.prob_softmax:
            formulae_scores = formulae_scores_raw.masked_fill(invalid_formula, torch.finfo(formulae_scores_raw.dtype).min)
        else:
            formulae_scores = formulae_scores_raw * formulae_mask
        
        # formulae_scores are per-candidate logits.
        # formulae_probs are the actual candidate weights used for spectral projection.
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)
        formulae_probs = formulae_probs * formulae_mask

        if os.environ.get("DEBUG_FORMULAE_PROBS", "0") == "1" and (not hasattr(self, "_dbg_formulae_probs_done")):
            p = formulae_probs.detach()
            s = formulae_scores.detach()
            print(
                "[DEBUG_FORMULAE_PROBS]",
                "scores_min=", float(s.min().item()),
                "scores_max=", float(s.max().item()),
                "scores_mean=", float(s.mean().item()),
                "probs_min=", float(p.min().item()),
                "probs_max=", float(p.max().item()),
                "probs_mean=", float(p.mean().item()),
                "gt1e-2=", float((p > 1e-2).float().mean().item()),
                "gt1e-3=", float((p > 1e-3).float().mean().item()),
                "gt1e-4=", float((p > 1e-4).float().mean().item()),
            )
            if p.dim() >= 2 and p.shape[0] > 0 and p.shape[1] > 0:
                topv, topi = torch.topk(p[0], k=min(20, p.shape[1]), dim=-1)
                print("[DEBUG_FORMULAE_PROBS_TOP20_IDX]", topi.detach().cpu().tolist())
                print("[DEBUG_FORMULAE_PROBS_TOP20_VAL]", [float(v) for v in topv.detach().cpu().tolist()])
            self._dbg_formulae_probs_done = True

        if (
            os.environ.get("DEBUG_RANKING", "0") == "1"
            and torch.is_tensor(ranking_logits_for_diag)
            and (not hasattr(self, "_dbg_ranking_done"))
        ):
            r = ranking_logits_for_diag.detach()
            print(
                "[DEBUG_RANKING]",
                "logits_min=", float(r.min().item()),
                "logits_max=", float(r.max().item()),
                "logits_mean=", float(r.mean().item()),
                "logits_std=", float(r.std().item()),
            )
            if r.dim() >= 2 and r.shape[0] > 0 and r.shape[1] > 0:
                topv, topi = torch.topk(r[0], k=min(20, r.shape[1]), dim=-1)
                print("[DEBUG_RANKING_TOP20_IDX]", topi.detach().cpu().tolist())
                print("[DEBUG_RANKING_TOP20_VAL]", [float(v) for v in topv.detach().cpu().tolist()])
            self._dbg_ranking_done = True

        # Stage G: project candidate probabilities back to dense spectrum bins.
        # This is the key mapping from "formula-level score" to final m/z intensity vector.
        spect_bin_n = self.spect_bin.get_num_bins()
        proj_impl = os.environ.get('FORMULA_PROJ_IMPL', 'scatter').strip().lower()

        if proj_impl == 'sparse' and mass_shift_probs is None:
            # Backward-compatible implementation (slower, but useful for debugging parity).
            out_y = []
            for batch_i in range(BATCH_N):
                sparse_mat_matrix = peak_indices_intensities_to_sparse_matrix(
                    formulae_peaks_mass_idx[batch_i],
                    formulae_peaks_intensity[batch_i],
                    spect_bin_n,
                )
                y = sparse_mat_matrix.float() @ formulae_probs[batch_i].float()
                out_y.append(y)
            spect_out = torch.stack(out_y)
        else:
            spect_out = project_formula_probs_to_spectrum_dense(
                formulae_probs,
                formulae_peaks_mass_idx,
                formulae_peaks_intensity,
                spect_bin_n,
                formulae_mask=formulae_mask,
                mass_shift_probs=mass_shift_probs,
                mass_shift_offsets=self.mass_shift_offsets,
            )

        if os.environ.get("DEBUG_SPECT_OUT", "0") == "1" and (not hasattr(self, "_dbg_spect_out_done")):
            x = spect_out.detach()
            print(
                "[DEBUG_SPECT_OUT]",
                "min=", float(x.min().item()),
                "max=", float(x.max().item()),
                "mean=", float(x.mean().item()),
                "nnz>1e-8=", float((x > 1e-8).float().mean().item()),
                "nnz>1e-6=", float((x > 1e-6).float().mean().item()),
                "nnz>1e-4=", float((x > 1e-4).float().mean().item()),
            )
            if x.dim() >= 2 and x.shape[0] > 0 and x.shape[1] > 0:
                topv, topi = torch.topk(x[0], k=min(20, x.shape[1]), dim=-1)
                print("[DEBUG_SPECT_OUT_TOP20_IDX]", topi.detach().cpu().tolist())
                print("[DEBUG_SPECT_OUT_TOP20_VAL]", [float(v) for v in topv.detach().cpu().tolist()])
            self._dbg_spect_out_done = True

        pred_exact_peaks = None
        if formulae_peaks is not None:
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

            pred_exact_peaks = project_formula_probs_to_exact_sparse(
                formulae_probs=formulae_probs,
                formulae_peaks=formulae_peaks,
                formulae_mask=formulae_mask,
                min_formula_prob=exact_min_prob,
                topk_formula=exact_topk,
                ranking_scores=ranking_logits_for_exact if ranking_logits_for_exact is not None else selector_logits_for_exact,
            )

        if (
            os.environ.get("DEBUG_EXACT_PEAKS", "0") == "1"
            and pred_exact_peaks is not None
            and (not hasattr(self, "_dbg_exact_done"))
        ):
            for bi in range(min(2, len(pred_exact_peaks))):
                arr = pred_exact_peaks[bi]
                if arr is None:
                    print(f"[DEBUG_EXACT_PEAKS] sample={bi} empty")
                    continue
                try:
                    arr_np = np.asarray(arr, dtype=np.float32).reshape(-1, 2)
                except Exception:
                    print(f"[DEBUG_EXACT_PEAKS] sample={bi} malformed")
                    continue
                if arr_np.size == 0:
                    print(f"[DEBUG_EXACT_PEAKS] sample={bi} empty")
                    continue
                mz = arr_np[:, 0]
                inten = arr_np[:, 1]
                print(
                    f"[DEBUG_EXACT_PEAKS] sample={bi}",
                    "n=", int(arr_np.shape[0]),
                    "mz_min=", float(np.min(mz)),
                    "mz_p10=", float(np.percentile(mz, 10)),
                    "mz_p50=", float(np.percentile(mz, 50)),
                    "mz_p90=", float(np.percentile(mz, 90)),
                    "mz_max=", float(np.max(mz)),
                    "int_sum=", float(np.sum(inten)),
                )
                topi = np.argsort(-inten)[:20]
                print(f"[DEBUG_EXACT_PEAKS_TOP20_MZ] sample={bi}", mz[topi].tolist())
                print(f"[DEBUG_EXACT_PEAKS_TOP20_INT] sample={bi}", inten[topi].tolist())
            self._dbg_exact_done = True

        if getattr(self, 'normalize_1_output', False):
            # L1 normalize
            spect_out = spect_out / (torch.sum(torch.abs(spect_out), dim=1, keepdim=True) + 1e-6)

        support_os_input = torch.cat([cond, mol_embed, candidate_pool_summary], dim=-1)

        # Stage H: return training/evaluation artifacts used by trainer diagnostics.
        out_dict = {
            'spect_out': spect_out,
            'formulae_probs': formulae_probs,
            'formulae_scores': formulae_scores,
            'global_formulae_scores_raw': global_formulae_scores_raw,
            'local_formula_logits': local_formula_logits,
            'local_break_logits_raw': local_formula_logits,
            'formulae_scores_raw': formulae_scores_raw,
            'base_score_raw': base_score_raw,
            'formulae_encoded': formulae_encoded,
            'vert_att_reduce': vert_att_reduce,
            'candidate_x': candidate_x,
            'peak_summary_feat': peak_summary_feat,
            'peak_summary_repr': peak_summary_repr,
            'formulae_aux_feat': formulae_aux_feat,
            'pred_exact_peaks': pred_exact_peaks,
            'selector_logits': selector_logits_for_diag,
            'selector_logits_raw': selector_logits_raw,
            'ranking_logits': ranking_logits_for_diag,
            'ranking_logits_raw': ranking_logits_raw,
            'cond_embed': support_os_input,
            'cond_only_embed': cond,
            'mol_embed': mol_embed,
            'candidate_pool_summary': candidate_pool_summary,
            'candidate_x_std_mean': candidate_x_std_mean,
            'hard_mask': hard_mask,
        }
        if isinstance(local_break_stats, dict):
            out_dict['local_break_stats'] = local_break_stats
        if debug_formulae_branch_stats is not None:
            out_dict['formulae_debug_stats'] = debug_formulae_branch_stats
        return out_dict
# Class overview: MolAttentionNetLowRank encapsulates a reusable component in this module.
class MolAttentionNetLowRank(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 mixture_weight_method = 'method1', 
                 rank_n = 1, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetLowRank, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            

        
        self.out_encoding = nn.ModuleList([nn.Sequential(nn.LayerNorm(formula_encoding_n +  g_feat_in),
                                                         nn.Linear(formula_encoding_n +  g_feat_in, internal_d),
                                                         nn.ReLU(),
                                                         nn.Linear(internal_d, internal_d),
                                                         nn.ReLU(), 
                                                         nn.LayerNorm(internal_d)              ,
                                                         nn.Linear(internal_d, 1)) for _ in range(rank_n)])
        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax

        self.mixture_weight_method = mixture_weight_method
        if mixture_weight_method in ['method1', 'method2', 'method3', 'method4']:
            self.mixture_norm = nn.LayerNorm(g_feat_in)
            self.mixture_weights_l1 = nn.Linear(g_feat_in, g_feat_in)
            self.mixture_weights_l2 = nn.Linear(g_feat_in, rank_n)


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        atom_counts = vert_mask_in.sum(dim=1)
        

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        formulae_scores = torch.cat([l(combined) for l in self.out_encoding], -1)
        
        # combined = self.combine_norm(combined)
        # x = F.relu(self.f_combine_l1(combined))
        # x = F.relu(self.f_combine_l2(x))
        # x = self.norm2(x)
        # formulae_scores = self.f_combine_score(x).squeeze(2)

        if self.mixture_weight_method == 'method1':
            mixture_weights = torch.softmax(self.mixture_weights_l1(self.mixture_norm(masked_vert_feat).mean(dim=1)), dim=1)
        elif self.mixture_weight_method == 'method2':
            # vertex feats use max 
            mw = self.mixture_weights_l1(goodmax(self.mixture_norm(masked_vert_feat), dim=1))
            mixture_weights = torch.softmax(mw, dim=1)
        elif self.mixture_weight_method == 'method3':
            # proper weighting of vertex feats?
            mixture_weights = torch.softmax(self.mixture_weights_l1(self.mixture_norm(masked_vert_feat).sum(dim=1) / atom_counts.unsqueeze(-1)), dim=1)
        elif self.mixture_weight_method == 'method4':
            # proper weighting of vertex feats?
            mw = F.relu(self.mixture_weights_l1(masked_vert_feat)).sum(dim=1) / atom_counts.unsqueeze(-1)
            mw = self.mixture_norm(mw)
            mw = self.mixture_weights_l2(mw)
            mixture_weights = torch.softmax(mw, dim=1)

            
        #print("mixture_weights.shape=", mixture_weights.shape, "formulae_scores.shape=", formulae_scores.shape)
        #print("unsqueeze", mixture_weights.unsqueeze(1).shape)
        
        #mixture_weights_expand =  mixture_weights.unsqueeze(1).expand(-1, formulae_scores.shape[1], -1)
        
        formulae_probs = (torch.softmax(formulae_scores, dim=1) * mixture_weights.unsqueeze(1)).sum(dim=2)
        

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return spect_out
    

# Class overview: MolAttentionNetOuter encapsulates a reusable component in this module.
class MolAttentionNetOuter(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 spect_bin_n = 512):

        
        super( MolAttentionNetOuter, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in*2, embedding_key_size)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        self.possible_formulae_bn = nn.BatchNorm1d(formula_encoding_n)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in*2)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in*2, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax
        
    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in, 
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape
        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae_flat_oh = self.possible_formulae_bn(possible_formulae_flat_oh)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        

        vert_feat_pairwise = torch.cat([masked_vert_feat.unsqueeze(1).expand(-1, ATOM_N, -1, -1), 
                                           masked_vert_feat.unsqueeze(2).expand(-1, -1, ATOM_N,  -1), ], -1)

        assert adj_oh.shape == (BATCH_N, 4, ATOM_N, ATOM_N)
        
        adj_oh_collapse = goodmax(adj_oh, dim=1)
        
        
        # encoded inputs
        vert_encoded_pairwise = self.embed_g_feat(vert_feat_pairwise) ## BATCH_N x ATOM_N x F
        
        formulae_encoded = self.embed_formulae_feat(possible_formulae) ## BATCH_N x MAX_FORMULA_N x F
        
        dot_prod = (formulae_encoded.unsqueeze(1).unsqueeze(1) * vert_encoded_pairwise.unsqueeze(3)).sum(dim=-1)
        dot_prod = dot_prod 
        

        dot_prod_flat = dot_prod.reshape(BATCH_N, ATOM_N * ATOM_N, MAX_FORMULAE_N)
        
        weighting_flat = torch.softmax(dot_prod_flat, dim=1)
        weighting = weighting_flat.reshape(BATCH_N, ATOM_N, ATOM_N, MAX_FORMULAE_N)
        

        vert_feat_pairwise = vert_feat_pairwise * adj_oh_collapse.unsqueeze(-1)
        #print('vert_feat_pairwise.shape=', vert_feat_pairwise.shape)

        vert_att_reduce = torch.einsum('nijf,nijd->ndf', vert_feat_pairwise, weighting)
        
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        mass_matrices = [create_mass_matrix_oh(formulae_masses[:, :, i, 0].round().long(),
                                               self.spect_bin_n,
                                               formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]
        mass_matrix = torch.sum(torch.stack(mass_matrices, -1), -1)


        spect_out = torch.einsum("ij,ijk->ik", formulae_probs, mass_matrix)
        return spect_out


    

# Class overview: MolLesion encapsulates a reusable component in this module.
class MolLesion(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 norm = 'layer', 
                 vert_reduce_method = 'mean-naive', 
                 spect_bin_n = 512):

        
        super( MolLesion, self).__init__()

        if norm == 'layer':
            norm_class = nn.LayerNorm
        elif norm == 'batch':
            norm_class = nn.BatchNorm1d
        self.norm = norm_class(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        # self.formulae_norm = nn.LayerNorm(formula_encoding_n)
        # self.formulae_collapse_l = nn.Linear(formula_encoding_n,formula_encoding_n)
        
        
        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = norm_class(internal_d)

        self.combine_norm = norm_class(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax

        self.vert_reduce_method = vert_reduce_method

    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh,  formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        possible_formulae_normed = possible_formulae
        formulae_num_atoms = possible_formulae.sum(dim=-1)

        # f_n = self.formulae_norm(F.relu(self.formulae_collapse_l(possible_formulae_normed )).sum(dim=1))
        # num_formulae =  (formulae_num_atoms > 0).float().sum(dim=1)

        # f_n = f_n / num_formulae.unsqueeze(-1) 
        
        # encoded inputs

        # [步骤1] 这里不使用 formula-aware attention，直接采用简化的分子级聚合。
        _ = self.embed_g_feat(vert_feat_in)
        
        #vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        # vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        if self.vert_reduce_method == 'mean-naive':
            
            vert_att_reduce = masked_vert_feat.mean(dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'sum-naive':
            
            vert_att_reduce = masked_vert_feat.sum(dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'max':
            
            vert_att_reduce = goodmax(masked_vert_feat, dim=1).unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        elif self.vert_reduce_method == 'mean-masked':
            
            vert_att_reduce = masked_vert_feat.sum(dim=1)
            num_vertices = vert_mask_in.sum(dim=1)
            vert_att_reduce = vert_att_reduce / num_vertices.unsqueeze(-1)

            vert_att_reduce = vert_att_reduce.unsqueeze(1).expand(-1, MAX_FORMULAE_N, -1)
        
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x1 = F.relu(self.f_combine_l1(combined))
        x2 = F.relu(self.f_combine_l2(x1))
        #x = self.norm2(x2) + x1
        x = self.norm2(x2)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    
# Class overview: MolCombineFormulaPreReduce encapsulates a reusable component in this module.
class MolCombineFormulaPreReduce(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolCombineFormulaPreReduce, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        
        
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(internal_d)
        self.f_combine_l1 = nn.Linear(internal_d, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size)

        self.vert_f_combine_l1 = nn.Linear(formula_encoding_n + g_feat_in, internal_d)
        self.vert_f_combine_l2 = nn.Linear(internal_d, internal_d)
        
                                           
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)
        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape
        
        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1) # BATCH_N, MAX_FORMULAE_N, FORMULAE_ENCODING
        


        vert_expand = masked_vert_feat.unsqueeze(2).expand(-1, -1, MAX_FORMULAE_N, -1)
        form_expand = possible_formulae.unsqueeze(1).expand(-1, ATOM_N, -1,-1)
        vert_form_cat = torch.cat([vert_expand, form_expand], -1)
        vf1 = F.relu(self.vert_f_combine_l1(vert_form_cat))
        vf2 = F.tanh(self.vert_f_combine_l2(vf1))
        
        

        combined = self.combine_norm(vf2.sum(dim=1))
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return spect_out

    

# Class overview: MolFormulaEltReduce encapsulates a reusable component in this module.
class MolFormulaEltReduce(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True, 
                 spect_bin_n = 512):

        
        super( MolFormulaEltReduce, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        #self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        # self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
        #                                      bias=formula_embed_bias)

        # if not formula_embed_train:
        #     self.embed_formulae_feat.weight.requires_grad = False
        #     self.embed_formulae_feat.bias.requires_grad = False
            

        self.vert_elt_lin = nn.Linear(g_feat_in, g_feat_in)

        self.norm2 = nn.LayerNorm(internal_d)

        self.embed_formulae = nn.Linear(formula_encoding_n, internal_d)
        
        self.combine_norm = nn.LayerNorm(g_feat_in + internal_d)
        self.f_l1 = nn.Linear(g_feat_in + internal_d, internal_d)
        self.f_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x VERT_F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        vert_element_oh : BATCH_N x MAX_FORMULAE_N x ELEMENT_N 

        for this to work FORMULA_ENCODING_N == ELEMENT_N
        """

        BATCH_N, ATOM_N, VERT_F = vert_feat_in.shape

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        _, MAX_FORMULAE_N, FORMULA_ENCODING_N = possible_formulae.shape
        BATCH_N, _, ELEMENT_N = vert_element_oh.shape

        assert ELEMENT_N == FORMULA_ENCODING_N
        
        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae_accum = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)


        vert_feat_contract_by_elt = torch.einsum('ijk,ije->iek', masked_vert_feat, vert_element_oh) # (BATCH_N, ELEMENT_N, F)
        vert_feat_contract_by_elt_mean = vert_feat_contract_by_elt / (vert_element_oh.sum(dim=1).unsqueeze(-1) + 0.001)
        #print(vert_feat_contract_by_elt_mean.shape)
        ## IDEA: Boolean of element counts
        ## IDEA: transforms inside of here
        vert_feat_contract_by_elt_mean = F.tanh(self.vert_elt_lin(vert_feat_contract_by_elt_mean))

        assert vert_feat_contract_by_elt_mean.shape == (BATCH_N, ELEMENT_N, VERT_F)

        ## IDEA: There are a lot of different ways of combining/weighting these things
        # now combine with formula
        formula_feat_contract = torch.einsum('ijk,ikl->ijl', possible_formulae.float(), vert_feat_contract_by_elt_mean)
        assert formula_feat_contract.shape == (BATCH_N, MAX_FORMULAE_N, VERT_F)
        formula_feat_contract = formula_feat_contract / (possible_formulae.float().sum(dim=2).unsqueeze(2) + 1)
        

        combined = self.combine_norm(torch.cat([formula_feat_contract, torch.tanh(self.embed_formulae(possible_formulae_accum))], -1))
        x = F.relu(self.f_l1(combined))
        x = F.relu(self.f_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = torch.softmax(formulae_scores, dim=-1)


        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    
    

# Class overview: MolAttentionNetOHSparseHighway encapsulates a reusable component in this module.
class MolAttentionNetOHSparseHighway(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 graph_layer_n= 4, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparseHighway, self).__init__()

        g_feat_in = g_feat_in * graph_layer_n
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh):
        """
        vert_feat_in : BATCH_N x ATOM_N x F x VERT_LAYER_N
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """
        vert_feat_in = torch.sigmoid(vert_feat_in )
        
        BATCH_N, ATOM_N, VERT_F, VERT_LAYER_N = vert_feat_in.shape
        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1).unsqueeze(-1)        
        vert_feat_in_flat = vert_feat_in.reshape(BATCH_N, ATOM_N, -1)
        masked_vert_feat_flat = masked_vert_feat.reshape(BATCH_N, ATOM_N, -1)
        

        # one-hot encode possible formulae
        _, MAX_FORMULAE_N, FORMULAE_INPUT_INCODING_N = possible_formulae.shape
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs
        vert_encoded = self.embed_g_feat(vert_feat_in_flat)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat_flat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        formulae_num_atoms = possible_formulae.sum(dim=-1)
        assert formulae_num_atoms.shape == (BATCH_N, MAX_FORMULAE_N)
        formulae_present_mask = (formulae_num_atoms > 0).float()
        formulae_scores_masked = formulae_scores + (1-formulae_present_mask)*-100

        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores_masked, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores_masked)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    

# Class overview: MolAttentionNetOHSparseExtraFormulaFeat encapsulates a reusable component in this module.
class MolAttentionNetOHSparseExtraFormulaFeat(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 formula_frag_count = 1,
                 formula_frag_count_bool = False, 
                 spect_bin_n = 512):

        
        super( MolAttentionNetOHSparseExtraFormulaFeat, self).__init__()

        self.formula_frag_count = formula_frag_count
        self.formula_frag_count_bool = formula_frag_count_bool
        
        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        if formula_frag_count > 0:
            formula_encoding_n += formula_frag_count
            
        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_norm = nn.LayerNorm(formula_encoding_n +  g_feat_in)
        self.f_combine_l1 = nn.Linear(formula_encoding_n +  g_feat_in, internal_d)
        self.f_combine_l2 = nn.Linear(internal_d, internal_d)
        self.f_combine_score = nn.Linear(internal_d, 1)


        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        possible_formulae = torch.cat([possible_formulae, formula_frag_count], -1)
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)
        combined = torch.cat([possible_formulae, vert_att_reduce], -1)
        combined = self.combine_norm(combined)
        x = F.relu(self.f_combine_l1(combined))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}


    
    


# Class overview: MolAttentionGRUDifferentSM encapsulates a reusable component in this module.
class MolAttentionGRUDifferentSM(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 internal_d = 512,
                 formulae_prob_transform = 'softmax',
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
                 spect_bin_n = 512):

        
        super( MolAttentionGRUDifferentSM, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n

        if formulae_prob_transform == 'softmax':
            self.formulae_prob_transform = nn.Softmax(dim=-1)
        elif formulae_prob_transform == 'sigsoftmax':
            self.formulae_prob_transform = SigSoftmax(dim=-1)
        elif formulae_prob_transform == 'sigmoid':
            self.formulae_prob_transform = nn.Sigmoid()
        


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)
        
        
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)


        # [步骤3] 通过注意力分数，把整个分子的原子特征浓缩成了特定于该“候选分子式”的隐特征
        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae.reshape(-1, possible_formulae.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        formulae_probs = self.formulae_prob_transform(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}



    
# Class overview: MolAttentionGRUNormalize encapsulates a reusable component in this module.
class MolAttentionGRUNormalize(nn.Module):
    """
    Per-vertex features to sparse points of peak, mass

    Input:
    BATCH_N x ATOM_N x F

    Output:
    BATCH_N x SPECT_BIN

    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, g_feat_in,
                 embedding_key_size = 16,
                 formula_oh_sizes = [20, 20, 20, 20, 20],
                 formula_oh_accum = True,
                 formula_oh_normalize_1=False, 
                 formula_oh_normalize_2=False, 
                 
                 internal_d = 512,
                 prob_softmax = True,
                 g_embed_train = True,
                 g_embed_bias = True, 
                 formula_embed_train= True,
                 formula_embed_bias=True,
                 gru_layer_n = 1,
                 linear_layer_n = 2,
                 spect_bin_n = 512):

        
        super( MolAttentionGRUNormalize, self).__init__()

        self.norm = nn.LayerNorm(g_feat_in)

        self.embed_g_feat = nn.Linear(g_feat_in, embedding_key_size, bias= g_embed_bias)
        formula_encoding_n = np.sum(formula_oh_sizes)

        self.formula_to_oh = StructuredOneHot(formula_oh_sizes, formula_oh_accum)
        
        self.embed_formulae_feat = nn.Linear(formula_encoding_n, embedding_key_size,
                                             bias=formula_embed_bias)
        self.formula_oh_normalize_1 = formula_oh_normalize_1
        self.formula_oh_normalize_2 = formula_oh_normalize_2

        if not formula_embed_train:
            self.embed_formulae_feat.weight.requires_grad = False
            self.embed_formulae_feat.bias.requires_grad = False
            
        
        self.norm2 = nn.LayerNorm(internal_d)

        self.combine_layers = nn.ModuleList([nn.GRUCell(formula_encoding_n, g_feat_in) for _ in range(gru_layer_n)])
        
        self.f_combine_l1 = nn.Linear(g_feat_in, internal_d)
        self.f_combine_l2 = nn.Sequential(*[nn.Sequential(nn.Linear(internal_d, internal_d),
                                                          nn.ReLU()) for _ in range(linear_layer_n)])
        
        self.f_combine_score = nn.Linear(internal_d, 1)

        
        self.spect_bin_n = spect_bin_n
        self.prob_softmax = prob_softmax


    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, vert_feat_in, vert_mask_in,
                possible_formulae, formulae_masses,
                vert_element_oh, adj_oh, formula_frag_count):
        """
        vert_feat_in : BATCH_N x ATOM_N x F
        vert_feat_mask : BATCH_N x ATOM_N  
        possible_formulae : BATCH_N x MAX_FORMULAE_N x FORMULA_ENCODING_N
        formula_masses : BATCH_N x MAX_FORMULAE_N x NUM_MASSES x 2  (mass, peak intensity) 
        """

        masked_vert_feat = vert_feat_in * vert_mask_in.unsqueeze(-1)

        # one-hot encode possible formulae
        possible_formulae_flat = possible_formulae.reshape(-1, possible_formulae.shape[-1])
        possible_formulae_flat_oh = self.formula_to_oh(possible_formulae_flat)
        possible_formulae = possible_formulae_flat_oh.reshape(possible_formulae.shape[0],
                                                              possible_formulae.shape[1], -1)

        possible_formulae2 = possible_formulae
        if self.formula_oh_normalize_1:
            possible_formulae2 = possible_formulae / (possible_formulae.sum(axis=1).unsqueeze(1) + 0.1)
            
        if self.formula_oh_normalize_2:
            possible_formulae2 = possible_formulae / (possible_formulae.sum(axis=2).unsqueeze(2) + 0.1)
            
        # encoded inputs

        # [步骤1] 分别将“图表示”和“化学式”映射到相同的隐空间特征维度，准备计算注意力
        vert_encoded = self.embed_g_feat(vert_feat_in)
        formulae_encoded = self.embed_formulae_feat(possible_formulae)

        dot_prod = (formulae_encoded.unsqueeze(1) * vert_encoded.unsqueeze(2)).sum(dim=-1)

        # [步骤2] 取点积计算图特征和候选分子式之间的匹配度 (Attention 分数)
        weighting = torch.softmax(dot_prod, dim=1)
        
        vert_att = masked_vert_feat.unsqueeze(3) * weighting.unsqueeze(2)
        vert_att_reduce = vert_att.sum(dim=1).permute(0, 2, 1) # (BATCH_N, FORMULAE_N, F)


        # [步骤3] 通过注意力分数，把整个分子的原子特征浓缩成了特定于该“候选分子式”的隐特征
        x = vert_att_reduce

        for l in self.combine_layers:
            x = l(possible_formulae2.reshape(-1, possible_formulae2.shape[-1]),
                  x.reshape(-1, x.shape[-1]))\
                  .reshape(x.shape)
        x = F.relu(self.f_combine_l1(x))
        x = F.relu(self.f_combine_l2(x))
        x = self.norm2(x)
        formulae_scores = self.f_combine_score(x).squeeze(2)
        
        if self.prob_softmax:
            formulae_probs = torch.softmax(formulae_scores, dim=-1)
        else:
            formulae_probs = torch.sigmoid(formulae_scores)

        #t1 = time.time()
        sparse_mass_matrices = [create_mass_matrix_sparse(formulae_masses[:, :, i, 0].round().long(),
                                                   self.spect_bin_n,
                                                   formulae_masses[:, :, i, 1]
        )\
                         .to(vert_feat_in.device) for i in range(formulae_masses.shape[2])]

        
        sparse_mass_matrix = sparse_mass_matrices[0]
        for m in sparse_mass_matrices[1:]:
            sparse_mass_matrix += m

        spect_out = mat_matrix_sparse_mm(sparse_mass_matrix, formulae_probs)
        # torch.cuda.synchronize()
        # t2 = time.time()
        # print(f"end of net took {(t2-t1)*1000:3.2f}ms ")

        return {'spect_out': spect_out,
                'formulae_probs' : formulae_probs}

