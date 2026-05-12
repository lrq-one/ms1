# File overview: This module is part of the MassSpecGym/RASSP codebase.
# Purpose: Model architecture and loss definitions used by the RASSP/MS prediction pipeline.


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import scipy.stats
import time


# Class overview: WeightedLoss encapsulates a reusable component in this module.
class WeightedLoss(nn.Module):
    """
    Different weightings
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, func='l2',
                 config = {},
                 log_true = False,
                 loss_pow = 1,
                 extra_loss_args= {}, 
                 **kwargs):
        super(WeightedLoss, self).__init__()

        self.swap_arg= False

        if func == 'kl':
            self.loss = nn.KLDivLoss(log_target=True, reduction='none')
        elif func == 'l2':
            self.loss = nn.MSELoss(reduction='none')
        elif func == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        elif func == 'l1smooth':
            self.loss = nn.SmoothL1Loss(reduction='none', **extra_loss_args)
        elif func == 'subtract':
            self.loss = lambda x, y : y -x 

        self.config = config

        self.log_true = log_true

        self.pos = 0
        self.loss_pow  = loss_pow
        
    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, res, spect, input_mask_t, **kwargs): 

        wc = self.config
        pred_spect = res['spect']
        SPECT_N = spect.shape[1]

        true_spect_np = spect.to('cpu').detach().numpy()
        if self.log_true:
            true_spect_np = np.exp(true_spect_np)

        if wc['kind'] == 'beta':
            
            w = np.clip(scipy.stats.beta.pdf(true_spect_np, wc['alpha'], wc['beta']) + wc.get('offset', 0.0), 
                        a_max=4, a_min=0)
        elif wc['kind'] == 'mass':
            w = np.linspace(0, 1, SPECT_N) ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
            w = w / np.max(w)
        elif wc['kind'] == 'prob':
            w = true_spect_np ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
        elif wc['kind'] == 'inv-prob':
            w = 1 - true_spect_np ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)
        elif wc['kind'] == 'prob_mass':
            w0 = true_spect_np ** wc.get('prob_power', 1.0)
            w0 = w0 + wc.get('prob_offset', 1.0)
            w1 = np.linspace(0, 1, SPECT_N) ** wc.get('mass_power', 1.0)
            w1 = w1 + wc.get('mass_offset', 1.0)

            w = w0 * w1 
            w = w + wc.get('offset', 0.0)
            w = w / np.max(w)
        
        elif wc['kind'] == 'sparse':
            
            is_zero = (true_spect_np < wc.get('zero_threshold', 1e-3)).astype(np.float32)
            w = is_zero 
            #w = np.linspace(0, 1, SPECT_N) ** wc.get('power', 1.0)
            w = w + wc.get('offset', 1.0)

        elif wc['kind'] == 'noop':
            w = np.ones_like(true_spect_np)

        w_t = torch.tensor(w.astype(np.float32)).to(spect.device)
        l = self.loss(pred_spect, spect)
        
        if self.loss_pow != 1:
            l = torch.pow(l, self.loss_pow)

        l_batch = torch.sum(w_t * l, dim=1)

        out = res.copy()
        out['true_spect'] = spect
        out['w_t'] = w_t
        out['l'] = l
        out['l_batch'] = l_batch


        self.pos += 1
        return l_batch.mean()


# Class overview: WeightedMSELoss encapsulates a reusable component in this module.
class WeightedMSELoss(nn.Module):
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, mass_scale=1.0,
                 intensity_pow=0.5, pred_eps = 1e-9,
                 **kwargs):
        super(WeightedMSELoss, self).__init__()

        self.mass_scale = mass_scale
        self.intensity_pow = intensity_pow
        self.loss = nn.MSELoss()
        self.pred_eps = pred_eps
        print("pred_eps=", pred_eps)

    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']


        w = torch.arange(SPECT_N).to(true_spect.device) * self.mass_scale

        eps = self.pred_eps
        
        pred_weighted = (pred_spect+eps)**self.intensity_pow
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+eps)**self.intensity_pow
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm

        return self.loss(pred_weighted_normed, true_weighted_normed)
    
    
# Class overview: CustomL1Loss encapsulates a reusable component in this module.
class CustomL1Loss(nn.Module):
    ## eqn from Adams paper
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, foo=None, **kwargs):
        super(CustomL1Loss, self).__init__()

        self.loss = nn.L1Loss()

    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']


        w = torch.arange(SPECT_N).to(true_spect.device)

        pred_weighted = (pred_spect+1e-9).sqrt() #* w.reshape(1, -1)
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+1e-9).sqrt() #* w.reshape(1, -1)
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm

        return self.loss(pred_weighted_normed, true_weighted_normed)


# Class overview: CustomWeightedMSELoss encapsulates a reusable component in this module.
class CustomWeightedMSELoss(nn.Module):
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, mass_scale=0.0,
                 intensity_pow=0.5,
                 pred_eps = 1e-9,
                 weight_scheme = None,
                 func = 'l2', 
                 loss_pow = 1, 
                 invalid_mass_weight = 0.0, 
                 **kwargs):
        super(CustomWeightedMSELoss, self).__init__()

        self.mass_scale = mass_scale
        self.intensity_pow = intensity_pow
        self.func = func
        if self.func == 'l1':
            self.loss = nn.L1Loss(reduce='none')
        elif self.func == 'smoothl1':
            self.loss = nn.SmoothL1Loss(reduce='none')
        elif self.func == 'bce':
            self.loss = nn.BCELoss(reduce='none')
        elif self.func == 'l2':
            self.loss = nn.MSELoss(reduce='none')
        else:
            raise ValueError(f"Unknown loss func {func}")
        self.pred_eps = pred_eps
        self.weight_scheme = weight_scheme
        self.invalid_mass_weight = invalid_mass_weight
        self.loss_pow = loss_pow

    # Function overview: __call__ handles a specific workflow step in this module.
    def __call__(self, res, true_spect, input_mask_t, **kwargs): 
        SPECT_N = true_spect.shape[1]
        pred_spect = res['spect']

        mw = 1 + torch.arange(SPECT_N).to(true_spect.device) * self.mass_scale
        mw = mw.unsqueeze(0)

        w = 1

        
        eps = self.pred_eps
        
        pred_weighted = (pred_spect+eps)**self.intensity_pow * mw * w
        pred_weighted_norm = torch.sqrt((pred_weighted**2).sum(dim=1)).unsqueeze(-1)
        
        true_weighted = (true_spect+eps)**self.intensity_pow * mw * w
        
        true_weighted_norm = torch.sqrt((true_weighted**2).sum(dim=1)).unsqueeze(-1)

        pred_weighted_normed = pred_weighted / pred_weighted_norm
        true_weighted_normed = true_weighted / true_weighted_norm
        

        l = self.loss(pred_weighted_normed, true_weighted_normed)

        lw = 1+ (true_spect < 0.01).float() * self.invalid_mass_weight
        return ((l*lw)**self.loss_pow).mean()


# Class overview: TopPeakRankingLoss encapsulates a reusable component in this module.
class TopPeakRankingLoss(nn.Module):
    """
    Pairwise ranking loss for sparse MS/MS peaks.

    The loss treats the top-T true bins as positives and the lowest-intensity
    bins as negatives, then encourages pred[pos] > pred[neg] with a softplus
    margin objective.
    """
    # Function overview: __init__ handles a specific workflow step in this module.
    def __init__(self, top_t=12, neg_k=32, margin=0.0, negative_threshold=1e-3, use_intensity_weights=True):
        super(TopPeakRankingLoss, self).__init__()
        self.top_t = int(top_t)
        self.neg_k = int(neg_k)
        self.margin = float(margin)
        self.negative_threshold = float(negative_threshold)
        self.use_intensity_weights = bool(use_intensity_weights)

    # Function overview: forward handles a specific workflow step in this module.
    def forward(self, pred_spect, true_spect, input_mask_t=None, **kwargs):
        if pred_spect is None or true_spect is None:
            return torch.tensor(0.0)

        if not torch.is_tensor(pred_spect) or not torch.is_tensor(true_spect):
            raise TypeError("TopPeakRankingLoss expects tensor inputs")

        pred_spect = pred_spect.float()
        true_spect = true_spect.float()

        if pred_spect.dim() != 2:
            pred_spect = pred_spect.reshape(pred_spect.shape[0], -1)
        if true_spect.dim() != 2:
            true_spect = true_spect.reshape(true_spect.shape[0], -1)

        valid_batch = torch.any(true_spect > 0, dim=1)
        if not torch.any(valid_batch):
            return pred_spect.new_zeros(())

        pred_spect = pred_spect[valid_batch]
        true_spect = true_spect[valid_batch]

        batch_n, spect_n = true_spect.shape
        top_t = min(max(self.top_t, 1), spect_n)
        pos_idx = torch.topk(true_spect, k=top_t, dim=1).indices
        pos_scores = torch.gather(pred_spect, 1, pos_idx)
        pos_true = torch.gather(true_spect, 1, pos_idx)

        pos_mask = torch.zeros_like(true_spect, dtype=torch.bool)
        pos_mask.scatter_(1, pos_idx, True)

        neg_mask = (true_spect <= self.negative_threshold) & (~pos_mask)
        neg_count = int(neg_mask.sum(dim=1).min().item())
        if neg_count <= 0:
            neg_mask = ~pos_mask
            neg_count = int(neg_mask.sum(dim=1).min().item())

        neg_k = min(max(self.neg_k, 1), neg_count)
        if neg_k <= 0:
            return pred_spect.new_zeros(())

        neg_values = true_spect.masked_fill(~neg_mask, float("inf"))
        neg_idx = torch.topk(-neg_values, k=neg_k, dim=1).indices
        neg_scores = torch.gather(pred_spect, 1, neg_idx)

        pairwise_gap = pos_scores.unsqueeze(-1) - neg_scores.unsqueeze(-2)
        pairwise_loss = F.softplus(self.margin - pairwise_gap)

        if self.use_intensity_weights:
            pos_weight = pos_true.clamp_min(0.0)
            pos_weight = pos_weight / pos_weight.sum(dim=1, keepdim=True).clamp_min(1e-8)
        else:
            pos_weight = torch.full_like(pos_true, 1.0 / float(top_t))

        loss_per_sample = (pairwise_loss * pos_weight.unsqueeze(-1)).sum(dim=(1, 2)) / float(neg_k)
        return loss_per_sample.mean()
