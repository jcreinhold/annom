#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.models

holds the architecture for estimating uncertainty

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 12, 2019
"""

__all__ = ['HotNet',
           'LRSDNet',
           'OrdNet']

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from synthtorch import Unet
from .loss import *

logger = logging.getLogger(__name__)


class OrdNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on ordinal regression in pytorch

    Args:
        ord_params (Tuple[int,int,int]): parameters for ordinal regression (start,end,n_bins) [Default=None]
    """
    def __init__(self, n_layers:int, ord_params:Tuple[int,int,int]=None, **kwargs):
        # setup and store instance parameters
        self.ord_params = ord_params
        super().__init__(n_layers, **kwargs)
        self.criterion = OrdLoss(ord_params, self.dim == 3)
        self.n_output += 1

    def _finish(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        xh, t = self._add_noise(self.finish[0][0](x)), self._add_noise(self.finish[1][0](x))
        xh, t = self.finish[0][1](xh), torch.clamp(self.finish[1][1](t), min=1e-6)
        return (xh / t, t)

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        n_classes = self.ord_params[2]
        ksz = (3,3,3) if self.semi_3d > 0 else self.kernel_sz
        kszf = tuple([1 for _ in self.kernel_sz])
        f = nn.ModuleList([self._conv_act(in_c, in_c, ksz, 'softmax' if self.softmax else self.act, self.norm),
                           self._conv(in_c, n_classes, kszf, bias=bias)])
        t = nn.ModuleList([self._conv_act(in_c, in_c, ksz, self.act, self.norm),
                           self._conv(in_c, 1, kszf, bias=False),
                           nn.Softplus()])
        return nn.ModuleList([f, t])

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        xhdt, t = self.forward(x)
        return torch.cat((self.criterion.predict(xhdt), t), dim=1)


class HotNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on vanilla regression in pytorch
    """
    def __init__(self, n_layers:int, monte_carlo:int=50, min_logvar:float=np.log(1e-6), laplacian:bool=True,
                 beta:float=1., **kwargs):
        self.n_samp = monte_carlo or 50
        self.mlv = min_logvar
        self.laplacian = laplacian
        super().__init__(n_layers, enable_dropout=True, **kwargs)
        if beta > 0:
            self.criterion = HotGaussianLoss(beta) if not laplacian else HotLaplacianLoss(beta)
        else:
            self.criterion = HotMSEOnlyLoss() if not laplacian else HotMAEOnlyLoss()
        self.n_output += 2

    def _finish(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        xh = self.finish[0](x)
        s = torch.clamp(self.finish[1](x), min=self.mlv)
        return xh, s

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        ksz = (3,3,3) if self.semi_3d > 0 else self.kernel_sz
        kszf = tuple([1 for _ in self.kernel_sz])
        f = nn.Sequential(self._conv_act(in_c, in_c, ksz, 'softmax' if self.softmax else self.act, self.norm),
                          self._conv(in_c, out_c, kszf, bias=bias))
        s = nn.Sequential(self._conv_act(in_c, in_c, ksz, self.act, self.norm),
                          self._conv(in_c, out_c, kszf, bias=bias))
        return nn.ModuleList([f, s])

    def _calc_uncertainty(self, yhat, s) -> Tuple[torch.Tensor,torch.Tensor]:
        epistemic = yhat.var(dim=0, unbiased=True)
        aleatoric = torch.mean(torch.exp(s),dim=0) if not self.laplacian else torch.mean(2*torch.exp(s)**2,dim=0)
        return epistemic, aleatoric

    def predict(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat = torch.stack([o[0] for o in out]).cpu().detach()
        s = torch.stack([o[1] for o in out]).cpu().detach()
        e, a = self._calc_uncertainty(yhat, s)
        return torch.cat((torch.mean(yhat, dim=0), e, a), dim=1)

    def freeze(self):
        super().freeze()
        for p in self.finish[0].parameters(): p.requires_grad = False


class LRSDNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet for low-rank and sparse decomposition in pytorch
    """
    def __init__(self, n_layers:int, penalty:Tuple[float,float]=(1,1), **kwargs):
        super().__init__(n_layers, **kwargs)
        if penalty is None: penalty = (1,1)
        self.criterion = LRSDecompLoss(penalty[0], penalty[1])
        self.n_output += 1

    def _finish(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        return self.finish[0](x), self.finish[1](x)

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        ksz = (3,3,3) if self.semi_3d > 0 else self.kernel_sz
        kszf = tuple([1 for _ in self.kernel_sz])
        lr = nn.Sequential(self._conv_act(in_c, in_c, ksz, 'softmax' if self.softmax else self.act, self.norm),
                          self._conv(in_c, out_c, kszf, bias=bias))
        s  = nn.Sequential(self._conv_act(in_c, in_c, ksz, self.act, self.norm),
                          self._conv(in_c, out_c, kszf, bias=bias))
        return nn.ModuleList([lr, s])

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.cat(self.forward(x), dim=1)


class BurnNet(nn.Module):
    """
    defines a N-D (multinomial) variational U-Net
    """
    def __init__(self, n_layers:int, zdim:int=5, temperature:float=0.67, **kwargs):
        super().__init__()
        self.zdim = zdim
        self.temperature = temperature
        n_output = kwargs.pop('n_output', 1)
        self.encoder = Unet(n_layers, enable_dropout=True, n_output=zdim, **kwargs)
        _ = kwargs.pop('n_input')
        self.decoder = Unet(n_layers, enable_dropout=True, n_input=zdim, n_output=n_output, **kwargs)

    def forward(self, x:torch.Tensor, **kwargs):
        z = self.encoder.forward(x, **kwargs)
        zgs = self.sample_gumbel_softmax(z)
        x = self.decoder.forward(zgs, **kwargs)
        return x

    def sample_gumbel_softmax(self, alpha, eps=1e-12):
        if self.training:
            # Sample from gumbel distribution
            unif = torch.rand(alpha.size()).to(alpha.device)
            gumbel = -torch.log(-torch.log(unif + eps) + eps)
            # Reparameterize to create gumbel softmax sample
            log_alpha = torch.log(alpha + eps)
            logit = (log_alpha + gumbel) / self.temperature
            return F.softmax(logit, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            _, max_alpha = torch.max(alpha, dim=1)
            one_hot_samples = torch.zeros(alpha.size())
            # On axis 1 of one_hot_samples, scatter the value 1 at indices
            # max_alpha. Note the view is because scatter_ only accepts 2D
            # tensors.
            one_hot_samples.scatter_(1, max_alpha.view(-1, 1).detach().cpu(), 1)
            one_hot_samples = one_hot_samples.to(alpha.device)
            return one_hot_samples

