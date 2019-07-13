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

from synthtorch import Unet
from .loss import HotGaussianLoss, HotLaplacianLoss, LRSDecompLoss, OrdLoss

logger = logging.getLogger(__name__)


class OrdNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on ordinal regression in pytorch

    Args:
        ord_params (Tuple[int,int,int]): parameters for ordinal regression (start,end,n_bins) [Default=None]
        device (torch.device): device to place new parameters/tensors on [Default=None]
    """
    def __init__(self, n_layers:int, ord_params:Tuple[int,int,int]=None, **kwargs):
        # setup and store instance parameters
        self.ord_params = ord_params
        super().__init__(n_layers, **kwargs)
        self.criterion = OrdLoss(ord_params, self.is_3d)
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
        self.criterion = HotGaussianLoss(beta) if not laplacian else HotLaplacianLoss(beta)
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
        epistemic = torch.mean(yhat**2,dim=0) - torch.mean(yhat,dim=0)**2
        aleatoric = torch.mean(torch.exp(s),dim=0) if not self.laplacian else torch.mean(2*torch.exp(s)**2,dim=0)
        return epistemic, aleatoric

    def predict(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat = torch.stack([o[0] for o in out]).cpu().detach()
        s = torch.stack([o[1] for o in out]).cpu().detach()
        e, a = self._calc_uncertainty(yhat, s)
        return torch.cat((torch.mean(yhat, dim=0), e, a), dim=1)


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
