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
from scipy.ndimage import sobel
import torch
from torch import nn

from synthtorch import Unet
from .errors import AnnomError
from .loss import HotLoss, HotLaplacianLoss, LRSDecompLoss, OrdLoss

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

    def forward(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_skip(x, return_temp) if not self.no_skip else self._fwd_no_skip(x, return_temp)
        return x

    def _fwd_skip(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_skip_nf(x)
        xh, t = x, x
        for fxh in self.finish[0][0:2]: xh = self._add_noise(fxh(xh))
        for ft in self.finish[1][0:2]:   t = self._add_noise(ft(t))
        xh = self.finish[0][2](xh)
        t  = self.finish[1][2](t)
        return xh / torch.clamp(t, min=1e-6) if not return_temp else t

    def _fwd_no_skip(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_no_skip_nf(x)
        xh, t = x, x
        for fxh in self.finish[0][0:2]: xh = self._add_noise(fxh(xh))
        for ft in self.finish[1][0:2]:   t = self._add_noise(ft(t))
        xh = self.finish[0][2](xh)
        t  = self.finish[1][2](t)
        return xh / torch.clamp(t, min=1e-6) if not return_temp else t

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        n_classes = self.ord_params[2]
        ksz = tuple([1 for _ in self.kernel_sz])
        f = nn.ModuleList([self._conv_act(in_c, in_c, self.kernel_sz, self.act, self.norm),
                           self._conv_act(in_c, in_c, self.kernel_sz, 'softmax' if self.softmax else self.act, self.norm),
                           self._conv(in_c, n_classes, ksz, bias=bias)])
        t = nn.ModuleList([self._conv_act(in_c, in_c, self.kernel_sz, self.act, self.norm),
                           self._conv_act(in_c, in_c, self.kernel_sz, self.act, self.norm),
                           self._conv(in_c, 1, ksz, bias=False),
                           nn.Softplus()])
        return nn.ModuleList([f, t])

    def predict(self, x:torch.Tensor, return_temp:bool=False, **kwargs) -> torch.Tensor:
        y_hat = self.forward(x, return_temp)
        if not return_temp: y_hat = self.criterion.predict(y_hat)
        return y_hat


class LRSDNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet for low-rank and sparse decomposition in pytorch
    """
    def __init__(self, n_layers:int, penalty:Tuple[float,float]=(1,1), **kwargs):
        super().__init__(n_layers, **kwargs)
        if penalty is None: penalty = (1,1)
        self.criterion = LRSDecompLoss(penalty[0], penalty[1])

    def forward(self, x:torch.Tensor, return_sparse:bool=False) -> torch.Tensor:
        x = self._fwd_skip(x, return_sparse) if not self.no_skip else self._fwd_no_skip(x, return_sparse)
        return x

    def _fwd_skip(self, x:torch.Tensor, return_sparse:bool=False) -> torch.Tensor:
        x = self._fwd_skip_nf(x)
        x = (self.finish[0](x), self.finish[1](x)) if not return_sparse else self.finish[1](x)
        return x

    def _fwd_no_skip(self, x:torch.Tensor, return_sparse:bool=False) -> torch.Tensor:
        x = self._fwd_no_skip_nf(x)
        x = (self.finish[0](x), self.finish[1](x)) if not return_sparse else self.finish[1](x)
        return x

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        ksz = tuple([1 for _ in self.kernel_sz])
        lr = self._conv(in_c, out_c, ksz, bias=bias)
        s = self._conv(in_c, out_c, ksz, bias=bias)
        return nn.ModuleList([lr, s])

    def predict(self, x:torch.Tensor, return_sparse:bool=False, **kwargs) -> torch.Tensor:
        return self.forward(x)[1] if return_sparse else self.forward(x)[0]


class HotNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on vanilla regression in pytorch
    """
    def __init__(self, n_layers:int, monte_carlo:int=50, min_logvar:float=np.log(1e-6), laplacian:bool=True,
                 uncertainty:str='predictive', **kwargs):
        self.n_samp = monte_carlo or 50
        self.mlv = min_logvar
        self.laplacian = laplacian
        self.uncertainty = uncertainty if uncertainty is not None else 'predictive'
        if self.uncertainty not in ('predictive', 'aleatoric', 'epistemic'):
            raise AnnomError(f'Uncertainty type: {self.uncertainty}, not valid')
        logger.debug(f'Uncertainty type: {self.uncertainty}')
        super().__init__(n_layers, enable_dropout=True, **kwargs)
        self.criterion = HotLoss() if not laplacian else HotLaplacianLoss()

    def _finish(self, x:torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
        xh = self.finish[0](x)
        s = torch.clamp(self.finish[1](x), min=self.mlv)
        return xh, s

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        lksz = tuple([1 for _ in self.kernel_sz])
        f = self._conv(in_c, out_c, lksz, bias=bias)
        s = self._conv(in_c, out_c, lksz, bias=bias)
        return nn.ModuleList([f, s])

    def _calc_uncertainty(self, yhat, s) -> torch.Tensor:
        epistemic = torch.mean(yhat**2,dim=0) - torch.mean(yhat,dim=0)**2
        aleatoric = torch.mean(torch.exp(s),dim=0) if not self.laplacian else torch.mean(2*torch.exp(s)**2,dim=0)
        if self.uncertainty == 'epistemic': return epistemic
        elif self.uncertainty == 'aleatoric': return aleatoric
        else: return (epistemic + aleatoric)

    def predict(self, x:torch.Tensor, return_temp:bool=False, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat = torch.stack([o[0] for o in out])
        s = torch.stack([o[1] for o in out])
        if return_temp: return self._calc_uncertainty(yhat, s)
        return torch.mean(yhat, dim=0)
