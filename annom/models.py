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
        x = (self.finish[0](x) / torch.clamp(self.finish[1](x), min=1e-6)) if not return_temp else self.finish[1](x)
        return x

    def _fwd_no_skip(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_no_skip_nf(x)
        x = (self.finish[0](x) / torch.clamp(self.finish[1](x), min=1e-6)) if not return_temp else self.finish[1](x)
        return x

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        n_classes = self.ord_params[2]
        fc = self._conv(in_c, n_classes, 1, bias=bias)
        fc_temp = nn.Sequential(self._conv(in_c, 1, 1, bias=bias), nn.Softplus())
        return nn.ModuleList([fc, fc_temp])

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
        lr = self._conv(in_c, out_c, 1, bias=bias)
        s = self._conv(in_c, out_c, 1, bias=bias)
        return nn.ModuleList([lr, s])

    def predict(self, x:torch.Tensor, return_sparse:bool=False, **kwargs) -> torch.Tensor:
        return self.forward(x)[1] if return_sparse else self.forward(x)[0]


class HotNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on vanilla regression in pytorch
    """
    def __init__(self, n_layers:int, n_samp:int=50, min_logvar:float=np.log(1e-6), edge:bool=True, laplacian:bool=True, **kwargs):
        self.n_samp = n_samp
        self.mlv = min_logvar
        self.edge = edge
        self.laplacian = laplacian
        super().__init__(n_layers, enable_dropout=True, **kwargs)
        self.criterion = HotLoss() if not laplacian else HotLaplacianLoss()

    def forward(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor,torch.Tensor]:
        x = self._fwd_skip(x, **kwargs) if not self.no_skip else self._fwd_no_skip(x, **kwargs)
        return x

    def _fwd_skip(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor,torch.Tensor]:
        if self.edge: edge = self._edge(x)
        x = self._fwd_skip_nf(x)
        if self.edge: x = torch.cat((x, edge), dim=1)
        xh = self.finish[0][1](self._add_noise(self.finish[0][0](x)))
        s  = torch.clamp(self.finish[1][1](self._add_noise(self.finish[1][0](x))),min=self.mlv)
        return xh, s

    def _fwd_no_skip(self, x:torch.Tensor, **kwargs) -> Tuple[torch.Tensor,torch.Tensor]:
        if self.edge: edge = self._edge(x)
        x = self._fwd_no_skip_nf(x)
        if self.edge: x = torch.cat((x, edge), dim=1)
        xh = self.finish[0][1](self._add_noise(self.finish[0][0](x)))
        s  = torch.clamp(self.finish[1][1](self._add_noise(self.finish[1][0](x))),min=self.mlv)
        return xh, s

    def _edge(self, x):
        xn = x.cpu().detach().numpy()
        em = torch.cat([torch.from_numpy(sobel(xn, axis=i)) for i in range(-2,0)], dim=1).to(x.device)  # only 2d
        return em

    def _final(self, in_c:int, out_c:int, out_act:Optional[str]=None, bias:bool=False):
        if self.edge: in_c = in_c + 2
        f = nn.ModuleList([self._conv_act(in_c, in_c, 3, self.act, self.norm),
                           nn.Sequential(self._conv_act(in_c, in_c, 3, self.act, self.norm),
                                         self._conv(in_c, out_c, 1, bias=bias))])
        s = nn.ModuleList([self._conv_act(in_c, in_c, 3, self.act, self.norm),
                           nn.Sequential(self._conv_act(in_c, in_c, 3, self.act, self.norm),
                                         self._conv(in_c, out_c, 1, bias=False))])
        return nn.ModuleList([f, s])

    def _calc_uncertainty(self, yhat, s) -> torch.Tensor:
        v1 = torch.mean(yhat**2,dim=0) - torch.mean(yhat,dim=0)**2
        v2 = torch.mean(torch.exp(s),dim=0) if not self.laplacian else torch.mean(2*torch.exp(s)**2,dim=0)
        samp_var = v1 + v2
        return samp_var

    def predict(self, x:torch.Tensor, return_temp:bool=False, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat = torch.stack([o[0] for o in out])
        s = torch.stack([o[1] for o in out])
        if return_temp: return self._calc_uncertainty(yhat, s)
        return torch.mean(yhat, dim=0)


