#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.models

holds the architecture for estimating uncertainty

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 12, 2019
"""

__all__ = ['LRSDNet',
           'OrdNet']

import logging
from typing import Optional, Tuple

import torch
from torch import nn

from synthnn import Unet
from .loss import LRSDecompLoss, OrdLoss

logger = logging.getLogger(__name__)


class OrdNet(Unet):
    """
    defines a 2d or 3d uncertainty-calculating unet based on ordinal regression in pytorch

    Args:
        ord_params (Tuple[int,int,int]): parameters for ordinal regression (start,end,n_bins) [Default=None]
        device (torch.device): device to place new parameters/tensors on [Default=None]
    """
    def __init__(self, n_layers:int, ord_params:Tuple[int,int,int]=None, device:torch.device=None, **kwargs):
        # setup and store instance parameters
        self.ord_params = ord_params
        self.device = device
        super().__init__(n_layers, **kwargs)
        self.criterion = OrdLoss(ord_params, device, self.is_3d)

    def forward(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_skip(x, return_temp) if not self.no_skip else self._fwd_no_skip(x, return_temp)
        return x

    def _fwd_skip(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_skip_nf(x)
        x = self.finish[0](x) / torch.clamp(self.finish[1](x), min=1e-6) if not return_temp else self.finish[1](x)
        return x

    def _fwd_no_skip(self, x:torch.Tensor, return_temp:bool=False) -> torch.Tensor:
        x = self._fwd_no_skip_nf(x)
        x = self.finish[0](x) / torch.clamp(self.finish[1](x), min=1e-6) if not return_temp else self.finish[1](x)
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
    def __init__(self, n_layers:int, l_lmbda:float=1, s_lmbda:float=1, **kwargs):
        super().__init__(n_layers, **kwargs)
        self.criterion = LRSDecompLoss(l_lmbda, s_lmbda)

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

