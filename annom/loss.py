#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.loss

define loss functions for neural network training
specific for anomaly detection

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 11, 2018
"""

__all__ = ['LRSDecompLoss',
           'OrdLoss']

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LRSDecompLoss(nn.Module):
    def __init__(self, l_lmbda=1, s_lmbda=1):
        super().__init__()
        self.l_lmbda = l_lmbda
        self.s_lmbda = s_lmbda

    @staticmethod
    def _to_mtx(x):
        device = x.get_device() if x.is_cuda else 'cpu'
        H, W, D = x.size()  # assume 3d
        out = torch.zeros((H*W, D), requires_grad=True, dtype=x.dtype).to(device)
        with torch.no_grad():
            for d in range(D):
                out[:,d] = x[...,d].flatten()  # manually do this to ensure that slices are stacked correctly
        return out

    def forward(self, yhat, y):
        device = y.get_device() if y.is_cuda else 'cpu'
        L, S = yhat
        batch_size = L.shape[0]
        assert L.shape[1] == 1
        l1_penalty = self.s_lmbda * torch.norm(S, 1)  # to induce sparsity on S
        lr_penalty = torch.tensor([0.], requires_grad=True, dtype=y.dtype).to(device)
        for i in range(batch_size):
            lr_penalty = lr_penalty + torch.norm(self._to_mtx(L[i,0,...]), 'nuc')
        return F.mse_loss(L + S, y) + self.s_lmbda * l1_penalty + self.l_lmbda * lr_penalty


class OrdLoss(nn.Module):
    def __init__(self, params:Tuple[int,int,int], device:torch.device, is_3d:bool=False, lmbda:Tuple[float,float]=(1,1)):
        super(OrdLoss, self).__init__()
        start, stop, n_bins = params
        self.ce_weight, self.mae_weight = lmbda
        self.device = device
        self.bins = np.linspace(start, stop, n_bins-1, endpoint=False)
        self.tbins = self._linspace(start, stop, n_bins, is_3d).to(self.device)
        self.mae = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _linspace(start:int, stop:int, n_bins:int, is_3d:bool) -> torch.Tensor:
        rng = np.linspace(start, stop, n_bins, dtype=np.float32)
        trng = torch.from_numpy(rng[:,None,None])
        return trng if not is_3d else trng[...,None]

    def _digitize(self, x:torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.digitize(x.cpu().detach().numpy(), self.bins)).squeeze().to(self.device)

    def predict(self, yd_hat:torch.Tensor) -> torch.Tensor:
        p = F.softmax(yd_hat, dim=1)
        intensity_bins = torch.ones_like(yd_hat) * self.tbins
        y_hat = torch.sum(p * intensity_bins, dim=1, keepdim=True)
        return y_hat

    def forward(self, yd_hat:torch.Tensor, y:torch.Tensor):
        yd = self._digitize(y)
        CE = self.ce(yd_hat, yd)
        y_hat = self.predict(yd_hat)
        MAE = self.mae(y_hat, y)
        return self.ce_weight * CE + self.mae_weight * MAE
