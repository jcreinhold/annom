#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.loss

define loss functions for neural network training
specific for anomaly detection

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 11, 2018
"""

__all__ = ['Burn2MSELoss',
           'Burn2MAELoss',
           'HotGaussianLoss',
           'HotLaplacianLoss',
           'HotMAEOnlyLoss',
           'HotMSEOnlyLoss',
           'LRSDecompLoss',
           'OCMAELoss',
           'OCMSELoss',
           'OrdLoss',
           'SVDDMAELoss',
           'SVDDMSELoss',
           'Unburn2GaussianLoss',
           'Unburn2LaplacianLoss']

import logging
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


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
    def __init__(self, params:Tuple[int,int,int], is_3d:bool=False, lmbda:Tuple[float,float]=(1,1)):
        super(OrdLoss, self).__init__()
        start, stop, n_bins = params
        self.ce_weight, self.mae_weight = lmbda
        self.bins = np.linspace(start, stop, n_bins-1, endpoint=False)
        self.tbins = self._linspace(start, stop, n_bins, is_3d)
        self.mae = nn.L1Loss()
        self.ce = nn.CrossEntropyLoss()

    @staticmethod
    def _linspace(start:int, stop:int, n_bins:int, is_3d:bool) -> torch.Tensor:
        rng = np.linspace(start, stop, n_bins, dtype=np.float32)
        trng = torch.from_numpy(rng[:,None,None])
        return trng if not is_3d else trng[...,None]

    def _digitize(self, x:torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(np.digitize(x.cpu().detach().numpy(), self.bins)).squeeze().to(x.device)

    def predict(self, yd_hat:torch.Tensor) -> torch.Tensor:
        p = F.softmax(yd_hat, dim=1)
        intensity_bins = torch.ones_like(yd_hat) * self.tbins.to(yd_hat.device)
        y_hat = torch.sum(p * intensity_bins, dim=1, keepdim=True)
        return y_hat

    def forward(self, yd_hat:torch.Tensor, y:torch.Tensor):
        yd_hat = yd_hat[0]  # second entry in tuple is temperature
        yd = self._digitize(y)
        CE = self.ce(yd_hat, yd)
        y_hat = self.predict(yd_hat)
        MAE = self.mae(y_hat, y)
        return self.ce_weight * CE + self.mae_weight * MAE


class HotLoss(nn.Module):
    def __init__(self, beta:float=1.):
        super(HotLoss, self).__init__()
        self.beta = beta

    def forward(self, out:torch.Tensor, y:torch.Tensor):
        raise NotImplementedError

    def extra_repr(self): return f'beta={self.beta}'


class HotMSEOnlyLoss(nn.Module):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        yhat, _ = out
        loss = F.mse_loss(yhat, y)
        return loss


class HotMAEOnlyLoss(nn.Module):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        yhat, _ = out
        loss = F.l1_loss(yhat, y)
        return loss


class HotGaussianLoss(HotLoss):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        if isinstance(out[0], tuple): out = out[0]
        yhat, s = out
        loss = torch.mean(0.5 * (torch.exp(-s) * F.mse_loss(yhat, y, reduction='none') + self.beta * s))
        return loss


class HotLaplacianLoss(HotLoss):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        if isinstance(out[0], tuple): out = out[0]
        yhat, s = out
        loss = torch.mean(np.sqrt(2) * (torch.exp(-s) * F.l1_loss(yhat, y, reduction='none')) + self.beta * s)
        return loss


class Burn2MSELoss(HotLoss):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        x1, x2, z1, z2, _, _ = out
        mse_loss1 = F.mse_loss(x1, y[:,0:1,...])
        mse_loss2 = F.mse_loss(x2, y[:,1:2,...])
        z_penalty = F.mse_loss(F.softmax(z1,dim=1), F.softmax(z2,dim=1), reduction='sum')
        return mse_loss1 + mse_loss2 + self.beta * z_penalty


class Burn2MAELoss(HotLoss):
    def forward(self, out:torch.Tensor, y:torch.Tensor):
        x1, x2, z1, z2, _, _ = out
        mae_loss1 = F.l1_loss(x1, y[:,0:1,...])
        mae_loss2 = F.l1_loss(x2, y[:,1:2,...])
        z_penalty = F.mse_loss(F.softmax(z1,dim=1), F.softmax(z2,dim=1), reduction='sum')
        return mae_loss1 + mae_loss2 + self.beta * z_penalty


class Unburn2Loss(HotLoss):
    def _loss(self, yhat, s, y):
        raise NotImplementedError

    def forward(self, out:torch.Tensor, y:torch.Tensor):
        yhat1, s1 = out[0][0]
        yhat2, s2 = out[1][0]
        z1, z2 = out[0][1], out[1][1]
        loss1 = self._loss(yhat1, s1, y[:,0:1,...])
        loss2 = self._loss(yhat2, s2, y[:,1:2,...])
        z_penalty = F.mse_loss(F.softmax(z1,dim=1), F.softmax(z2,dim=1), reduction='sum')
        return loss1 + loss2 + self.beta * z_penalty


class Unburn2GaussianLoss(Unburn2Loss):
    def _loss(self, yhat, s, y):
        return torch.mean(0.5 * (torch.exp(-s) * F.mse_loss(yhat, y, reduction='none') + s))


class Unburn2LaplacianLoss(Unburn2Loss):
    def _loss(self, yhat, s, y):
        return torch.mean(np.sqrt(2) * (torch.exp(-s) * F.l1_loss(yhat, y, reduction='none')) + s)


class OCLoss(HotLoss):
    def _loss(self, yhat, y):
        raise NotImplementedError

    def forward(self, out, y):
        yhat, c, ctv = out
        ysz = yhat.shape[2:]
        y = F.interpolate(y, ysz, mode='bilinear' if len(ysz) == 2 else 'trilinear', align_corners=True)
        rp = self._loss(yhat, y)
        if self.beta >= 0:
            nb = c.shape[0] // 2
            ct = torch.ones(nb*2, dtype=torch.long, device=c.device)
            ct[:nb] = 0
            bce = F.cross_entropy(c, ct)
            pd = torch.mean(F.pdist(ctv))
            accuracy = torch.mean((torch.argmax(c,dim=1)==ct).float())
            logger.info(f'CE: {bce.item():.2e}, PD: {pd.item():.2e}, '
                        f'RP: {rp.item():.2e}, Acc: {accuracy.item():.5f}')
        return (bce + self.beta * (pd + rp)) if self.beta >= 0 else rp


class OCMSELoss(OCLoss):
    def _loss(self, yhat, y):
        return F.mse_loss(yhat, y)


class OCMAELoss(OCLoss):
    def _loss(self, yhat, y):
        return F.l1_loss(yhat, y)


class SVDDLoss(nn.Module):
    def __init__(self, sz, beta:float=1.):
        super().__init__()
        self.beta = beta
        self.register_buffer('c', torch.ones(1, sz, dtype=torch.float32))
        self.register_buffer('is_c_set', torch.zeros(1, dtype=torch.uint8))

    def _loss(self, yhat, y):
        raise NotImplementedError

    def forward(self, out, y):
        yhat, phi = out
        ysz = yhat.shape[2:]
        y = F.interpolate(y, ysz, mode='bilinear' if len(ysz) == 2 else 'trilinear', align_corners=True)
        rp = self._loss(yhat, y)
        if self.beta >= 0:
            nb = phi.shape[0] // 2
            if torch.sum(self.is_c_set) == 0:
                with torch.no_grad():
                    self.c *= torch.mean(phi[nb:], dim=0, keepdim=True)
                    self.is_c_set += 1
                    logger.info('Set c in SVDD loss.')
            c = torch.ones_like(phi) * self.c
            yi = torch.ones(nb*2, dtype=phi.dtype, device=phi.device)
            yi[:nb] = -1
            z = torch.mean(F.mse_loss(phi, c, reduction='none'), dim=1)
            z[:nb] += 1e-6  # avoid division by zero
            z = z ** yi
            z[:nb] *= 0.5  # weight the fake anomalies less
            svdd = torch.mean(z)
            logger.info(f'SVDD: {svdd.item():.2e}, RP: {rp.item():.2e}')
        return (svdd + self.beta * rp) if self.beta >= 0 else rp

    def extra_repr(self): return f'beta={self.beta}'


class SVDDMSELoss(SVDDLoss):
    def _loss(self, yhat, y):
        return F.mse_loss(yhat, y)


class SVDDMAELoss(SVDDLoss):
    def _loss(self, yhat, y):
        return F.l1_loss(yhat, y)
