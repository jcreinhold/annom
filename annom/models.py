#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.models

holds the architecture for estimating uncertainty

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 12, 2019
"""

__all__ = ['BurnNet',
           'Burn2Net',
           'Burn2NetP12',
           'Burn2NetP21',
           'HotNet',
           'LAutoNet',
           'LavaNet',
           'Lava2Net',
           'LRSDNet',
           'OCNet1',
           'OCNet2',
           'OrdNet',
           'Unburn2Net',
           'UnburnNet']

import logging
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from synthtorch import Unet
from synthtorch.util import get_act
from .errors import *
from .layers import *
from .loss import *
from .util import *

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
    def __init__(self, n_layers:int, monte_carlo:int=50, min_logvar:float=np.log(1e-6), beta:float=1., **kwargs):
        self.n_samp = monte_carlo or 50
        self.mlv = min_logvar
        super().__init__(n_layers, enable_dropout=True, **kwargs)
        self.laplacian = use_laplacian(self.loss)
        if beta > 0:
            self.criterion = HotGaussianLoss(beta) if not self.laplacian else HotLaplacianLoss(beta)
        else:
            self.criterion = HotMSEOnlyLoss() if not self.laplacian else HotMAEOnlyLoss()
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

    def get_metrics(self, x, y, eps=1e-6, **kwargs):
        """ get uncertainties and other metrics during training for analysis """
        state = self.training
        self.eval()
        with torch.no_grad():
            yhat, s, ep, al = self.predict(x)
            loss = self.criterion((yhat, s), y)
            yhat, s = yhat.detach().cpu(), s.detach().cpu()
            ep, al = ep.detach().cpu(), al.detach().cpu()
            sb = ep / (al + eps)
            eu, au = ep.mean(), al.mean()
            su = sb.mean()
        self.train(state)
        return loss, (yhat, s), (ep, al, sb), (eu, au, su)


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
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, loss:str=None, **kwargs):
        super().__init__()
        self.zdim = latent_size
        self.temperature = temperature
        n_output = kwargs.pop('n_output', 1)
        self.encoder = Unet(n_layers, enable_dropout=True, n_output=latent_size, **kwargs)
        _ = kwargs.pop('n_input')
        self.decoder = Unet(n_layers, enable_dropout=True, n_input=latent_size, n_output=n_output, **kwargs)
        self.laplacian = use_laplacian(loss)
        self.criterion = HotMSEOnlyLoss() if not self.laplacian else HotMAEOnlyLoss()
        self.n_output = n_output + latent_size
        del self.encoder.criterion, self.decoder.criterion

    def forward(self, x:torch.Tensor, **kwargs):
        z = self.encoder.forward(x, **kwargs)
        z = self.sample_gumbel_softmax(z)
        x = self.decoder.forward(z, **kwargs)
        return (x, z)

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.cat(self.forward(x), dim=1)

    def sample_gumbel_softmax(self, x, eps=1e-12):
        if self.training:
            x = F.log_softmax(x,dim=1)
            # Sample from gumbel distribution
            unif = torch.rand_like(x)
            gumbel = -torch.log(-torch.log(unif + eps) + eps)
            # Reparameterize to create gumbel softmax sample
            x = (x + gumbel) / self.temperature
            return F.softmax(x, dim=1)
        else:
            # In reconstruction mode, pick most likely sample
            idxs = torch.argmax(x, dim=1, keepdim=True)
            one_hot_samples = torch.zeros_like(x)
            one_hot_samples.scatter_(1, idxs, 1.)
            return one_hot_samples

    def freeze(self):
        self.encoder.freeze()
        for p in self.encoder.finish.parameters(): p.requires_grad = False
        self.decoder.freeze()


class UnburnNet(BurnNet):
    """
    defines a N-D (multinomial) variational U-Net with uncertainty
    """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, loss:str=None,
                 monte_carlo:int=50, min_logvar:float=np.log(1e-6), beta:float=1., **kwargs):
        super().__init__(n_layers, latent_size, temperature, loss, **kwargs)
        n_output, _ = kwargs.pop('n_output', 1), kwargs.pop('n_input')
        self.decoder = HotNet(n_layers, monte_carlo, min_logvar, beta, n_input=latent_size, n_output=n_output, **kwargs)
        self.criterion = self.decoder.criterion
        del self.decoder.criterion
        self.n_output += 2
        self.n_samp = monte_carlo or 50

    def forward(self, x:torch.Tensor, **kwargs):
        z = self.encoder.forward(x, **kwargs)
        zg = self.sample_gumbel_softmax(z)
        x = self.decoder.forward(zg, **kwargs)
        return (x, z, zg)

    def _calc_uncertainty(self, yhat, s) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decoder._calc_uncertainty(yhat, s)

    def predict(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat = torch.stack([o[0][0] for o in out]).cpu().detach()
        s = torch.stack([o[0][1] for o in out]).cpu().detach()
        z = torch.stack([o[2] for o in out]).cpu().detach()
        del out  # try to save memory
        e, a = self._calc_uncertainty(yhat, s)
        return torch.cat((torch.mean(yhat, dim=0), e, a, torch.mean(z, dim=0)), dim=1)


class Unburn2Net(nn.Module):
    """
    defines a N-D (multinomial) variational U-Net with uncertainty for two inputs, outputs
    """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, loss:str=None,
                 monte_carlo:int=50, min_logvar:float=np.log(1e-6), beta:float=1., **kwargs):
        super().__init__()
        _, _ = kwargs.pop('n_output', 1), kwargs.pop('n_input')
        self.vae1 = UnburnNet(n_layers, latent_size, temperature, loss, monte_carlo, min_logvar, beta,
                              n_input=1, n_output=1, **kwargs)
        self.vae2 = UnburnNet(n_layers, latent_size, temperature, loss, monte_carlo, min_logvar, beta,
                              n_input=1, n_output=1, **kwargs)
        del self.vae1.criterion, self.vae2.criterion
        self.laplacian = use_laplacian(loss)
        self.criterion = Unburn2GaussianLoss(beta) if not self.laplacian else Unburn2LaplacianLoss(beta)
        self.n_output = self.vae1.n_output + self.vae2.n_output
        self.n_samp = monte_carlo or 50

    def forward(self, x:torch.Tensor, **kwargs):
        x1, x2 = x[:,0:1,...], x[:,1:2,...]
        x1 = self.vae1.forward(x1)
        x2 = self.vae1.forward(x2)
        return x1, x2

    def _calc_uncertainty(self, yhat, s) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.vae1._calc_uncertainty(yhat, s)

    def _extract(self, out, i):
        yhat = torch.stack([o[i][0][0] for o in out]).cpu().detach()
        s = torch.stack([o[i][0][1] for o in out]).cpu().detach()
        zg = torch.stack([o[i][2] for o in out]).cpu().detach()
        return yhat, s, zg

    def predict(self, x:torch.Tensor, **kwargs) -> torch.Tensor:
        out = [self.forward(x) for _ in range(self.n_samp)]
        yhat1, s1, z1 = self._extract(out, 0)
        yhat2, s2, z2 = self._extract(out, 1)
        del out  # try to save memory
        e1, a1 = self._calc_uncertainty(yhat1, s1)
        e2, a2 = self._calc_uncertainty(yhat2, s2)
        return torch.cat((torch.mean(yhat1, dim=0), e1, a1, torch.mean(z1, dim=0),
                          torch.mean(yhat2, dim=0), e2, a2, torch.mean(z2, dim=0)), dim=1)

    def freeze(self):
        self.vae1.freeze()
        self.vae2.freeze()


class Burn2Net(BurnNet):
    """
    defines a N-D (multinomial) variational U-Net for two inputs, outputs
    """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, beta:float=1., **kwargs):
        super().__init__(n_layers, latent_size, temperature, **kwargs)
        del self.encoder, self.decoder
        ni, no = kwargs.pop('n_input'), kwargs.pop('n_output')
        if ni != no or ni != 2: raise AnnomError('Burn2Net requires the number of input and output both to be 2.')
        self.encoder1 = Unet(n_layers, enable_dropout=True, n_input=1, n_output=latent_size, **kwargs)
        self.encoder2 = Unet(n_layers, enable_dropout=True, n_input=1, n_output=latent_size, **kwargs)
        self.decoder1 = Unet(n_layers, enable_dropout=True, n_input=latent_size, n_output=1, **kwargs)
        self.decoder2 = Unet(n_layers, enable_dropout=True, n_input=latent_size, n_output=1, **kwargs)
        self.criterion = Burn2MSELoss(beta) if not self.laplacian else Burn2MAELoss(beta)
        self.n_output = 2 + (2 * latent_size)
        del self.encoder1.criterion, self.decoder1.criterion, self.encoder2.criterion, self.decoder2.criterion

    def forward(self, x:torch.Tensor, **kwargs):
        x1, x2 = x[:,0:1,...], x[:,1:2,...]
        z1 = self.encoder1.forward(x1, **kwargs)
        z2 = self.encoder2.forward(x2, **kwargs)
        zg1 = self.sample_gumbel_softmax(z1)
        zg2 = self.sample_gumbel_softmax(z2)
        y1 = self.decoder1.forward(zg1, **kwargs)
        y2 = self.decoder2.forward(zg2, **kwargs)
        return (y1, y2, z1, z2, zg1, zg2)

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        x1, x2, _, _, zg1, zg2 = self.forward(x)
        return torch.cat((x1, x2, zg1, zg2), dim=1)

    def freeze(self):
        self.encoder1.freeze()
        for p in self.encoder1.finish.parameters(): p.requires_grad = False
        self.decoder1.freeze()
        self.encoder2.freeze()
        for p in self.encoder2.finish.parameters(): p.requires_grad = False
        self.decoder2.freeze()


class Burn2NetP12(Burn2Net):
    """ predict second class from first """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, **kwargs):
        super().__init__(n_layers, latent_size, temperature, **kwargs)
        self.n_output = 1 + latent_size

    def forward(self, x:torch.Tensor, **kwargs):
        z = self.encoder1.forward(x, **kwargs)
        zg = self.sample_gumbel_softmax(z)
        y = self.decoder2.forward(zg, **kwargs)
        return (y, zg)

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.cat(self.forward(x), dim=1)


class Burn2NetP21(Burn2Net):
    """ predict first class from second """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, **kwargs):
        super().__init__(n_layers, latent_size, temperature, **kwargs)
        self.n_output = 1 + latent_size

    def forward(self, x:torch.Tensor, **kwargs):
        z = self.encoder2.forward(x, **kwargs)
        zg = self.sample_gumbel_softmax(z)
        y = self.decoder1.forward(zg, **kwargs)
        return (y, zg)

    def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return torch.cat(self.forward(x), dim=1)


class LavaNet(BurnNet):
    """
    defines a N-D (multinomial) variational U-Net, like burnnet with small decoder
    """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, loss:str=None, **kwargs):
        super().__init__(n_layers, latent_size, temperature, loss, **kwargs)
        self.decoder = self.decoder._final(latent_size, self.decoder.n_output,
                                           self.decoder.out_act, self.decoder.enable_bias)

    def freeze(self):
        self.encoder.freeze()
        for p in self.encoder.finish.parameters(): p.requires_grad = False


class Lava2Net(Burn2Net):
    """
    defines a N-D (multinomial) variational U-Net, like burn2net with small decoder
    """
    def __init__(self, n_layers:int, latent_size:int=5, temperature:float=0.67, beta:float=1., **kwargs):
        super().__init__(n_layers, latent_size, temperature, beta, **kwargs)
        self.decoder1 = self.decoder1._final(latent_size, self.decoder1.n_output,
                                             self.decoder1.out_act, self.decoder1.enable_bias)
        self.decoder2 = self.decoder2._final(latent_size, self.decoder2.n_output,
                                             self.decoder2.out_act, self.decoder2.enable_bias)

    def freeze(self):
        self.encoder1.freeze()
        for p in self.encoder1.finish.parameters(): p.requires_grad = False
        self.encoder2.freeze()
        for p in self.encoder2.finish.parameters(): p.requires_grad = False


class LAutoNet(Unet):
    """ use a 1-d vector as latent space in this network """
    def __init__(self, n_layers:int, img_dim:Tuple[int], loss:str=None, latent_size:int=128, **kwargs):
        kwargs['affine'] = False
        kwargs['enable_bias'] = False
        kwargs['input_connect'] = False
        kwargs['no_skip'] = True
        super().__init__(n_layers, loss=loss, **kwargs)
        self.img_dim = img_dim
        self.laplacian = use_laplacian(loss)
        self.criterion = HotMAEOnlyLoss() if self.laplacian else HotMSEOnlyLoss()

        # handle creating new bridge (to 1-d latent space vector)
        del self.bridge
        self.latent_size = latent_size
        with torch.no_grad():
            img_dim_z = self._encode(torch.zeros((1, self.n_input, *img_dim), dtype=torch.float32))[0].shape
        self.fsz = img_dim_z[1:]
        self.esz = int(np.prod(self.fsz))
        logger.debug(f'Size after Conv = {self.fsz}; Encoding size = {self.esz}')

        # Latent vector
        self.latent_fc1 = nn.Sequential(
            nn.Linear(self.esz, latent_size, bias=False),
            nn.BatchNorm1d(latent_size, affine=False),
            get_act(self.act, inplace=kwargs['inplace']))
        self.latent_fc2 = nn.Linear(latent_size, latent_size, bias=False)

        # Back to conv
        self.decode_fc1 = nn.Sequential(
            nn.Linear(latent_size, latent_size, bias=False),
            nn.BatchNorm1d(latent_size, affine=False),
            get_act(self.act, inplace=kwargs['inplace']))
        self.decode_fc2 = nn.Sequential(
            nn.Linear(latent_size, self.esz, bias=False),
            nn.BatchNorm1d(self.esz, affine=False),
            get_act(self.act, inplace=kwargs['inplace']))

        # replace first upsampconv to not reduce channels
        nc = int(2 ** (self.channel_base_power + n_layers - 1))
        if not self.all_conv: self.upsampconvs[0] = self._upsampconv(nc, nc)

    def _dropout(self, x):
        return F.dropout(x, self.dropout_prob, training=self.enable_dropout, inplace=self.inplace)

    def _encode(self, x):
        sz = [x.shape]
        if self.semi_3d: x = self._add_noise(self.init_conv(x))
        for si in self.start: x = self._add_noise(si(x))
        x = self._down(x, 0)
        if self.all_conv: x = self._add_noise(x)
        for i, dl in enumerate(self.down_layers, 1):
            if self.resblock: xr = x
            for dli in dl: x = self._add_noise(dli(x))
            sz.append(x.shape)
            x = self._down((x + xr) if self.resblock else x, i)
            if self.all_conv: x = self._add_noise(x)
        return x, sz

    def _decode(self, x, sz):
        nb = x.size(0)
        x = self._dropout(self.decode_fc1(x))
        x = self.decode_fc2(x).view(nb, *self.fsz)
        x = self._up(self._add_noise(x), sz[-1][2:], 0)
        if self.all_conv: x = self._add_noise(x)
        for i, ul in enumerate(self.up_layers, 1):
            if self.resblock: xr = x
            for uli in ul: x = self._add_noise(uli(x))
            x = self._up((x + xr) if self.resblock else x, sz[-i-1][2:], i)
            if self.all_conv: x = self._add_noise(x)
        if self.resblock: xr = x
        for eli in self.end: x = self._add_noise(eli(x))
        if self.resblock: x = x + xr
        return self._finish(x)

    def _latent(self, x):
        return self.latent_fc2(self._dropout(self.latent_fc1(x.view(x.size(0), self.esz))))

    def forward(self, x, **kwargs):
        x = self._interp(x, self.img_dim)
        x, sz = self._encode(x)
        z = self._latent(x)
        return self._decode(z, sz), z

    def predict(self, x, *args, **kwargs):
        """ predict from a sample `x` """
        return self._interp(self.forward(x)[0], x.shape[2:])

    def freeze(self):
        """ freeze encoder """
        for p in self.start.parameters(): p.requires_grad = False
        for p in self.down_layers.parameters(): p.requires_grad = False
        if self.all_conv:
            for p in self.downsampconvs.parameters(): p.requires_grad = False
        if self.semi_3d > 0:
            for p in self.init_conv.parameters(): p.requires_grad = False

    def _interp(self, x, sz):
        if x.shape[2:] != sz:
            x = F.interpolate(x, sz, mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)
        return x


class OCNet1(LAutoNet):
    """
    one-class classifier network
    """
    def __init__(self, n_layers:int, img_dim:Tuple[int], latent_size:int=50, loss:str=None,
                 beta:float=1., temperature:float=0.01, monte_carlo:int=1, **kwargs):
        use_bias = kwargs['enable_bias']
        super().__init__(n_layers, img_dim, loss, latent_size, **kwargs)
        self.classifier = nn.Linear(latent_size, 2, bias=use_bias)
        self.laplacian = use_laplacian(loss)
        self.criterion = OCMAELoss(beta) if self.laplacian else OCMSELoss(beta)
        self.temperature = temperature
        self.gradients = None
        self.n_samp = monte_carlo if self.dropout_prob > 0 else 1
        self.n_output = self.n_output + (1 if self.n_samp == 1 else 3)  # account for var, grad, and grad var images

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations(self, x):
        x = self._interp(x, self.img_dim)
        x, _ = self._encode(x)
        return x

    def forward(self, x:torch.Tensor, **kwargs):
        x = self._interp(x, self.img_dim)
        x, sz = self._encode(x)
        z = self._latent(x)
        x = self._decode(z, sz)
        zf = torch.cat((torch.randn_like(z)*self.temperature,z), dim=0)  # create fake (oos) data at the origin
        c = self.classifier(zf)
        return x, z, c

    def _fwd_predict(self, x:torch.Tensor):
        x = self._interp(x, self.img_dim)
        x, sz = self._encode(x)
        h = x.register_hook(self.activations_hook)
        z = self._latent(x)
        c = self.classifier(z)
        x = self._decode(z, sz)
        return x, c

    def _gradcam(self, x):
        self.zero_grad()
        with torch.enable_grad():
            x, c = self._fwd_predict(x)
            c[0,0].backward()
            reduce_dims = [0, 2, 3] if self.dim == 2 else [0, 2, 3, 4]
            pooled_gradients = torch.mean(self.gradients, dim=reduce_dims, keepdim=True)
            activations = self.get_activations(x).detach()
            activations *= pooled_gradients
            heatmap = torch.mean(activations, dim=1, keepdim=True)
            F.relu_(heatmap).div_(heatmap.max()+1e-6)
        return x, c, heatmap

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = [self._gradcam(x) for _ in range(self.n_samp)]
        yhats = torch.stack([o[0] for o in out]).cpu().detach()
        yhat = torch.mean(yhats, dim=0)
        yvar = torch.var(yhats, dim=0, unbiased=True)
        c = torch.mean(torch.stack([o[1] for o in out]).cpu().detach(), dim=0)
        heatmaps = torch.stack([o[2] for o in out]).cpu().detach()
        heatmap = torch.mean(heatmaps, dim=0)
        hmvar = torch.var(heatmaps, dim=0, unbiased=True)
        yhat = self._interp(yhat, x.shape[2:])
        yvar = self._interp(yvar, x.shape[2:])
        heatmap = self._interp(heatmap, x.shape[2:])
        hmvar = self._interp(hmvar, x.shape[2:])
        pred = torch.argmax(c, dim=1)
        logger.info(f'Prediction: {pred.detach().cpu().numpy().squeeze()}')
        out = (yhat, heatmap) if self.n_samp == 1 else (yhat, yvar, heatmap, hmvar)
        return torch.cat(out, dim=1)


class OCNet2(LAutoNet):

    def __init__(self, n_layers:int, img_dim:Tuple[int], latent_size:int=50, loss:str=None,
                 beta:float=1., temperature:float=0.1, monte_carlo:int=1, **kwargs):
        super().__init__(n_layers, img_dim, loss, latent_size, **kwargs)
        id = np.asarray(img_dim)
        sz_range = list(zip(np.around(0.25*id),np.around(0.5*id)))
        self.block = RandomBlock(sz_range, thresh=0, int_range=None, is_3d=self.dim == 3)
        self.criterion = SVDDMAELoss(latent_size, beta, temperature) if self.laplacian else \
                         SVDDMSELoss(latent_size, beta, temperature)
        self.n_samp = monte_carlo if self.dropout_prob > 0 else 1
        self.n_output = self.n_output + (1 if self.n_samp == 1 else 3)  # account for var, grad, and grad var images

    def forward(self, x:torch.Tensor, **kwargs):
        nb = x.size(0)
        x = self._interp(x, self.img_dim)
        with torch.no_grad():
            xa = torch.stack([self.block(xi) for xi in x.cpu().detach()], dim=0).to(x.device)
            x = torch.cat((xa+0.5*torch.randn_like(xa), x), dim=0)
        x, sz = self._encode(x)
        z = self._latent(x)
        return self._decode(z[nb:], sz), z

    def _fwd_predict(self, x):
        x = self._interp(x, self.img_dim)
        x, sz = self._encode(x)
        z = self._latent(x)
        x = self._decode(z, sz)
        return x, z

    def _grad_img(self, x):
        self.zero_grad()
        x = x.detach()
        x.requires_grad = True
        with torch.enable_grad():
            out = self._fwd_predict(x)
            err = F.mse_loss(out[1], self.criterion.c * torch.ones_like(out[1]))
            err.backward()
        grad = x.grad.detach()
        return out, torch.mean(torch.abs(grad), dim=1, keepdim=True)

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        out = [self._grad_img(x) for _ in range(self.n_samp)]
        yhats = torch.stack([o[0][0] for o in out]).cpu().detach()
        yhat = torch.mean(yhats, dim=0)
        yvar = torch.mean(torch.var(yhats, dim=0, unbiased=True), keepdim=True, dim=1)
        z = torch.mean(torch.stack([o[0][1] for o in out]), dim=0)
        grads = torch.stack([o[1] for o in out]).cpu().detach()
        grad = torch.mean(grads, dim=0)
        gvar = torch.var(grads, dim=0, unbiased=True)
        yhat = self._interp(yhat, x.shape[2:])
        yvar = self._interp(yvar, x.shape[2:])
        dist = F.mse_loss(z, torch.ones_like(z)*self.criterion.c)
        logger.info(f'Anomaly score: {dist.item():.3e}')
        with open('scores.txt', 'a') as f: f.write(f'{dist.item():.8f}\n')
        out = (yhat, grad) if self.n_samp == 1 else (yhat, yvar, grad, gvar)
        return torch.cat(out, dim=1)
