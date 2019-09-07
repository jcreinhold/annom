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
           'LavaNet',
           'Lava2Net',
           'LRSDNet',
           'OCNet',
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
from .errors import *
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


class OCNet(Unet):
    """
    one-class classifier network
    """
    def __init__(self, n_layers:int, img_dim:Tuple[int], loss:str=None, beta:float=1., **kwargs):
        _ = kwargs.pop('no_skip')
        super().__init__(n_layers, loss=loss, no_skip=True, **kwargs)
        self.img_dim = img_dim
        z = self._test_z()
        self.z_sz = z.shape[2:]
        zs = np.asarray(self.z_sz) // 2
        nc = int(2 ** (self.channel_base_power + n_layers))
        no = int(2 ** self.channel_base_power)
        s = (2,2) if self.dim == 2 else (2,2,2)
        clsf = [*self._conv_act(nc, no, norm='weight', seq=False, stride=s)]
        while np.all(zs > 24):
            clsf.extend(self._conv_act(no, no, norm='weight', seq=False, stride=s))
            zs //= 2
        self.classifier = nn.Sequential(*clsf)
        self.o_sz = self._o_size(z)
        self.out = nn.Linear(np.prod(self.o_sz), 2)
        self.n_output = (self.n_output + 1) if not kwargs['color'] else (self.n_output * (4/3))  # for grad image
        self.laplacian = use_laplacian(loss)
        self.criterion = OCMAELoss(beta) if self.laplacian else OCMSELoss(beta)
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def get_activations(self, x):
        x = self._interp(x, self.img_dim)
        z, sz = self._encode(x)
        c = self.classifier(z)
        return c

    def forward(self, x:torch.Tensor, add_oos:bool=True, hook:bool=False, **kwargs):
        x = self._interp(x, self.img_dim)
        z, sz = self._encode(x)
        x = self._decode(z, sz)
        if add_oos: z = torch.cat((torch.randn_like(z[0:1,...])*0.1,z), dim=0)
        c = self.classifier(z)
        if hook: h = c.register_hook(self.activations_hook)
        c = torch.flatten(c, start_dim=1)
        c = self.out(c)
        return x, c

    def _test_z(self):
        with torch.no_grad():
            x = torch.randn(1, self.n_input, *self.img_dim, dtype=torch.float32)
            z, _ = self._encode(x)
        return z

    def _o_size(self, z):
        with torch.no_grad():
            o = self.classifier(z)
        return o.shape[1:]

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
        x = self._add_noise(self.bridge[0](x))
        x = self.bridge[1](x)
        return x, sz

    def _decode(self, x, sz):
        x = self._up(self._add_noise(x), sz[-1][2:], 0)
        if self.all_conv: x = self._add_noise(x)
        for i, (ul, s) in enumerate(zip(self.up_layers, reversed(sz)), 1):
            if self.attention is not None: x = self.attn[i-1](x)
            if self.resblock: xr = x
            for uli in ul: x = self._add_noise(uli(x))
            x = self._up((x + xr) if self.resblock else x, sz[-i-1][2:], i)
            if self.all_conv: x = self._add_noise(x)
        if self.attention is not None: x = self.attn[-1](x)
        if self.resblock: xr = x
        for eli in self.end: x = self._add_noise(eli(x))
        if self.resblock: x = x + xr
        return self._finish(x)

    def _gradcam(self, x):
        self.zero_grad()
        with torch.enable_grad():
            x, c = self.forward(x, add_oos=False, hook=True)
            c[:,0].backward()
            reduce_dims = [0, 2, 3] if self.dim == 2 else [0, 2, 3, 4]
            pooled_gradients = torch.mean(self.gradients, dim=reduce_dims, keepdim=True)
            activations = self.get_activations(x).detach()
            activations *= pooled_gradients
            heatmap = torch.mean(activations, dim=1, keepdim=True)
            F.relu_(heatmap).div_(heatmap.max())
        return x, c, heatmap

    def _interp(self, x, sz):
        return F.interpolate(x, sz, mode='bilinear' if self.dim == 2 else 'trilinear', align_corners=True)

    def predict(self, x:torch.Tensor, *args, **kwargs) -> torch.Tensor:
        yhat, c, heatmap = self._gradcam(x)
        yhat = self._interp(yhat, x.shape[2:])
        heatmap = self._interp(heatmap, x.shape[2:])
        logger.info(f'Prediction: {list(torch.argmax(c, dim=1).detach().cpu().squeeze())}')
        return torch.cat((yhat, heatmap), dim=1)
