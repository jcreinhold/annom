#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.layers

define layers for annom models

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Sep 16, 2019
"""

__all__ = ['SelfAttentionWithMap']

import math

import torch
from torch import nn


class SelfAttentionWithMap(nn.Module):
    """ Self attention layer that returns attn map """
    def __init__(self, n_channels:int):
        super().__init__()
        no = n_channels // 8 if (n_channels // 8) > 0 else 1
        self.query = nn.utils.spectral_norm(nn.Conv1d(n_channels, no, 1))
        self.key   = nn.utils.spectral_norm(nn.Conv1d(n_channels, no, 1))
        self.value = nn.utils.spectral_norm(nn.Conv1d(n_channels, n_channels, 1))
        self.gamma = nn.Parameter(torch.Tensor([0.]))

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2],-1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = torch.softmax(torch.bmm(f.permute(0,2,1).contiguous(), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        attn_map = torch.mean(beta, dim=2).unsqueeze(1).view(*size).contiguous()  # which pixels are attended to most
        return o.view(*size).contiguous(), attn_map
