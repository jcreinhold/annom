#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.util

holds utilities for annom functions and classes

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 08, 2019
"""

__all__ = ['RandomBlock',
           'use_laplacian']

import random

import numpy as np


def use_laplacian(loss:str):
    if loss is not None:
        laplacian = True if loss.lower() == 'mae' else False
    else:
        laplacian = False
    return laplacian



class RandomBlock:
    """ add random blocks of random intensity to an image """
    def __init__(self, sz_range, thresh=None, int_range=None, is_3d=False):
        self.int = int_range
        self.sz = sz_range if all([isinstance(szr, (tuple,list)) for szr in sz_range]) else \
                  (sz_range, sz_range, sz_range) if is_3d else (sz_range, sz_range)
        self.thresh = thresh
        self.is_3d = is_3d

    def block2d(self, x):
        *_, hmax, wmax = x.shape
        mask = np.where(x >= (x.mean() if self.thresh is None else self.thresh))
        c = np.random.randint(0, len(mask[1]))  # choose the set of idxs to use
        h, w = [m[c] for m in mask[1:]]  # pull out the chosen idxs (2D)
        sh, sw = random.randrange(*self.sz[0]), random.randrange(*self.sz[1])
        oh = 0 if sh % 2 == 0 else 1
        ow = 0 if sw % 2 == 0 else 1
        if h+(sh//2)+oh >= hmax: h = hmax - (sh//2) - oh
        if w+(sw//2)+ow >= wmax: w = wmax - (sw//2) - ow
        if h-(sh//2) < 0: h = sh//2
        if w-(sw//2) < 0: w = sw//2
        int_range = self.int if self.int is not None else (x.min(), x.max()+1)
        x[...,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow] = np.random.uniform(*int_range)
        return x

    def block3d(self, x):
        *_, hmax, wmax, dmax = x.shape
        mask = np.where(x >= (x.mean() if self.thresh is None else self.thresh))
        c = np.random.randint(0, len(mask[1]))  # choose the set of idxs to use
        h, w, d = [m[c] for m in mask[1:]]  # pull out the chosen idxs (2D)
        sh, sw, sd = random.randrange(*self.sz[0]), random.randrange(*self.sz[1]), random.randrange(*self.sz[2])
        oh = 0 if sh % 2 == 0 else 1
        ow = 0 if sw % 2 == 0 else 1
        od = 0 if sd % 2 == 0 else 1
        if h+(sh//2)+oh >= hmax: h = hmax - (sh//2) - oh
        if w+(sw//2)+ow >= wmax: w = wmax - (sw//2) - ow
        if d+(sd//2)+od >= dmax: d = dmax - (sd//2) - od
        if h-(sh//2) < 0: h = sh//2
        if w-(sw//2) < 0: w = sw//2
        if d-(sd//2) < 0: d = sd//2
        int_range = self.int if self.int is not None else (x.min(), x.max()+1)
        x[...,h-sh//2:h+sh//2+oh,w-sw//2:w+sw//2+ow,d-sd//2:d+sd//2+od] = np.random.uniform(*int_range)
        return x

    def __call__(self, x):
        return self.block2d(x) if not self.is_3d else self.block3d(x)

    def __repr__(self):
        s = '{name}(sz={sz}, int_range={int}, thresh={thresh}, is_3d={is_3d})'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)
