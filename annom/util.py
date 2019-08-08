#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
annom.util

holds utilities for annom functions and classes

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Aug 08, 2019
"""

__all__ = ['use_laplacian']


def use_laplacian(loss:str):
    if loss is not None:
        laplacian = True if loss.lower() == 'mae' else False
    else:
        laplacian = False
    return laplacian
