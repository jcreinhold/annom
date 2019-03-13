#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
tests.test_loss

test the annom loss functions

Author: Jacob Reinhold (jacob.reinhold@jhu.edu)

Created on: Mar 11, 2019
"""

import unittest

import torch

from annom.loss import HotLoss, LRSDecompLoss, OrdLoss


class TestLoss(unittest.TestCase):

    def setUp(self):
        pass

    def test_hot(self):
        hl = HotLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = hl(x, y)
        self.assertEqual(loss.item(), 0)

    def test_lrsd(self):
        lrsd = LRSDecompLoss()
        x, y = (torch.zeros((2,1,2,2,2)), torch.zeros((2,1,2,2,2))), torch.zeros((2,1,2,2,2))
        loss = lrsd(x, y)
        self.assertEqual(loss.item(), 0)

    def test_ord(self):
        ol = OrdLoss((0,1,2), 'cpu', is_3d=False)
        x, y = torch.zeros((2,2,2,2)), torch.zeros((2,2,2))
        x[:,0,:,:] = 1
        x[:,0,0,0] = 0
        x[:,1,0,0] = 1
        y[:,0,0] = 1
        loss = ol(x, y)
        self.assertLess(loss.item(), 2)

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()

