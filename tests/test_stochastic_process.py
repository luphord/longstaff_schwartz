# -*- coding: utf-8 -*-

'''Tests for stochastic process implementations.'''

import unittest

import numpy as np
from numpy.random import RandomState
from scipy.stats import kstest

from longstaff_schwartz.stochastic_process import BrownianMotion


class TestRegressionBasis(unittest.TestCase):
    '''Tests for stochastic process implementations.'''

    def setUp(self):
        self.bm = BrownianMotion(mu=0.123, sigma=0.456)
        self.rnd = RandomState(1234)

    def test_brownian_motion_distribution(self):
        '''Test terminal distribution of Brownian Motion.'''
        t = np.linspace(0, 20, 20)
        n = 200
        x = self.bm.simulate(t, n, self.rnd)
        self.assertEqual((t.size, n), x.shape)
        self.assertEqual(n, x[-1, :].size)
        terminal_dist = self.bm.distribution(t[-1])
        test_result = kstest(x[-1, :], terminal_dist.cdf)
        self.assertGreater(test_result.pvalue, 0.4)
