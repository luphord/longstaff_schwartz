# -*- coding: utf-8 -*-

'''Tests for regression basis.'''

import unittest
import numpy as np
from numpy.polynomial import Polynomial

from longstaff_schwartz.regression_basis import PolynomialRegressionBasis


class TestRegressionBasis(unittest.TestCase):
    '''Tests for regression basis.'''

    def test_polynomial_components(self):
        '''Test polynomial components.'''
        for n in range(10):
            regr = PolynomialRegressionBasis(n)
            self.assertEqual(len(regr.basis_functions), n + 1)

    def test_np_polynomial_api_compatibility(self):
        x = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.])
        y = np.array([0., -1., -1.4, -1.6, -1.2, -0.5, .9, 1.6, 2.1, 2.2, 2.3])
        p = Polynomial.fit(x, y, 4)
        r = PolynomialRegressionBasis(4).fit(x, y)
        self.assertTrue(np.allclose(p(x), r(x)))
        px, py = p.linspace()
        rx, ry = r.linspace()
        self.assertTrue(np.allclose(px, rx))
        self.assertTrue(np.allclose(py, ry))
        px, py = p.linspace(123, [-1, 1])
        rx, ry = r.linspace(123, [-1, 1])
        self.assertTrue(np.allclose(px, rx))
        self.assertTrue(np.allclose(py, ry))
        coef = p.convert(domain=[-1, 1]).coef
        self.assertTrue(np.allclose(coef, r.beta))
