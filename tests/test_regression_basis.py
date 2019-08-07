# -*- coding: utf-8 -*-

'''Tests for regression basis.'''

import unittest

from longstaff_schwartz.regression_basis import PolynomialRegressionBasis


class TestRegressionBasis(unittest.TestCase):
    '''Tests for regression basis.'''

    def test_polynomial_components(self):
        '''Test polynomial components.'''
        for n in range(10):
            regr = PolynomialRegressionBasis(n)
            self.assertEqual(len(regr.basis_functions), n + 1)
