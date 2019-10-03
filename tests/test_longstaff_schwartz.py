#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for `longstaff_schwartz` package.'''


import unittest
from click.testing import CliRunner

import numpy as np
from numpy.polynomial import Polynomial
from numpy.random import RandomState

from longstaff_schwartz import cli
from longstaff_schwartz.algorithm import \
    longstaff_schwartz_american_option_quadratic, \
    ls_american_option_quadratic_iter, \
    longstaff_schwartz
from longstaff_schwartz.stochastic_process import GeometricBrownianMotion


# Test values from chapter 1, Numerical Example
# of originial Longstaff-Schwartz paper
X = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]]).T
t = np.array([0, 1, 2, 3])
r = 0.06
strike = 1.1
coef2 = np.array([-1.070, 2.983, -1.814])  # -1.813 in paper
coef1 = np.array([2.038, -3.335, 1.356])


# functions to plug into general algorithm to arrive at
# original paper special case examples


def american_put_payoff(spot):
    return np.maximum(strike - spot, 0.0)


def constant_rate_df(t_from, t_to):
    return np.exp(-r * (t_to - t_from))


def fit_quadratic(x, y):
    return Polynomial.fit(x, y, 2)


def itm(payoff, spot):
    return payoff > 0


class TestLongstaff_schwartz(unittest.TestCase):
    '''Tests for `longstaff_schwartz` package.'''

    def test_command_line_interface(self):
        '''Test the CLI.'''
        runner = CliRunner()
        result = runner.invoke(cli.main)
        assert result.exit_code == 0
        help_result = runner.invoke(cli.main, ['--help'])
        assert help_result.exit_code == 0
        assert '--help  Show this message and exit.' in help_result.output

    def test_longstaff_schwartz_paper_example(self):
        '''Test against example value of original Longstaff-Schwartz paper'''
        value = longstaff_schwartz_american_option_quadratic(X, t, r, strike)
        self.assertEqual(0.1144, np.round(value, 4))
        df = np.exp(-r * (t[-1] - t[0]))
        european_value = np.maximum(strike - X[-1, :], 0.0).mean() * df
        self.assertEqual(0.0564, np.round(european_value, 4))

    def test_longstaff_schwartz_paper_example_intermediate_values(self):
        intermediate = list(ls_american_option_quadratic_iter(X, t, r, strike))
        cashflow, x, fitted, continuation, exercise, ex_idx = intermediate[0]
        fitted_coef2 = np.round(fitted.convert(domain=[-1, 1]).coef, 3)
        self.assertTrue(np.allclose(coef2, fitted_coef2))
        cashflow, x, fitted, continuation, exercise, ex_idx = intermediate[1]
        fitted_coef1 = np.round(fitted.convert(domain=[-1, 1]).coef, 3)
        self.assertTrue(np.allclose(coef1, fitted_coef1))

    def test_longstaff_schwartz_paper_example_for_general_algorithm(self):
        '''Test general algorithm against example value
           of original Longstaff-Schwartz paper'''
        value = longstaff_schwartz(X, t, constant_rate_df, fit_quadratic,
                                   american_put_payoff, itm)
        self.assertEqual(0.1144, np.round(value, 4))
        df = np.exp(-r * (t[-1] - t[0]))
        european_value = american_put_payoff(X[-1, :]).mean() * df
        self.assertEqual(0.0564, np.round(european_value, 4))

    def test_general_against_specific_algorithm(self):
        '''Choose parameters for general algorithm implementation
           such that they math the american put option specific code
        '''
        gbm = GeometricBrownianMotion(mu=r, sigma=0.1)
        rnd = RandomState(1234)
        t = np.linspace(0, 5, 20)
        n = 50
        x = gbm.simulate(t, n, rnd)
        general = longstaff_schwartz(x, t, constant_rate_df, fit_quadratic,
                                     american_put_payoff, itm)
        specific = longstaff_schwartz_american_option_quadratic(x, t, r,
                                                                strike)
        self.assertAlmostEqual(general, specific)
        allpathregr = longstaff_schwartz(x, t, constant_rate_df, fit_quadratic,
                                         american_put_payoff)
        self.assertNotAlmostEqual(general, allpathregr)
