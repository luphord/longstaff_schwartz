#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Tests for `longstaff_schwartz` package.'''


import unittest
from click.testing import CliRunner

import numpy as np

from longstaff_schwartz import cli
from longstaff_schwartz.algorithm import \
    longstaff_schwartz_american_option_quadratic


test_X = np.array([
    [1.00, 1.09, 1.08, 1.34],
    [1.00, 1.16, 1.26, 1.54],
    [1.00, 1.22, 1.07, 1.03],
    [1.00, 0.93, 0.97, 0.92],
    [1.00, 1.11, 1.56, 1.52],
    [1.00, 0.76, 0.77, 0.90],
    [1.00, 0.92, 0.84, 1.01],
    [1.00, 0.88, 1.22, 1.34]]).T

test_t = np.array([0, 1, 2, 3])

test_r = 0.06

test_strike = 1.1


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
        value = longstaff_schwartz_american_option_quadratic(test_X, test_t,
                                                             test_r,
                                                             test_strike)
        self.assertEqual(0.1144, np.round(value, 4))
        df = np.exp(-test_r * (test_t[-1] - test_t[0]))
        european_value = np.maximum(test_strike - test_X[-1, :], 0.0).mean() \
            * df
        self.assertEqual(0.0564, np.round(european_value, 4))
