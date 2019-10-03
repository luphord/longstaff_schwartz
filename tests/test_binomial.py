import unittest

import numpy as np

from longstaff_schwartz.binomial import BinomialModel, create_binomial_model, \
    call_payoff, european_call_price, european_put_price, american_call_price,\
    american_put_price, american_put_exercise_barrier, \
    american_put_exercise_barrier_fitted


class TestBinomial(unittest.TestCase):
    '''Tests for `binomial` package.'''

    def test_european_call_price_decrease_with_strike(self):
        mdl = create_binomial_model(0.1, 0.02, 100, 1, n=100)
        last = None
        for s in range(90, 110):
            current = european_call_price(mdl, s)
            if last is not None:
                self.assertLess(current, last)
            last = current

    def test_thayer_watkins_example(self):
        '''Test example values by Thayer Watkins in
           http://www.sjsu.edu/faculty/watkins/binomial.htm
        '''
        u = 1 + 0.1
        d = 1 - 0.1
        r = np.log(1 + 0.05)  # = 0.05 with discrete compounding
        S0 = 100
        T = 1
        n = 1
        strike = 95
        mdl = BinomialModel(u, d, r, S0, T, n)
        npv = european_call_price(mdl, strike)
        self.assertEqual(10.71, np.round(npv, 2))

    def test_cox_ross_rubinstein_example(self):
        '''Test example values by Cox, Ross, Rubinstein in
           "Option pricing: A simplified approach", chapter 4
        '''
        S0 = 80
        n = 3
        T = 3
        strike = 80
        u = 1.5
        d = 0.5
        r = np.log(1.1)
        mdl = BinomialModel(u, d, r, S0, T, n)
        npv = european_call_price(mdl, strike)
        self.assertEqual(34.080, np.round(npv, 2))  # 34.065 in paper
        expected_payoff = np.array([0, 0, 90 - 80, 270 - 80])
        payoff = call_payoff(strike)(mdl.ST)
        self.assertTrue(np.allclose(expected_payoff, payoff))
        self.assertTrue(np.allclose(0.6, mdl.q))

    def test_put_call_parity(self):
        expected = []
        actual = []
        for sigma in [0.05, 0.1, 0.2]:
            for r in [-0.02, 0.0, 0.01, 0.1]:
                for S0 in [50, 100, 5000]:
                    mdl = create_binomial_model(0.1, 0.02, 100, 1, n=100)
                    for strike in [80, 90, 100, 110, 120]:
                        expected.append(mdl.S0 -
                                        strike * np.exp(-mdl.r * mdl.T))
                        actual.append(european_call_price(mdl, strike) -
                                      european_put_price(mdl, strike))
        self.assertTrue(np.allclose(expected, actual))

    def test_american_european_call_equality(self):
        european = []
        american = []
        for sigma in [0.05, 0.1, 0.2]:
            for r in [-0.02, 0.0, 0.01, 0.1]:
                for S0 in [50, 100, 5000]:
                    mdl = create_binomial_model(0.1, 0.02, 100, 1, n=100)
                    for strike in [80, 90, 100, 110, 120]:
                        european.append(european_call_price(mdl, strike))
                        american.append(american_call_price(mdl, strike))
        self.assertTrue(np.allclose(european, american))

    def test_american_european_put_difference(self):
        european = []
        american = []
        for sigma in [0.05, 0.1, 0.2]:
            for r in [-0.02, 0.0, 0.01, 0.1]:
                for S0 in [50, 100, 5000]:
                    mdl = create_binomial_model(0.1, 0.02, 100, 1, n=100)
                    for strike in [80, 90, 100, 110, 120]:
                        european.append(european_put_price(mdl, strike))
                        american.append(american_put_price(mdl, strike))
        aamerican = np.array(american)
        aeuropean = np.array(european)
        self.assertTrue((aamerican > aeuropean).all())
        # test for some "real" difference, threshold 1.23 is rather arbitrary
        self.assertGreater((aamerican - aeuropean).max(), 1.23)

    def test_ucsd_example(self):
        '''Test American put example data found in
           https://www.math.ucsd.edu/~p1tong/ProgramForFun/
           Binomial%20Tree%20model%20for%20an%20American%20Put%20option.xls
        '''
        S0 = 50
        n = 5
        T = 0.4167
        strike = 50
        sigma = 0.4
        r = 0.1
        mdl = create_binomial_model(sigma, r, S0, T, n)
        npv = american_put_price(mdl, strike)
        self.assertAlmostEqual(4.48859919382338, npv)

    def test_deep_itm_american_out(self):
        '''Test a deep in the money american out option.'''
        S0 = 100
        n = 5
        T = 5
        strike = 500
        sigma = 0.4
        r = 0.1
        mdl = create_binomial_model(sigma, r, S0, T, n)
        npv = american_put_price(mdl, strike)
        self.assertAlmostEqual(strike - S0, npv)

    def test_american_exercise_barrier(self):
        '''Test exercise barrier of american out option.'''
        S0 = 100
        n = 100
        T = 5
        strike = 100
        sigma = 0.4
        r = 0.1
        mdl = create_binomial_model(sigma, r, S0, T, n)
        exercise_barrier = american_put_exercise_barrier(mdl, strike)
        last_even = np.nan
        last_odd = np.nan
        for i, s in enumerate(exercise_barrier):
            if i % 2 == 0:
                if not np.isnan(last_even):
                    self.assertGreaterEqual(s, last_even)
                last_even = s
            elif i % 2 == 1:
                if not np.isnan(last_odd):
                    self.assertGreaterEqual(s, last_odd)
                last_odd = s

    def test_american_exercise_barrier_fitted(self):
        '''Test fitted exercise barrier of american out option.'''
        S0 = 100
        n = 100
        T = 5
        strike = 100
        sigma = 0.4
        r = 0.1
        mdl = create_binomial_model(sigma, r, S0, T, n)
        barrier = american_put_exercise_barrier_fitted(mdl, strike, 2)
        barrier = barrier.convert(domain=[-1, 1])
        self.assertGreater(barrier.coef[0], 0)
        self.assertLessEqual(barrier.coef[1], 0)
        self.assertGreater(barrier.coef[2], 0)
