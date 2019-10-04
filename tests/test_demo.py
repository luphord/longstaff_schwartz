import unittest


class TestDemo(unittest.TestCase):
    def test_readme_demo(self):
        '''Test usage demo including imports'''
        from longstaff_schwartz.algorithm import longstaff_schwartz
        from longstaff_schwartz.stochastic_process \
            import GeometricBrownianMotion
        import numpy as np

        # Model parameters
        t = np.linspace(0, 5, 100)  # timegrid for simulation
        r = 0.01  # riskless rate
        sigma = 0.15  # annual volatility of underlying
        n = 50  # number of simulated paths

        # Simulate the underlying
        gbm = GeometricBrownianMotion(mu=r, sigma=sigma)
        rnd = np.random.RandomState(1234)
        x = gbm.simulate(t, n, rnd)  # x.shape == (t.size, n)

        # Payoff (exercise) function
        strike = 0.95

        def put_payoff(spot):
            return np.maximum(strike - spot, 0.0)

        # Discount factor function
        def constant_rate_df(t_from, t_to):
            return np.exp(-r * (t_to - t_from))

        # Approximation of continuation value
        def fit_quadratic(x, y):
            return np.polynomial.Polynomial.fit(x, y, 2, rcond=None)

        # Selection of paths to consider for exercise
        # (and continuation value approxmation)
        def itm(payoff, spot):
            return payoff > 0

        # Run valuation of American put option
        npv_american = longstaff_schwartz(x, t, constant_rate_df,
                                          fit_quadratic, put_payoff, itm)

        # European put option for comparison
        npv_european = constant_rate_df(t[0], t[-1]) * put_payoff(x[-1]).mean()

        # Check results
        assert np.round(npv_american, 4) == 0.0702
        assert np.round(npv_european, 4) == 0.0598
        assert npv_american > npv_european
