# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats.distributions import norm, rv_frozen


class BrownianMotion:
    '''Brownian Motion (Wiener Process) with optional drift.'''
    def __init__(self, mu: float=0.0, sigma: float=1.0):
        self.mu = mu
        self.sigma = sigma

    def simulate(self, t: np.array, n: int, rnd: np.random.RandomState) \
            -> np.array:
        assert t.ndim == 1, 'One dimensional time vector required'
        assert t.size > 0, 'At least one time point is required'
        dt = np.concatenate((t[0:1], np.diff(t)))
        assert (dt >= 0).all(), 'Increasing time vector required'
        # transposed simulation for automatic broadcasting
        W = rnd.normal(size=(n, t.size))
        W_drift = (W * np.sqrt(dt) * self.sigma + self.mu * dt).T
        return np.cumsum(W_drift, axis=0)

    def distribution(self, t: float) -> rv_frozen:
        return norm(self.mu * t, self.sigma * np.sqrt(t))
