# -*- coding: utf-8 -*-

import numpy as np


class PolynomialRegressionFunction:
    def __init__(self, exponent):
        self.exponent = exponent

    def __str__(self):
        return f'x**{self.exponent}'

    def __call__(self, x):
        return x ** self.exponent


class RegressionBasis:
    def __init__(self, regression_functions):
        self.regression_functions = regression_functions

    def __str__(self):
        return ' + '.join(str(f) for f in self.regression_functions)

    def apply(self, x):
        for f in self.regression_functions:
            yield f(x)

    def __call__(self, x):
        assert x.ndim == 1
        x = x.reshape((x.shape[0], 1))
        return np.concatenate(tuple(self.apply(x)), axis=1)


class PolynomialRegressionBasis(RegressionBasis):
    def __init__(self, degree):
        super().__init__([PolynomialRegressionFunction(i)
                          for i in range(degree + 1)])
        self.degree = degree
