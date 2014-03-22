#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_stochastic_approximation
----------------------------------

Tests for `stochastic_approximation` module.
"""

import unittest

from stochastic_approximation import stochastic_approximation
import numpy as np


class Test_SA(unittest.TestCase):

    def setUp(self):
        self.initial = 10
        self.learning_rate = 10
        self.n_iter = 1000
        self.stepsize = 0.1
        self.a = 0.1
        self.noise_var = 0.01
        self.lb = -50
        self.ub = 50
        self.Dx = 100.0
        self.M = 10.0
        self.L = 0.0
        self.response_function = lambda x: -self.a*x**2 + np.random.normal(scale=self.noise_var, size=1)
        self.gradient_function = lambda x: -self.a*x*2 + np.random.normal(scale=self.noise_var, size=1)

    def test_kw(self):
        kw = stochastic_approximation.KW(self.initial, self.learning_rate, self.stepsize,
                                         self.response_function, self.n_iter, lb=self.lb,
                                         ub=self.ub)
        self.assertTrue(abs(kw.optimize()) < 5)

    def test_kw_path(self):
        kw = stochastic_approximation.KW(self.initial, self.learning_rate, self.stepsize,
                                         self.response_function, self.n_iter, lb=self.lb,
                                         ub=self.ub, sample_path=True)
        self.assertTrue(len(kw.optimize()) == self.n_iter + 1)

    def test_rm(self):
        rm = stochastic_approximation.RM(self.initial, self.learning_rate, self.gradient_function,
                                         self.n_iter, lb=self.lb, ub=self.ub)
        self.assertTrue(abs(rm.optimize() < 5))

    def test_star_sa(self):
        star = stochastic_approximation.StarSA(self.initial, self.learning_rate, self.stepsize,
                                               self.response_function, self.gradient_function,
                                               self.n_iter, lb=self.lb, ub=self.ub)
        self.assertTrue(abs(star.optimize() < 5))

    def test_rsa(self):
        rsa = stochastic_approximation.RSA(self.initial, self.Dx, self.M, self.gradient_function,
                                           self.n_iter, lb=self.lb, ub=self.ub)
        self.assertTrue(abs(rsa.optimize() < 5))

    def test_iasa(self):
        iasa = stochastic_approximation.IASA(self.initial, self.learning_rate, self.gradient_function,
                                             self.n_iter, lb=self.lb, ub=self.ub)
        self.assertTrue(abs(iasa.optimize() < 5))

    def test_acsa(self):
        acsa = stochastic_approximation.ACSA(self.initial, self.L, self.gradient_function,
                                             self.n_iter, lb=self.lb, ub=self.ub)
        self.assertTrue(abs(acsa.optimize() < 5))

    def tearDown(self):
        pass

if __name__ == '__main__':
    unittest.main()