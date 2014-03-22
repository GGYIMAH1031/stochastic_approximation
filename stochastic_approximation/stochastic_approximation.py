#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Python Implementation of Various Stochastic Approximation Algorithms
"""

import numpy as np


class StochasticApproximation(object):
    """ Stochastic Approximation """
    def __init__(self, initial, learning_rate, n_iter, **kwargs):
        self.initial = initial
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.result = np.zeros(n_iter + 1)
        self.result[0] = initial
        self.lb = kwargs.get('lb', None)
        self.ub = kwargs.get('ub', None)
        self.sample_path = kwargs.get('sample_path', False)

    def project(self, current):
        if (self.ub is not None) and (current > self.ub):
            current = self.ub
        elif (self.lb is not None) and (current < self.lb):
            current = self.lb
        return current


class KW(StochasticApproximation):
    """ Kiefer Wolfwitz Stochastic Approximation. """

    def __init__(self, initial, learning_rate, stepsize, response_function, n_iter=100, **kwargs):

        super(KW, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.response_function = response_function
        self.stepsize = stepsize

    def optimize(self):
        current = self.initial
        stepsize = self.stepsize

        for n in xrange(1, self.n_iter + 1):
            c_n = stepsize / (n**0.25)
            upper = current + c_n
            lower = current - c_n
            f_upper = self.response_function(upper)
            f_lower = self.response_function(lower)
            gradient = (f_upper - f_lower) / c_n
            current = current + self.learning_rate / n * gradient
            current = self.project(current)
            if self.sample_path:
                self.result[n] = current
        if self.sample_path:
            return self.result
        return current


class RM(StochasticApproximation):
    """ Robbins Monro Stochastic Approximation """

    def __init__(self, initial, learning_rate, gradient_function, n_iter=100, **kwargs):

        super(RM, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.gradient_function = gradient_function

    def optimize(self):
        current = self.initial

        for n in xrange(1, self.n_iter+1):
            gradient = self.gradient_function(current)
            current = current + self.learning_rate / n * gradient
            current = self.project(current)
            if self.sample_path:
                self.result[n] = current
        if self.sample_path:
            return self.result
        return current


class StarSA(StochasticApproximation):
    """ STAR Stochastic Approximation """

    def __init__(self, initial, learning_rate, stepsize, response_function,
                 gradient_function, n_iter=1000, **kwargs):
        super(StarSA, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.stepsize = stepsize
        self.response_function = response_function
        self.gradient_function = gradient_function

    def optimize(self):
        current = self.initial

        for n in xrange(1, self.n_iter+1):
            c_n = self.stepsize / (n**0.25)
            upper = current + c_n
            lower = current - c_n
            # alpha = sigma_g**2*c_n**2/(sigma_f**2+sigma_g**2*c_n**2)
            alpha = 0.5
            f_upper = self.response_function(upper)
            f_lower = self.response_function(lower)
            fd_gradient = (f_upper - f_lower) / c_n
            di_gradient = 0.5*(self.gradient_function(upper) +
                               self.gradient_function(lower))
            gradient = alpha * fd_gradient + (1 - alpha) * di_gradient
            current = current + self.learning_rate / n * gradient
            current = self.project(current)
            if self.sample_path:
                self.result[n] = current
        if self.sample_path:
            return self.result
        return current


class RSA(StochasticApproximation):
    """ Robust Stochastic Approximation """

    def __init__(self, initial, Dx, M, gradient_function, n_iter=1000, **kwargs):

        learning_rate = Dx / M
        super(RSA, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.gradient_function = gradient_function

    def optimize(self):
        current = self.initial

        for n in xrange(1, self.n_iter+1):
            current = current + self.learning_rate / np.sqrt(n) * self.gradient_function(current)
            current = self.project(current)
            if self.sample_path:
                self.result[n] = current
        if self.sample_path:
            return self.result
        return current


class IASA(StochasticApproximation):
    """Iterative Averaging Stochastic Approximation.
    """

    def __init__(self, initial, learning_rate, gradient_function, n_iter=1000, **kwargs):
        super(IASA, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.gradient_function = gradient_function

    def optimize(self):
        current = self.initial
        current_set = np.array([current])
        for n in xrange(1, self.n_iter+1):
            current = current + self.learning_rate / n * self.gradient_function(current)
            current = self.project(current)
            current_set = np.append(current_set, current)
            if self.sample_path:
                self.result[n] = current
        if self.sample_path:
            return self.result

        return np.mean(current_set)


class ACSA(StochasticApproximation):
    """ Accelerated Stochastic Approximation """

    def __init__(self, initial, L, gradient_function, n_iter, **kwargs):
        learning_rate = None
        super(ACSA, self).__init__(initial, learning_rate, n_iter, **kwargs)
        self.gradient_function = gradient_function
        self.L = L

    def optimize(self):
        current = self.initial
        aggregate = self.initial
        N = self.n_iter
        gamma = np.sqrt((self.L**2 + 0.1) * N * (N+1) * (N+2) / (1.5 * self.initial))
        # result = np.array([aggregate])
        for n in xrange(1, self.n_iter+1):
            alpha = 2.0 / (n + 1)
            medium = alpha * current + (1 - alpha) * aggregate
            current = current + n / gamma * self.gradient_function(medium)
            aggregate = alpha*current + (1-alpha)*aggregate
            current = self.project(current)
            if self.sample_path:
                self.result[n] = aggregate
        if self.sample_path:
            return self.result
        return aggregate