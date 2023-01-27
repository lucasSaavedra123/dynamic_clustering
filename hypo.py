from numpy import prod, zeros, random, array, cumsum, sqrt
from math import exp
from numpy import linspace, array
import matplotlib.pyplot as plt

#https://github.com/kevinzagalo/hypoexponential

class Hypoexponential:
    # only works if the parameters eta are distinct
    def __init__(self, eta):
        self._eta = eta
        self._prod_eta = []
        self._weights = []
        for i, eta_i in enumerate(self._eta):
            tmp_list = list(self._eta[:i])
            tmp_list.extend(self._eta[i+1:])
            self._prod_eta.append(prod([(eta_j - eta_i) for eta_j in tmp_list]))
            self._weights.append(prod([1 / (1 - eta_i/eta_j) for eta_j in tmp_list]))

    def pdf(self, x):
        return [prod(self._eta) * sum([exp(-eta_i * xx) / self._prod_eta[i]
                                      for i, eta_i in enumerate(self._eta)]) for xx in x]

    def cdf(self, x):
        return [sum([(1-exp(-eta_j * xx)) * self._weights[j] for j, eta_j in enumerate(self._eta)]) for xx in x]

    @property
    def weights(self):
        return self._weights

    @property
    def params(self):
        return {'rates': self._eta}

    def sample(self, n_sample=1):
        S = zeros((n_sample, len(self._eta)))
        for i, eta_i in enumerate(self._eta):
            S[:, i] = random.exponential(1/eta_i, size=n_sample)
        return S.sum(axis=1)
