# classes for bayesian optimization (min)

# import relevant packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
from sklearn.metrics import mean_squared_error
from itertools import product
from gp_para import gp_tuning
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import collections
from scipy.stats import norm
from scipy.stats import gamma

# stop showing warning messages
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

class GPUCB():

    def __init__(self, meshgrid, X, Y, beta=100., delta=0.05, dim=2, pre_defined_gp=None):
        '''
        meshgrid: Output from np.methgrid.
        e.g. np.meshgrid(np.arange(-1, 1, 0.1), np.arange(-1, 1, 0.1)) for 2D space
        with |x_i| < 1 constraint.
        beta (optional): Hyper-parameter to tune the exploration-exploitation
        balance. If beta is large, it emphasizes the variance of the unexplored
        solution solution (i.e. larger curiosity)
        '''
        # 参考 https://github.com/tushuhei/gpucb/blob/master/gpucb.py
        self.meshgrid = np.array(meshgrid)
        self.experimen_result = Y
        self.beta = beta
        self.iteration = 1
        self.dim = dim
        self.delta = delta
        self.acquired_pts = []
        self.x_only = []
        self.y_only = []  # for plotting
        if pre_defined_gp:
            self.gp = pre_defined_gp
        else:
            self.gp = GaussianProcessRegressor()
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []

    def argmin_ucb(self):
        res = np.argmin(self.mu - self.sigma * np.sqrt(self.beta))
        return res

    def learn(self):
        grid_idx = self.argmin_ucb()
        self.sample(self.X_grid[grid_idx])
        self.beta = 2 * np.log(
            self.X_grid.shape[0] * self.X_grid.shape[1] * (self.iteration ** 2) * (np.pi ** 2) / (6 * self.delta))
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
        self.iteration += 1

    def sample(self, x):
        t = self.experimen_result[tuple(x)]
        self.acquired_pts.append((x, t))
        self.y_only.append(t)
        self.x_only.append(x)
        self.X.append(x)
        self.T.append(t)


class EI():

    def __init__(self, meshgrid, X, Y, beta=100., delta=0.05, dim=2, pre_defined_gp=None):
        # beta, delta, and dim are not used in EI
        self.meshgrid = np.array(meshgrid)
        self.experimen_result = Y
        self.iteration = 1
        self.acquired_pts = []
        self.x_only = []
        self.y_only = []  # for plotting
        if pre_defined_gp:
            self.gp = pre_defined_gp
        else:
            self.gp = GaussianProcessRegressor()
        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []
        self.next_pt = self.X_grid[0]

    def learn(self):
        self.sample(self.next_pt)
        y_min = min(self.T)
        self.gp.fit(self.X, self.T)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
        var2 = np.maximum(self.sigma, 1e-8 + 0 * self.sigma)
        z = (y_min - self.mu) / np.sqrt(var2)
        self.next_pt = self.X_grid[np.argmax((y_min - self.mu) * norm.cdf(z) + np.sqrt(var2) * norm.pdf(z))]

        self.iteration += 1

    def sample(self, x):
        t = self.experimen_result[tuple(x)]
        self.acquired_pts.append((x, t))
        self.y_only.append(t)
        self.x_only.append(x)
        self.X.append(x)
        self.T.append(t)


class RGPUCB():

    def __init__(self, meshgrid, X, Y, beta=100., delta=0.05, dim=2, pre_defined_gp=None):
        # delta defines the scale parameter (usually denoted theta) for gamma distribution
        # larger scale means more exploration
        # beta and dim are not used
        self.meshgrid = np.array(meshgrid)
        self.experimen_result = Y
        self.beta = beta
        self.kappa = 1
        self.iteration = 5
        self.dim = dim
        self.delta = delta
        self.acquired_pts = []
        self.x_only = []
        self.y_only = []  # for plotting
        if pre_defined_gp:
            self.gp = pre_defined_gp
        else:
            self.gp = GaussianProcessRegressor()

        self.X_grid = self.meshgrid.reshape(self.meshgrid.shape[0], -1).T
        self.mu = np.array([0. for _ in range(self.X_grid.shape[0])])
        self.sigma = np.array([0.5 for _ in range(self.X_grid.shape[0])])
        self.X = []
        self.T = []
        self.next_pt = self.X_grid[0]

    def learn(self):
        self.sample(self.next_pt)
        self.gp.fit(self.X, self.T)
        # shape parameter for gamma distribution
        self.kappa = np.log((1 / np.sqrt(2 * np.pi)) * (self.iteration ** 2 + 1)) / np.log(1 + self.delta / 2)
        # beta sampled from gamma(self.kappa, self.delta)
        self.beta = rng.gamma(self.kappa, self.delta)
        self.mu, self.sigma = self.gp.predict(self.X_grid, return_std=True)
        self.next_pt = self.X_grid[np.argmin(self.mu - np.sqrt(self.beta) * self.sigma)]
        self.iteration += 1

    def sample(self, x):
        t = self.experimen_result[tuple(x)]
        self.acquired_pts.append((x, t))
        self.y_only.append(t)
        self.x_only.append(x)
        self.X.append(x)
        self.T.append(t)

# usage

# search_space = np.meshgrid(np.arange(0, 1.1, 0.2))
#
# test = GPUCB(search_space, X, lookup, beta=20., delta=0.1, dim=2,
#              pre_defined_gp=gp)
#
# for i in range(300):
#     test.learn()