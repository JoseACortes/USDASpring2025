import numpy as np
import scipy.optimize

def identity(X, _):
    return np.ones_like(X[:, 0])

def const(X, t):
    return np.ones_like(X[:, 0])*t

def linear(X, c, d):
    x, y, z = X.T
    return c * z + d

def plane(X, a, b, c, d):
    x, y, z = X.T
    return a * x + b * y + c * z + d

def quad(X, a, b, c, d):
    x, y, z = X.T
    return a * x**2 + b * y**2 + c * z**2 + d

def quad_plane(X, a, b, c, d, e, f, g):
    x, y, z = X.T
    return a * x**2 + b * y**2 + c * z**2 + d * x + e * y + f * z + g

def fit_function(X, y, func):
    popt, _, infodict, __, ___ = scipy.optimize.curve_fit(func, X, y, full_output=True)
    return popt, infodict