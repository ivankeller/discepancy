# A module to simulate rain on interval [0, 1]

import matplotlib.pyplot as plt
import numpy as np
import functools


def rejection_sampling(f):
    """Sample distribution given the density function f (assumed to be f: [0, 1] -> [0, 1]).
    f does not need to be normalized.

    Parameters
    ----------
    f : function
        the function to sample

    Returns
    -------
    sample : float

    """
    u = np.random.sample()
    v = np.random.sample()
    if v < f(u):
        return u
    else:
        return rejection_sampling(f)


def plot_density(p, size=3, linewidth=4):
    """Plot density function on interval [0, 1].

    Parameters
    ----------
    p : function
        the density function to plot (must be define on [0, 1])
    size : int (optional)
        the size of the figure
    linewidth : int (optional)
        the line width
    """
    x = np.linspace(0, 1, 1000)
    y = np.vectorize(p)(x)
    plt.figure(figsize=(size, size))
    plt.plot(x, y, linewidth=linewidth)


def unif(x, p0=0.5, r=0.1):
    """Define indicative function of interval in [0, 1] given center and radius.

    Parameters
    ----------
    x : float in [0, 1]
        function variable
    p0 : float in [0, 1] (optional)
        center
    r : float (optional)
        radius

    Returns
    -------
    0 or 1 : value of function on x
    """
    if max(0, p0 - r) < x < min(1, p0 + r):
        return 1
    else:
        return 0


def unif_segmented(x, segments):
    """Define indicative function of union of intevals in [0, 1].

    Parameters
    ----------
    x : float in [0, 1]
        function variable
    segments : list of intervals in [0, 1]
        ex: [[0.1, 0.2], [0.5, 1.]]

    Returns
    -------
    0 or 1 : value of function on x

    """
    in_seg = [seg[0] < x < seg[1] for seg in segments]
    return functools.reduce( (lambda x, y: x or y), in_seg )


def unif_pieces(x, centroids, r):
    """Define indicative function of union of intervals given centroids and a unique radius.

    Parameters
    ----------
    x : float in [0, 1]
        function variable
    centroids : list of floats in [0, 1]
    r : float
        unique radius

    Returns
    -------
    0 or 1 : value of function on x
    """
    segments = [[max(0, c-r), min(1, c+r)] for c in centroids]
    return unif_segmented(x, segments)


def density_drops(x, centroids, fill=0.5):
    """Define the probability density function of next drop given previous drop locations.

    Parameters
    ----------
    x : float in [0, 1]
        function variable
    centroids : list of floats in [0, 1]
        location of previous drops on the interval [0, 1]
    fill : float in [0, 1] (optional)
        the fraction of total forbidden space on interval [0, 1] for next drop

    Returns
    -------
    0 or 1 : value of function on x
    """
    nb_drops = len(centroids)
    r = fill / (2 * nb_drops)
    return 1 - unif_pieces(x, centroids, r)
