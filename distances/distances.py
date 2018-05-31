import numpy as np
import sys

class Point(object):
    """"A point in the plane.
    
    Attributes
    ----------
    x, y : float
        Point coodinates
        
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y
        

class Distance(object):
    """Distance between two points.
    
    Attributes
    ----------
    euclidian, manhattan : float
        distances between the two points.
        
    
    """
    def __init__(self, p1, p2):
        """
        Parameters
        ----------
        p1, p2 : Point
        
        """
        self.euclidian = self.compute_euclidian(p1, p2)
        self.manhattan = self.compute_manhattan(p1, p2)
        
    def compute_euclidian(self, p1, p2):
        return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)
    
    def compute_manhattan(self, p1, p2):
        return np.abs(p1.x - p2.x) + np.abs(p1.y - p2.y)
 

def min_mask(matrix):
    """Return the mask matrix of minima for each row of a matrix.
    
    Parameters
    ----------
    matrix : 2d numpy.array
        The matrix on which to compute the minima
    
    Return
    ------
    min_mat : 2d bool numpy.array of same shape than `matrix`
        binary matrix with True on each row indicating the minimum value in the corresponding row of the given matrix.
        
    Example
    -------
    >>> min_mask(np.array([[1, 2, 3],
    ...                    [2, 2, 0]]))
    array([[ True, False, False],
           [False, False,  True]], dtype=bool)
    
    """
    min_mat = np.zeros(matrix.shape, dtype=bool)
    min_mat[np.arange(matrix.shape[0]), matrix.argmin(axis=1)] = 1
    return min_mat

def min_indexes(matrix):
    """Return the indices of minima of each row of a matrix.
    
    If there is tie minima in a row , returns the index of the first one (minimum index).
    
    Parameters
    ----------
    matrix : 2d numpy.array
    
    Returns
    -------
    dict {'row_idx': 1d numpy.array, 'col_idx': 1d numpy.array}
        The row and column indices of minima of the given matrix.
    
    Examples
    --------
    >>> min_indexes(np.array([[2, 5, 1],
    ...                       [0, 0, 2]]))
    {'row_idx': array([0, 1]), 'col_idx': array([2, 0])}
    
    """
    return {'row_idx': np.arange(matrix.shape[0]), 'col_idx': matrix.argmin(axis=1)}

def deltas(vect):
    """Return the matrix of pairwise deltas between coordinates of a vector.
    
    Delta matrix element d_ij = (x_i - x_j) for all x_i, x_j in vect.
    
    Parameters
    ----------
    vect : 1d numpy.array
    
    Return
    ------
    2d numpy.array of shape (len(vect), len(vect))
        The square matrix of pairwise differences
        
    Examples
    --------
    >>> vect = np.array([0, 1, 3])
    ... deltas(vect)
    array([[ 0, -1, -3],
           [ 1,  0, -2],
           [ 3,  2,  0]])
    
    """
    size = len(vect)
    return np.repeat(vect, size, axis=0).reshape((size, size)) - np.tile(vect, size).reshape((size, size))

def pairwise_distances_euclidean(points):
    """Return the matrix of pairwise euclidean distances between points.
    
    Parameters
    ----------
    points: 2d numpy.array
        The coordinates of the point, points[0, :] and points[1, :] corresponding to x's and y's respectively.
    
    Returns
    -------
    2d numpy.array
        square matrix of pairwise euclidian distances between points.
        
    Examples
    --------
    >>> pairwise_distances_euclidean(np.array([[0, 1, 2], [0, 1, 0.5]]))
    array([[ 0.        ,  1.41421356,  2.06155281],
           [ 1.41421356,  0.        ,  1.11803399],
           [ 2.06155281,  1.11803399,  0.        ]])
    
    """
    xs, ys = points
    return np.sqrt(deltas(xs)**2 + deltas(ys)**2)

def pairwise_distances_manhattan(points):
    """Return the matrix of pairwise manahttan distances between points.
    
    idem as pairwise_distances_euclidean but for Manhattan distance.
        
    Examples
    --------
    >>> pairwise_distances_manhattan(np.array([[0, 1, 2], [0, 1, 0.5]]))
    array([[ 0. ,  2. ,  2.5],
           [ 2. ,  0. ,  1.5],
           [ 2.5,  1.5,  0. ]])
           
    """
    xs, ys = points
    return np.abs(deltas(xs)) + np.abs(deltas(ys))

def min_distances(points, distance='euclidean'):
    """Return the distances of nearest point for each point.
    
    For each point in `points` return the distance of the nearest point in `points`
    
    Parameters
    ----------
    points: 2d numpy.array
        The coordinates of the point, points[0, :] and points[1, :] corresponding to x's and y's respectively.
    
    Returns
    -------
    1d numpy.array
        
    Examples
    --------
    >>> min_distances(np.array([[0, 1, 2], [0, 1, 0.5]]))
    array([ 1.41421356,  1.11803399,  1.11803399])
        
    """
    if distance == 'euclidean':
        dist_func = pairwise_distances_euclidean
    elif distance == 'manhattan':
        dist_func = pairwise_distances_manhattan
    else:
        print("{} is not an available distance".format(distance))
        sys.exit(1)
    pdists = dist_func(points)
    #replace 0's by infinity
    pdists[pdists == 0] = np.inf
    return np.min(pdists, axis=0)