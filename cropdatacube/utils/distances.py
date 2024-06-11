import math
import numpy as np

def euclidean_distance(p1,p2):
    return math.sqrt(
        math.pow(p1[0] - p2[0],2) + math.pow(p1[1] - p2[1],2))


def calculate_distance_matrix(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calculate the half-distance matrix between two sets of coordinates using Euclidean distance.
    
    Parameters
    ----------
    coords1 : np.ndarray
        First set of coordinates with shape (N, D), where N is the number of points and D is the dimensionality.
    coords2 : np.ndarray
        Second set of coordinates with shape (N, D), where N is the number of points and D is the dimensionality.
    
    Returns
    -------
    np.ndarray
        A matrix of shape (N, N) containing the half-distance values.
    
    Raises
    ------
    AssertionError
        If the number of points in `coords1` and `coords2` are not the same.
    """
    coordsshape = coords1.shape[0]
    assert  coordsshape == coords2.shape[0], 'Coordinates must have the same number of points.'
    
    distances = np.zeros((coordsshape,coordsshape))
    for i in range(coordsshape):
        p1 = coords1[i]
        for j in range(i,coordsshape):
            p2 = coords2[j]
            distances[i,j] = euclidean_distance(p1,p2)
    
    return distances