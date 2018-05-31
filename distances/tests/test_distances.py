import numpy as np
from distances.distances import *

##### Point and Distance
p1 = Point(0, 1)
p2 = Point(1, 0.5)
dist = Distance(p1, p2)

def test_compute_euclidian():
    assert dist.euclidian == np.sqrt(5) / 2
    
def test_compute_manhattan():
    assert dist.manhattan == 1.5

test_compute_euclidian()
test_compute_manhattan()
    
##### min
mat = np.array([[1, 2, 3], [2, 2, 0]])  

def test_min_mask():
    expected = np.array([[ True, False, False], [False, False, True]])
    result = min_mask(mat)
    assert (result == expected).all()
    
def test_min_indexes():
    expected = {'row_idx': np.array([0, 1]), 'col_idx': np.array([0, 2])}
    result = min_indexes(mat)
    for key in result.keys():
        assert (expected[key] == result[key]).all
    
test_min_mask()
test_min_indexes()

