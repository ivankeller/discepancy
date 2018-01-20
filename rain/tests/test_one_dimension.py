from rain.one_dimension import unif

def test_unif():
    assert unif(0.1) == 0
    assert unif(0.5) == 1
