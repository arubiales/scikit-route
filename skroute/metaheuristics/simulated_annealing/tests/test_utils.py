# import os, sys

# sys.path.append("/".join(os.path.abspath("").split("/")[:-1]))WW
from skroute.metaheuristics.simulated_annealing._base_simulated_annealing._utils_sa import (_check_temp, _normalize, _swap_route)
import numpy as np
import pytest
xfail = pytest.mark.xfail(strict=True)

class TestUtils:

    def test_check_temp(self):
        small_num = 1e-24
        assert isinstance(_check_temp(500, 600, 1), bool), "_check_temp must return a boolean"
        assert _check_temp(500, 600, np.inf), "_check_temp validator function is wrong, should be True"
        assert not _check_temp(500, 501, small_num), "_check_temp validator function is wrong, should be False"

    def test_normalize(self):
        a = 0.2
        b = 0.94
        one = 1
        zero = 0
        result = _normalize(a)
        assert result < 1 and result > 0.90, "_normalize must be between 0.9 and 1"
        assert isinstance(result, float), "_normalize must return a float"
        assert result and _normalize(b) > b, "_normalize must give back a biggest number"
        assert np.isclose(_normalize(b), 0.99, atol=1e-2), "_normalize result is not correct"
        assert _normalize(one) != one and _normalize(zero) != zero, "_normalize 0 or 1 nevers returns 0 or 1"


    def test_swap_route(self):
        a = [1,2,3,4,5,6]
        r1 = 2
        r2 = 4
        compare =  [1, 2, 5, 4, 3, 6]
        result = _swap_route(a, r1, r2)
        assert isinstance(result, list), "_swap_route must return a list"
        assert isinstance(result[3], int), "_swap_route must return a list of ints"
        assert sum(result) == sum(a), "_swap_route must return all the routes IDs"
        assert result == compare,  "_swap_route must swap 4 and 5 with theese arguments"
        assert _swap_route(a, r2, r1) == a, "_swap_route arguments r1 and r2 must be conmmutative"
        

    