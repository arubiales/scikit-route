
import numpy as np
import pytest
xfail = pytest.mark.xfail(strict=True)
from skroute._utils._utils import _cost
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_alicante_murcia
import pandas as pd
import pytest
from skroute.heuristics.brute import BruteForce

df_alicante_barcelona = load_alicante_murcia()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_alicante_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_alicante_barcelona, "id_origin", "id_destinity", "hours")
route = list(time_matrix.keys())

class TestBruteForce:

    def test_instanciate(self):
        bf = BruteForce()
        assert str(type(bf)) == "<class 'skroute.heuristics.brute._base_brute_force._base_brute_force.BruteForce'>", "Brute force is not instanciate correctly"

    def test_parameters(self):
        esa1 = BruteForce(max_time_work=10)
        esa2 = BruteForce(max_time_work=10, extra_cost=2, people=3)
        assert esa1, "Brute force is not instanciate correctly"
        assert esa2, "Brute force is not instanciate correctly"

    def test_improve(self):
        bf = BruteForce(max_time_work=8, extra_cost=0, people=1)
        result = bf.fit(route, time_matrix, fuel_matrix)
        assert result == (328.47003173828125, [10000002, 70, 71, 72, 86, 92, 97, 99, 10000002]) , "The algorithm is not giving a perfect result"
