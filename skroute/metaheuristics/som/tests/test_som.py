import pytest
xfail = pytest.mark.xfail(strict=True)

import numpy as np
from skroute.datasets import load_barcelona
from skroute.metaheuristics.som._utils_som import *
from skroute.metaheuristics.som import SOM
from skroute._utils._utils import _cost
from skroute.preprocessing import dfcolumn_to_dict


df_barcelona = load_barcelona()["DataFrame"]
cost_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")
df_barcelona = df_barcelona[["id_origin", "lat_origin", "lon_origin"]].drop_duplicates()


nodes = []
for n in df_barcelona.to_numpy():
    n = list(n)
    n[0] = int(n[0])
    n[1] = float(n[1])
    n[2] = float(n[2])
    nodes.append(tuple(n))

nodes = tuple(nodes)


class TestSom:
    def test_instantiate(self):
        som = SOM()
        assert "<class 'skroute.metaheuristics.som.som.SOM'>" == str(type(som))

    def test_parameters(self):
        som1 = SOM()
        som2 = SOM(1000, 2000, 0.8, lr=0.7)
        som3 = SOM(lr_decay=0.2)
        assert som1, "SOM instantiate is not correct"
        assert som2, "SOM instantiate is not correct"
        assert som3, "SOM instantiate is not correct"

    @xfail
    def test_parameters_false_1(self):
        SOM(0.3)

    @xfail
    def test_parameters_false_2(self):
        SOM(10, 0.2)

    @xfail
    def test_parameters_false_3(self):
        SOM(10, 10, 2)

    @xfail
    def test_parameters_false_4(self):
        SOM(10, 10, 0,8, 2)

    @xfail
    def test_parameters_false_5(self):
        SOM(10, 10, 0.8, 0.4, 3)

    def test_improve(self):
        som = SOM()
        result = som.fit(nodes)
        assert 425 >_cost(result, 19, time_matrix, cost_matrix, np.inf, 10, 1), "Som is not improving the route"

