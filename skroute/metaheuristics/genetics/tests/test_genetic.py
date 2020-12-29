from skroute.metaheuristics.genetics import Genetic
from skroute.metaheuristics.genetics._base_genetics._utils_genetic import _cost
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
import pandas as pd
import pytest

df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")

route = list(time_matrix.keys())


class TestGenetic:

    def test_instanciate(self):
        ga = Genetic()
        assert str(type(ga)) == "<class 'skroute.metaheuristics.genetics._base_genetics._base_genetic.Genetic'>"

    def test_parameters(self):
        ga1 = Genetic(p_c=0.7,p_m=0.5, pop=200, gen=1600, k=3, early_stopping=50, max_time_work = 6.34, extra_cost=23.3, people=3)
        ga2 = Genetic(p_c=0.7,p_m=0.5, pop=200, gen=1600, k=3)
        ga3 = Genetic(0.7, 0.5, 200, 154354, 54, None)
        assert ga1
        assert ga2
        assert ga3

    def test_improve(self):
        ga = Genetic(p_c=0.6, p_m=0.4, gen=2000, pop=200, early_stopping=300)
        result = ga.fit(route, time_matrix, fuel_matrix)
        first_result = _cost(route, 19, time_matrix, fuel_matrix,8, 10, 1)
        assert result[0] < first_result, "The algorithm is not improving a random result."
        assert result[0] < 470., "The algorithm is not improving correctly."
        assert ga.history_, "Attribute history_ is not working"
        assert len(ga.history_) < 2000, "early_stopping is not working"
