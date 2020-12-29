from skroute.metaheuristics.genetics import EnsembleGenetic
from skroute._utils._utils import _cost
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
import pytest

df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")

route = list(time_matrix.keys())

class TestEnsembleGenetic:

    def test_instanciate(self):
        eg = EnsembleGenetic()
        assert str(type(eg)) == "<class 'skroute.metaheuristics.genetics._base_ensemble_genetics.EnsembleGenetic'>"

    def test_parameters(self):
        eg1 = EnsembleGenetic(p_c=0.7,p_m=0.5, pop=200, gen=1600, k=3, early_stopping=50, max_time_work = 8, extra_cost=23.3, people=3, n_jobs=4, n_genetics=4)
        eg2 = EnsembleGenetic(p_c=0.7,p_m=0.5, pop=200, gen=1600, k=3, n_jobs=2, n_genetics=2)
        eg3 = EnsembleGenetic(2, 0.7, 0.5, 200, 154354, 54, None)
        assert eg1
        assert eg2
        assert eg3

    def test_improve(self):
        eg = EnsembleGenetic(n_genetics=4, n_jobs=4, p_c=0.6, p_m=0.4, gen=2000, pop=200, early_stopping=300)
        result = eg.fit(route, time_matrix, fuel_matrix)
        first_result = _cost(route, 19, time_matrix, fuel_matrix, 8, 10, 1)
        assert result[0] < first_result, "The algorithm is not improving a random result."
        assert result[0] < 470., "The algorithm is not improving correctly."
