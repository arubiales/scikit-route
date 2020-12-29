from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
from skroute._utils._utils import _cost
from skroute.metaheuristics.tabu_search._tabu_search import TabuSearch
import pytest
xfail = pytest.mark.xfail(strict=True)


df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")
random_route = list(dict.fromkeys(fuel_matrix).keys())

class TestTabuSearch:

    def test_instanciate(self):
        ts = TabuSearch()
        assert str(type(ts)) == "<class 'skroute.metaheuristics.tabu_search._tabu_search.TabuSearch'>"

    def test_parameters(self):
        ts1 = TabuSearch(searchs=1500, p_m=0.5, tabu_length=20, tabu_var=10, max_time_work = 6.34, extra_cost=23.3, people=3)
        ts2 = TabuSearch(searchs=500,p_m=0.5, tabu_length=100, tabu_var=50)
        ts3 = TabuSearch(100, 0.2, 200, 54, 24.)
        assert ts1
        assert ts2
        assert ts3

    def test_improve(self):
        ts = TabuSearch(searchs=2000, p_m=0.65, tabu_length=20, tabu_var=10)
        result = ts.fit(random_route, time_matrix, fuel_matrix)
        first_result = _cost(random_route, 19, time_matrix, fuel_matrix,8, 10, 1)
        assert result[0] < first_result, "The algorithm is not improving a random result."
        assert result[0] < 470., "The algorithm is not improving correctly."
        assert ts.history_, "Attribute history_ is not working"
