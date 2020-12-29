from skroute.datasets import load_madrid
from skroute.preprocessing import dfcolumn_to_dict
from skroute._utils._utils import _cost, _final_cost


df_points_distances = load_madrid()["DataFrame"]

fuel_costs = dfcolumn_to_dict(df_points_distances, "id_origin", "id_destinity", "cost")
time_costs = dfcolumn_to_dict(df_points_distances, "id_origin", "id_destinity", "hours")
route_example = list(fuel_costs.keys()) 
    
    
class TestUtils:
    def test_cost(self):
        res = _cost(route_example, 18, time_costs, fuel_costs, 8, 12.45, 3)
        assert round(res, 2) == 1711.29, "The result of _cost function is not correct"
        assert isinstance(res, float), "The _cost function must return a float"


    def test_final_cost(self):
        res = _final_cost(route_example, 18, time_costs, 8)
        assert res == [10000016,3,6,8,9,10,28,29,33,58,60,61,62,66,10000016,67,75,85,88,10000016], "The result of _final_cost function is not correct"
        assert isinstance(res, list), "The _final_cost function must return a list"
        assert res[-1] == res[0], "The _final_cost function last element must be the same that the first"
        assert all(isinstance(num, int) for num in res), "The _final_cost function must return a list of integers"

