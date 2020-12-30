from skroute.preprocessing import dfcolumn_to_dict
from skroute.metaheuristics.tabu_search._utils_tabu import (create_combinations, cost_combinations, find_best_route,
                          random_swap_indices, cross_route, var_lower_length)
from skroute.datasets import load_barcelona
from skroute._utils._utils import _cost
import pytest
xfail = pytest.mark.xfail(strict=True)

df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")
random_route = list(dict.fromkeys(fuel_matrix).keys())




class TestUtilsTabuSearch:

    def test_setup(self):
        global combinations
        global costs
        try:
            combinations = create_combinations(random_route, random_route[0])
        except:
            assert False, "create_combinations input parameters are failing"

        try:
            costs = cost_combinations(combinations, random_route[0], 19, time_matrix, fuel_matrix, 8, 0, 1)
        except:
            assert False, "cost_combinations input parameters are failing"


    def test_create_combinations(self):
        assert type(combinations) == list, "create_combinations must return a list of lists"
        assert type(combinations[0]) == list, "create_combinations must return a list of lists"
        assert all(type(n) == int for comb in combinations for n in comb), "create_combinations must return a list of list formed by integers"
        assert random_route not in combinations, "create_combinations is not working properly because return the same route entered"
        assert all(random_route[0] == comb[0] for comb in combinations), "create_combination is moving the start route from the first position"
        assert all(len(comb) == 19 for comb in combinations), "create_combination is changing the length of the route"
        assert all(node in comb for comb in combinations for node in random_route), "create_combination is not keeping the same nodes ID"
        assert len(combinations) == 136, "create_combinations is creating more/less combinations than expected"

    def test_cost_combinations(self):
        assert len(costs) == 2, "cost_cobinations must be lenght 2"
        assert len(costs[0]) == len(costs[1]), "cost_combinations length of costs and lenght of "
        assert len(costs[0]) == 136, "cost_combinations length of costs"
        assert type(costs) == tuple, "cost_combinations must return a tuple"
        assert all(type(c) == tuple for c in costs), "cost_combinations must be a tuple of tuples"
        assert all(type(c) == float for c in costs[0]), "cost_combinations first tuple must be a tuple of floats"
        assert all(type(route) == list for route in costs[1]), "cost_combinations second tuple must be a tuple of lists"
        assert all(type(n) == int for route in costs[1] for n in route), "cost_combinations second tuple must be a tuple of list with integers"
        assert _cost(costs[1][0], 19, time_matrix, fuel_matrix, 8, 0, 1) == costs[0][0], "The _cost function is not working correctly"
        assert costs[0] == tuple(sorted(costs[0])), "The cost tuple returned is not sorted"

    def test_find_best_route(self):
        assert find_best_route(costs[0], costs[1]), "find_best_route input parameters are failing"
        best_route = find_best_route(costs[0], costs[1])
        assert type(best_route) == tuple, "find_best_route must return a tuple"
        assert type(best_route[0]) == float and type(best_route[1]) == list, "find_best_route, must return a tuple with a float and a list"
        assert len(best_route[1]) == 19, "find_best_route can't modify the length of the route"
        assert all(type(i) == int for i in best_route[1]), "find_best_route must return a route of integers IDs"
        assert best_route[0] == min(costs[0]), "find_best_route must return the minimun cost route"
        assert best_route[1] == costs[1][0], "find_best_route is not returning the best route"

    def test_random_swap_indices(self):
        
        assert random_swap_indices(19)
        swap_idx = random_swap_indices(19)
        assert len(swap_idx) == 2, "random_swap_indices must have length of two"
        assert type(swap_idx) == tuple, "random_swap_indices must return a tuple"
        assert type(swap_idx[0]) == int and type(swap_idx[1]) == int, "random_swap_indices must return a two integers inside the tuple"
        assert swap_idx[0] != swap_idx[1], "random_swap_indices must return two differents numbers"
    
    def test_cross_route(self):
        random_route_2 = random_route[:]
        route_changed = cross_route(random_route_2, 19)
        assert not route_changed, "cross_route must return None because the change is inplace"
        assert type(random_route_2) == list, "cross_route must keep a list"
        assert all(type(n) == int for n in random_route_2), "cross_route must keep integers ids route"
        assert random_route_2 != random_route, "cross_route must change the route passed"
        assert len(random_route_2) == len(random_route), "cross_route can't change the lenght of the route"
