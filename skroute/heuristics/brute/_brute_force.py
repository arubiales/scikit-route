from itertools import permutations
import os, sys
sys.path.append("/".join(os.path.abspath("").split("/")[:-1]))
from ..._validators import _intenger_validator
from ..._utils._utils import _cost, _final_cost

#NOTE: Pasar a Cython

class BruteForce:

    def __init__(self, max_time_work=8, extra_cost=0, people=1):
        self.max_time_work = max_time_work
        self.extra_cost = extra_cost
        self.people = people

    def fit(self, route_example, time_matrix, cost_matrix):
        length = len(route_example)
        route_to_permute = route_example[1:]
        home = route_example[0]
        min_cost = _cost(route_example, length, time_matrix, cost_matrix, self.max_time_work, self.extra_cost, self.people)
        min_route = route_example
        for route in permutations(route_to_permute):
            list_route = list(route)
            list_route.insert(0, home)
            route_cost = _cost(list_route, length, time_matrix, cost_matrix, self.max_time_work, self.extra_cost, self.people)
            if route_cost < min_cost:
                min_route = list_route
                min_cost  = route_cost

        return min_cost, _final_cost(min_route, length, time_matrix, self.max_time_work)

