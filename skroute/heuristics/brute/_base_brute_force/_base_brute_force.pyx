#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
"""
Brute Force

Notes
-----
Maybe we could speed it by creating C++ unordered dicts and vectors but Python dicts look-up
are really fast and the append method is very fast too. Also in the future add contiguos arrays
could speed up (C arrays)


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""
from itertools import permutations
import os, sys
sys.path.append("/".join(os.path.abspath("").split("/")[:-1]))
sys.path.append("/".join(os.path.abspath("").split("/")[:-2]))
from ...._utils._utils import _cost, _final_cost
from ...._validators._validators import (_intenger_validator, _float_validator,
                          _validate_dict_of_dicts, _validate_route_example)


cdef class BruteForce():
    """
BruteFoce is a heuristic algorithm that checks all possible combinatios from 
a given example route. This algorithm always give back the best solution
but the time increase dramatically with the length of the route, it's not 
recommended use BruteForce with routes with lenght higher than 14. The 
total computations of the algorith could be calculted making the factorial
of the length of the route.


Parameters
-----------

max_time_work: float32, default=8
    the number of ours that a employ can work per day. For example
    if it's 8 hours, the algorithm will force that a route have to
    finish after the 8 hours have been completed, making the
    employeed come back home. it's a time constraint.

extra_cost: float32, default=0
    if it's 0 anything happend. If it's >0 in combination with
    max_time_work when the max_time_work is reached, extra_cost is
    applied. This add a extra cost to the solution each time that
    max_time_work is reached. It's like extra pay for the worker 
    each time max_time_work is completed (journey).

people: int32, default=1
    The number of people that you use in each route, for example if
    you need two truck drivers. that's another contstraint. That
    will multiply the time_costs and the extra_cost. Not the travel
    cost because it's assumed that both go in the same vehicle.

--------------------------------------------------------------------------

Methods:
--------

fit(self, list route_example, dict time_matrix, dict fuel_matrix)
    Execute the algorithm and give back the best solution

    Parameters
    ----------
    route_example: list
        it's a list that contain a random example of ints ids routes. But it's
        mandatory that the origin place, be the first id of the list

    time_costs: dict
        it's a dict of dicts with the points to visit as a keys and
        the value is another dictionary with the points to visit and
        the value of going for the first point to the second. It's a
        dictionary representation of a matrix. For example if we
        have 3 ID 1, 2 and 3 the dict will be like this:

            {
            1:{
                1:0,
                2:x,
                3:y
                },
            2:{
                1:x,
                2:0,
                3:z
                },
            3:{
                1:y,
                2:z,
                3:0
                }
            }

        This is just a (3, 3) symmetric matrix with the cost in
        time from one point to another with column and index.

            1   2   3
        1   0   x   y
        2   x   0   z
        3   y   z   0
    
    fuel_costs: dict
        For a extended explanation go up to time_matrix. it's
        the fuel cost to go from one point to another.

    Returns:
    --------
    The best route possible.
    """
    
    cdef public:
        float max_time_work
        float extra_cost
        int people

    def __init__(self, float max_time_work=8.0, float extra_cost=10.0, int people=1):
        self.max_time_work = _float_validator(max_time_work, u"max_time_work")
        self.extra_cost = _float_validator(extra_cost, u"extra_cost")
        self.people = _intenger_validator(people,  u'people')


    cpdef tuple fit(self, list route_example, dict time_costs, dict fuel_costs):
        _validate_dict_of_dicts(time_costs, u"time_cost")
        _validate_dict_of_dicts(fuel_costs, u"fuel_cost")
        _validate_route_example(route_example, time_costs, fuel_costs)

        cdef:
            int length = len(route_example)
            list route_to_permute = route_example[1:]
            int home = route_example[0]
            list min_route = route_example, route_to_list = []
            tuple route_to_compute = ()
            float min_cost = _cost(min_route, length, time_costs, fuel_costs, self.max_time_work, self.extra_cost, self.people), new_route_computed = 0.0

        for route_to_compute in permutations(route_to_permute, length):
            route_to_list = list(route_to_compute)
            route_to_list.insert(0, home)
            new_route_computed = _cost(route_to_list, length, time_costs, fuel_costs, self.max_time_work, self.extra_cost, self.people)
            if new_route_computed < min_cost:
                min_route = route_to_list
                min_cost  = new_route_computed

        return min_cost, _final_cost(min_route, length, time_costs, self.max_time_work)
