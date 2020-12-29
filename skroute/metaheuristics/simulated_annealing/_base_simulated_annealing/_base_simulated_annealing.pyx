#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
"""
Simulated Annealing class

Notes
-----
Maybe we could speed it by creating C++ unordered dicts and vectors but Python dicts look-up
are really fast and the append method is very fast too. Also in the future add contiguos arrays
could speed up (C arrays)

- Its not returning sometimes the rute accordingly to the cost.


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""

from ._utils_sa import _check_temp, _swap_route, _normalize
from ...._utils._utils import _cost,_final_cost
from ...._validators._validators import (_zero_one_validator, _intenger_validator, _float_validator,
                          _validate_dict_of_dicts, _validate_route_example)
import random as rd

cdef class SimulatedAnnealing():
    """
The Metaheuristic algorithm, Simmulated Annealing class


Parameters:
-----------

temp: float32, default=12
    The temperature parameter. In the algorithm the temperature will 
    decrease till reach the tol parameter to give you the best
    solution.

neighbours: int32 , default=250
    Given a route the exchanges that will make in at the same temperature.
    You can think as the neighbours that will visit at a particular
    temperature.

delta: float32, default=0.78
    A number between 0 and 1 the highest the number more slow the
    the temperature decrease, and more solution the algorithm try.
    If it's close to 0 the temperature decrease fast and the
    algorithm converge faster but look for less optimal solutions

tol: float32, default=1.29
    The tolerance, the minimun temperature allow by the algorithm
    one time the "temp" is equal to "tol" the algorithm finish.

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

fit(self, list route_example, dict time_matrix, dict cost_matrix)
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
    A very optimized route that solve the MTSPTW problem.


---------------------------------------------------------------------------

Attributes
----------

history_: The best solution in each iteration of the
          generation or the decrease of the loss.

Example:
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.simmulated_annealing import SimulatedAnnealing

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
sa = SimulatedAnnealing()

#random route - time_matrix - cost_matrix
result = sa.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(sa.history_)
    """
    cdef public:
        int neighbours
        float delta
        float temp
        float tol
        float max_time_work
        float extra_cost
        int people
        list history_

    def __init__(self, float temp=12.0, int neighbours=250, float delta=0.78,  float tol=1.29,
                    float max_time_work=8.0, float extra_cost=1.0, int people=1):
        _zero_one_validator(delta, u"delta")
        self.neighbours = _intenger_validator(neighbours, u"neighbours")
        self.delta = _normalize(delta)
        self.temp = _float_validator(temp, u"temp")
        self.tol = _float_validator(tol, u"tol")
        self.max_time_work = _float_validator(max_time_work, u"max_time_work")
        self.extra_cost = _float_validator(extra_cost, u"extra_cost")
        self.people = _intenger_validator(people,  u'people')
        self.history_ = []

    cpdef tuple fit(self, list route, dict time_matrix, dict cost_matrix):
        _validate_dict_of_dicts(time_matrix, u"time_cost")
        _validate_dict_of_dicts(cost_matrix, u"fuel_cost")
        _validate_route_example(route, time_matrix, cost_matrix)

        cdef:
            list best_route = route[:]
            int length_route = len(route)
            int length_route_less_one = length_route-1
            float cost0 = _cost(route, length_route, time_matrix, cost_matrix, self.max_time_work, self.extra_cost, self.people)
            float min_cost = cost0
            int j=0, temporal=0, r1 = 0, r2=0
            float cost1=0.0, temp = self.temp
            list history_ = []
         
        while temp > self.tol:
            for j in range(self.neighbours):
                r1 = int(rd.random()*length_route_less_one)+1
                r2 = int(rd.random()*length_route_less_one)+1
                if r1 != r2:
                    route = _swap_route(route, r1, r2)
                    cost1 = _cost(route, length_route, time_matrix, cost_matrix, self.max_time_work, self.extra_cost, self.people)
                    if cost1 < cost0:
                        cost0 = cost1
                    else:
                        if _check_temp(cost0, cost1, temp): 
                            cost0 = cost1
                        else:
                            route = _swap_route(route, r1, r2)

            temp *= self.delta

            history_.append(cost0)
            if cost0 < min_cost:
                min_cost = cost0
                best_route = route

        self.history_ = history_
        best_route = _final_cost(best_route, length_route, time_matrix, self.max_time_work)
        return min_cost, best_route
