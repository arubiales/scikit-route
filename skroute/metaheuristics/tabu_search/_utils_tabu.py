from itertools import combinations
import numpy as np
from ..._utils._utils import _cost
# from skroute._utils._utils import _cost
import random as rd

prueba = [1,2,3,4]

def create_combinations(rnd_sol_1, start_id):
    """
Create pair combinations from one route

Parameters
-----------
rnd_sol_1: list
    A list of ints with the Node's IDs

start_id_list: list
     A list with one integer. The ID representing the start Node
    """

    store_all_combinations = []
    rnd_sol_1 = rnd_sol_1[1:]
    route_index = range(len(rnd_sol_1)-1)
    list_of_n = list(combinations(route_index, 2))

    for swap1, swap2 in list_of_n:
        x_swap = rnd_sol_1[:]
        x_swap[swap1], x_swap[swap2] = x_swap[swap2], x_swap[swap1]
        store_all_combinations.append([start_id] + x_swap)
    
    return store_all_combinations


def cost_combinations(all_combinations, start_id, lenght, time_matrix, cost_matrix, max_time_work, extra_cost, people):
    """
Compute and order the cost of a list of routes

Parameters
-----------
all_combinations: list
    A list with differents routes

lenght: int
    The lenght that have all routes

time_matrix: dict
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

cost_matrix: dict
    For a extended explanation go up to time_matrix. it's
    the fuel cost to go from one point to another.

max_time_work: float32, default=8
    the number of ours that a employ can work per day. For example if it's 8 hours, the algorithm will
    force that a route have to finish after the 8 hours have been completed, making the employeed come
     back home. it's a time constraint.

extra_cost: float32, default=0
    if it's 0 anything happend. If it's >0 in combination with max_time_work when the max_time_work is 
    reached, extra_cost is applied. This add a extra cost to the solution each time that max_time_work is
    reached. It's like extra pay for the worker each time max_time_work is completed (journey).

people: int32, default=1
    The number of people that you use in each route, for example if you need two truck drivers. that's 
    another contstraint. That will multiply the time_costs and the extra_cost. Not the travel cost because
    it's assumed that both go in the same vehicle

    """
    cost_routes = []
    for route in all_combinations:
        route_copy = route[:]
        route_copy.append(start_id)
        cost_route = _cost(route_copy, lenght, time_matrix, cost_matrix, max_time_work, extra_cost, people)
        cost_routes.append(cost_route)

    cost_routes, routes = zip(*sorted(zip(cost_routes, all_combinations), key=lambda x: x[0]))
    return cost_routes, routes

def find_best_route(all_cost, all_routes):
    """
Find the best route of all routes.

Parameters
-----------

all_cost: list
    A list of integers with the cost of different routes

all_routes: list
    A list with the different routes, cost and route must have the same index position

Return
-------
A tuple with the best cost and the best route
    """
    cost_best_route = np.inf
    for i in range(len(all_cost)):
        if all_cost[i] < cost_best_route:
            cost_best_route = all_cost[i]
            best_route = all_routes[i]
    return cost_best_route, best_route


def random_swap_indices(length):
    """
Generate two random number between 0 and length

Parameters
length: int
    The highest random number that can be generated
    """
    rand_1 = int(rd.random()*length)+1
    rand_2 = int(rd.random()*length)+1
    while rand_1==rand_2:
        rand_2 = int(rd.random()*length)+1
    return rand_1, rand_2


def cross_route(mutated_route, length):
    """
Make a INPLACE reverse of the route passed

Parameters
-----------
mutated_route: list
    The route to mutate

lenght: int
    The lenght of the route to mutate

    """
    rand_1, rand_2 = random_swap_indices(length)

    if rand_1 > rand_2:
        mutated_route[rand_2:rand_1] = reversed(mutated_route[rand_2:rand_1])
    else:
        mutated_route[rand_1:rand_2] = reversed(mutated_route[rand_1:rand_2])


def var_lower_length(tabu_lenght, tabu_var):
    """
Validation function that assert that tabu_var isn't higher number than tabu_lenght

Parameters:
-----------
tabu_lenght: int
tabu_var: int
    """
    
    if tabu_var <= tabu_lenght:
        return tabu_var
    else:
        raise ValueError("tabu_var can't be higher than tabu_lenght")
