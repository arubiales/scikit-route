#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
"""
utils: Functions to help compute genetics algorithms 

Notes
-----
Maybe we could speed it by creating C++ unordered dicts and vectors but Python dicts look-up are 
really fast and the append method is very fast too. Also in the future add contiguos arrays could
speed up (C arrays)


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""


import numpy as np
import random as rd
from ...._utils._utils import _cost, _final_cost

cpdef tuple _generate_random_pop(list initial_route, int pop, int length_route, dict time_costs, dict fuel_costs, float max_time_work, float extra_cost, int people):

    """
Generate pop random routes from a random given initial route

Parameters:
-----------

initial_route: list of ints
    Contains the ID of the route that we want to measure de cost from one point
    (ID) to the following (ID)

pop: int
    The number of population (different random solutions), more
    population, more probabilities to find different solutions but 
    make the algorithm more time expensive. If the number is too
    low, the solution will be bad, because the algorithm can't 
    iterate over the loss function with the enough amount of data

length_route: int
    The length of the route wich means all the IDs for a route

time_costs: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix. For example if we
    have 3 ID 1, 2 and 3 the dict will be like this:

        {
        1:{
            1:0,
            2:x
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
    For a extended explanation go up to time_costs. it's
    the fuel cost to go from one point to another.

Returns:
--------
The route and the cost of the a route

    """
    cdef:
        list n_list = [],
        list n_list_cost = []
        list aux_shuffle= initial_route[1:]
        int i=0

    for i in range(pop):
        np.random.shuffle(aux_shuffle)
        initial_route[1:] = aux_shuffle
        n_list.append(initial_route.copy())
        n_list_cost.append(_cost(initial_route, length_route, time_costs, fuel_costs, max_time_work, extra_cost, people))

    return n_list_cost, n_list


cpdef int _tournament_selection(list price_pop, int k, int pop):
    """
Take a subsample of size "k" from the list of prices and choose the one with lowest price index.
This is knowed as tournament selection in the algorithm

Parameters:
-----------

price_pop: list of ints
    it's a list of all the prices of all the routes

k: int
    The number of individual of our populations (solutions) that will fight for being the best.
    When we are inside of ageneration the algorithm take a subsample of numbers k. The best
    k (individual) solution is choosen for the next generation.

pop: int
    The number of population (different random solutions), more
    population, more probabilities to find different solutions but 
    make the algorithm more time expensive. If the number is too
    low, the solution will be bad, because the algorithm can't 
    iterate over the loss function with the enough amount of data

Returns:
--------
The index of the "k" subsample with less cost.

    """


    cdef:
        int i=0, selected_idx = int(rd.random()*pop), other_idx = 0
        float selected = price_pop[selected_idx], other = 0.0

    for i in range(k-1):
        other_idx = int(rd.random()*pop)
        other = price_pop[other_idx]
        if other < selected:
            selected_idx = other_idx
            selected = other
    return selected_idx


cpdef tuple _crossover(list Parent_1, list Parent_2, int length_route, int Cr_1):
    """
Take two parents and create a son that is a combination of boths. Basically change  the order of
the route based on a random number

Parameters:
-----------

Parent_1: list of ints
    A list of ints that represent a parent that will be croosover with the other parent

Parent_2: list of ints
    A list of ints that represent a parent that will be croosover with the other parent

Cr_1: int
    A random generated number. The function will use for swap the positions between the route

Returns:
--------
A tuple with two chids (two routes)
    """


    cdef:
        int i = 0
        list Child_1 = Parent_1[0:1]
        list Child_2 = Parent_2[0:1] 

    for i in range(Cr_1, length_route):
        Child_1.append(Parent_1[i])
        Child_2.append(Parent_2[i])

    for i in range(1, Cr_1):
        Child_1.append(Parent_1[i])
        Child_2.append(Parent_2[i])

    return Child_1, Child_2


cpdef tuple _mutate(list Child_1, list Child_2, length_route, int index_1_child_1, int index_2_child_1):

    """
Take two parents and in both, will swap position

Parameters:
-----------

Child_1: list of ints
    A list of ints that represent a child that will mutate, swapping the position of two numbers

Child_2: list of ints
    A list of ints that represent a child that will mutate, swapping the position of two numbers

index_1_child_1: int
    A random generated number. The function will use for swap the positions between the route

index_2_child_1: int
    A random generated number. The function will use for swap the positions between the route

Returns:
--------
A tuple with two chids (two routes)
    """

    cdef:
        list mutate_child_1 = Child_1.copy()
        list mutate_child_2 = Child_2.copy()


    mutate_child_1[index_1_child_1] = Child_1[index_2_child_1]
    mutate_child_1[index_2_child_1] = Child_1[index_1_child_1]

    mutate_child_2[index_1_child_1] = Child_2[index_2_child_1]
    mutate_child_2[index_2_child_1] = Child_2[index_1_child_1]

    return mutate_child_1, mutate_child_2


cpdef tuple _elitism(list new_population, int pop, int length_route, dict time_costs, dict fuel_costs, float max_time_work, float extra_cost, int people):
    """
Take the worst routes (most expensives) and are replaced by the best of all generation. This
represent the natural selection where the weekest individual death and the strongest live.

Parameters:
-----------

new_population: list of ints
The new population generated after tournament_selection, crossover and mutation for compute
their cost

pop: int
    The number of population (different random solutions), more population, more probabilities
    to find different solutions but make the algorithm more time expensive. If the number is too
    low, the solution will be bad, because the algorithm can't iterate over the loss function
    with the enough amount of data

length_route: int
    The length of the route wich means all the IDs for a route

time_costs: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix. For example if we
    have 3 ID 1, 2 and 3 the dict will be like this:

        {
        1:{
            1:0,
            2:x
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
    For a extended explanation go up to time_costs. it's
    the fuel cost to go from one point to another.


Returns:
--------
A tuple with the population after one generation.
    """
    cdef:
        list price_pop = []
        list individual_population_min = new_population[0]
        float value_max = 0.0
        list individual_population_max = new_population[0]
        int i=0
        float value_min = 0.0

    for i in range(pop):
        price_pop.append(_cost(new_population[i], length_route, time_costs, fuel_costs, max_time_work, extra_cost, people))
    
    value_min = price_pop[0]
    value_max = price_pop[0]

    for i in range(1, pop):
        if price_pop[i] < value_min:
            value_min = price_pop[i]
            individual_population_min = new_population[i]
        elif price_pop[i] > value_max:
            value_max = price_pop[i]
            individual_population_max = new_population[i]

    # cambiamos los menos aptos por los más aptos
    for i in range(pop):
        if price_pop[i] == value_max:
            price_pop[i] = value_min
            new_population[i] = individual_population_min
    
    return value_min, individual_population_min, price_pop, new_population
    
cpdef tuple _final_result(list history, int num_gen, int length_route, dict time_costs, float max_time_work):
    cdef:
        best_value = history[0][0]
        best_route = history[0][1]

    for i in range(1, num_gen):
        if history[i][0] < best_value:
            best_value = history[i][0]
            best_route = history[i][1]

    best_route = _final_cost(best_route, length_route, time_costs, max_time_work)
    return best_value, best_route
