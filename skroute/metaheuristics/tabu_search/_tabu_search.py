# import sys
# sys.path.append("/home/rubiales/Desktop/Projects/scikit-route/skroute/_validators")

# from _validators import (_intenger_validator,_zero_one_validator, _float_validator,
#                                         _validate_dict_of_dicts, _validate_route_example)


from ..._validators._validators import (_intenger_validator,_zero_one_validator, _float_validator,
                                        _validate_dict_of_dicts, _validate_route_example)
from ..simulated_annealing._base_simulated_annealing._utils_sa import _swap_route

import random as rd
from collections import deque
from ._utils_tabu import *
from ..._utils._utils import _final_cost

# from _utils_tabu import *


# def _swap_route(route, r1, r2):

#     temporal = 0
#     temporal = route[r1]
#     route[r1] = route[r2]
#     route[r2] = temporal
#     return route




class TabuSearch():
    """
Instantaite the Tabu Search class.

Tabu Search is a metaheuristic algorithm that solve TSPTW/VRPTW 

Parameters:
------------
searchs: int, default=1250
    The number of neighbourhood route searchs, more searchs will lead to a better result but with an 
    associate time computation cost.

p_m : float, default=0.6
    it's mutate, random probabilies are choosen when it will take place. If the random numbers are 2
    and 8 for example, the numbers located at that index will swap positions.

tabu_length: int, default=45
    it's the lenght of tabu list where routes are stored in order to avoid visits them again.

tabu_var: int, default=10
    Variations in the lenght of tabu_length, can't be higher number than tabu_length

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

ts = TabuSearch(searchs= 1250, p_m = 0.6, tabu_length=45, tabu_var=10, max_time_work=6, extra_cost=12.83)

--------------------------------------------------------------------------------------------------------------

Methods:
--------

fit(self, list random_route, dict time_matrix, dict cost_matrix)
    Execute the algorithm and give back the best solution

    Parameters
    ----------
    random_route: list
        it's a list that contain a random example of ints ids routes. But it's
        mandatory that the origin place, be the first id of the list

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

    Returns:
    --------
    A very optimized route that solve the TSPTW/VRPTW problem.

--------------------------------------------------------------------------------------------------------------

Attributes
----------

history_: The best solution in each iteration of the searchs or the decrease of the loss.

Example
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristic.tabu_search import TabuSearch

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
ts = TabuSearch(searchs= 1250, p_m = 0.6, tabu_length=45, tabu_var=10, max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = ts.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(ts.history_)
    """


    def __init__(self, searchs=1250, p_m=0.6, tabu_length=45, tabu_var=10, max_time_work=8., people=1, extra_cost=1.):
        self.searchs = _intenger_validator(searchs, u"searchs")
        self.p_m = _zero_one_validator(p_m, u"p_m")
        self.tabu_length = _intenger_validator(tabu_length, u"lenght_tabu")
        self.tabu_var = var_lower_length(tabu_length, _intenger_validator(tabu_var, u"tabu_var"))
        self.max_time_work = _float_validator(max_time_work, u"max_time_work")
        self.people = _intenger_validator(people,  u'people')
        self.extra_cost = _float_validator(extra_cost, u"extra_cost")

    def fit(self, random_route, time_matrix, cost_matrix):
        _validate_dict_of_dicts(time_matrix, u"time_matrix")
        _validate_dict_of_dicts(cost_matrix, u"cost_matrix")
        _validate_route_example(random_route, time_matrix, cost_matrix)

        start_id = random_route[0]
        length = len(random_route)
        length_less1 = length - 1
        all_solutions_route = []
        all_solutions_cost = []
        tabu_costs = deque()
        tabu_routes = deque()
        low_tabu = self.tabu_length - self.tabu_var
        high_tabu = self.tabu_length + self.tabu_var
        rnd_sol_1 = random_route[:]
        

        for i in range(1, self.searchs):
            all_routes = create_combinations(rnd_sol_1, start_id)

            cost_routes, all_routes = cost_combinations(all_routes, start_id, length, time_matrix, cost_matrix, self.max_time_work, self.people, self.extra_cost) #TODO: hay que cambiar por self
            best_cost_it = cost_routes[0]
            best_route_it = all_routes[0]

            if all_solutions_cost:
                best_cost, best_route = find_best_route(all_solutions_cost, all_solutions_route) 
                t = 0
                while best_cost_it in tabu_costs and t < length_less1:
                    if best_cost < best_cost_it:
                        best_cost_it = best_cost
                        best_route_it = best_route[:]
                        break
                    else:
                        best_cost_it = cost_routes[t]
                        best_route_it = all_routes[t]
                        t +=1
                
            
            if len(tabu_costs) >= self.tabu_length:
                if best_cost_it not in tabu_costs:
                    while len(tabu_costs) >= self.tabu_length:
                        tabu_costs.popleft()
                        tabu_routes.popleft()
                tabu_costs.append(best_cost_it)
                tabu_routes.append(best_route_it)
            else:
                tabu_routes.append(best_route_it)
                tabu_costs.append(best_cost_it)
            
            all_solutions_cost.append(best_cost_it)
            all_solutions_route.append(best_route_it)

            mod_iterations_1 = i%20  
            mod_iterations_2 = i%40 
            mod_iterations_3 = i%30 

            if mod_iterations_3 == 0 and best_cost_it > best_cost: 
                best_cost_it = best_cost
                best_route_it = best_route[:]

            if rd.random() <= self.p_m:
                best_route_it = best_route_it[:]
                cross_route(best_route_it, length)
                rnd_sol_1 = best_route_it
            else:
                rnd_sol_1 = best_route_it[:]

            
            if mod_iterations_2 == 0:
                rand_1, rand_2 = random_swap_indices(length_less1)
                rand_1_1, rand_3_1 = random_swap_indices(length_less1)
                rand_3, rand_4 = random_swap_indices(length_less1)
                
                best_route = _swap_route(best_route[:], rand_1, rand_2)
                best_route = _swap_route(best_route[:], rand_1_1, rand_2)
                best_route = _swap_route(best_route[:], rand_3, rand_4)
                best_route = _swap_route(best_route[:], rand_3_1, rand_4)

                rnd_sol_1 = best_route[:]

            
            if mod_iterations_1 == 0:
                self.tabu_length = rd.randint(low_tabu, high_tabu)

        
        final_cost, final_route = find_best_route(all_solutions_cost, all_solutions_route)
        final_route = _final_cost(final_route, length, time_matrix, self.max_time_work)
        self.history_ = list(zip(all_solutions_cost, all_solutions_route))
        return final_cost, final_route

