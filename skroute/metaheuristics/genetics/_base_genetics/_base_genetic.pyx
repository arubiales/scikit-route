#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
"""
Genetic Algorithm class

Notes
-----
Maybe we could speed it by creating C++ unordered dicts and vectors but Python dicts look-up
are really fast and the append method is very fast too. Also in the future add contiguos arrays
could speed up (C arrays)


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""


import random as rd
from ._utils_genetic import (_generate_random_pop, _tournament_selection, _crossover, _mutate, _elitism,
                     _final_result)

from ...._validators._validators import (_zero_one_validator, _intenger_validator, _early_stopping_validator, _float_validator,
                          _validate_dict_of_dicts, _validate_route_example)

cdef class Genetic():
    """
Instantiate the Genetic Algorithm class

Parameters:
-----------

p_c: float32, default=0.6
    between 0 and 1 the probability of crossover. In each generation
    at start the algorithm generate two random probabilities these
    random probabilies are the choosen when crossover will take
    place. If the random number is 4 that mean that the number in
    position and the following numbers will be now at start. And the
    first four numbers will be at the end.

p_m : float32, default=0.4
    it's mutate, random probabilies are choosen when it will take
    place. If the random numbers are 2 and 8 for example, the 
    numbers located at that index will swap positions.

pop: int32, default=400
    The number of population (different random solutions), more
    population, more probabilities to find different solutions but 
    make the algorithm more time expensive. If the number is too
    low, the solution will be bad, because the algorithm can't 
    iterate over the loss function with the enough amount of data

gen: int32 Default=1000
    The number of generations. More generations make the algorithm
    achieve better solution but also make the algorithm take more
    time. This can be solved with early_stoping parameter to make
    the algorithm stop when is not improving. If the number of
    generation is low the algorithm may not converge

k: int32, default=3
    The number of individual populations (solutions) that
    will fight for being the best. When we are inside of a
    generation the algorithm take a subsample of numbers k. The best
    k (individual) solution is choosen as a parten for this generation.

early_stopping: int32, default=None
    If it's None, the algorithm will finish when the last generation  
    is performed. If it's other unsigned integer (positive integer
    number) if the algorithm can't improve the result while X
    generations (where x is early_stopping number) the algorithm
    will stop.

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

fit(list route_example, dict time_matrix, dict fuel_matrix)
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

Example
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.genetics import Genetic

df_barcelona = load_barcelona()["DataFrame"]

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
ga = Genetic(p_m = 0.3, pop=400, gen=2000, k=5, early_stoping=100, max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = ga.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(ga.history_)
    """
    cdef public:
        float p_c
        float p_m
        int pop
        int gen
        int k
        object early_stopping
        float max_time_work
        float extra_cost
        int people
        list history_
    

    def __init__(self, float p_c=0.6, float p_m=0.4, pop=400, gen=1600, k=3, early_stopping=None,
                  float max_time_work=8.0, float extra_cost=1.0, people=1):
        self.p_c = _zero_one_validator(p_c, u'p_c')
        self.p_m = _zero_one_validator(p_m, u'p_m')
        self.pop = _intenger_validator(pop, u'pop')
        self.gen = _intenger_validator(gen, u'gen')
        self.k = _intenger_validator(k, u'k')
        self.early_stopping = _early_stopping_validator(early_stopping)
        self.max_time_work = _float_validator(max_time_work, u"max_time_work")
        self.extra_cost = _float_validator(extra_cost, u"extra_cost")
        self.people = _intenger_validator(people,  u'people')
        self.history_ = []


    
    cpdef tuple fit(self, route_example, time_costs, fuel_costs):
        _validate_dict_of_dicts(time_costs, u"time_cost")
        _validate_dict_of_dicts(fuel_costs, u"fuel_cost")
        _validate_route_example(route_example, time_costs, fuel_costs)
        cdef:
            int i=0, j=0, o=0, rand1=0, rand=0
            list Parent_1=[], Parent_2=[], Child_1=[], Child_2=[], best_route=[], value=[], New_Population=[]
            float best_value = 0.0
            int mid_pop = self.pop // 2
            int early_stopping_counter = 0
            int length_route = len(route_example)
            list population = [], price_pop =[]
            int length_child = length_route-1
            float value_min = 0.0, min_history = 0.0
            list population_min = []
            list history = []
            list value_history = []


        price_pop, population = _generate_random_pop(route_example, self.pop, length_route, time_costs, fuel_costs,
                                                     self.max_time_work, self.extra_cost, self.people)
        if self.pop < self.k:
            self.k = self.pop
        for i in range(self.gen):
            New_Population = []  
            rand1 = int(rd.random()*length_child)+1
            rand2 = int(rd.random()*length_child)+1
            if rand1 != rand2:
                for j in range(mid_pop):  
                    # Tournament Selection
                    Parent_1 = population[_tournament_selection(price_pop, self.k, self.pop)]
                    Parent_2 = population[_tournament_selection(price_pop, self.k, self.pop)]
                    if rd.random() < self.p_c:
                        if i % 2 == 0:
                            Child_1, Child_2 = _crossover(Parent_1, Parent_2, length_route, rand1)
                        else:
                            Child_1, Child_2 = _crossover(Parent_1, Parent_2, length_route, rand2)
                    else:  
                        Child_1 = Parent_1
                        Child_2 = Parent_2
                
                    if rd.random() < self.p_m:
                        Child_1, Child_2 = _mutate(Child_1, Child_2, length_route, rand1, rand2)

                    New_Population.append(Child_1)
                    New_Population.append(Child_2)
            else:
                for j in range(mid_pop): 
                    Parent_1 = population[_tournament_selection(price_pop, self.k, self.pop)]
                    Parent_2 = population[_tournament_selection(price_pop, self.k, self.pop)]
                    New_Population.append(Parent_1)
                    New_Population.append(Parent_2)

            value_min, population_min, price_pop, population = _elitism(New_Population, self.pop, length_route, time_costs, fuel_costs, self.max_time_work, self.extra_cost, self.people)
            history.append([value_min, population_min])
            value_history.append(value_min)

            if self.early_stopping:
                if min_history > 0.0:
                    if value_min < min_history:
                        early_stopping_counter = 0
                        min_history = value_min
                    else:
                        early_stopping_counter += 1
                        if early_stopping_counter == self.early_stopping:
                            self.history_ = value_history
                            return _final_result(history, len(value_history), length_route, time_costs, self.max_time_work)

                else:
                    min_history = value_min

        self.history_ = value_history
        return _final_result(history, self.gen, length_route, time_costs, self.max_time_work)

    


