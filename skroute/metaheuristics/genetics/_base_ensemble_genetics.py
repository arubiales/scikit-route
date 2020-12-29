#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
from ._base_genetics._base_genetic import Genetic
from ..._validators._validators import _intenger_validator
from multiprocessing import cpu_count, Pool

class EnsembleGenetic(Genetic):
    """
Instantiate the EnsembleGenetic class that inherit from Genetic class

Parameters:
-----------
n_genetics: int, default=10
    The  number of Genetics algorithm that will be thrown, more
    algorithms you will achieve a better result, but will be more
    slower

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

n_jobs: int default=0
    The number of jobs to work in parallel the more workers, the
    faster will go. Use this parameter with caution, maybe can 
    collapse the computer if you select a lot jobs. 


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
    A very optimized route that solve the MTSPTW problem.

---------------------------------------------------------------------------


Example
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.genetics import EnsembleGenetic

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
eg = EnsembleGenetic(n_genetics=20, n_jobs=4, p_m = 0.6, pop=400, gen=2000, max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = eg.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)
    """
    
    def __init__(self, n_genetics:int= 10,  p_c:float=0.6, p_m:float=0.4, pop:int=400, gen:int=1000, k:int=3, early_stopping=None,
                  max_time_work:float=8., extra_cost:float=10., people:int=1, n_jobs:int=1):
        super().__init__(p_c, p_m, pop, gen, k, early_stopping, max_time_work, extra_cost, people)
        self.n_genetics = _intenger_validator(n_genetics, "n_genetics")
        self.n_jobs = _intenger_validator(n_jobs, "n_jobs")

    def fit(self, route_example:list, time_matrix:dict, cost_matrix:dict) -> list:
        if self.n_jobs > 1:
            pool = Pool(processes=self.n_jobs)
            list_to_pool = []
            for i in range(self.n_genetics):
                list_to_pool.append([Genetic(self.p_c, self.p_m, self.pop, self.gen, self.k, self.early_stopping,
                                    self.max_time_work, self.extra_cost, self.people),
                 route_example, time_matrix, cost_matrix])
            list_results = pool.starmap(Genetic.fit, list_to_pool)
            min_value = list_results[0][0]
            result = list_results[0]
            for i in range(self.n_genetics):
                if list_results[i][0] < min_value:
                    result = list_results[i]
            return result
        else:
            result = []
            value = []
            for i in range(self.n_genetics):
                algo_result = super().fit(route_example, time_matrix, cost_matrix)
                result.append(algo_result)
                value.append(algo_result[0])
            return result[value.index(min(value))]

