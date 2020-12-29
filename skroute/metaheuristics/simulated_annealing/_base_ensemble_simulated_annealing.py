
from ._base_simulated_annealing._base_simulated_annealing import SimulatedAnnealing
from multiprocessing import Pool
from ..._validators._validators import _intenger_validator

class EnsembleSimulatedAnnealing(SimulatedAnnealing):
    """
The Metaheuristic algorithm, Simmulated Annealing class.

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

Attributes
----------

history_: The best solution in each iteration of the
          generation or the decrease of the loss.

Example:
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.simmulated_annealing import EnsembleSimulatedAnnealing

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
esa = EnsembleSimulatedAnnealing()

#random route - time_matrix - cost_matrix
result = esa.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(esa.history_)
    """

    def __init__(self, n_simulateds=20,  temp:float=12.0, neighbours:int=250, delta:float=0.78, tol:float=1.29,
                max_time_work:float=8.0, extra_cost:float=10.0, people:int=1, n_jobs:int=1):
        super().__init__(temp, neighbours, delta, tol, max_time_work, extra_cost, people)
        self.n_simulateds = _intenger_validator(n_simulateds, "n_simulateds")
        self.n_jobs = _intenger_validator(n_jobs, "n_jobs")
    
    def fit(self, route_example:list, time_matrix:dict, cost_matrix:dict) -> list:
        if self.n_jobs > 1:
            pool = Pool(processes=self.n_jobs)
            list_to_pool = []
            for i in range(self.n_simulateds):
                list_to_pool.append([SimulatedAnnealing(self.temp, self.neighbours, self.delta, self.tol,
                                                        self.max_time_work, self.extra_cost, self.people),
                                    route_example, time_matrix, cost_matrix])
            list_results = pool.starmap(SimulatedAnnealing.fit, list_to_pool)
            min_value = list_results[0][0]
            result = list_results[0]
            for i in range(self.n_simulateds):
                if list_results[i][0] < min_value:
                    result = list_results[i]
            return result
        else:
            result = []
            value = []
            for i in range(self.n_simulateds):
                algo_result = super().fit(route_example, time_matrix, cost_matrix)
                result.append(algo_result)
                value.append(algo_result[0])
            return result[value.index(min(value))]

