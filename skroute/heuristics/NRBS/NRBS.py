"""
Simulated Annealing class

Notes
-----
In the future we can improve the speed with Cython

- Its not returning sometimes the rute accordingly to the cost.


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""

from ._utils_nrbs import _compute_mean_std_dicts, _priority_function, _priority_connection, Node, _check_route
import os, sys
sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-3]))
from ..._validators._validators import (_float_validator, _intenger_validator, _validate_route_example, _validate_dict_of_dicts)
from ..._utils._utils import _cost
from copy import deepcopy
import numpy as np
#Node Ranking Based on Stats
class NRBS:
    """
An Heuristic Algorithm, Node Ranking Based on Stats (NRBS) Class

Parameters:
------------

mean_priority: float
    Strongly recommended a number between 0 and 2, The ponderation of the mean for 
    all points, higher number make points with higher mean (more distance from
    others) have large values and priority selection

std_priority: float
    Strongly recommended a number between 0 and 2, The ponderation of the standar
    deviation for all points, higher number make points with higher deviation
    (more diferents of distance between points) have larger values and priority
    selection

mean_connection: float
    Strongly recommended a number between 0 and 2, The ponderation of the mean for 
    all points, higher number make points with higher mean (more distance from
    others) have large values and priority to connect

std_connection: float
    Strongly recommended a number between 0 and 2, The ponderation of the standar
    deviation for all points, higher number make points with higher deviation
    (more diferents of distance between points) have larger values and priority
    to connect

distance_weight: float
    Strongly recommended a number between 0 and 2, The ponderation of the distance 
    between all points, higher distance_weight make points with higher distances
    between them have large values, this reduce the influce of the mean_connection
    and the std_connections parameters depending of the distance.


---------------------------------------------------------------------------------------
Methods

fit(start_point_id, ids_route, cost_matrix)
    train the algorithm and give back tghe best solution

    Parameters
    -----------
    start_point_id: int
        The point of start
    
    ids_route: list
        A list with all points

    cost_matrix: dict
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

    Returns
    --------
    A very optimized route that solve the TSP/VRP problem.
    """

    def __init__(self, mean_priority, std_priority, mean_connection, std_connection, distance_weigth):
        self._mean_priority = _float_validator(mean_priority, "mean_priority")
        self._std_priority = _float_validator(std_priority, "std_priority") 
        self._mean_connection = _float_validator(mean_connection, "mean_Connection")
        self._std_connection = _float_validator(std_connection, "std_connection")
        self._distance_weigth = _float_validator(distance_weigth, "distance_weight")

    def fit(self, start_node_id, ids_node, cost_matrix):
        _intenger_validator(start_node_id, "start_point_id")
        _validate_dict_of_dicts(cost_matrix, "cost_matrix")
        _validate_route_example(ids_node, cost_matrix, cost_matrix)
        _lenght_route = len(ids_node)

        mean_dict, std_dict = _compute_mean_std_dicts(ids_node, cost_matrix, _lenght_route)
        priority_routes = _priority_function(ids_node, mean_dict, std_dict, self._mean_priority, self._std_priority)
        priority_routes_conections = _priority_connection(priority_routes, cost_matrix, mean_dict, std_dict, self._mean_connection, self._std_connection, self._distance_weigth)

        #Cogemos y agrupamos cada ruta con su conexión mejor y tenemos un counter_dict que cuenta cuando están conectadas y cuando no
        node_values = {i:Node(i) for i in ids_node}

        for i in range(2):
            for k, _ in priority_routes.items():
                if len(node_values[k]) < 2:
                    temp_connection = priority_routes_conections[k]
                    temp_keys = list(temp_connection.keys())
                    sol_found = False
                    counter = 0
                    while counter < _lenght_route-1 and not sol_found:
                        to_connect = temp_keys[counter]
                        temp_node_values = deepcopy(node_values)
                        if len(node_values[to_connect]) < 2:
                            temp_node_values[k].neighbours = to_connect
                            temp_node_values[to_connect].neighbours = k
                            if to_connect not in node_values[k].neighbours and _check_route(temp_node_values, _lenght_route):
                                node_values[k].neighbours = to_connect
                                node_values[to_connect].neighbours = k
                                sol_found = True
                        counter += 1
        result = _check_route(node_values, 19)
        cost = _cost(result, _lenght_route, cost_matrix, cost_matrix, np.inf, 0, 1)
        result.append(start_node_id)
        return cost, result
        



