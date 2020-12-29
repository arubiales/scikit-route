
def _std(var:list, mean:float, length:int):
    """
Compute the Standar Deviation of a point

Parameters:
------------

var: list
    The variance between all points

mean: float
    The mean of the point

lenght: int
    The lenght of all routes

Return
-------
Return the standar deviation of a point
    """

    summatory = 0
    for i in range(length):
        summatory += (var[i] - mean)**2

    return (summatory / length)**0.5

def _priority_function(random_route:list, mean_routes:dict, std:dict, mean_weight:float, std_weight:float)-> dict:
    """
Give us the order of  priority routes to select to look for connections

Parameters
-----------

random_route: list
    The list of all points to connect

mean_routes: dict
    A dict with the mean distance/cost of all points one each other

std_routes: dict
    A dict with the standar deviation distance/cost of all points one each other

mean_weight: float
    The ponderation of the mean for all points, higher number make points
    with higher mean (more distance from others) have large values and priorice
    their connections

std_weight: float
    The ponderation of the standar deviation for all points, higher number
    make points with higher deviation (more diferents of distance between points)
    have larger values and priorice their connections

Return
-------
A Dict of oredered by priority
    """

    priorities = {}
    for id_route in random_route:
        priorities[id_route] = (mean_routes[id_route]**mean_weight) * (std[id_route]**std_weight)

    ordered_prior_route = dict(sorted(priorities.items(),reverse=True, key=lambda x: x[1]))
    return ordered_prior_route

def _priority_connection(priority_route:dict, cost_matrix, mean_routes, std_routes, mean_weight, std_weigth, distance_weight) -> dict:
    """
Give us the priority connections dictionary for each route

Parameters
-----------

priority_route: dict
    The dictionary with the priority routes

cost_matrix: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix.

mean_routes: dict
    A dict with the mean distance/cost of all points one each other

std_routes: dict
    A dict with the standar deviation distance/cost of all points one each other

mean_weight: float
    The ponderation of the mean for all points, higher number make points
    with higher mean (more distance from others) have large values and priorice
    their connections

std_weight: float
    The ponderation of the standar deviation for all points, higher number
    make points with higher deviation (more diferents of distance between points)
    have larger values and priorice their connections

distance_weight: float
    The ponderation of the distance between all points, higher distance_weight 
    make points with higher distances between them have large values

Return
-------
A dict of dicts every point connections ordered by priority
    """



    dict_prior_connections_ordered = {}
    base_order = priority_route.keys()
    for id_route in base_order:
        prior_connections = {}
        for given_route in base_order:
            if id_route != given_route:
                prior_connections[given_route] = ((mean_routes[given_route]**mean_weight) * (std_routes[given_route]**std_weigth)) /  cost_matrix[given_route][id_route] ** distance_weight
        dict_prior_connections_ordered[id_route] = dict(sorted(prior_connections.items(), reverse=True, key=lambda x: x[1]))
    return dict_prior_connections_ordered

def _compute_mean_std_dicts(random_route, cost_matrix, lenght_route):
    """
Compute the mean and the standar deviation distance between all points
one each other

Parameters
----------
random_route: list
    A list of ints with all our points

cost_matrix: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix.

length_route: int
    The amount of points

Return
-------
A tuple with two dicts and the mean and standar deviation of all points
    """
    mean_routes = {}
    std_routes = {}
    for id_route in random_route:
        values = list(cost_matrix[id_route].values())
        mean = sum(values) / lenght_route
        mean_routes[id_route] = mean
        std_routes[id_route] = _std(values, mean, lenght_route)
    return mean_routes, std_routes


def _check_route(node_dictionary, length_route):
    """
Check if the connection between two points create a circular route before the algorithm
finish

Parameters
-----------
node_dictionary dict
    A dictionary with the key as integer representing the node point and a value
    as a Node object

length_route int
    The amount of points


Returns
-----------
 There is two possibles returns
    * A boolean if the route its possible or not
    * if it's a circular route, but we have cover all points (length), returns the
    route path.
    """

    for k in node_dictionary.keys():
        if node_dictionary[k].neighbours:
            path = [node_dictionary[k].node, node_dictionary[k].neighbours[0]]
            counter = 0
            while counter < length_route-1:
                last_route = path[-1]
                next_route = node_dictionary[last_route]
                next_route_neighbours = next_route.neighbours
                #Comprobamos si están los dos en el path, si están hay una ruta circular, por lo que retornamos false
                if len(next_route_neighbours) == 2:
                    if next_route_neighbours[0] in path and next_route_neighbours[1] in path:
                        if len(path) != length_route:
                            return False
                        else:
                            return path
                    elif next_route_neighbours[0] in path:
                        path.append(next_route_neighbours[1])
                        counter += 1
                    elif next_route_neighbours[1] in path:
                        path.append(next_route_neighbours[0])
                        counter +=1
                else: #SI un nodo no tiene dos rutas, entonces no puede ser circular  por loq ue pasamos
                    counter +=1
    return True



def _define_true_route(ids_route):
    """
Only for developt Purposes, a true good route.
    """
    true_dict = {i:Node(i) for i in ids_route}
    true_dict[10000007].neighbours = 46
    true_dict[10000007].neighbours = 47
    true_dict[47].neighbours = 30
    true_dict[47].neighbours = 10000007
    true_dict[30].neighbours = 23
    true_dict[30].neighbours = 47
    true_dict[23].neighbours = 59
    true_dict[23].neighbours = 30
    true_dict[59].neighbours = 4
    true_dict[59].neighbours = 23
    true_dict[4].neighbours = 91
    true_dict[4].neighbours = 59
    true_dict[91].neighbours = 25
    true_dict[91].neighbours = 4
    true_dict[25].neighbours = 1
    true_dict[25].neighbours = 91
    true_dict[1].neighbours = 31
    true_dict[1].neighbours = 25
    true_dict[31].neighbours = 5
    true_dict[31].neighbours = 1
    true_dict[5].neighbours = 12
    true_dict[5].neighbours = 31
    true_dict[12].neighbours = 7
    true_dict[12].neighbours = 5
    true_dict[7].neighbours = 26
    true_dict[7].neighbours = 12
    true_dict[26].neighbours = 27
    true_dict[26].neighbours = 7
    true_dict[27].neighbours = 65
    true_dict[27].neighbours = 26
    true_dict[65].neighbours = 44
    true_dict[65].neighbours = 27
    true_dict[44].neighbours = 32
    true_dict[44].neighbours = 65
    true_dict[32].neighbours = 46
    true_dict[32].neighbours = 44
    true_dict[46].neighbours = 10000007
    true_dict[46].neighbours = 32
    return true_dict


class NodeValueError(Exception):
    """
NodeValueError Exception is throw when a node have two neighbours and is added 
one more
    """
    def __init__(self, node, neighbours):
        self.node = node
        self.neigbours = neighbours
    def __str__(self):
        return f"A node only can have two neighbours. The node {self.node} has {self.neigbours} neighbours"



class Node:
    """
Node Class

This Class create a Node, a node is a point that can have a maximum of two connections

Parameters
-----------

node int
    The point that will be connected with two other points

--------------------------------------------------------------------------------------

Attributes
-----------
#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
neighbours list
    neigbours is a list of int, each time you add a neighbours you to the list is a
    connection with the node. Remember that a node only can have a maximum of two
    neighbours


--------------------------------------------------------------------------------------

Special Methods
---------------

__repr__ str
    The Node and their neighbours

__len__ int
    Number of neighbours

--------------------------------------------------------------------------------------

Example
--------

node1 = Node(1)

#Add neighbours
node1.neighbours = 4
node1.neighbours = 7

#Print a node
print(node1)

# length of node
print(len(node1))
    """

    def __init__(self, node):
        self.node = node
        self._neighbours = []

    def __repr__(self):
        return str((self.node, self.neighbours))

    def __len__(self):
        return len(self._neighbours)

    @property
    def neighbours(self):
        return self._neighbours

    @neighbours.setter
    def neighbours(self, x):
        if len(self._neighbours) < 2:
            self._neighbours.append(x)
        else:
            raise NodeValueError(self.node, self.neighbours)
        

