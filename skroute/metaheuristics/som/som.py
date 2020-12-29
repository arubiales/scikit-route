from ._utils_som import *
from ._som_validators import * 
import tensorflow as tf
from ..._validators._validators import _intenger_validator, _zero_one_validator
from tqdm import tqdm



class SOM:
    """
SOM (Self Organizing Map) class is a Neuronal Network approach to the TSP/VRP problem

Parameters:
------------
    
units: int, default=None
    The neurons of our Neural Network, if it's None the units will be the length of the route
    multiplyed by eight

radius: int, default=None
    The area of search to obtain new neighbours, if it's None the radius will be the length of
    the route multiplyed by eight

radius_decay: float, default=0.9991
    The decay of the radius. In order to find the most optimal solution, at the end is good
    to only look the most near neighbours.

lr: float, default=0.8
    The learning rate, is how aggressive the update of the weight is, higher learning rate, more
    aggressive is the update of the weights

lr_decay: float, default=0.9991
    The decay of the learning rate. In order to find the most optimal solution, at the end is good
    to have a low learning rate.


--------------------------------------------------------------------------

Methods:
--------

fit(nodes, epochs)
    Execute the algorithm and give back the best route find it

nodes: tuple
    A tuple of tuples, each tuple is a Node with the first element the ID, the sencond the latitude
    and the third then longitude. For example if we have a route with three points, the tuple will
    be like this:

    nodes = (
        (1, 0.459887, 14.345767),
        (2, 0.634534, 12.575462),
        (3, 0.256765, 9.734435),
    )


epochs: int, default=10_000
    The times that the Neural network will update the weights trying to find the optimal solution


Returns
--------
A very optimized route that solve the TSP/VRP problem.


Example
--------
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import normalize, df_to_tuple
from sklearn_route.metaheuristics.som import SOM

df_barcelona = load_barcelona()["DataFrame"]
df_barcelona = df_barcelona[["id_origin", "lat_origin", "lon_origin"]].drop_duplicates()
df_barcelona = normalize(df_barcelona, "lat_origin", "lon_origin")

route = df_to_tuple(df_Barcelona)

som = SOM()
result = som.fit(route)

#Printing the best route
print(result)
    """

    def __init__(self, units=None, radius=None, radius_decay=0.9991, lr=0.8, lr_decay=0.9991):
        self.units = None if not units else _intenger_validator(units, "units")
        self.radius = None if not radius else _intenger_validator(radius, "radius")
        self.radius_decay= _zero_one_validator(radius_decay, "radius_decay")
        self.lr = _zero_one_validator(lr, "lr")
        self.lr_decay = _zero_one_validator(lr_decay, "lr_decay")


    def fit(self, nodes, epochs=10_000):
        epochs = _intenger_validator(epochs, "epochs")
        validate_node_input(nodes)
        length = len(nodes)
        if self.units is None:
            self.units = 8 * length

        if self.radius is None:
            self.radius = length * 8
        
        weights = generate_weights(self.units)


        for i in tqdm(range(epochs)):
            node = nodes[random_gen(length)][1:]
            node_idx = get_closest(weights, node)
            gaussian = get_neighbor(node_idx, int(self.radius//10), weights.shape[0])
            weights += tf.expand_dims(tf.cast(gaussian, tf.float32), axis=1) * self.lr * (node - weights)

            self.lr *= self.lr_decay
            self.radius *= self.radius_decay
            if self.radius < 1:
                print('\nRadius has completely decayed, finishing execution',
                'at {} iterations'.format(i))
                break
            if self.lr < 0.001:
                print('\nLearning rate has completely decayed, finishing execution',
                'at {} iterations'.format(i))
                break
        route = []
        for node in nodes:
            route.append(get_closest(weights, node[1:]))

        route = [int(i) for i in list(list(zip(*list(zip(*sorted(zip(route, nodes))))[1]))[0])]

        return route
