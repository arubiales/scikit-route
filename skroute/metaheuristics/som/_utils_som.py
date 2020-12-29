import tensorflow as tf
import numpy as np


def generate_weights(size):
    """
Create random weights for the neural network

Parameters:
------------
size int
    The number of weights of the neural network

Return:
-------
A two dimensional tensor of weights
    """
    return tf.random.uniform(shape=[size, 2])

def get_neighbor(center, radix, domain):
    """
Based on the center, the radix and the domain to find best neighbourhood and
help update the weights in the future.

Parameters:
------------

center int
    The id of the node to find the neighbors

radix int
    The radiux of search neighbors

domain array
    The weights that will be used to find networks

Return:
--------
A gaussian distribution that will be used to updates the weights on the future

    """

    if radix < 1:
        radix = 1

    deltas = tf.abs(center - tf.range(domain))
    distances = tf.minimum(deltas, domain - deltas)

    return tf.exp(-(distances * distances) / (2* (radix * radix)))


def euclidean_distance(weights, node):
    """
Based on weights compute the euclidean distance between nodes

Parameters:
------------

weights array
    The weights used to compute the distance between nodes

nodes tuple
    A node with the latitude and the longitude to compute the distance
 
 Return:
 -------
 The distance between a node and the rest
    """
    return tf.linalg.norm(weights - node, axis=1)

def get_closest(weights, node):
    """
Based on weights returns the closest neigbour to the node

Parameters:
------------

weights array
    The weights used to compute the distance between nodes

nodes tuple
    A node with the latitude and the longitude to compute the distance
 
 Return:
 -------
 The position index of the best neighbor of node
    """
    return tf.cast(tf.math.argmin(euclidean_distance(weights, node)), tf.int32)

def random_gen(length):
    """
    Generate a random number between 0 and length
    """
    return int(tf.random.uniform(shape=[1])*length)

