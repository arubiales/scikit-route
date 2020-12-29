#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
"""
Utils for Simmulated Annealing algorithm


Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""



import numpy as np
import random as rd


cpdef bint _check_temp(float cost0, float cost1, float temp):
    """
The Mathematical expression that check if we accept a bad cost

Parameters:
-----------

cost0: float32
    The cost of the initial route that is better than the new one

cost1: float32
    The cost of the new route that is worst than the older one

temp: float32
    The current temperature
    """

    return rd.random() < np.exp((cost0-cost1)/temp)


cpdef float _normalize(float x):
    """
it's used to normalize the temp parameter of simmulated annealing. Take
a parameter between 0 and 1 and gives back a number between 0.9 and 1

Parameters:
-----------

x: float32
    The normalized number
    """
    if x == 1:
        x = 0.999999
    elif x == 0:
        x = 0.000001
    return x * 0.1 + 0.9


cpdef list _swap_route(list route, int r1, int r2):
    """
Given two numbers swap the positions in a list between.

Parameters:
-----------

route: list
    A list representing a route

r1: int32
    The position one that will be swap in the list for the position 2

r2: int32 
    The position one that will be swap in the list for the position 2
    """


    cdef int temporal = 0
    temporal = route[r1]
    route[r1] = route[r2]
    route[r2] = temporal
    return route
