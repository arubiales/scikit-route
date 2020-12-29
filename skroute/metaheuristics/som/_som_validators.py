import numpy as np

def validate_node_input(nodes):
    if type(nodes) != tuple:
        raise TypeError("fit method input must be a tuple of tuples")
    
    float_types = set([float, np.float, np.float32, np.float64])
    length = len(nodes)
    ids = []
    lats = []
    lons = []
    for node in nodes:
        if type(node) != tuple:
            raise TypeError("fit method input must be a tuple of tuples")
        if type(node[0]) != int:
            raise TypeError("fit method tuples first element must be integer")
        if type(node[1]) not in float_types:
            raise TypeError(f"fit method tuples second element must be a float representing the latitude {type(node[1])}") 
        if type(node[2]) not in float_types:
            raise TypeError("fit method tuples third element must be a float representing longitude")
        ids.append(node[0])
        lats.append(node[1])
        lons.append(node[2])
                
    if len(list(set(nodes))) != length:
        raise ValueError("Nodes can't be duplicated, there are nodes with same ids, latitude or longitude")
