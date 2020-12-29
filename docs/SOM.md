# Self Organizing Maps (SOM)

Can be used to solve de TSP/VRP problem

The main **advantages** of SOM over others solutions are:
* Very good results even with a huge amount of points
* It's computed using only the point ID, latitude and longitude

The main **disvantages** of SOM over other solutions are:
*  The result is a very optimized route but you don't know the cost
*  Computationally more expensive than other solutions
*  Constrains (time/capacity) are not available in this algorithm 
*  Penalties are not available in this algorithm (time/capacity)

## Global understunding of the Algorithm
SOM is metaheuristic aproach wich mean is not deterministic, this lead that every time you run the algorithm you cound find a different solution.

This algorithm is based on Self Organizing Maps from [Tehuvo Kohonen](https://sci2s.ugr.es/keel/pdf/algorithm/articulo/1990-Kohonen-PIEEE.pdf) and the implementation in Python is based on the [Diego Vicente](https://diego.codes/post/som-tsp/) solution.  

The main formula of the alorithm if the search formula, to explore new combinations, combined with a learning rate decay in order to minimize the searchs over the iterations. Our algorithm can be expressed as:

$$nt+1=nt+αt⋅g(we,ht)⋅Δ(e,nt)$$

Where $\delta$ is the learning rate, $g$ is the gaussian function that look for a winner in a radius of $h$ 


## SOM

### Hyper - Parameters

!!! note ""
    **sklearn_route.metaheuristics.som.SOM***(units=None, radius=None, radius_decay=0.9991, lr=0.8, lr_decay=0.9991)*


 
* ```units```: int, default=None
>The number of Neurons of the SOM, if it's None the algorithm takes the number of nodes multiply by eight. More Numbers of neuron, better results but with a time penalty
* ```radius```: int, defaultNone
>The radius of search, if it's None the algorithm takes the number of nodes multiply by eight. The radius will decrease with the radius_decay parameter so it's good to have a high
radius at start to find throug all nodes at the begining
* ```radius_decay```: float, default=0.9991
>The decay of the radius per epoch wich means the decrease of the radius
* ```lr```: float, default=0.8
>The learning rate, is how aggressive the update of the weight is, higher learning rate, more
    aggressive is the update of the weights

* ```lr_decay```: float, default=0.9991
>The decay of the learning rate. In order to find the most optimal solution, at the end is good to have a low learning rate.

### Method

* ```fit(nodes, epochs)```:
>Execute the algorithm and give back the best route find it

    1. ```nodes```: tuple
>A tuple of tuples, each tuple is a Node with the first element the ID, the sencond the latitude
        and the third then longitude. For example if we have a route with three points, the tuple will
        be like this:

        nodes = (  
            (1, 0.459887, 14.345767),  
            (2, 0.634534, 12.575462),  
            (3, 0.256765, 9.734435),  
        )  
    <br>

    2. ```epochs```: int, default=10_000
>The times that the Neural network will update the weights trying to find the optimal solution

### Example SOM 

```
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
```
