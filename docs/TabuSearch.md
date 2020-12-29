# Genetics

The main advantage of this Tabu Search over others solutions are:
* Easy to learn and to use, inspired in Scikit Learn library.
* Very well documented.
* Give very good results

## Global understunding of the Algorithm
Tabu Search is metaheuristic aproach wich mean is not deterministic this lead that every time you run the result could be different but will be very near of the **global optimum.**

The algorithm is called Tabu Search because it have a Tabu List, who makes that if a generated route have been prove recently, the algorithm will check another diferent not in the Tabu List. Also the algorithm have a probability to mutate the route, this create a reverse a subroute of the initial route, that make the algorithm visits other nears solutions.

## Tabu Search

### Hyper - Parameters

!!! note ""
    **sklearn_route.metaheuristics.tabu_search.TabuSearch**(searchs=1250, p_m=0.6, tabu_length=45, tabu_var=10, max_time_work=8,
                                                            people=1, extra_cost=0)

* ```searchs```: int, default=1250
>The number of neighbourhood route searchs, more searchs will lead to a better result but with an associate time computation cost.

* ```p_m```: float, default=0.6
>it's mutate, random probabilies are choosen when it will take place. If the random numbers are 2 and 8 for example, the numbers located at that index will swap positions.

* ```tabu_length```: int, default=45
>it's the lenght of tabu list. Tabu list is a list of routes that the alogirhtm is not going to visit because have been visited already

* ```tabu_var```: int, default=10
>Variations in the lenght of tabu_length, can't be higher number than tabu_length

* ```max_time_work```: float32, default=8
>The number of ours that a employ can work per day. For example
if it's 8 hours, the algorithm will force that a route have to
finish after the 8 hours have been completed, making the
employeed come back home. it's a time constraint.

* ```extra_cost```: float32, default=0
>If it's 0 anything happend. If it's > 0 in combination with
max_time_work when the max_time_work is reached, extra_cost is
applied. This add a extra cost to the solution each time that
max_time_work is reached. It's like extra pay for the worker 
each time max_time_work is completed (journey).

* ```people``` int32, default=1
>The number of people that you use in each route, for example if
you need two truck drivers. that's another contstraint. That
will multiply the time_costs and the extra_cost. Not the travel
cost because it's assumed that both go in the same vehicle.


### Method

* ```.fit(route_example, time_costs, fuel_costs)```:
>This method train the algorithm, we need to pass the following data:  
        1. *route_example*: Is a list random route to initiate the algorithm.
        2. *time_cost*: it's a dict of dicts that represent a diagonal matrix with the times between all points
        3. *fuel_cost*: it's a dict of dicts that represent a diagonal matrix with the costs between all points

### Attribute
* ```history_```:
> it's a list with the best cost in each generation. The loss function.

### Example with Tabu Search Algorithm.

```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.tabu_search import TabuSearch

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
ts = TabuSearch(p_m = 0.3, pop=400, gen=2000, k=5, p_c early_stoping=100,
            max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = ts.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(ts.history_)
```

### Understunding the data needed by the algorithm
The **dicts of dicts** time_cost and fuel_cost, are dicts of dicts for performance reason. these dicst could be also represented as diagonal symmetric matrix, columns and index represent the cost of the point. Here an example:

```
easy_route = {
            1:{
                1:0,
                2:x
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

#We can see clearly the matrix with pandas
import pandas as pd

#Visualice the diagonal symmetric matrix
print(pd.DataFrame(easy_route))
```

