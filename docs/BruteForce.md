# Brute Force

The main advantage of this Brute Force algorithm over others solutions are:

* Easy to learn and to use, inspired in Scikit Learn library.
* Heuristic aproach, wicht mean it's deterministic, and check all possible solutions to give the best.
* Very well documented.

This algorithm is very easy to understund, it takes all the possible combinations and choose the best one.

## Brute Force

### Hyper Parameters

!!! note ""
    **sklearn_route.brute.BruteForce***(max_time_work=8.0, extra_cost=10.0, people=1)*

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


### Example with Brute Force algorithm.


```
from sklearn_route.datasets import load_alicante_murcia
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.brute import BruteForce

df_alicante_murcia = load_alicante_murcia()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_alicante_murcia, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcedf_alicante_murcialona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
bf = BruteForce(max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = bf.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)
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

**Output:**  
![mini_matrix](images/diagonal_matrix_mini.png)

It's important to understund that the cost matrix **must be the computation of all the costs** that the user of the algorithm want to take in count **per person**. In the majority of cases the cost will be composed by:

* The cost of the fuel for one person to go from one point to another.
* The cost of the hours of work needed from one point to another.

If we have multiple persons doing the same route, the `people` parameter will multiply the cost by the number of persons and will increase the cost with the `extra_cost` in case of the ```max_time_work``` paremeter will be surpased.

Also note that `extra_cost` **could be used as a maximun capacity** of a truck for example in the case of material transports problems.

The other dict of dicts `time_matrix` parameter will be only used by the algorithm to compute the `max_time_work` and compute the final route of the algorithm with the times that the salesman/truck have finished their journey.

### Caveats for beginners
It's always common for beginners think "If I use this algorithm I will get the best result (not a suboptimal like with others metaheuristics approachs)" This is true but the main problem of this algorithm  is that is very computationally expensive, for routes with more than 15 places you will need to make **more than 1 Trillion operations** (15 factorial because the first one don't count, because is always home) that's imposible to do for machines today, even for 15 (14!) places will be 87 Billions operations, probably it's needed 1 day of computation. That's why we encourage to not use on routes with **more than twelve places** and never use it with more than 15.