# Node Ranking Based on Stats (NRBS)

The main advataje of NRBS over other solutions are:

* Easy to lean and to use inspired in Scikit Learn library
* Heuritic approach, wich mean it's deterministic, NRBS returns always the same solution with the same Hyperparameters
* Very Fast and with very good results

## Global understunding of the Algorithm
NRBS is based on [this paper](https://arxiv.org/pdf/1608.01716.pdf). Basically this algorithm base the **connections between points of a route in the mean and standar deviation of the distance** from ones point each others. For that we have two formulas:

1. Priority Formula: given the mean and the standar deviation of a point distance to the rest of points, Priority function raise them to an exponent and multiply both. The node with the highest priority is the first choosen to select a connection, and it's continue till the node with less priority

$$P_i = \mu_i^\alpha \sigma_i^\beta$$

2. Connection Formula: based on the mean, the standar deviation and the distance from one node to other, give us back the connection ratting, the nodes with highest connections are choosed to connect. The Exponents are the different hyper parametrs in this formula

$$ C_j = \frac{\mu_j^\delta \sigma_j^\epsilon}{d_{j}^{i^\lambda}}$$


**Caveat:** This alorithm solve the TSP/VRP problem, by now it's not available with constraints but it will be in the future. 

## NRBS Hyper Parameters

!!! note ""
    **sklearn_route.genetics.Genetic***(p_c=0.6, p_m=0.4, pop=400, gen=1600, k=3, early_stopping=None,
                    max_time_work=8.0, extra_cost=10.0, people=1)*

* `mean_priority`: float
>Must be a number greater than zero, it is the exponent of the Priority formula mean, higher number make points with higher mean (more distance from others) have large values and priority selection, strongly recommended a number between 0 and 2.

* `std_priority`: float
>Must be a number greater than zero, it is the exponent of the Priority formula standar deviation, higher number make points with higher deviation (more diferents of distance between points) have larger values and priority selection, Strongly recommended a number between 0 and 2

* `mean_connection`: float
>Must be a number greater than zero, it is the exponent of the Connection formula mean, higher number make points with higher mean (more distance from others) have large values and priority connection, strongly recommended a number between 0 and 2.

* `std_connection`: float
>Must be a number greater than zero, it is the exponent of the Connections formula standar deviation, higher number make points with higher deviation (more diferents of distance between points) have larger values and priority connection, Strongly recommended a number between 0 and 2

* `distance_weight`: float
>Must be a number greater than zero, it is the exponent of the Connections formula standar deviation, higher number make points with higher mean and deviation haves less importance and lower ratting at connection  have larger Strongly recommended a number between 0 and 2

## Method

`.fit(route_example, time_costs, fuel_costs):`
> This method train the algorithm, we need to pass the following parameters:  
`start_node_id`: The starting node.  
`ids_node`: a list with all the nodes ids.   
`cost_matrix`: a dict of dicts that represent a diagonal matrix with the times between all points. 

## Example with NRBS algorithm

```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.heuristics import NRBS 

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())
start_id = route_example[0]

nrbs = NRBS(mean_priority=0.3, pop=1.5, gen=0.4, k=2, p_c early_stoping=1)

#random route - time_matrix - cost_matrix
result = nrbs.fit(start_id, route_example, cost_matrix)

#Print the cost and the route.
print(result)
```


### Understunding the data needed by the algorithm
The **dicts of dicts** matrix_cost, are dicts of dicts for performance reason. these dicts could be also represented as diagonal symmetric matrix, columns and index represent the cost of the point. Here an example:

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