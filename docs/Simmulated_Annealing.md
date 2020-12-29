# Simulated Annealing
The main advantage of Simulated Annealing over others metaheuristics algorithms is his speed. It can be instantiate and solved in less than a tenth of a second, that give the oportunity to make a lot of search in the loss function to improve the results. Also as other Algorithms of this library it's based on Scikit Learn so is very easy to use.

## Global understunding of the Algorithm

Genetic Algorithm is a metaheuristic aproach wich mean is not deterministic, this lead that every time you run the result could be different but will be very near of the global optimum.

This algorith is used on the metal cooling to decrease the **temperature** to the optimal number. The temperature is decresed by a **delta** parameter, each time the temperature decrease by the delta parameter, th e algorithm looks for a number **neighbours** of combination of routes. When the temperature decrease to the minimun temperature **tolerated** the algorithms stop

**tolerance** it's like iterations, given a temperature *X*, more tolerance, less iterations, less tolerance, moree iterations.

## Simmulated annealing

!!! note ""
    **sklearn_route.simulated_annealing.SimulatedAnnealing***(temp=12.0, neighbours=250, delta=0.78, tol=1.29,
                max_time_work=8.0, extra_cost=10.0, people=1)*


### Hyper -  Parameters
* `temp`: float32, default=12
>The temperature parameter. In the algorithm the temperature will 
decrease till reach the tol parameter to give you the best
solution. If the temp is hight the algorithm is slower and 
look for more posible solutions if it's low the algorithm it's
faster but look on less solutions. Note that is not always good
for the optimization look more solutions, because this parameter
is combined with delta and tol.

* `neighbours`: int32 , default=250
>Given a route the exchanges that will make in at the same temperature.
You can think as the neighbours that will visit at a particular
temperature. If the `neighbours` is hight the algorithm is slower and 
look for more posible solutions if it's low the algorithm it's
faster but look on less solutions. If it's too hight you probably will
try the sames solutions over and over again.

* `delta`: float32, default=0.78
>A number between 0 and 1 the highest the number more slow the
the temperature decrease, and more solution the algorithm try.
If it's close to 0 the temperature decrease fast and the
algorithm converge faster but look for less optimal solutions

* `tol`: float32, default=1.29
>The tolerance, the minimun temperature allow by the algorithm
one time the "temp" is equal to "tol" the algorithm finish.

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

### Examples with Simulated Annealing.

```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.simulated_annealing import SimulatedAnnealing

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm
sa = SimulatedAnnealing(temp=10, neighbours=300, delta=0.82, tol=0.2,
                        max_time_work=8, extra_cost=12.83, people=2)

#random route - time_matrix - cost_matrix
result = sa.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(sa.history_)
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
It's always common for beginners think "If I increase the time, with high `temp`, high `neighbours` and low `delta` 
the results will be better" This is not true, you have to find the optimal parameters for your especific problem.
For example if your data is compose by 4 differents places to visit, don't have sense to have 250 neighbours, because
you don't have a lot of differents combinations possible.

Other case it's if you have for example a too high temperature, and the places are very near one each other then
probably the algorithm will never converge because hight temperature will lead to not accept worst solutions when
the algorithm is converging.


## Ensemble Simulated Annealing

!!! note ""
    **sklearn_route.simulated_annealing.EnsembleSimulatedAnnealing***(n_simulateds=20,  temp=12.0, neighbours=250, delta=0.78, tol=1.29,
                max_time_work=8.0, extra_cost=10.0, people=1, n_jobs=1)*

Ensemble Simulated Annealing is a bagging of Simulated Annealing models, so you can refer to the documentation above to see how it works. Basically are a *X* number of Genetics estimator, this help to get a better result than the Simulated Annealing alogorithm, the computationall cost will depend of the number of workers (n_jobs) you use. In the case you use the same `n_jobs` as `n_simulateds` you will be as fast in time as Simulated Anealing

As said above, the algorith is pretty much the same, so here there are only the new hyper parameters.

### Hyper - Parameters.

* `n_simulateds`: int, default=10 
>The number of Simulated Annealing algorithms that will be thrown. The more algorithms better result could be achieve, but's will be computationally more expensive (this can be solved/mitigated with the n_jobs parameter)
* `n_jobs`: int, default=1
>The number of workers (threads) that will use the algorithm in parallel, by default is one, and must be at least one to the algorithm run. Use this parameter with caution, maybe can collapse the computer if you select a lot jobs. 

### Example with Ensemble Simulated Annealing

```
from sklearn_route.datasets import load_valencia
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.genetics import EnsembleGenetic

df_valencia = load_valencia()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_valencia, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_valencia, "id_origin", "id_destinity", "cost")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())

#Instantiate the algorithm with 18 simulated annealing algorithm and 6 jobs
esa = EnsembleSimulatedAnneling(n_simulateds=18, temp = 18, neighbours=300, max_time_work=6, extra_cost=12.83, n_jobs=6)

#random route - time_matrix - cost_matrix
result = esa.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)
```
