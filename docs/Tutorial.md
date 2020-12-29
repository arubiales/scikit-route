# Tutorial
This is a quick tutorial based on examples to learn to use quickly the differents algorithms of the library
## Genetic

Genetic is a metaheuristic algorithm wich mean that it's not deterministic, so every time the algorithm output will be a different solution near of the unknown best solution.

This Algorithm is focused in solve the MTSPTW problem (**M**ulti **T**ravel **S**ales **P**roblem with **T**ime **W**indows). This problem is about a person that need to cover a number of points to visit but have a limit of working hours (time constraint) and extra cost. If it's the case, this algorithm is the perfect solution:


### Load the modules
First of all we load the modules that we will use:
```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.genetics import Genetic
```
If you want to know more about these modules you can visit:

* [Datasets](Datasets.md)
* [Preprocessin](Preprocessing.md)
* [Genetics](Genetics.md)


### Pre-processing data
Here we put the data in the format required by the algorithm. In this case we use the ```matrix_to_dict``` function to transform our *id_points* cost in fuel and in time to dicts of dicts.
You can see the docs of all functions, using the ```help()``` command. Also there are more detailed examples in the docs webpage.

```
df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Create a random route, it's will needed to initiate the algorithm
route_example = list(dict.fromkeys(cost_matrix_df).keys())
```

### Running the Algorithm
Here we see how the algorithm operation is the same as Scikit Learn. You can see all the Hyperparamenters, what they do, and examples with the builtin ```help()``` function. Also there are more detailed examples in the docs webpage.

In this case we choose an **algorithm with Methaeuristic** aproach to the problem. There are other algorithms in the package that can solve this problem or have another approach.

```
#Instantiate the algorithm
ga = Genetic(p_m = 0.3, pop=400, gen=2000, k=5, p_c early_stoping=100, max_time_work=6, extra_cost=12.83)

#random route - time_matrix - cost_matrix
result = ga.fit(route_example, time_matrix, cost_matrix)

#Printing the best route
print(result)

#Printing the loss function
print(ga.history_)
```

In the Genetic Algorithm we have a lot of Hyperparameters to config, for a full understanding of the Algorithm please refer to [Genetics](Genetics.md)

## Clustering
## Genetics
## Simmulated Annealing
## Brute Force
## Utilities