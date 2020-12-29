 [![Build Status](https://travis-ci.org/arubiales/scikit-route.svg?branch=main)](https://travis-ci.org/arubiales/scikit-route) [![codecov](https://codecov.io/gh/arubiales/scikit-route/branch/main/graph/badge.svg?token=1CKqr34fsc)](https://codecov.io/gh/arubiales/scikit-route) ![](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8-blue)

![Logo](docs/images/logo.png "Title")

**WIP Package**

**Scikit Route** is a Python module create with Python and Cython to optimize route and travel problems and there are a lot of different alogithms in the package for all the different purposes.

**Scikit Route have the spirit and the soul of Scikit Learn** wich mean that is fully oriented to the programmer, easy to use and to understund, you can create algorithm with a few lines of code, and the modules, submodules, methods of the algorithms, etc, are the same than Scikit Learn wich will make even easyer to use this library if you previously know Scikit Learn.

This project was started in 2020 by Alberto Rubiales and one of the main goals is to be integrated in the future with Scikit Learn project.

Anyone is free to help and developt the package.

## Installation
### Dependencies
scikit_route requires:
* Python (>=3.6)
* Numpy (>=1.19)
* Pandas (>=1.1)
* Scikit-Learn (>=0.23)
* Cython (>=0.29)

Others version could work but it's not guaranted.

Also to Allow functions with plotting capabilites requires Matplotlib (>=3.1)

### User installation

The easiest way to install scikit_route is using ```pip```:
```pip install scikit_route```

or ```conda```:

```conda instal -c conda-forge scikit_route```

### Fast User Guide
This is an example on how to use **Scikit Route** and show how easy it is.

Here the **MTSPTW problem is solved**, we need to create a good path for our commercials in the City in Barcelona and we have time restrictions (in Spain is not legal work more than 8h) and cost restrictions. We will use the Genetic algorithm wich is a algorithm that fit very well for our approach.

#### Load the modules
First of all we load the modules that we will use:
```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.genetics import Genetic
```


#### Pre-processing data
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

#### Running the Algorithm
Here we see how the algorithm operation is the same as Scikit Learn. You can see all the Hyperparamenters, what they do, and examples with the builtin ```help()``` function. Also there are more detailed examples in the docs webpage.

In this case we choose an **algorithm with Metaheuristic** aproach to the problem. There are other algorithms in the package that can solve this problem or have another approach.

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

### Conclusions

Sklearn Route is library to **solve differents types of route problems, easy to use and fast to learn.** If you want to know more about the different algorithms, hyperparameters, how it works and more examples, **the package have a very deep and detailed documentations** with a lot of examples in the official documentation here.
