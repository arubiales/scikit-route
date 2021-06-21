#Preprocessing

The preprocessing module could be confused if it's compared with the Scikit Learn Preprocessing module. In this case this module is not a Mathematical module for preprocesing data, it's a module oriente to preprocesed data in order to feed the algorithms, wich means that it's not make it with mathematical purposes only with structure purposes.

The different functions of the preprocessing module are:

## matrix_to_dict

```matrix_to_dict```: is a function that convert a diagonal matrix with origins -> destiny points from a dataframe to a dicts of dicts. This dict of dict is needed for the ```Genetic``` algorithm and the ```EnsembleGenetic``` algorithm in order to compute their results. The **parameters** of the function are:

* ```dataframe```: the dataframe that will be converted
* ```id_origin```: the name of the column with the id of the origin points 
* ```id_destinity```: the name of the column with the id of the destinity points
* ```compute_column```: the column data that will be taken in order to create the data of the diagonal matrix

We have the following Dataframe (Barceloa dataframe from datasets):
![Dataset example](images/dataset_example.png)

Use example:

```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")

#Dataset - id of origin - id of destiny - column to transform (in this case the cost)
cost_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")
```

So the dict of dicts represent a diagonal matrix. it's transformed in a dict of dicts for performance reasons. Only for learning purpose we will represent the dict of dicts as a diagonal matrix:

```
import pandas as pd

df = pd.DataFrame(cost_matrix_df)
```

And the result:

![Dataframe diagona example](images/diagonal_matrix_example.png)

This matrix represent the cost of go from one point (origin id) to other (destinity id). This cost is computed with the hours and the cost of fuel per kilometer.

## normalize

Take a dataframe with latitude and longitude columns and normalize the latitude and longitude, works very well with SOM algorithm, The **parameters** of the function are:

* df_nodes: the dataframe with the data
* lat: name of the latitude column
* lon: name of the longitude conlumn


**Use example**
```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import normalize, df_to_tuple
from sklearn_route.metaheuristics.som import SOM

df_barcelona = load_barcelona()["DataFrame"]
df_barcelona = df_barcelona[["id_origin", "lat_origin", "lon_origin"]].drop_duplicates()
df_barcelona = normalize(df_barcelona, "lat_origin", "lon_origin")
```

## CostScraper
To use this class is needed to have a an account in google maps (you have to pay to google for every extraction). Please visit https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
for more info.

This class is used when there is a dataset with latitude and longitude and is needed to know the distance in time. The **parameters** are:

* api (str): The API credential to the distance matrix Google service
* nodes (list of lists): a list of list in which each list contain in the following order [id, latitude, longitude]. Each list represent a node. CostScraper takes 
    all the info and combine every node with the rest, using symmetric routes
    (same distance go and back). This is done with itertools.combinations.
    The most important part is respect the order of the tuple. This is an input example:
```
    nodes = [
        [1, 26.0336111, 51.2002778],
        [2, 26.0488889, 51.0569444],
        [3, 26.05, 51.25],
        [4, 26.0502778, 51.2975],
        [5, 26.0505556, 51.1358333],
    ]
```

* mode (str): {"driving","walking","bicycling","transit"} default="driving"
    The mode of transport that Google API Distance Matrix offer.

**Methods**

* scrap: with the credentials, the nodes and the mode, scrap all distances from all nodes to
    others and return a tuple of connections between all nodes and the time and meters
    needed. Example:
```
    nodes_connected = [
    (1,
    25.01,
    51.0394444,
    'Al Karaana Road، Al Kir anah, Qatar',
    2,
    25.2847222,
    51.5552778,
    'Ras Bu Abboud Street، Doha, Qatar',
    71104,
    3712),
    (1,
    25.01,
    51.0394444,
    'Al Karaana Road، Al Kir anah, Qatar',
    3,
    25.2861111,
    51.5041667,
    'Barwa Alsadd, C Ring Rd, Doha, Qatar',
    66279,
    3287),
    ...
    ]
```
The elements of each tuple are:
    - First node id
    - Latitude of first node
    - Longitude of first node
    - Address of first node
    - Second node id (The connection will be first node with second)
    - Latitude of seconde node
    - Longitude of sencond node
    - Address of sencond node
    - Distance in seconds from first node to second node and viceversa (symmetric)
    - Distance in meters  from first node to second node and viceversa (symmetric)

Caveat!
The None values are distances not availables in the Google API.  So sometimes
it's needed to continue processing the result of this method by hand.

* to_pickle (str): The path with the .pkl extension to save the results scraped, strongly 
    recommended to use, in order to not to lose the information.

* pandas: The data are parse to a Pandas.DataFrame


## matrix_parse
Parse a list of tuples connected with the format of the CostScraper output into a matrix. Which is a list of tuples, each tuple representing a connection between two nodes. The **parameters** are


* connections (list of lists): the nodes to parse
* mode (str): wich is the measure to connect the nodes, the two options are "meters" or "seconds"

## matrix to dict
2D array with cost of each node with the rest, return a dict of dicts ready for 
the algorithm

**Example**
```
from sklearn_route.datasets import load_barcelona
from sklearn_route.preprocessing import matrix_to_dict
from sklearn_route.metaheuristics.genetics import Genetic

df_barcelona = load_barcelona()

#Dataset - id of origin - id of destiny - column to transform (in this case the hour)
time_matrix = matrix_to_dict(df_barcelona, "id_origin", "id_destinity", "hora")
```

## df_to_tuple

Parse a Dataframe into node of tuples, this helps some algorithms to be faster
