import numpy as np
import pandas as pd
from itertools import combinations
# import plotly.express as px
# import plotly.offline as pyo
from googlemaps import Client
from time import sleep
import pickle
import warnings
from tqdm import tqdm

def dfcolumn_to_dict(dataframe, id_origin, id_destinity, compute_column):
    """
Takes a DataFrame and return a dict of dicts that represent a 2D matrix with costs

Parameters
-----------
dataframe: DataFrame
    The dataframe

id_origin: str
    The column of the DataFrame with the id of origin

id_destinity: str
    The column of the DataFrame with the id of destinity

compute_columns: str
    The columns with the values to compute time/distance
    """

    columns = dataframe[id_origin].drop_duplicates().values
    cluster_len = len(dataframe[id_destinity].drop_duplicates())
    empty_matrix = np.zeros([cluster_len, cluster_len])
    start = 0
    count = 0
    finish = cluster_len
    for n in range(1, cluster_len):
        count += 1
        empty_matrix[n-1][n-1:cluster_len] = dataframe[compute_column][start:finish]
        start = finish
        finish = start + cluster_len - count
    clusters_cost_matrix = pd.DataFrame(empty_matrix + empty_matrix.T, columns=columns, index=columns).to_dict()

    return clusters_cost_matrix

def normalize(df_nodes, lat, lon):
    """
Take a Dataframe and normalize the latitude and the longitude columns, works very well with
SOM algorithm.

Parameters
-----------
df_nodes: dataframe
    The dataframe with the latitude and longitude columns, can have more columns.

lat: string
    The name of the latitude column

lon: string
    The name of the longitude column

Return
-------
The columns latitude and longitude normalize
    """
    ratio = (df_nodes[lat].max() - df_nodes[lat].min()) / (df_nodes[lon].max() - df_nodes[lon].min()), 1
    ratio = np.array(ratio) / max(ratio)
    norm = df_nodes[[lat, lon]].apply(lambda c: (c - c.min()) / (c.max() - c.min()))
    df_nodes[[lat, lon]] = norm.apply(lambda p: ratio * p, axis=1)
    return df_nodes

def df_to_tuple(dataframe):
    """
Parse a DataFrame into node of tuples
    """
    nodes = []
    for n in dataframe.to_numpy():
        n = list(n)
        n[0] = int(n[0])
        nodes.append(tuple(n))

    return tuple(nodes)



# def _swap_point(string):
#     string = str(string).replace(".", "")
#     string = string[:2] + "." + string[2:]
#     return float(string)

# def _plot_points(df, lat, lon, save=False, path=None):
#     fig = px.scatter_mapbox(df, lat='latitude', lon='longitude')
#     fig.update_layout(mapbox_style="open-street-map")
#     fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
#     fig.show(filename=path)
#     if save:
#         pyo.plot(fig, filename=path, auto_open=False)



class DataLossWarning(UserWarning):
    def __str__(self):
            return " Warning DataLossWarning: The data has None values, that have been deleted. Please check your data if you don't want to loss nodes"


class CostScraper:  
    """
CostScraper uses the Google API to extract the distance between points.

YOU NEED A GOOGLE ACCOUNT, AND YOU HAVE TO PAY GOOGLE FOR EVERY EXTRACTION. Please
visit https://developers.google.com/maps/documentation/distance-matrix/usage-and-billing
for more info.

Scikit Route is a completly free open software package, we don't have any economic
interest on this. It's just to help people to have and easy way to get the distance in time
and in meters. Scikit Route don't encourage or promote the use of the Google API.

Parameters:
-----------

api: str
    The API credential to the distance matrix Google service

nodes: list
    A list of lists with the ID, latitude and longitude of every node. CostScraper takes 
    all the info and combine every node with the rest, using symmetric routes
    (same distance go and back). This is done with itertools.combinations.
    The most important part is respect the order of the tuple. This is an input example:

    nodes = [
        [1, 26.0336111, 51.2002778],
        [2, 26.0488889, 51.0569444],
        [3, 26.05, 51.25],
        [4, 26.0502778, 51.2975],
        [5, 26.0505556, 51.1358333],
    ]

mode: str, {"driving","walking","bicycling","transit"} default="driving"
    The mode of transport that Google API Distance Matrix offer.

------------------------------------------------------------------------------------------------------------

Method:
---------

scrap():
    With the credentials, the nodes and the mode, scrap all distances from all nodes to
    others and return a tuple of connections between all nodes and the time and meters
    needed. Example:
    
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

to_pickle(filename):
    Parameter
    ---------
    filename: The path with the .pkl extension to save the results scraped, strongly 
    recommended to use, in order to not to lose the information.

pandas(): 
    The data are parse to a Pandas.DataFrame
    """

    def __init__(self, api, nodes, mode="driving"):
        self._api = Client(api)
        self._nodes = combinations(nodes, r=2)
        self.mode = mode
        self._ids = [n[0] for n in nodes]

    @property
    def api(self):
        return self._api.key

    @api.setter
    def api(self, credential):
        self._api = Client(credential)

    @api.deleter
    def apit(self):
        del self._api

    @property
    def nodes(self):
        return self._nodes
    
    @nodes.setter
    def nodes(self, values):
        self._nodes = combinations(values, r=2)
        self._ids = [n[0] for n in values]
    
    @nodes.deleter
    def nodes(self):
        del self._nodes

    def scrap(self):
        """
        Al ejecutar este método se envia la petición a Google y este nos devuelve las distancias,
        en metros y segundos. **CUIDADO EJECUTAR ESTE MÉTODO TIENE UN COSTE MONETARIO**
        """        

        scraped_conections = []
        for connection in tqdm(self._nodes):
            sleep(0.01)
            res_dict = self._api.distance_matrix(tuple(connection[0][1:]), tuple(connection[1][1:]), mode=self.mode)
            
            elements = res_dict["rows"][0]["elements"][0]
            if res_dict["status"] == "OK" and elements["status"] == "OK":
                scraped_conections.append((*connection[0], res_dict["origin_addresses"][0], *connection[1],
                                            res_dict["destination_addresses"][0], elements["distance"]["value"],
                                            elements["duration"]["value"]))
            else:
                scraped_conections.append(tuple([None]*10))
        
        self.scraped_conections = scraped_conections
        return scraped_conections
    
    def to_pickle(self, filename):
        if ".pkl" == filename[-4:]:
            raise TypeError(f'filename must finish in ".pkl" this is the pickle extension and is finishing {filename[:-4]}')
        with open(filename, "wb") as f:
            pickle.dump(self.scraped_conections, f)

    def pandas(self):
        columns = ["id_origin", "lat_origin", "lon_origin", "address_origin", "id_destinity", 
                  "lat_destinity", "lon_destinity", "address_destinity", "meters", "seconds"]
        return pd.DataFrame(self.scraped_conections, columns=columns)



def matrix_parse(connections, mode="meters"):
    """
Parse a list of tuples connected with the format of the CostScraper output into a matrix.
Which is a list of tuples each tuple representing a connection between two nodes

Parameters:
------------

connections: list
    The list of tuples connection  with the format of the CostScraper output into a matrix.
    Which is a list of tuples each tuple representing a connection between two nodes. Example:

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

    The elements of each tuple are:
    - First node id
    - Latitude of first node
    - Longitude of first node
    - Address of first node
    - Second node id (The connection will be first node with second)
    - Latitude of seconde node
    - Longitude of sencond node
    - Address of sencond node
    - Distance in meters from first node to second node and viceversa (symmetric)
    - Distance in seconds  from first node to second node and viceversa (symmetric)

mode: str {"meters", "time"} default="meters"
    If meters, the idx -2 of each tuple is selected else, the idx -1 is selected.
    """
    ids = []
    check_none = False
    for con in connections:
        if not check_none and None in con:
            check_none = True
        ids.append(con[0])
    
    if check_none:
        warnings.warn(str(DataLossWarning), DataLossWarning)

    length_connections = len(list(filter(lambda x: x[0] == ids[0], connections))) +1
    ids = dict.fromkeys([con[0] for con in connections if con[0] is not None])

    matrix = np.empty((length_connections, length_connections))

    if mode == "seconds":
        idx = -1
    elif mode == "meters":
        idx = -2
    else:
        raise ValueError('Please select an available mode. The modes availables are "meters" and "seconds"')

    counter = 0
    for node_id in ids:
        selected = filter(lambda x: x[0] == node_id, connections)
        matrix[counter, counter+1:] = [sel[idx] for sel in selected]
        counter +=1

    return matrix

def matrix_to_dict(matrix):
    """
2D array with cost of each node with the rest, return a dict of dicts ready for 
the algorithm

Parameters
-----------
matrix: 2d array
    the matrix that will be converted into a dict of dicts.

Return
-------
dict of dicts with the connections between nodes.

    """
    clean_dict = {}
    for key in dict_of_dicts:
        temp_dict = {}
        for k, v in dict_of_dicts[key].items():
            if v !=0:
                temp_dict[k] = v
        clean_dict[key] = temp_dict
