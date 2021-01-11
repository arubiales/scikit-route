# import os, sys
# _path = "/".join(os.path.join(os.path.dirname(__file__) + "/").split("/")[:-2]) + "/"
# sys.path.append(_path)

from skroute.datasets import _datasets
import inspect
import pandas as pd
import numpy as np
from copy import deepcopy


function_lists = [func_name for func_name, func in _datasets.__dict__.items() if inspect.isfunction(func) and "load_" in func_name]
all_datasets = [getattr(_datasets, func)() for func in function_lists]

class TestDataset:

    def test_loads(self):
        assert len(all_datasets) == 32, "dataset module can't load all datasets"
        assert all([True for dst in all_datasets for k in dst.keys() if k in ["DataFrame", "DESCR", "feature_names"]]), \
            "dataset module can't load all datasets correctly"
        assert all([True for dst in all_datasets for value in list(dst.values()) if value is not None]), \
            "dataset module can't load all datasets correctly"

    def test_types(self):
        assert all([type(dst) == dict for dst in all_datasets]), "datasets functions must be all dictionaries"
        assert all([type(dst["DataFrame"]) == pd.DataFrame for dst in all_datasets]), "datasets functions must return a DataFrame"
        assert all([type(dst["DESCR"]) == str for dst in all_datasets]), "datasets functions must return a description string"
        assert all([type(dst["feature_names"]) == str for dst in all_datasets]), "datasets functions must return the feature names"

    def test_dataframes(self):
        columns_cost = ['id_origin', 'id_destinity', 'lat_origin', 'lon_origin', 'lat_destiniy',
       'lon_destinity', 'cluster', 'origin', 'destinity', 'meters', 'secs',
       'hours', 'kilometers', 'cost']
        columns_latlon = ['id', 'latitude', 'longitude']
        dtypes_cost = [
            np.int64,np.int64, np.float64, np.float64, np.float64, np.float64,
            np.int64, np.object, np.object, np.int64, np.int64, np.float64,
            np.float64,np.float64
        ]
        dtypes_latlon = [np.int64, np.float64, np.float64]

        assert all([all(dst["DataFrame"].columns == columns_cost) for dst in all_datasets[:5]]), \
                "dataset functions DataFrames columns are not correct due to the function or the data"
        assert all([all(dst["DataFrame"].columns == columns_latlon) for dst in all_datasets[5:]]), \
                "dataset functions DataFrames columns are not correct due to the function or the data"
        assert all([all(dst["DataFrame"].columns == columns_cost) for dst in all_datasets[:5]]), \
                "dataset functions DataFrames columns are not correct due to the function or the data"
        assert all([all(dst["DataFrame"].columns == columns_latlon) for dst in all_datasets[5:]]), \
                "dataset functions DataFrames columns are not correct due to the function or the data"

        assert all([all(dst["DataFrame"].dtypes == dtypes_cost) for dst in all_datasets[:5]]), \
                "dataset functions DataFrame types are not correct due to the function or the data"    
        assert all([all(dst["DataFrame"].dtypes == dtypes_latlon) for dst in all_datasets[5:]]), \
                "dataset functions DataFrame types are not correct due to the function or the data"

    def test_parameters(self):
        all_datasets_small = [getattr(_datasets, func)("small") for func in function_lists[5:]]
        all_datasets_medium = [getattr(_datasets, func)("medium") for func in function_lists[5:]]
        all_datasets_small_2 = [getattr(_datasets, func)("small") for func in function_lists[5:]]
        all_datasets_medium_2 = [getattr(_datasets, func)("medium") for func in function_lists[5:]]
        all_datasets_big = deepcopy(all_datasets[5:])
        length_dataset = len(all_datasets_small_2)

        assert all([len(all_datasets_small[i]["DataFrame"]) < len(all_datasets_medium[i]["DataFrame"]) < len(all_datasets_big[i]["DataFrame"]) \
                for i in range(length_dataset)]), "the parameter load of the functions from dataset module is not working correctly"
        assert all([all(all_datasets_small[i]["DataFrame"] == all_datasets_small_2[i]["DataFrame"]) for i in range(length_dataset)]), \
               "The dataset load functions, must always load the same datasets"
        assert all([all(all_datasets_medium[i]["DataFrame"] == all_datasets_medium_2[i]["DataFrame"]) for i in range(length_dataset)]), \
                "The dataset load functions, must always load the same datasets"

    def test_docs(self):
        all_docs = [getattr(_datasets, func).__doc__ for func in function_lists]
        text_func1 = ["small", "big", "medium", "takes", "longitude", "latitude", "Func1"]
        text_func2 = ["latitude","longitude", "Func2"]

        assert  all([True for doc in all_docs[:5] for word in text_func1 if word in text_func1]), "_docstring_decorator main info have change"
        assert  all([True for doc in all_docs[5:] for word in text_func2 if word in text_func2]), "_docstring_decorator main info have change"
