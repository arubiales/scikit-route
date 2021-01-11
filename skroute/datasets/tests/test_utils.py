import os
import pandas as pd

from skroute.datasets import load_japan, load_morocco
_path = "/".join(os.path.join(os.path.dirname(__file__) + "/").split("/")[:-2]) + "/"
from skroute.datasets._utils_datasets import _loader, _read_txt, _read_tsp, _docstring_decorator


class TestUtils:

    def test_loader(self):
        load_jap = _loader(_path + "_data/_latitude_longitude/ja9847.tsp")
        load_mor = _loader(_path + "_data/_latitude_longitude/mo14185.tsp")
        load_ire = _loader(_path + "_data/_latitude_longitude/ei8246.tsp")

        load_jap_sm = _loader(_path + "_data/_latitude_longitude/ja9847.tsp", "small")
        load_mor_sm = _loader(_path + "_data/_latitude_longitude/mo14185.tsp", "small")
        load_ire_sm = _loader(_path + "_data/_latitude_longitude/ei8246.tsp", "small")

        load_jap_mid = _loader(_path + "_data/_latitude_longitude/ja9847.tsp", "medium")
        load_mor_mid = _loader(_path + "_data/_latitude_longitude/mo14185.tsp", "medium")
        load_ire_mid = _loader(_path + "_data/_latitude_longitude/ei8246.tsp", "medium")
                
        assert type(load_jap) and type(load_mor) and type(load_ire) == dict, "_loader function must return a dictionary"
        assert len(load_jap) and len(load_ire) and len(load_mor) == 3, "_loader must return a dictionary must have 3 keys"
        assert load_jap.keys() == load_ire.keys() == load_mor.keys(), "_loader must return a dictionary always with the same keys"
        assert type(load_jap["DataFrame"]) and type(load_mor["DataFrame"]) and type(load_ire["DataFrame"]) == pd.DataFrame
        assert type(load_jap["DESCR"]) == str, "_loader dict value DESCR must be a str"
        assert type(load_jap["feature_names"]) == str, "_loader dict value feature_names must be a str"
        assert load_jap["DESCR"] == load_mor["DESCR"] == load_ire["DESCR"], "_loader must return always the same DESCR values"
        assert load_jap["feature_names"] == load_mor["feature_names"] == load_ire["feature_names"], "_loader must return the same feature_names value"
        assert (all(load_jap["DataFrame"].columns == load_mor["DataFrame"].columns) and 
               all(load_mor["DataFrame"].columns == load_ire["DataFrame"].columns)), "All datasets loaded by loader must have sames columns names"
        assert load_jap["DataFrame"].columns.tolist() == ['id', 'latitude', 'longitude'], "Columns names loaded by _loader are incorrect"
        assert len(load_jap["DataFrame"]) > len(load_jap_mid["DataFrame"]) > len(load_jap_sm["DataFrame"]), "_loader mode parameter is not working correctly"
        assert len(load_mor["DataFrame"]) > len(load_mor_mid["DataFrame"]) > len(load_mor_sm["DataFrame"]), "_loader mode parameter is not working correctly"
        assert len(load_ire["DataFrame"]) > len(load_ire_mid["DataFrame"]) > len(load_ire_sm["DataFrame"]), "_loader mode parameter is not working correctly"


    def test_read_txt(self):
        columns_costs = _read_txt(_path+ "columns_costs.txt")
        columns_lon_lat = _read_txt(_path + "columns_lon_lat.txt")
        columns_descr = _read_txt(_path + "DESCR.txt")

        assert type(columns_costs) and type(columns_lon_lat) and type(columns_descr) == str, "_read_txt must return a str"
        assert columns_descr != columns_lon_lat != columns_costs, "_read_txt must return differents strings if the txt are differents"
    
    def test_read_tsp(self):
        df_tsp = _read_tsp(_path + "_data/_latitude_longitude/ar9152.tsp")
        assert type(df_tsp) == pd.DataFrame, "_read_tsp must return a dataframe"
        assert df_tsp.columns.to_list() == ["id", "latitude", "longitude"], "_read_tsp must return a dataframe"
        assert all([type(n) == float for n in df_tsp.iloc[0].tolist()]), "_read_tsp must clean the dataset info of the beginning"
        assert all([type(n) == float for n in df_tsp.iloc[-1].tolist()]), "_read_tsp must clean the useless data of the end"

    def test_docstring_decorator(self):

        @_docstring_decorator()
        def test_func1():
            return "test_function"
        
        @_docstring_decorator("cost")
        def test_func2():
            return "test_function"

        test_func1_res = test_func1()
        test_func2_res = test_func2()
        a = help(test_func1)
        test_func1.__doc__
        text_func1 = ["small", "big", "medium", "takes", "longitude", "latitude", "Func1"]
        text_func2 = ["latitude","longitude", "Func2"]

        assert type(test_func1_res) and type(test_func2_res) == str, "_docstring_decorator modify the result of the function"
        assert test_func1_res and test_func2_res == "test_function", "_docstring_decorator modify the result of the function"
        assert  all([word in test_func1.__doc__ for word in text_func1]), "_docstring_decorator main info have change"
        assert  all([word in test_func2.__doc__ for word in text_func2]), "_docstring_decorator main info have change"
        assert test_func1.__doc__ != test_func2.__doc__, "_dostring_decorator parameter must change the docstring of the functions"

