"""
Datasets

Authors
-------
2020: Alberto Rubiales <al.rubiales.b@gmail.com>
"""

import pandas as pd
from ._utils_datasets import _read_tsp, _read_txt, _loader, _docstring_decorator
import os, sys

_path = os.path.join(os.path.dirname(__file__) + "/", )

# #Borrar esto, es para jupyter
# sys.path.append(_path)
# from _utils_datasets import _read_tsp, _read_txt, _loader, _docstring_decorator




@_docstring_decorator("cost")
def load_alicante_murcia():
    return {
        "DataFrame":pd.read_pickle(_path + "_data/_money_cost/Alicante-Murcia_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns_costs.txt")
    }


@_docstring_decorator("cost")
def load_barcelona():
    return {
        "DataFrame":pd.read_pickle(_path + "_data/_money_cost/Barcelona_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path +"/columns_costs.txt")
    }

@_docstring_decorator("cost")
def load_madrid():
    return {
        "DataFrame":pd.read_pickle(_path + "_data/_money_cost/Madrid_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns_costs.txt")
    }

@_docstring_decorator("cost")
def load_valencia():
    return {
        "DataFrame":pd.read_pickle(_path + "_data/_money_cost/Valencia.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns_costs.txt")
    }

@_docstring_decorator("cost")
def load_costs_qatar():
    return{
        "DataFrame":pd.read_pickle(_path + "_data/_money_cost/Valencia.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns_costs.txt")
    }  

@_docstring_decorator("latlon")
def load_argentina(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ar9152.tsp", mode)

@_docstring_decorator("latlon")
def load_burma(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/bm33708.tsp", mode)

@_docstring_decorator("latlon")
def load_china(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ch71009.tsp", mode)

@_docstring_decorator("latlon")
def load_canada(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ca4663.tsp", mode)

@_docstring_decorator("latlon")
def load_djibouti(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/dj38.tsp", mode)

@_docstring_decorator("latlon")
def load_egypt(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/eg7146.tsp", mode)

@_docstring_decorator("latlon")
def load_ireland(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ei8246.tsp", mode)

@_docstring_decorator("latlon")
def load_finland(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/fi10639.tsp", mode)

@_docstring_decorator("latlon")
def load_greece(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/gr9882.tsp", mode)

@_docstring_decorator("latlon")
def load_honduras(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ho14473.tsp", mode)

@_docstring_decorator("latlon")
def load_italy(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/it16862.tsp", mode)

@_docstring_decorator("latlon")
def load_japan(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ja9847.tsp", mode)

@_docstring_decorator("latlon")
def load_kazakhstan(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/kz9976.tsp", mode)

@_docstring_decorator("latlon")
def load_luxembourg(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/lu980.tsp", mode)

@_docstring_decorator("latlon")
def load_morocco(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/mo14185.tsp", mode)

@_docstring_decorator("latlon")
def load_oman(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/mu1979.tsp", mode)

@_docstring_decorator("latlon")
def load_nicaragua(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/nu3496.tsp", mode)

@_docstring_decorator("latlon")
def load_panama(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/pm8079.tsp", mode)

@_docstring_decorator("latlon")
def load_qatar(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/qa194.tsp", mode)

@_docstring_decorator("latlon")
def load_rwanda(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/rw1621.tsp", mode)

@_docstring_decorator("latlon")
def load_sweden(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/sw24978.tsp", mode)

@_docstring_decorator("latlon")
def load_tanzania(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/tz6117.tsp", mode)

@_docstring_decorator("latlon")
def load_uruguay(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/uy734.tsp", mode)

@_docstring_decorator("latlon")
def load_vietnam(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/vm22775.tsp", mode)

@_docstring_decorator("latlon")
def load_sahara(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/wi29.tsp", mode)

@_docstring_decorator("latlon")
def load_yemen(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/ym7663.tsp", mode)

@_docstring_decorator("latlon")
def load_zimbabwe(mode="big"):
    return _loader(_path + "_data/_latitude_longitude/zi929.tsp", mode)

