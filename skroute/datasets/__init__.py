import pandas as pd
import os
_path = os.path.join(os.path.dirname(__file__))

__all__ = ["load_alicante_murcia", "load_barcelona", "load_madrid", "load_valencia"]
#TODO: describe functions




def load_alicante_murcia():
    return {
        "DataFrame":pd.read_pickle(_path + "/Alicante-Murcia_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns.txt")
    }

def load_barcelona():
    return {
        "DataFrame":pd.read_pickle(_path + "/Barcelona_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path +"/columns.txt")
    }

def load_madrid():
    return {
        "DataFrame":pd.read_pickle(_path + "/Madrid_places.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns.txt")
    }

def load_valencia():
    return {
        "DataFrame":pd.read_pickle(_path + "/Valencia.pkl"),
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns.txt")
    }        


def load_argentina(mode="big"):
    return _loader("/ar9152.tsp", mode)

def load_burma(mode="big"):
    return _loader("/bm33708.tsp", mode)

def load_china(mode="big"):
    return _loader("/ch71009.tsp", mode)

def load_canada(mode="big"):
    return _loader("/ca4663.tsp", mode)

def load_djibouti(mode="big"):
    return _loader("/dj38.tsp", mode)

def load_egypt(mode="big"):
    return _loader("/eg7146.tsp", mode)

def load_ireland(mode="big"):
    return _loader("/ei8246.tsp", mode)

def load_finland(mode="big"):
    return _loader("/fi10639.tsp", mode)

def load_greece(mode="big"):
    return _loader("/gr9882.tsp", mode)

def load_honduras(mode="big"):
    return _loader("/ho14473.tsp", mode)

def load_italy(mode="big"):
    return _loader("/it16862.tsp", mode)

def load_japan(mode="big"):
    return _loader("/ja9847.tsp", mode)

def load_kazakhstan(mode="big"):
    return _loader("/kz9976.tsp", mode)

def load_luxembourg(mode="big"):
    return _loader("/lu980.tsp", mode)

def load_morocco(mode="big"):
    return _loader("/mo14185.tsp", mode)

def load_oman(mode="big"):
    return _loader("/mu1979.tsp", mode)

def load_nicaragua(mode="big"):
    return _loader("/nu3496.tsp", mode)

def load_panama(mode="big"):
    return _loader("/pm8079.tsp", mode)

def load_qatar(mode="big"):
    return _loader("/qa194.tsp", mode)

def load_rwanda(mode="big"):
    return _loader("/rw1621.tsp", mode)

def load_sweden(mode="big"):
    return _loader("/sw24978.tsp", mode)

def load_tanzania(mode="big"):
    return _loader("/tz6117.tsp", mode)

def load_uruguay(mode="big"):
    return _loader("/uy734.tsp", mode)

def load_vietnam(mode="big"):
    return _loader("/vm22775.tsp", mode)

def load_sahara(mode="big"):
    return _loader("/wi29.tsp", mode)

def load_yemen(mode="big"):
    return _loader("/ym7663.tsp", mode)

def load_zimbabwe(mode="big"):
    return _loader("/zi929.tsp", mode)



def _loader(url, mode="big"):
    dictionary_dataset = {
        "DESCR": _read_txt(_path + "/DESCR.txt"),
        "feature_names": _read_txt(_path + "/columns.txt")
    }
    if mode == "big":
        dictionary_dataset["DataFrame"] = _read_tsp(_path + url)
        return dictionary_dataset
    elif mode == "medium":
        df =_read_tsp(_path + url)
        dictionary_dataset["DataFrame"] = df.sample(int(len(df) *0.2), random_state=2019)
        return dictionary_dataset
    elif mode== "small":
        df =_read_tsp(_path + url)
        dictionary_dataset["DataFrame"] = df.sample(int(len(df) *0.005), random_state=2019)
        return dictionary_dataset
    else:
        raise ValueError(f"The mode '{mode}' is not availabre, the modes availables are: big, medium, small")




def _read_txt(path:str) -> list:
    """sumary_line
    
    Keyword arguments:
    argument -- description
    Return: return_description
    """
    
    with open(path, "r") as file:
        readed_file = file.read()
    return readed_file


def _read_tsp(url):
    return pd.read_csv(url, skiprows=7, skipfooter=1, names=["id", "latitude", "longitude"], sep="\s")
