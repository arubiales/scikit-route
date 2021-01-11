import pandas as pd

def _loader(path, mode="big"):
    real_path = "/".join(path.split("/")[:-3])
    dictionary_dataset = {
        "DESCR": _read_txt(real_path + "/DESCR.txt"),
        "feature_names": _read_txt(real_path + "/columns_lon_lat.txt")
    }
    if mode == "big":
        dictionary_dataset["DataFrame"] = _read_tsp(path)
        return dictionary_dataset
    elif mode == "medium":
        df =_read_tsp(path)
        dictionary_dataset["DataFrame"] = df.sample(int(len(df) *0.2), random_state=2019)
        return dictionary_dataset
    elif mode== "small":
        df =_read_tsp(path)
        dictionary_dataset["DataFrame"] = df.sample(int(len(df) *0.005), random_state=2019)
        return dictionary_dataset
    else:
        raise ValueError(f"The mode '{mode}' is not availabre, the modes availables are: big, medium, small")


def _read_txt(path:str) -> str:
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





def _docstring_decorator(type="latlon"):
    """
    Decorates datasets functions

    Parameters:
    -----------
    type: str default="latlon"
        select the type of data to load, "cost" or "latlon"
    """
    def docs(func, type=type):
        if type=="latlon":
            func.__doc__ = f"""
            Load a number the latitude and longitude of places that belong to {func.__name__.split("_")[-1].capitalize()} country

            Parameters:
            ------------
            mode: str default="big"
                Have three types of parameters:
                * "small": takes the 0.5% of the data from the country
                * "medium": takes the 20% of the data from the country
                * "big": takes all data of the country
            """

        elif type=="cost":
            func.__doc__ = f"""
            Load a number the latitude and longitude of places that belong to {func.__name__.split("_")[-1].capitalize()} country
            """
        else:
            raise ValueError('paremeter type only allows two values, "latlon" or "cost"')
        return func
    return docs
