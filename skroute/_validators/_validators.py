
def _zero_one_validator(x, function):
    message = function + " must be a float between zero and one"
    if x < 0.0 or x > 1.0:
        raise ValueError(message)
    return x


def _intenger_validator(x, function):
    message = function + " must be positive intenger"
    if isinstance(x, int) and x > 0:
        return x
    raise TypeError(message)


def _early_stopping_validator(x):
    if x:
        if x <=0 or not isinstance(x, int):
            raise ValueError("Early stopping must be a positive number bigger than 0 or None for disable it")
        return x
    return x


def _float_validator(x, function):
    message = function + " must be a positive float"
    if x < 0 or type(x) != float:
        raise ValueError(message)
    return x


def _validate_dict_of_dicts(x, function):
    message = function + " must be a dict of dict representing a simmetric matrix"

    if not isinstance(x, dict) and not isinstance(x[list(x.keys())], dict):
        raise TypeError(message)

    for k in x.keys():
        if x[k].keys() != x.keys():
            raise ValueError(message)

def _validate_route_example(x, fuel_costs, time_costs):
    for i in x:
        if not isinstance(i, int):
            raise TypeError("The route provided must be a list of integers")

    x_sorted = sorted(x)
    dict_sorted = sorted(list(fuel_costs.keys()))

    if x_sorted != dict_sorted:
        raise ValueError("The ints in the random route provided must be in the dicts as keys")

    if len(x) == len(list(fuel_costs.keys())) == len(list(time_costs.keys())):
        pass
    else:
        raise ValueError("The dicts must have the same dimension and the route example must have the same lenght as the dicts keys")
