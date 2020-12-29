#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cpdef float _cost(list route, int length_route, dict time_costs, dict fuel_costs, float max_time_work, float extra_cost, int people):

    """
Function that measure the value of a route. In the algorithm compute the value of
all solutions per generation

Parameters:
-----------

route: list of ints
    Contains the ID of the route that we want to measure de cost from one point
    (ID) to the following (ID)

length_route: int
    The length of the route wich means all the IDs for a route

time_costs: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix. For example if we
    have 3 ID 1, 2 and 3 the dict will be like this:

        {
        1:{
            1:0,
            2:x
            3:y
            },
        2:{
            1:x,
            2:0,
            3:z
            },
        3:{
            1:y,
            2:z,
            3:0
            }
        }

    This is just a (3, 3) symmetric matrix with the cost in
    time from one point to another with column and index.

        1   2   3
    1   0   x   y
    2   x   0   z
    3   y   z   0

fuel_costs: dict
    For a extended explanation go up to time_costs. it's
    the fuel cost to go from one point to another.

Returns:
--------
The cost of the given route

Note:
    it's possible to improve the time speed creating a cache

    """
    cdef: 
        float distance_fuel = 0.0
        float distance_time = 0.0
        int extra_count = 0, i=0, real_length_route = length_route-1, p0= route[0]
    
    for i in range(real_length_route):
        origin = route[i]
        destiny = route[i+1]
        if distance_time < max_time_work:
            distance_time += time_costs[origin][destiny]
            distance_fuel += fuel_costs[origin][destiny]

        else:
            distance_time = time_costs[origin][p0] + time_costs[p0][destiny]
            distance_fuel += fuel_costs[origin][p0] + fuel_costs[p0][destiny]
            extra_count +=1
            
    distance_fuel +=  fuel_costs[p0][route[real_length_route]]
    return distance_fuel * people + extra_count * extra_cost * people



cpdef list _final_cost(list route, int length_route, dict time_costs, float max_time_work):

    """
Function that measure the value of the final route of the algorithm including the come back home
for the employes.

Parameters:
-----------

route: list of ints
    Contains the ID of the route that we want to measure de cost from one point
    (ID) to the following (ID)

length_route: int
    The length of the route wich means all the IDs for a route

time_costs: dict of dicts
    it's a dict of dicts with the points to visit as a keys and
    the value is another dictionary with the points to visit and
    the value of going for the first point to the second. It's a
    dictionary representation of a matrix. For example if we
    have 3 ID 1, 2 and 3 the dict will be like this:

        {
        1:{
            1:0,
            2:x
            3:y
            },
        2:{
            1:x,
            2:0,
            3:z
            },
        3:{
            1:y,
            2:z,
            3:0
            }
        }

    This is just a (3, 3) symmetric matrix with the cost in
    time from one point to another with column and index.

        1   2   3
    1   0   x   y
    2   x   0   z
    3   y   z   0

Returns:
--------
The route and the cost of the a route


    """
    cdef: 
        list new_route = []
        int i=0
        float distance_time = 0
        int real_length = length_route-1, p0= route[0]
    
    for i in range(real_length):
        origin = route[i]
        destiny = route[i+1]
        if distance_time < max_time_work:
            new_route.append(origin)
            distance_time += time_costs[origin][destiny]
        else:
            new_route.append(origin)
            new_route.append(p0)
            distance_time = time_costs[origin][p0] + time_costs[p0][destiny]

    new_route.append(route[real_length])
    new_route.append(p0)

    return new_route
