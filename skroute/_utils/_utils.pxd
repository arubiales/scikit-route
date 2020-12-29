#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cpdef float _cost(list route, int length_route, dict time_costs, dict fuel_costs, float max_time_work ,float extra_cost, int people)
cpdef list _final_cost(list route, int lenth_route, dict time_costs, float max_time_work)
