#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cpdef bint _check_temp(float cost0, float cost1, float temp)
cpdef float _normalize(float x)
cpdef list _swap_route(list route, int r1, int r2)