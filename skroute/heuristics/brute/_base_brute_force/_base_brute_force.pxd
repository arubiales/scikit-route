#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cdef class BruteForce():
    cdef public:
        float max_time_work
        float extra_cost
        int people

    cpdef tuple fit(self, list route, dict time_costs, dict fuel_costs)
