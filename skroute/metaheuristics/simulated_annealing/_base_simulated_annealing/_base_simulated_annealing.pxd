#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cdef class SimulatedAnnealing():
    cdef public:
        int neighbours
        float delta
        float temp
        float tol
        float max_time_work
        float extra_cost
        int people
        list history_

    cpdef tuple fit(self, list route, dict time_costs, dict fuel_costs)
