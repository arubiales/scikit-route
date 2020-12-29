#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cdef class Genetic():
    cdef public:
        float p_c
        float p_m
        int pop
        int gen
        int k
        object early_stopping
        float max_time_work
        float extra_cost
        int people
        list history_



    cpdef tuple fit(self, route_example, time_costs, fuel_costs)
