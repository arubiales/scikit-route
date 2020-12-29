#cython: language_level = 3, boundscheck = False, wraparound = False, nonecheck = False, embedsignature=True
cpdef float _cost(list route, int length_route, dict time_costs, dict fuel_costs, float max_time_work ,float extra_cost, int people)
cpdef list _final_cost(list route, int lenth_route, dict time_costs, float max_time_work)
cpdef tuple _generate_random_pop(list initial_route, int pop, int length_route, dict time_costs, dict fuel_costs, float max_time_work, float extra_cost, int people)
cpdef int _tournament_selection(list price_pop, int k, int pop)
cpdef tuple _crossover(list Parent_1, list Parent_2, int lenth_route, int Cr_1)
cpdef tuple _mutate(list Child_1, list Child_2, lenth_route, int index_1_child_1, int index_2_child_1)
cpdef tuple _elitism(list new_population, int pop, int length_route, dict time_costs, dict fuel_costs, float max_time_work, float extra_cost, int people)
cpdef tuple _final_result(list history, int num_gen, int length_route, dict time_costs, float max_time_work)
