from skroute.metaheuristics.genetics._base_genetics._utils_genetic import (_crossover, _generate_random_pop, _tournament_selection, 
                                    _mutate, _elitism, _final_result)
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_madrid
import pandas as pd

df_points_distances = load_madrid()["DataFrame"]

fuel_costs = dfcolumn_to_dict(df_points_distances, "id_origin", "id_destinity", "cost")
time_costs = dfcolumn_to_dict(df_points_distances, "id_origin", "id_destinity", "hours")
route_example = list(fuel_costs.keys())
pop = _generate_random_pop(route_example, 200, 18, time_costs, fuel_costs, 8, 12.45, 3)[1]
random_population = _generate_random_pop(route_example, 200, 18, time_costs, fuel_costs,8, 12.45, 3)
example_reversed = route_example[4:] + route_example[:4] 
route_example = list(fuel_costs.keys())

class TestUtils:


    def test_generate_random_pop(self):
        res = _generate_random_pop(route_example, 200, 18, time_costs, fuel_costs, 8, 12.45, 3)
        assert isinstance(res, tuple), "_generate_random_pop must be a tuple"
        assert len(res) == 2, "_generate_random_pop must return two lists one with prices and other with routes"
        assert len(res[0]) == 200 and len(res[1])== 200, "The length of both lists of _generate_random_pop must be the pop parameter"
        assert isinstance(res[0], list) and isinstance(res[1], list), "_generate_random_pop must return a tuple of lists"
        assert all(isinstance(num, float) for num in res[0]) and  all(isinstance(element, list) for element in res[1]), "The first list of _generate_random_pop must be a list of floats and the second one must be a list of lists"
        assert all(isinstance(num, int) for element in res[1] for num in element), "The second list of _generate_random_pop must be a list of integers"
        assert any(595.33 != num for num in res[0]) and any(route_example != element for element in res[1]), "_generate_random_pop should generate different routes with differents costs"
        assert res[1][-0] != res[1][-1], "_generate_random_pop must return a list with unique numbers (not with the come back)"


    def test_tournament_selection(self):
        res = _tournament_selection(pop, 3, 200)
        assert isinstance(res, int), "_tournament_selection must return an integer"
        assert res < 200, "_tournamente_selection can't return a integer higher than pop parameter"


    def test_crossover(self):
        res = _crossover(route_example, example_reversed, 18, 7)
        
        assert isinstance(res, tuple), "_crossover must return a tuple"
        assert res[0] != route_example and res[1] != example_reversed, "_crossover must resturn distinc lists"
        assert len(res) == 2, "_croosover must return two lists of routes"
        assert isinstance(res[0], list) and isinstance(res[1], list), "_croosover must return two lists"
        assert all(isinstance(num, int) for element in res for num in element), "_croosover must return a tuple list of integers"


    def test_mutate(self):
        res = _mutate(route_example, example_reversed, 18, 3, 11)

        assert isinstance(res, tuple), "_mutate must return a tuple"
        assert res[0] != route_example and res[1] != example_reversed
        assert len(res) == 2, "_croosver must return two lists of routes"
        assert isinstance(res[0], list) and isinstance(res[1], list), "_croosover must return a tuple of two lists"
        assert all(isinstance(num, int) for element in res for num in element), "_mutate must return a tuple list of integers"

    
    def test_elitism(self):
        res = _elitism(pop, 200, 18, time_costs, fuel_costs,8, 12.45, 3)
        res[3]
        assert isinstance(res, tuple)
        assert len(res) == 4, "_elitism must return a tuple with four elements min_value, min_route, costs, routes"
        assert isinstance(res[0], float) and isinstance(res[1], list) and isinstance(res[2], list) and isinstance(res[3], list), "_elitism must return a float, and three lists"
        assert all(isinstance(num, float) for num in res[2]) and all(isinstance(element, list) for element in res[3]) and all(isinstance(element, int) for element in res[1]), "_elitism must return a float and three lists, the first one and the third one must be a list of integers and the second one a list of lists"
        assert all(isinstance(num, int) for element in res[3] for num in element), "_elitism third list must be a list of integers"
        assert len(res[2]) == 200 and len(res[3]) == 200, "_elitism must return two list of length equal to pop"


    def test_final_result(self):
        random_population_zipped = list(zip(*random_population))
        res  = _final_result(random_population_zipped, 200, 18, time_costs, 8)

        assert isinstance(res, tuple), "_final_result must return a tuple"
        assert len(res) == 2, "_final_result mus have a length of two"
        assert isinstance(res[0], float), "_final_result must return a float as first element of the tuple"
        assert isinstance(res[1], list), "_final_result must return a list as second element of the tuple"
        assert all(isinstance(num, int) for num in res[1]), "_final_result second element must be a list of integers"
        assert res[0] == min(random_population[0]), "_final_result route must be the lowest of all the list"
        assert len(list(set(res[1]))) < len(res[1]), "_final_result route must habe come backs (id repeated)"





