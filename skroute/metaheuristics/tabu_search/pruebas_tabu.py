import sys
sys.path.append("/home/rubiales/Desktop/Projects/scikit-route/skroute/metaheuristics/tabu_search")
from _tabu_search import TabuSearch
from skroute.datasets import load_barcelona, load_madrid, load_valencia, load_alicante_murcia, load_finland
from skroute.preprocessing import dfcolumn_to_dict
from _utils_tabu import *

df_barcelona = load_barcelona()["DataFrame"]
matrix_cost = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
matrix_time = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")

route_exampl = list(dict.fromkeys(matrix_cost).keys())


ts = TabuSearch()
ts.fit(route_exampl, matrix_time, matrix_cost)

combs = create_combinations(route_exampl, [route_exampl[0]])

cost_combs = cost_combinations(combs, 19, matrix_time, matrix_cost, 8, 10, 1)

