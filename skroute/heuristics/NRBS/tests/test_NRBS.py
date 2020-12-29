# import os, sys
# from posix import NGROUPS_MAX

# from numpy.core.numerictypes import issubclass_
# sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-2]))
# sys.path.append("/".join(os.path.abspath(__file__).split("/")[:-4]))
# from NRBS import NRBS
# from datasets import load_barcelona
# from preprocessing import matrix_to_dict
import random as rd
import pytest
xfail = pytest.mark.xfail(strict=True)
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
from skroute.heuristics.NRBS import NRBS


df_bcn = load_barcelona()["DataFrame"]
time_matrix = dfcolumn_to_dict(df_bcn, "id_origin", "id_destinity", "hours")
cost_matrix = dfcolumn_to_dict(df_bcn, "id_origin", "id_destinity", "cost")

ids_route = list(cost_matrix.keys())
start_point_id = ids_route[0]

length_route = len(ids_route)
params = [0.5]* 5


class TestNRBS:

    def test_instanciate(self):
        nrbs = NRBS(*params)
        assert isinstance(nrbs, NRBS)
        assert str(type(nrbs)) == "<class 'skroute.heuristics.NRBS.NRBS.NRBS'>"

    def test_parameters(self):
        nrbs1 = NRBS(rd.random(), rd.random(), rd.random(), rd.random(), rd.random())
        nrbs2 = NRBS(rd.random(), rd.random(), rd.random(), rd.random(), rd.random())
        nrbs3 = NRBS(rd.random(), rd.random(), rd.random(), rd.random(), rd.random())
        assert nrbs1
        assert nrbs2
        assert nrbs3
    
    @xfail(reason="NBRBS can't be initialized with strings")
    def test_false_parameters1(self):
       NRBS("abc", 1, 2, 3,4)
    @xfail(reason="NBRBS can't be initialized with strings")
    def test_false_parameters2(self):
       NRBS(1,"ddfe", 2, 3,4)
    @xfail(reason="NBRBS can't be initialized with strings")
    def test_false_parameters3(self):
       NRBS(1, 1, "edfw", 3,4)
    @xfail(reason="NBRBS can't be initialized with strings")
    def test_false_parameters4(self):
       NRBS(1, 1, 2, "adfs",4)
    @xfail(reason="NBRBS can't be initialized with strings")
    def test_false_parameters5(self):
       NRBS(1, 1, 2, 3,"few")

    def test_NRBS_fit1(self):
        nrbs = NRBS(*params)
        resultado = nrbs.fit(start_point_id, ids_route, cost_matrix)
        assert round(resultado[0], 2) == 446.59, ".fit() output cost of the route is not correct"
        assert resultado[1] == [10000007, 47, 30, 1, 31, 5, 12, 26, 7, 65, 27, 91, 4, 25, 23, 59, 46, 44, 32, 10000007], ".fit() route is not correct"

    @xfail(reason="NBRBS fit() first parameter must be a integer")
    def test_NRBS_fit2(self):
        nrbs = NRBS(*params)
        resultado = nrbs.fit(ids_route, ids_route, cost_matrix)

    @xfail(reason="NBRBS fit() second parameter must be a list")
    def test_NRBS_fit3(self):
        nrbs = NRBS(*params)
        resultado = nrbs.fit(ids_route, start_point_id, cost_matrix)

    @xfail(reason="NBRBS fit() third parameter must be a dict")
    def test_NRBS_fit(self):
        nrbs = NRBS(*params)
        resultado = nrbs.fit(ids_route, start_point_id, start_point_id)


    








