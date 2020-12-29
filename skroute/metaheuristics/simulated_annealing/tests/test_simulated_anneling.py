from skroute.metaheuristics.simulated_annealing._base_simulated_annealing._base_simulated_annealing import SimulatedAnnealing
import numpy as np
import pytest
xfail = pytest.mark.xfail(strict=True)
from skroute._utils._utils import _cost
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
import pandas as pd

df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")


class TestSimulatedAnnealing():

    def test_instanciate_SimulatedAnnealing(self):
        assert SimulatedAnnealing(), "SimulatedAnnealing instantiate error"
        assert SimulatedAnnealing(temp=50, neighbours=34, tol=1.28, people=43), "SimulatedAnnealing instantiate error"
        assert SimulatedAnnealing(max_time_work=12, extra_cost=1, delta=0.4), "SimulatedAnnealing instantiate error"
        sa = SimulatedAnnealing()
        assert str(type(sa)) == "<class 'skroute.metaheuristics.simulated_annealing._base_simulated_annealing._base_simulated_annealing.SimulatedAnnealing'>", "SimulatedAnnealing, class name fail"

    def test_fit(self):
        route = list(time_matrix.keys())
        sa = SimulatedAnnealing()
        result = sa.fit(route, time_matrix, fuel_matrix)
        assert isinstance(result, tuple), "fit method of SimulatedAnnealing must return a tuple"
        assert isinstance(result[0], float), "fit method of SimulatedAnnealing must return a tuple with float at index 0"
        assert isinstance(result[1], list), "fit method of SimulatedAnnealing must return a tuple with a list at index 1"
        assert isinstance(result[1][0], int), " fit method of SimulatedAnnealing must return a tuple with a list of integers at index 1"

    def test_atr(self):
        route = list(time_matrix.keys())
        sa = SimulatedAnnealing()
        sa.fit(route, time_matrix, fuel_matrix)
        assert sa.history_, "atribute history_ of SimulatedAnnealing is not working"
        assert isinstance(sa.history_, list), "atribute history of SimulatedAnnealing must be a list"
        assert isinstance(sa.history_[0], float), "atribute history of SimulatedAnnealing must be a list of floats"
    
    def test_improve(self):
        route = list(time_matrix.keys())
        sa = SimulatedAnnealing()
        random_cost = _cost(route, 19, time_matrix, fuel_matrix, 8, 10, 1)
        result = sa.fit(route, time_matrix, fuel_matrix)
        assert result[0] < random_cost, "The result of fitted SimulatedAnnealing is not improving a random result"
        assert result[0] < 475., "The result of fitted SimulatedAnnealing is very poor, something is wrong"
        