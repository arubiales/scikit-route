from skroute.metaheuristics.simulated_annealing._base_ensemble_simulated_annealing import EnsembleSimulatedAnnealing
import numpy as np
import pytest
xfail = pytest.mark.xfail(strict=True)
from skroute._utils._utils import _cost
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona
import pandas as pd
import pytest

df_barcelona = load_barcelona()["DataFrame"]
fuel_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "cost")
time_matrix = dfcolumn_to_dict(df_barcelona, "id_origin", "id_destinity", "hours")

route = list(time_matrix.keys())

class TestEnsembleSimulatedAnnealing:

    def test_instanciate(self):
        esa = EnsembleSimulatedAnnealing()
        assert str(type(esa)) == "<class 'skroute.metaheuristics.simulated_annealing._base_ensemble_simulated_annealing.EnsembleSimulatedAnnealing'>"

    def test_parameters(self):
        esa1 = EnsembleSimulatedAnnealing(n_jobs=4, n_simulateds=4)
        esa2 = EnsembleSimulatedAnnealing(temp=345, delta=0.99, n_simulateds=5, tol=10)
        esa3 = EnsembleSimulatedAnnealing(4, 14., 250, 0.76, 1.04, 8., 10., 2, 2)
        assert esa1
        assert esa2
        assert esa3

    def test_improve(self):
        esa = EnsembleSimulatedAnnealing(n_simulateds=4, temp=12.5, n_jobs=2)
        result = esa.fit(route, time_matrix, fuel_matrix)
        first_result = _cost(route, 19, time_matrix, fuel_matrix, 8, 10, 1)
        assert result[0] < first_result, "The algorithm is not improving a random result."
        assert result[0] < 470., "The algorithm is not improving correctly."
