import pytest
xfail = pytest.mark.xfail(strict=True)
from skroute.heuristics.NRBS._utils_nrbs import _std, _priority_connection, _priority_function, _compute_mean_std_dicts, _define_true_route, _check_route, Node, NodeValueError
from skroute.preprocessing import dfcolumn_to_dict
from skroute.datasets import load_barcelona


df_bcn = load_barcelona()["DataFrame"]
time_matrix = dfcolumn_to_dict(df_bcn, "id_origin", "id_destinity", "hours")
cost_matrix = dfcolumn_to_dict(df_bcn, "id_origin", "id_destinity", "cost")

ids_route = list(cost_matrix.keys())
start_point_id = ids_route[0]

length_route = len(ids_route)

real_std = [17.34, 14.23, 13.71,
15.55, 16.89, 15.5, 16.17,
13.82, 17.08, 19.88, 17.86,
14.49, 18.04, 17.36, 17.47,
17.78, 15.96, 17.94, 21.09]



class TestUtils:

    def test_std(self):
        std_routes = []
        for route in ids_route:
            values = list(cost_matrix[route].values())
            mean = sum(values) / length_route
            compute_std = _std(values, mean, length_route)
            std_routes.append(round(compute_std, 2))

        assert type(std_routes) == list, "_std must return a list"
        assert all(type(i) == float for i in real_std), "_std must return a list of floats"
        assert std_routes == real_std, "Standar Deviation results are not correc"
    
    def test_compute_mean_std_dicts(self):
        mean_std = _compute_mean_std_dicts(ids_route, cost_matrix, length_route)

        assert type(mean_std) == tuple, "_compute_mean_std_dicts must return a tuple"
        assert type(mean_std[0]) == dict and type(mean_std[1]) == dict , "_compute_mean_std_dicts must return a tupzle of dicts"
        assert all(type(x) == float for x in mean_std[0].values()), "_compute_mean_std_dicts mean's must be float"
        assert all(type(x) == float for x in mean_std[1].values()), "_compute_mean_std_dicts standar's deviation must be float"
        assert list(mean_std[0].keys()) == ids_route, "_compute_mean_std_dicts dictionarys must return same input routes"
        assert list(mean_std[1].keys()) == ids_route, "_compute_mean_std_dicts dictionarys must return same input routes"
        assert round(mean_std[0][1], 2) == 26.11, "_compute_mean_std_dicts don't compute the mean correctly"
        assert round(mean_std[1][1], 2) == 14.23, "_compute_mean_std_dicts dont't compute the standar deviation correctly"

    def test_priority_function(self):
        mean_std = _compute_mean_std_dicts(ids_route, cost_matrix, length_route)
        route_priority = _priority_function(ids_route, mean_std[0], mean_std[1], 1, 1)

        assert type(route_priority) == dict, "_priority_function dictionary must return a dictionary"
        assert all(i in ids_route for i in route_priority.keys()), "_priority_function dictionary must return same input routes"
        assert all(type(i) == int for i in route_priority.keys()), "_priority_function dictionary keys must be integers"
        assert list(route_priority.keys())[:5] == [91, 27, 26, 65, 4], "_priority_function compute formula is not correct"
        assert all(type(i) == float for i in route_priority.values()), "_priority_function values must be floats"
        assert list(route_priority.values()) == sorted(list(route_priority.values()), reverse=True), "_priority_function must return an order keys and values"

    def test_priority_connection(self):
        mean_std = _compute_mean_std_dicts(ids_route, cost_matrix, length_route)
        route_priority = _priority_function(ids_route, mean_std[0], mean_std[1], 1, 1)
        connections = _priority_connection(route_priority, cost_matrix, mean_std[0], mean_std[1], 0.5, 0.5, 0.5)

        assert type(connections) == dict, "_priority_connection must be a dictionary"
        assert all(type(connections[x]) == dict for x in ids_route), "_priority_connection must be a dictionary of dictionarys"
        assert all(type(x) == int for x in connections.keys()), "_priority_connection keys must be integers"
        assert all(x in ids_route for x in connections.keys()), "_priority_connection keys must be the points routes"
        assert all(type(x_2) == int for x_1 in connections.keys() for x_2 in connections[x_1]), "_priority_connection second dictionary values must be integers id routes"
        assert all(x not in connections[x].values() for x in connections.keys()), "The key of the dictionarys mustn't be in their values"
        assert all(x_2 in ids_route for x_1 in connections.keys() for x_2 in connections[x_1].keys()), "_priority_connection second dictionary keys must be the same routes id"
        assert all(type(x_2) == float for x_1 in connections.keys() for x_2 in connections[x_1].values()), "_priority_connection second dictionary values must be float"
        assert [round(i, 2) for i in list(connections[91].values())[:4]] == [2.72,2.62,2.53,2.45] , "_priority_connection compute values are incorrect, values don't respect the formula"
        assert  [connections[x_1].values() == sorted(connections[x_1].values(), reverse=True) in ids_route for x_1 in connections.keys()], "_priority_connection must return an order keys and values"

    def test_nodevalueerror(self):
        assert issubclass(NodeValueError, Exception)
        assert str(NodeValueError) == "<class 'skroute.heuristics.NRBS._utils_nrbs.NodeValueError'>"

    def test_node(self):
        node = Node(1)

        assert str(Node) == "<class 'skroute.heuristics.NRBS._utils_nrbs.Node'>", "Node __repr__ is not correct"
        assert str(node) == "(1, [])", "Node __repr__ is not correct"
        assert len(node) == 0, "Node __len__ is not correct"
        assert len(node.neighbours) == 0, "Node.neighbours method is not correct"
        
        node.neighbours = 91
        assert str(node) == "(1, [91])", "Node __repr__ is not correct"
        assert len(node) == 1, "Node __len__ is not correct"
        assert len(node.neighbours) == 1, "Node.neighbours method is not correct"
        assert node.neighbours == [91], "node.neighbours method not return the neighbours correctly"

        node.neighbours = 59
        assert str(node) == "(1, [91, 59])", "Node __repr__ is not correct"
        assert len(node) == 2, "Node __len__ is not correct"
        assert len(node.neighbours) == 2, "Node.neighbours method is not adding neighbours correctly"
        assert node.neighbours == [91, 59], "node.neighbours method not return the neighbours correctly"

        try:
            node.neighbours = 4
            assert False, "NodeValueError is not working, a Node can have a maximun of two neighbours"
        except NodeValueError:
            assert True

    def test_check_route(self):
        true_route = _define_true_route(ids_route)
        assert type(_check_route(true_route, length_route)) == list, "_check_route must return a route list if it's finished"
        assert _check_route(true_route, length_route), "_check_route must return a route list if it's finished"
        assert _check_route(true_route, length_route) == [10000007, 46, 32, 44, 65, 27, 26, 7, 12, 5, 31, 1, 25, 91, 4, 59, 23, 30, 47], "_check_route must return a route if it's finished"
        assert all(type(i) == int for i in _check_route(true_route, length_route)), "_check_route must return a route of ints if it's finished"

        incompleted_true_dict = {i:Node(i) for i in ids_route}
        assert _check_route(incompleted_true_dict, length_route), "_check_route must return True if the route is uncompleted"
        incompleted_true_dict[91].neighbours = 5
        incompleted_true_dict[5].neighbours = 91
        assert _check_route(incompleted_true_dict, length_route), "_check_route must return True if the route is uncompleted"
        incompleted_true_dict[5].neighbours = 27
        incompleted_true_dict[27].neighbours = 5
        incompleted_true_dict[27].neighbours = 91
        incompleted_true_dict[91].neighbours = 27
        assert not _check_route(incompleted_true_dict, length_route), "_check_route must return False if the route is circular before finishing"

