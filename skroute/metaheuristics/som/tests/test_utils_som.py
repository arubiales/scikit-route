from skroute.metaheuristics.som._utils_som import *

import scipy.stats as ss
import tensorflow as tf
import pytest

xfail = pytest.mark.xfail(strict=True)

weights = generate_weights(100)
nodes = tf.Variable([1., 2.])
dist = euclidean_distance(weights, nodes)
dist_expand = tf.expand_dims(dist, axis=1) 

print("VERSION DE TENSORFLOW", tf.__version__)

class TestUtils:

    def test_generate_weights(self):
        assert weights is not None, "generate_weights return Falsy"
        assert weights.shape == [100, 2], "generate_weights is not returning the correct shape"
        assert tf.is_tensor(weights), "generate_weights must return a tensor"
        assert weights.dtype == tf.float32, "generate_weights is not returning the correct data type"
        assert all(all(i) for i in weights < 1.00001), "generate_weights"
        assert 0.005 < ss.kstest(tf.reshape(weights, 200), ss.uniform(loc=0.0, scale=1.0).cdf)[1], \
            "generate_weights is not creating a random uniform distribution"
        
    def test_random_gen(self):
        rnd = [random_gen(19) for i in range(10000)]
        assert rnd, "random_gen return Falsy"
        assert len(rnd) == 10000, "random_gen is generating more than one random number"
        assert all(n < 19 and -1 < n for n in rnd), "random_gen is generating random numbers out of the lenght parameter"
        assert all(isinstance(n, int) for n in rnd), "random_gen is generating random numbers that are not integers"
    
    def test_euclidean_distance(self):
        assert dist.shape == 100, "euclidean_distance can't modify the shape of the weights"
        assert tf.is_tensor(dist), "euclidean_distance must return a tensor"
        assert dist.dtype == tf.float32, "euclidean_distance must return float32"
        assert all(all(dist_expand[w] > weights[w]) for w in range(100)), "euclidean_distance, is not computing the distance correctly"
        assert all(tf.math.equal(tf.linalg.norm(weights - nodes, axis=1), dist)), "euclidean_distance is not computing the distance correctly"

    def test_get_closes(self):
        closest = get_closest(weights, nodes)
        assert closest is not None, "get_closest is renturning Falsy"
        assert closest.dtype == tf.int32, "get_closest must return and int32, it's the index"
        assert tf.is_tensor(closest), "get_closest must be a tensor"
        assert closest.shape == [], "get_closest must return empty shape"
        assert closest == tf.cast(tf.math.argmin(dist), tf.int32), "get_closest must return the index of the highest value"

    def test_get_neighbor(self):
        neigbors = get_neighbor(1, 8, 100)
        assert neigbors is not None, "get_neighbor is returning Falsy"
        assert tf.is_tensor(neigbors), "get_neighbor must return a Tensor"
        assert neigbors.dtype == tf.float64, "get_neighbor must return a tf.float64 tensor"
        assert neigbors.shape == 100, "get_neighbor shape is not correct"
        assert all(neigbors < 1.0001), "get_neighbor computation is not correct"


        
        
        