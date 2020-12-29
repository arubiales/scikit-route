from sklearn.cluster import KMeans

class KMeansTruncate(KMeans):
    def __init__(self, n_clusters, init):
        super().__init__(n_clusters, init)
        self.max_iter = 1
