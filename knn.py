import numpy as np


class KNN:
    """
    K Nearest Neighbor.
    Parameters
    ----------
    k : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split and "random" to choose
    """
    def calculate_distance(self, test_row, doc_row):
        """
        Takes a test vector at a time and uses Euclidean distance to calculate its distance from sample vectors.
        :param list test_row: A vector of test document consisting binary representation of features
        :param list doc_row: A vector of sample document consisting
        :return float:
        """
        distance = 0.0
        for i in range(len(test_row)):
            distance += (test_row[i] - doc_row[i]) ** 2
        return np.sqrt(distance)

    def get_neighbors(self, train, test_row, num_neighbors: int = 5):
        """
        Calculates and sorts the nearest neighbors to test vector.
        :param list train: A list of lists consisting sample vectors
        :param list test_row: A vector consisting feature values
        :param int num_neighbors: The K parameter
        :return list: Nearest neighbors of test vector
        """
        distances = []
        for row in train:
            dist = self.calculate_distance(test_row, row)
            distances.append((row, dist))
        distances.sort(key=lambda tup: tup[1])
        neighbors = []
        for i in range(num_neighbors):
            neighbors.append(distances[i][0])
        return neighbors

    def get_vote(self, neighbors_data):
        """
        Selects K nearest vectors to test vector.
        :param list neighbors_data: Includes nearest neighbors vectors and their distances
        :return int: Either 0 for positive class or 1 for negative class
        """
        neighbors = [row[-1] for row in neighbors_data]
        predicted_class = max(set(neighbors), key=neighbors.count)
        return predicted_class

    def knn_predict(self, vectors, test_vector, k: int = 5):
        """
        Gets the K nearest neighbors and with that, predicts the class of the given test vector.
        :param list vectors: A list of lists consisting sample vectors
        :param list test_vector: A vector consisting feature values
        :param int k: Number of neighbors to be voted for class prediction as nearest neighbors
        :return int: Either 0 for positive class or 1 for negative class
        """
        neighbors = self.get_neighbors(vectors, test_vector, k)
        predicted_class = self.get_vote(neighbors)
        return predicted_class
