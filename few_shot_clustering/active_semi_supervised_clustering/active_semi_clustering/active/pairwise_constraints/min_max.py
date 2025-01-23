import copy
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

from .example_oracle import MaximumQueriesExceeded
from .explore_consolidate import ExploreConsolidate, TriangleExploreConsolidate


class MinMax(ExploreConsolidate):
    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        skeleton = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                skeleton.add(i)

        remaining = set()
        for i in range(n):
            if i not in skeleton:
                remaining.add(i)

        distances = euclidean_distances(X, X)
        kernel_width = np.percentile(distances, 20)

        pairwise_distances = euclidean_distances(X, X, squared=True)
        kernel_similarities = np.exp(-pairwise_distances / (2 * (kernel_width ** 2)))

        while True:
            try:
                max_similarities = np.full(n, fill_value=float('+inf'))
                
                for x_i in remaining:
                    max_similarities[x_i] = np.max(kernel_similarities[x_i, list(skeleton)])

                q_i = max_similarities.argmin()

                sorted_neighborhoods = reversed(sorted(neighborhoods, key=lambda neighborhood: np.max(kernel_similarities[q_i, list(neighborhood)])))

                for neighborhood in sorted_neighborhoods:
                    self.counts += 1
                    # print(f"Consolidate Count: {self.counts}")

                    if oracle.query(q_i, neighborhood[0]):
                        neighborhood.append(q_i)
                        break

                skeleton.add(q_i)
                if len(remaining) == 0:
                    return neighborhoods
                else:
                    remaining.remove(q_i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


class SimilarityFinder(ExploreConsolidate):

    def fit(self, X, oracle=None):

        edges = self._explore(X, self.n_clusters, oracle)
        self.pairwise_constraints_ = self._consolidate(edges, X, oracle)

        return self

    def _consolidate(self, edges, X, oracle):
        ml = []
        cl = []
        print("len(edges)")
        print(len(edges))

        for edge in edges:
            i, j = edge
            oracle_response = oracle.query(i, j)
            print('oracle_response:')
            print(oracle_response)
            if oracle_response == True:
                # print("enter ml")
                ml.append((i, j))
            elif oracle_response == False:
                cl.append((i, j))
                # print("enter cl")

            self.counts += 1
            print(f"Consolidate Count: {self.counts}")

        return ml, cl


def similarity(x, y, kernel_width):
    return np.exp(-((x - y) ** 2).sum() / (2 * (kernel_width ** 2)))

class TriangleFinder(TriangleExploreConsolidate):
    # def query(self, X, oracle=None):


    def fit(self, X, oracle=None):

        triangles = self._explore(X, self.n_clusters, oracle)
        self.pairwise_constraints_ = self._consolidate(triangles, X, oracle)

        return self

    def _consolidate(self, triangles, X, oracle):
        ml = []
        cl = []
        print("len(triangles)")
        print(len(triangles))

        for triangle in triangles:
            i, j, k = triangle
            oracle_response = oracle.query(i, j, k)
            if oracle_response == "a":
                ml.append((i, j))
                ml.append((i, k))
                ml.append((j, k))
            elif oracle_response == "b":
                ml.append((i, j))
                cl.append((i, k))
                cl.append((j, k))
            elif oracle_response == "c":
                cl.append((i, j))
                ml.append((i, k))
                cl.append((j, k))
            elif oracle_response == "d":
                cl.append((i, j))
                cl.append((i, k))
                ml.append((j, k))
            elif oracle_response == "e":
                cl.append((i, j))
                cl.append((i, k))
                cl.append((j, k))

            self.counts += 1
            print(f"Consolidate Count: {self.counts}")

        return ml, cl
