import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import time
from tqdm import tqdm

from itertools import combinations
import random
import heapq

import os
import json

from .helpers import get_constraints_from_neighborhoods
from .example_oracle import MaximumQueriesExceeded


class ExploreConsolidate:
    def __init__(self, n_clusters, mode, data, selection, encoder_name, **kwargs):
        self.n_clusters = n_clusters
        self.counts = 0
        self.mode = mode
        self.data = data
        self.selection = selection
        self.encoder_name = encoder_name

    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, self.n_clusters, oracle)
        neighborhoods = self._consolidate(neighborhoods, X, oracle)

        self.pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)

        return self

    def _explore(self, X, k, oracle, max_explore_ratio=0.5):

        data = self.data
        mode = self.mode
        selection = self.selection
        encoder_name = self.encoder_name

        cache_file_name = f'cache/{encoder_name}/{selection}/selected/{data}-{mode}.json'
        edges = self.select_edges(X, mode=self.mode, k=oracle.max_queries_cnt, cache_file=cache_file_name)
        print('oracle.max_queries_cnt:')
        print(oracle.max_queries_cnt)
        print('len(edges):')
        print(len(edges))

        return edges

    def compute_spanning_edge_centrality(self, X):
        """
        Compute the normalized vector z according to the provided formula.

        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).

        Returns:
        - z: The normalized vector z.
        """
        X_sum = X.sum(axis=0)  # Summation of all vectors in T
        z = 1 / (np.dot(X, X_sum))  
        return z

    def compute_weighted_degree(self, X):
        """
        Compute the weighted degree for each node based on the provided formula.
        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).
        Returns:
        - z: A vector representing the weighted degree of each node.
        """
        # Step 1: Compute the dot product matrix for all vectors
        dot_matrix = np.dot(X, X.T)  # This gives us the pairwise dot products
        
        # Step 2: Replace negative dot products with 0
        dot_matrix[dot_matrix < 0] = 1e-10  # Set all negative dot products to 0
        
        # Step 3: Compute the weighted degree by summing each row
        z = np.sum(dot_matrix, axis=1)  # Sum over each row to get the weighted degree for each node
        
        return z

    def compute_edge_score(self, z, edge):
        """
        Compute the score of an edge based on the fast SEC estimation formula.

        Parameters:
        - z: The vector of weighted degrees for each node.
        - edge: A tuple (i, j) representing the edge between node i and node j.

        Returns:
        - score: The SEC score of the edge (i, j).
        """
        i, j = edge
        score = (1 / z[i]) + (1 / z[j])  # Compute the SEC score
        return score


    def select_edges(self, X, k, mode="max-sum", cache_file=None):
        """
        Select edges based on the Greedy Edge Selection algorithm.

        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).
        - k: The number of edges to return.
        - mode: The selection mode ("max-sum" or "min-sum").
        - cache_file: Path to the cache file. If provided and not None, results are loaded from or saved to this file.

        Returns:
        - A stack S (list of tuples representing edges) containing the selected edges.
        """

        if cache_file and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            print(f"Loading edges from cache file: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_edges = json.load(f)
            return [tuple(edge) for edge in cached_edges]  # Convert back to tuples

        z = self.compute_weighted_degree(X)
        n = len(z)

        if mode == "max-sum":  
            sorted_indices = sorted(range(n), key=lambda i: z[i])

        # Create mapping from original indices to sorted indices and reverse mapping
        original_to_sorted = {original: sorted_pos for sorted_pos, original in enumerate(sorted_indices)}
        sorted_to_original = {sorted_pos: original for sorted_pos, original in enumerate(sorted_indices)}

        # Map z to the reordered space
        z_mapped = [z[idx] for idx in sorted_indices]

        limit = int((8 * k + 1)**0.5 - 1) // 2
        priorities = {t: idx + 1 for idx, t in enumerate(range(limit))}  # Only for nodes within range

        t1, t2 = 0, 1  # In mapped space
        stack = [(t1, t2)]
        priorities[t1] = 2  # Update t1 priority

        score = 0

        while len(stack) < k:
            # Get the last added edge from the stack
            t_a, t_b = stack[-1]

            b_minus = priorities.get(t1, float('inf'))  # p(t_a)
            b_plus = priorities.get(t_a + 1, float('inf')) if t_a + 1 < len(z_mapped) else float('inf')

            # Construct candidate edges
            candidates = []
            if b_minus != float('inf'):
                candidates.append((t1, b_minus))  # (t1, t_b-)
            if t_b + 1 < len(z_mapped):
                candidates.append((t_a, t_b + 1))  # (t_a, t_b+1)
            if b_plus != float('inf') and t_a + 1 < len(z_mapped):
                candidates.append((t_a + 1, b_plus))  # (t_a+1, t_b+)

            # Compute scores for candidate edges
            edge_scores = [(edge, self.compute_edge_score(z_mapped, edge)) for edge in candidates]

            # Select the best edge based on mode
            if not edge_scores:
                break  # No valid candidates
            if mode == "max-sum":
                best_edge, best_score = max(edge_scores, key=lambda x: x[1])  # Maximize score

            score += best_score
            # Push the best edge into the stack
            stack.append(best_edge)

            # Update priority of the newly added node
            t_x, t_y = best_edge
            priorities[t_x] = t_y + 1

        # Map the stack (edges) back to original indices
        mapped_stack = [(sorted_to_original[edge[0]], sorted_to_original[edge[1]]) for edge in stack]

        if cache_file:
            print(f"Saving edges to cache file: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(mapped_stack, f)

        return mapped_stack

    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:

            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(neighborhoods, key=lambda neighborhood: dist(i, neighborhood, X))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        self.counts += 1
                        # print(f"Count: {self.counts}")
                        break

                neighborhoods_union.add(i)
                remaining.remove(i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods


def dist(i, S, points):
    distances = np.array([np.sqrt(((points[i] - points[j]) ** 2).sum()) for j in S])
    return distances.min()


def batch_dist(Is, S, points):
    pairwise_distances = euclidean_distances(points[Is], points[S])
    min_distances = np.min(pairwise_distances, axis=1)
    return min_distances

class TriangleExploreConsolidate:
    def __init__(self, n_clusters, mode, data, selection, encoder_name, **kwargs):
        self.n_clusters = n_clusters
        self.counts = 0
        self.mode = mode
        self.data = data
        self.selection = selection
        self.encoder_name = encoder_name

    def fit(self, X, oracle=None):
        if oracle.max_queries_cnt <= 0:
            return [], []

        neighborhoods = self._explore(X, self.n_clusters, oracle)
        neighborhoods = self._consolidate(neighborhoods, X, oracle)

        self.pairwise_constraints_ = get_constraints_from_neighborhoods(neighborhoods)

        return self



    def _explore(self, X, k, oracle, max_explore_ratio=0.5):

        data = self.data
        mode = self.mode
        selection = self.selection
        encoder_name = self.encoder_name

        cache_file_name = f'cache/{encoder_name}/{selection}/selected/{data}-{mode}.json'
        triangles = self.select_triangles(X, mode=self.mode, k=oracle.max_queries_cnt, cache_file=cache_file_name)
        print('oracle.max_queries_cnt:')
        print(oracle.max_queries_cnt)

        return triangles

    def _consolidate(self, neighborhoods, X, oracle):
        n = X.shape[0]

        neighborhoods_union = set()
        for neighborhood in neighborhoods:
            for i in neighborhood:
                neighborhoods_union.add(i)

        remaining = set()
        for i in range(n):
            if i not in neighborhoods_union:
                remaining.add(i)

        while True:

            try:
                i = np.random.choice(list(remaining))

                sorted_neighborhoods = sorted(neighborhoods, key=lambda neighborhood: dist(i, neighborhood, X))

                for neighborhood in sorted_neighborhoods:
                    if oracle.query(i, neighborhood[0]):
                        neighborhood.append(i)
                        self.counts += 1
                        # print(f"Count: {self.counts}")
                        break

                neighborhoods_union.add(i)
                remaining.remove(i)

            except MaximumQueriesExceeded:
                break

        return neighborhoods

    def compute_spanning_edge_centrality(self, X):
        """
        Compute the normalized vector z according to the provided formula.

        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).

        Returns:
        - z: The normalized vector z.
        """
        X_sum = X.sum(axis=0)  # Summation of all vectors in T
        z = 1 / (np.dot(X, X_sum))  # Pointwise inverse as per the formula
        return z

    def compute_weighted_degree(self, X):
        """
        Compute the weighted degree for each node based on the provided formula.
        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).
        Returns:
        - z: A vector representing the weighted degree of each node.
        """
        # Step 1: Compute the dot product matrix for all vectors
        dot_matrix = np.dot(X, X.T)  # This gives us the pairwise dot products
        
        # Step 2: Replace negative dot products with 0
        dot_matrix[dot_matrix < 0] = 1e-10  # Set all negative dot products to 0
        
        # Step 3: Compute the weighted degree by summing each row
        z = np.sum(dot_matrix, axis=1)  # Sum over each row to get the weighted degree for each node
        
        return z

    def compute_edge_score(self, z, edge):
        """
        Compute the score of an edge based on the fast SEC estimation formula.

        Parameters:
        - z: The vector of weighted degrees for each node.
        - edge: A tuple (i, j) representing the edge between node i and node j.

        Returns:
        - score: The SEC score of the edge (i, j).
        """
        i, j = edge
        score = (1 / z[i]) + (1 / z[j])  # Compute the SEC score
        return score

    def compute_triangle_score(self, z, triangle):
        """
        Compute the score of a triangle based on the fast SEC estimation formula.

        Parameters:
        - z: The vector of weighted degrees for each node.
        - triangle: A tuple (i, j, k) representing the triangle formed by nodes i, j, and k.

        Returns:
        - score: The SEC score of the triangle (i, j, k).
        """
        i, j, k = triangle

        # Compute the SEC score for the triangle
        score = (1 / z[i]) + (1 / z[j]) + (1 / z[k])  # Sum of inverse degrees for the triangle nodes

        return score

    def heap_add(self, heap, elements):
        """
        Add new elements to a min-heap and maintain its continuity.

        Parameters:
        - heap: The min-heap (list).
        - elements: New elements to be added to the heap.

        Returns:
        - The updated min-heap.
        """
        h = heap[0]  # Equivalent to H.min()

        for element in elements:
            heapq.heappush(heap, element)

        if any(e == h + 1 for e in elements):  # Check ∃ e ∈ E s.t. e = h + 1
            while len(heap) > 1:
                beta = heapq.heappop(heap)  # β ← H.pop()
                if heap[0] != beta + 1:  # Check if H.min() ≠ β + 1
                    heapq.heappush(heap, beta)  # Push β back to the heap
                    break

        return heap

    def greedy_scan(self, heap_x, heap_y):
        min_y = heap_y[0]
        if min_y + 1 not in heap_x:
            return min_y + 1
        for gamma in range(min_y + 2, max(heap_y) + 1):
            if gamma not in heap_x and gamma not in heap_y:
                return gamma

    def select_triangles(self, X, k, mode="max-sum", cache_file=None):
        """
        Select triangles based on the Greedy Triangle Selection algorithm.

        Parameters:
        - X: The normalized embeddings matrix (each row is a normalized vector).
        - k: The number of triangles to return.
        - mode: The selection mode ("max-sum" or "min-sum").
        - cache_file: Path to the cache file. If provided, results are loaded from or saved to this file.

        Returns:
        - A list of triangles (tuples of indices, mapped back to original indices).
        """

        if cache_file and os.path.exists(cache_file) and os.path.getsize(cache_file) > 0:
            print(f"Loading triangles from cache file: {cache_file}")
            with open(cache_file, 'r') as f:
                cached_triangles = json.load(f)
            return [tuple(triangle) for triangle in cached_triangles]  # Convert back to tuples

        z = self.compute_weighted_degree(X)

        if mode == "max-sum":  
            sorted_indices = sorted(range(len(z)), key=lambda i: z[i])

        # Create mapping from original indices to sorted indices and reverse mapping
        original_to_sorted = {original: sorted_pos for sorted_pos, original in enumerate(sorted_indices)}
        sorted_to_original = {sorted_pos: original for sorted_pos, original in enumerate(sorted_indices)}

        # Map z to the reordered space
        z_mapped = [z[idx] for idx in sorted_indices]  # Reorder z based on sorted_indices

        heaps = {t: [t] for t in range(len(sorted_indices))}  # Use mapped indices
        stack = [(0, 1, 2)]  # Start with the first three sorted indices
        selected_triangles = [(0, 1, 2)]  # Track selected triangles in mapped indices

        heaps[0] = self.heap_add(heaps[0], [1, 2])
        heaps[1] = self.heap_add(heaps[1], [2])

        score = 0

        while len(selected_triangles) < k:
            # Get the latest added triangle
            t_a, t_b, t_c = stack[-1]

            # Compute b-, b°, b+ indices
            b_minus = heaps[t_a - 1][0] + 1 if t_a > 0 and heaps.get(t_a - 1) else None
            b_mid = heaps[t_a][0] + 1 if heaps.get(t_a) else None
            b_plus = heaps[t_a + 1][0] + 1 if t_a < len(heaps) - 1 and heaps.get(t_a + 1) else None

            # Compute c-, c°, c+ using GreedyScan (on mapped z)
            c_minus = self.greedy_scan(heaps[t_a - 1], heaps[b_minus]) if b_minus is not None else None
            c_mid = self.greedy_scan(heaps[t_a], heaps[b_mid]) if b_mid is not None else None
            c_plus = self.greedy_scan(heaps[t_a + 1], heaps[b_plus]) if b_plus is not None else None

            # Construct candidate triangles
            candidates = []
            if b_minus is not None and c_minus is not None:
                candidates.append((t_a - 1, b_minus, c_minus))
            if b_mid is not None and c_mid is not None:
                candidates.append((t_a, b_mid, c_mid))
            if b_plus is not None and c_plus is not None:
                candidates.append((t_a + 1, b_plus, c_plus))

            # Compute SEC scores for triangles (on mapped indices and z_mapped)
            triangle_scores = [(tri, self.compute_triangle_score(z_mapped, tri)) for tri in candidates]

            # Select the best triangle based on mode
            if not triangle_scores:
                break  # No valid candidates
            if mode == "max-sum":
                best_triangle, best_score = max(triangle_scores, key=lambda x: x[1])  # Maximize score

            score += best_score
            # Push the best triangle to the stack
            stack.append(best_triangle)
            selected_triangles.append(best_triangle)

            # Update heaps with the new triangle
            t_x, t_y, t_z = best_triangle
            heaps[t_x] = self.heap_add(heaps[t_x], [t_y, t_z])
            heaps[t_y] = self.heap_add(heaps[t_y], [t_z])

        mapped_triangles = [(sorted_to_original[t1], sorted_to_original[t2], sorted_to_original[t3])
                            for t1, t2, t3 in selected_triangles]

        if cache_file:
            print(f"Saving triangles to cache file: {cache_file}")
            with open(cache_file, 'w') as f:
                json.dump(mapped_triangles, f)

        return mapped_triangles
