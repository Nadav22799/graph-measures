import os
import pickle
from functools import partial
from itertools import permutations, combinations

import networkx as nx
import numpy as np
from bitstring import BitArray
# from  import NodeFeatureCalculator, FeatureMeta

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))
VERBOSE = False
DEBUG = False


class MotifsEdgeCalculator:
    def __init__(self, level, gnx):
        self._print_name = "motifs - "
        self._is_loaded = False
        self._features = {}
        # self._logger = EmptyLogger() if logger is None else logger
        self._logger = None
        self._gnx = gnx
        self._default_val = 0

        assert level in [3, 4], "Unsupported motif level %d" % (level,)
        self._level = level
        self._node_variations = {}
        self._all_motifs = None
        self._print_name += "_%d" % (self._level,)
        self._gnx = self._gnx.copy()
        self._load_variations()

    def is_relevant(self):
        return True

    def print_name(self):
        return "motif" + str(self._level)
        # print_name = super(MotifsNodeCalculator, cls).print_name()
        # if level is None:
        #     return print_name
        # return "%s_%d" % (print_name, level)

    def _load_variations_file(self):
        fname = "%d_%sdirected.pkl" % (self._level, "" if self._gnx.is_directed() else "un")
        fpath = os.path.join(BASE_PATH, "motif_variations", fname)
        return pickle.load(open(fpath, "rb"))

    def _load_variations(self):
        self._node_variations = self._load_variations_file()
        self._all_motifs = set(self._node_variations.values())

    # passing on all:
    #  * undirected graph: combinations [(n*(n-1)/2) combs - handshake lemma]
    #  * directed graph: permutations [(n*(n-1) perms - handshake lemma with respect to order]
    # checking whether the edge exist in the graph - and construct a bitmask of the existing edges
    def _get_group_number(self, nbunch):
        func = permutations if self._gnx.is_directed() else combinations
        # Reversing is a technical issue. We saved our node variations files
        bit_form = BitArray(self._gnx.has_edge(n1, n2) for n1, n2 in func(nbunch, 2))
        bit_form.reverse()
        return bit_form.uint

    # implementing the "Kavosh" algorithm for subgroups of length 3
    def _get_motif3_sub_tree(self, root):
        visited_vertices = {root: 0}
        visited_index = 1

        # variation - two neighbors of the root
        first_neighbors = set(nx.all_neighbors(self._gnx, root))
        for n1 in first_neighbors:
            visited_vertices[n1] = visited_index
            visited_index += 1

        for n1, n2 in combinations(first_neighbors, 2):
            if (visited_vertices[n1] < visited_vertices[n2]) and \
                    not (self._gnx.has_edge(n1, n2) or self._gnx.has_edge(n2, n1)):
                yield [root, n1, n2]

        # variation - one vertex of depth 1, one of depth 2

        for n1 in first_neighbors:
            last_neighbors = set(nx.all_neighbors(self._gnx, n1))
            for n2 in last_neighbors:
                if n2 in visited_vertices:
                    if visited_vertices[n1] < visited_vertices[n2]:
                        yield [root, n1, n2]
                else:
                    visited_vertices[n2] = visited_index
                    visited_index += 1
                    yield [root, n1, n2]

    # implementing the "Kavosh" algorithm for subgroups of length 4
    def _get_motif4_sub_tree(self, root):
        visited_vertices = {root: 0}

        # variation - three neighbors of the root
        neighbors_first_deg = set(nx.all_neighbors(self._gnx, root))
        neighbors_first_deg = list(neighbors_first_deg)

        for n1 in neighbors_first_deg:
            visited_vertices[n1] = 1
        for n1, n2, n3 in combinations(neighbors_first_deg, 3):
            group = [root, n1, n2, n3]
            yield group

        # variations - depths 1, 1, 2 and 1, 2, 2
        for n1 in neighbors_first_deg:
            # all neighbors adjacent to vertices of depth 1, that are not of depth 1 themselves, are of depth 2.
            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            neighbors_sec_deg = list(neighbors_sec_deg)
            for n in neighbors_sec_deg:
                if n not in visited_vertices:
                    visited_vertices[n] = 2

            # variation - depths 1, 1, 2
            for n2 in neighbors_sec_deg:
                for n11 in neighbors_first_deg:
                    if visited_vertices[n2] == 2 and n1 != n11:
                        edge_exists = (self._gnx.has_edge(n2, n11) or self._gnx.has_edge(n11, n2))
                        # avoid double-counting due to two paths from root to n2 - from n1 and from n11.
                        if (not edge_exists) or (edge_exists and n1 < n11):
                            group = [root, n1, n11, n2]
                            yield group

            # variation - depths 1, 2, 2
            for comb in combinations(neighbors_sec_deg, 2):
                if visited_vertices[comb[0]] == 2 and visited_vertices[comb[1]] == 2:
                    group = [root, n1, comb[0], comb[1]]
                    yield group

        # variation - one vertex of each depth (root, 1, 2, 3)
        for n1 in neighbors_first_deg:
            neighbors_sec_deg = set(nx.all_neighbors(self._gnx, n1))
            neighbors_sec_deg = list(neighbors_sec_deg)
            for n2 in neighbors_sec_deg:
                if visited_vertices[n2] == 1:
                    continue

                for n3 in set(nx.all_neighbors(self._gnx, n2)):
                    if n3 not in visited_vertices:
                        visited_vertices[n3] = 3
                        if visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group
                    else:
                        if visited_vertices[n3] == 1:
                            continue

                        if visited_vertices[n3] == 2 and not (self._gnx.has_edge(n1, n3) or self._gnx.has_edge(n3, n1)):
                            group = [root, n1, n2, n3]
                            yield group

                        elif visited_vertices[n3] == 3 and visited_vertices[n2] == 2:
                            group = [root, n1, n2, n3]
                            yield group

    def _order_by_degree(self, gnx=None):
        if gnx is None:
            gnx = self._gnx
        return sorted(gnx, key=lambda n: len(list(nx.all_neighbors(gnx, n))), reverse=True)

    def _calculate_motif(self):
        # consider first calculating the nth neighborhood of a node
        # and then iterate only over the corresponding graph
        motif_func = self._get_motif3_sub_tree if self._level == 3 else self._get_motif4_sub_tree
        sorted_nodes = self._order_by_degree()
        for node in sorted_nodes:
            for group in motif_func(node):
                group_num = self._get_group_number(group)
                motif_num = self._node_variations[group_num]
                yield group, group_num, motif_num
            if VERBOSE:
                self._logger.debug("Finished node: %s" % node)
            self._gnx.remove_node(node)

    def _update_edges(self, group, motif_num):
        # check - checks that the edge exist
        # if self._gnx.is_directed:
        #     check = lambda a1, a2: a1 != a2 and self._gnx.has_edge(a1, a2)
        # else:
        #     check = lambda a1, a2: (a1, a2) in self._features

        for v1 in group:
            for v2 in group:
                # print(v1, v2, check(v1, v2))
                # if check(v1, v2):
                if (v1, v2) in self._features:
                    self._features[(v1, v2)][motif_num] += 1

    def calculate(self, include=None):
        m_gnx = self._gnx.copy()
        motif_counter = {motif_number: 0 for motif_number in self._all_motifs}

        self._features = {edge: motif_counter.copy() for edge in self._gnx.edges()}

        for i, (group, group_num, motif_num) in enumerate(self._calculate_motif()):
            self._update_edges(group, motif_num)

            if (i + 1) % 1000 == 0 and VERBOSE:
                self._logger.debug("Groups: %d" % i)

        # print('Max num of duplicates:', max(self._double_counter.values()))
        # print('Number of motifs counted twice:', len(self._double_counter))

        self._gnx = m_gnx
        return self._features

    def _get_feature(self, element):
        all_motifs = self._all_motifs.difference({None})
        cur_feature = self._features[element]
        return np.array([cur_feature[motif_num] for motif_num in sorted(all_motifs)])


Motifs3EdgeCalculator = partial(MotifsEdgeCalculator, 3)
Motifs4EdgeCalculator = partial(MotifsEdgeCalculator, 4)
