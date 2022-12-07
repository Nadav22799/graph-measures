"""
Define the C-variables and functions from the C-files that are needed in Python
"""
import os
from ctypes import *
from functools import partial

from graphMeasures.edges_features.acc_features.acc_utils import get_edge_order, get_c_array, empty_c_array, build_chunks
from graphMeasures.edges_features.acc_features.graph_converter import convert_graph_to_db_format

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(CUR_PATH)
ACC_PATH = os.path.join(BASE_PATH, "acc/bin")    # TODO change it?


class AccMotifsEdgeCalculator:
    def __init__(self, level, gnx):
        self._gnx = gnx
        self._print_name = "motifs - "
        self.lib_name = os.path.join(ACC_PATH, "motif.so")

        motifs_lib = None
        try:
            motifs_lib = CDLL(self.lib_name)
        except Exception as e:
            print(e)
            print("Please check that you have the conditions to run accelerate calculation.")
            exit(1)

        self.c_func = motifs_lib.get_edge_motifs
        self._features = {}
        self._level = 3 if level not in [3, 4] else level
        self.directed = self._gnx.is_directed()

        if self._level == 3:
            self.motif_num = 13 if self.directed else 2
        else:
            self.motif_num = 199 if self.directed else 6

    def print_name(self):
        return "motif" + str(self._level)

    def run_edge_motifs(self, neighbors, offsets):
        """This function gets the parameters of the graph and the feature,
        and call the CPP function.

        Args:
            neighbors (list): neighbor list of the graph.
            offsets (lst): offsets list of the graph

        Returns:
            list: with the motifs of the edges.
        """
        nodes = len(offsets) - 1
        edges = len(neighbors)

        # Create the C parameters for the library
        c_nodes = c_uint(nodes)
        c_edges = c_uint(edges)
        c_directed = c_bool(self.directed)
        c_level = c_int(self._level)

        c_nbr = get_c_array(c_uint, neighbors)
        c_offsets = get_c_array(c_int64, offsets)
        c_out = empty_c_array(c_uint, self.motif_num * edges)

        self.c_func(c_nodes, c_edges, c_nbr, c_offsets, c_level, c_directed, c_out)
        return c_out[:]

    def calculate(self):
        offsets, neighbors = convert_graph_to_db_format(self._gnx)
        neighbors = [int(x) for x in neighbors]

        results = self.run_edge_motifs(neighbors, offsets)
        chunks = list(build_chunks(results, self.motif_num))
        edges = get_edge_order(neighbors, offsets)
        results = {}
        for edge, motifs in zip(edges, chunks):
            results[edge] = {i: motifs[i] for i in range(len(motifs))}

        return results


AccMotifs3EdgeCalculator = partial(AccMotifsEdgeCalculator, 3)
AccMotifs4EdgeCalculator = partial(AccMotifsEdgeCalculator, 4)
