import os
from functools import partial

import numpy as np
from ...features_algorithms.accelerated_graph_features.src import motif
from ...features_infra.feature_calculators import NodeFeatureCalculator, FeatureMeta
import networkx as nx

from .acc_utils import get_edge_order
from .graph_converter import convert_graph_to_db_format

CUR_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(CUR_PATH))


class MotifsNodeCalculator(NodeFeatureCalculator):
    def __init__(self, *args, level=3, gpu=False, device=2, edges=False, **kwargs):
        super(MotifsNodeCalculator, self).__init__(*args, **kwargs)
        assert level in [3, 4], "Unsupported motif level %d" % (level,)
        self._level = level
        self._gpu = gpu
        self._device = device
        self._print_name += "_%d" % (self._level,)
        self.edges = edges

    def is_relevant(self):
        return True

    @classmethod
    def print_name(cls, level=None):
        print_name = super(MotifsNodeCalculator, cls).print_name()
        if level is None:
            return print_name
        return "%s_%d_C_kernel" % (print_name, level)

    def _calculate(self, include=None):
        self._features = motif(self._gnx, level=self._level, gpu=self._gpu, cudaDevice=self._device, edges=self.edges)
        if self.edges:
            directed = nx.is_directed(self._gnx)
            offsets, neighbors = convert_graph_to_db_format(self._gnx)
            neighbors = [int(x) for x in neighbors]
            edges = get_edge_order(neighbors, offsets)
            results = {}
            for i, motifs in enumerate(self._features):
                if directed:
                    results[edges[i]] = motifs
                else:
                    same_edge = (edges[i][1], edges[i][0])
                    if same_edge in results:
                        for j in range(len(motifs)):
                            results[same_edge][j] += motifs[j]
                    else:
                        results[edges[i]] = motifs
            self._features = results

    def _get_feature(self, element):
        return np.array(self._features[element])


def nth_nodes_motif(motif_level, gpu, device, edges=False):
    return partial(MotifsNodeCalculator, level=motif_level, gpu=gpu, device=device, edges=edges)

def nth_edges_motif(motif_level, gpu, device, edges=True):
    return partial(MotifsNodeCalculator, level=motif_level, gpu=gpu, device=device, edges=edges)

feature_node_entry = {
    "motif3_c": FeatureMeta(nth_nodes_motif(3, gpu=False, device=2, edges=False), {"m3_c"}),
    "motif4_c": FeatureMeta(nth_nodes_motif(4, gpu=False, device=2, edges=False), {"m4_c"})
}
