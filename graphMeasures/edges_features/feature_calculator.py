import numpy as np
import pandas as pd
import networkx as nx

from graphMeasures.edges_features.pickle_manager import PickleManager

from graphMeasures.edges_features.py_features.motifs import Motifs3EdgeCalculator, Motifs4EdgeCalculator
from graphMeasures.edges_features.acc_features.motifs import AccMotifs3EdgeCalculator, AccMotifs4EdgeCalculator

meta = {
    "motif3": Motifs3EdgeCalculator,
    "motif4": Motifs4EdgeCalculator
}

acc_meta = {
    "motif3": AccMotifs3EdgeCalculator,
    "motif4": AccMotifs4EdgeCalculator
}


class FeatureCalculator:
    def __init__(self, features, gnx, acc=False, pkl_dir="pkl"):
        self.df = None
        self.features = features

        # We are mapping the nodes to ints for the calculation, and after we will map it to the original labels.
        self.gnx = nx.convert_node_labels_to_integers(gnx, label_attribute="old_label")
        # This is the matching between the original labels to the new ones.
        self.mapping = nx.get_node_attributes(self.gnx, "old_label")

        self.acc = acc
        self.calculators = set()  # To avoid duplicate calculation.
        self.results = []
        self.pickle_manager = PickleManager(pkl_dir)

        for feature in features:
            # Should be written better...
            if feature in meta.keys():
                if self.acc:
                    self.calculators.add(acc_meta[feature])
                else:
                    self.calculators.add(meta[feature])
            else:
                self.results.append((feature, {edge: np.nan for edge in self.gnx.edges}))

    def build(self):
        # Iterate each feature and run its calculator
        for calculator in self.calculators:
            calculator = calculator(self.gnx)
            res = calculator.calculate()
            name = calculator.print_name()
            self.results.append((name, res))

        self.organize()

    def map_back(self, edge):
        return self.mapping[edge[0]], self.mapping[edge[1]]

    def organize(self):
        # Return the graph to the original labels.
        self.gnx = nx.relabel_nodes(self.gnx, self.mapping, copy=False)

        # Organize the features results in a dataframe
        self.df = pd.DataFrame(index=list(self.gnx.edges))

        for name, res in self.results:
            arbitrary = next(iter(res.values()))
            if isinstance(arbitrary, dict):
                for key in arbitrary.keys():
                    if key is None:
                        continue

                    # This isn't working without the self.df.index.map...
                    self.df[f"{name}_{key}"] = self.df.index.map(
                        {self.map_back(edge): values[key] for edge, values in res.items()})
            else:
                self.df[name] = self.df.index.map(res)
