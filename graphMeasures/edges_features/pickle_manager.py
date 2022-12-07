import os
import pickle


class PickleManager:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir

        if not os.path.isdir(self.base_dir):
            os.makedirs(self.base_dir)

    def dump(self, graph_name: str, feature_name: str, feature):
        graph_path = os.path.join(self.base_dir, graph_name)

        if not os.path.isdir(graph_path):
            os.makedirs(graph_path)

        feature_path = os.path.join(graph_path, f"{feature_name}.pkl")
        with open(feature_path, 'wb') as f:
            pickle.dump(feature, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, graph_name: str, feature_name: str):
        feature_path = os.path.join(self.base_dir, graph_name, f"{feature_name}.pkl")

        if not os.path.exists(feature_path):
            return None

        with open(feature_path, 'rb') as f:
            return pickle.load(f)
