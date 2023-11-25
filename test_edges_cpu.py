import networkx as nx
from graphMeasures.edges_features.feature_calculator import FeatureCalculator

path = "./graph.txt"
gnx = nx.read_edgelist(path, delimiter=",", create_using=nx.Graph)
# acc signs if we will use the accelerated version.
calculator = FeatureCalculator(["motif3", "motif4"], gnx, acc=True)     
calculator.build()

# The result will be a pandas Dataframe named calculator.df.
print(calculator.df)
