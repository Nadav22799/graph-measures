# Using graphMeasures without FeatureCalculator

> This is another way you may use graphMeasures, but it is not recommended like the first one. 

The calculations require an input graph in NetworkX format, later referred as gnx, and a logger.
For this example, we build a gnx and define a logger:
```python
import networkx as nx
from graphMeasures import PrintLogger

gnx = nx.DiGraph()  # should be a subclass of Graph
gnx.add_edges_from([(0, 1), (0, 2), (1, 3), (3, 2)])

logger = PrintLogger("MyLogger")
```
On the gnx we have, we will want to calculate the topological features.
There are two options to calculate topological features here, depending on the number of features we want to calculate: 
* Calculate a specific feature:

```python
import numpy as np
# Import the feature. 
# If simple, import it from vertices folder, otherwise from accelerated_graph_features: 
from graphMeasures.features_algorithms.vertices import LouvainCalculator  

feature = LouvainCalculator(gnx, logger=logger)  
feature.build()  # The building happens here

mx = feature.to_matrix(mtype=np.matrix)  # After building, one can request to get features the a matrix 
```

* Calculate a set of features (one feature can as well be calculated as written here):

```python
import numpy as np
from graphMeasures.features_infra import GraphFeatures
from graphMeasures.features_infra.feature_calculators import FeatureMeta
from graphMeasures.features_algorithms.vertices import LouvainCalculator
from graphMeasures.features_algorithms.vertices import BetweennessCentralityCalculator

features_meta = {
   "louvain": FeatureMeta(LouvainCalculator, {"lov"}),
   "betweenness_centrality": FeatureMeta(BetweennessCentralityCalculator, {"betweenness"}),
}  # Hold the set of features as written here. 

features = GraphFeatures(gnx, features_meta, logger=logger) 
features.build()

mx = features.to_matrix(mtype=np.matrix)
```

**Note:** All the keys-values options that can be set in the `features_meta` variable can be found
in `graphMeasures.features_meta` or `graphMeasures.accelerated_features_meta`
```python
from graphMeasures import FeaturesMeta
# if one uses the accelerated calculation:
# from graphMeasures.accelerated_features_meta import FeaturesMeta
all_possible_features_meta = FeaturesMeta().NODE_LEVEL

# all possible features
print(all_possible_features_meta.keys())   
# get the value for louvain
louvain = all_possible_features_meta['louvain']   
# get the value for betweenness_centrality
betweenness_centrality = all_possible_features_meta['betweenness_centrality']
```