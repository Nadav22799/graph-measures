# Topological Graph Features

Topological feature calculators infrastructure.

## Calculating Features
This package helps one to calculate features for a given graph. All features are implemented in python codes, 
and some features have also an accelerated version written in C++. Among the accelerated features, one can find 
a code for calculating 3- and 4-motifs using VDMC, a distributed algorithm to calculate 3- and 4-motifs in a 
GPU-parallelized way.

## Versions
- Last version: 0.1.51 (most recommended)

## What Features Can Be Calculated Here?
The set of all vertex features implemented in graph-measures is the following:

| Feature                                | Feature's name in code                 | Is available in gpu? | Output size for directed graph | Output size for undirected graph |
|----------------------------------------|----------------------------------------|----------------------|--------------------------------|----------------------------------|
| Average neighbor degree                | average_neighbor_degree                | NO                   | N x 1                          | N x 1                            |
| Degree^                                | degree                                 | NO                   | N x 2                          | N x 1                            |
| In degree                              | in_degree                              | NO                   | N x 1                          | - - - - - - -                    |
| Out degree                             | out_degree                             | NO                   | N x 1                          | - - - - - - -                    |
| Louvain^^                              | louvain                                | NO                   | - - - - - - -                  | N x 1                            |
| Hierarchy energy                       | hierarchy_energy                       | NO                   |                                |                                  |
| Motifs3                                | motif3                                 | YES                  | N x 13                         | N x 2                            |
| Motifs4                                | motif4                                 | YES                  | N x 199                        | N x 6                            |
| K core                                 | k_core                                 | YES                  | N x 1                          | N x 1                            |
| Attraction basin                       | attractor_basin                        | YES                  | N x 1                          | - - - - - - -                    |
| Page Rank                              | page_rank                              | YES                  | N x 1                          | N x 1                            |
| Fiedler vector                         | fiedler_vector                         | NO                   | - - - - - - -                  | N x 1                            |
| Closeness centrality                   | closeness_centrality                   | NO                   | N x 1                          | N x 1                            |
| Eccentricity                           | eccentricity                           | NO                   | N x 1                          | N x 1                            |
| Load centrality                        | load_centrality                        | NO                   | N x 1                          | N x 1                            |
| BFS moments                            | bfs_moments                            | NO                   | N x 2                          | N x 2                            |
| Flow                                   | flow                                   | YES                  | N x 1                          | - - - - - - -                    |
| Betweenness centrality                 | betweenness_centrality                 | NO                   | N x 1                          | N x 1                            |
| Communicability betweenness centrality | communicability_betweenness_centrality | NO                   | - - - - - - -                  | N x ?                            |
| Eigenvector centrality                 | eigenvector_centrality                 | NO                   | N x 1                          | N x 1                            |
| Clustering coefficient                 | clustering_coefficient                 | NO                   | N x 1                          | N x 1                            |
| Square clustering coefficient          | square_clustering_coefficient          | NO                   | N x 1                          | N x 1                            |
| Generalized degree                     | generalized_degree                     | NO                   | - - - - - - -                  | N x 16                           |
| All pairs shortest path length         | all_pairs_shortest_path_length         | NO                   | N x N                          | N x N                            |

^ Degree - In the undirected case return the sum of the in degree and the out degree. <br>
^^Louvain - Implement Louvain community detection method, then associate to each vertex the number of vertices in its community.

Aside from those, there are some other [edge features](https://github.com/AmitKabya/graph-measures/tree/master/src/graphMeasures/features_algorithms/edges).
Some more information regarding the features can be found in the files of [features_meta](https://github.com/AmitKabya/graph-measures/blob/master/src/graphMeasures/features_meta).

## Dependencies
```
setuptools
networkx==2.6.3
pandas
numpy
matplotlib
scipy
scikit-learn
python-louvain
bitstring
future
torch
```

## How To Use The Accelerated Version (CPU/GPU)?
Both versions currently are not supported with the pip installation. \
To use the accelerated version, one must use <b>*Linux* operation system</b> and <b>*Anaconda* distribution</b>, with the follow the next steps:
1. Go to the [package's GitHub website](https://github.com/AmitKabya/graph-measures) and manually download:

   - The directory `graphMeasures`.
   - The python file `runMakefileACC.py`.

   *You might need to download a zip of the repository and extract the necessary files.*
2. Place both the file and the directory inside your project, and run `runMakefileACC.py`.
3. Move to the *boost environment*: `conda activate boost` (The environment was created in step 2).
4. Use the package as explained in the section `How To Use?`

## Installation Through pip
The full functionality of the package is currently available on a Linux machine, with a Conda environment.
- Linux + Conda<br>1. Go to base environment<br>2. If pip is not installed on your env, install it. Then, use pip to install the package
- Otherwise, pip must be installed.
```commandline
pip install graph-measures
```
**Note:** On Linux+Conda the installation might take longer (about 5-10 minuets) due to the compilation of the c++ files.
## How To Use?
Even though one has installed the package as `graph-measures`, The package should be imported from the code as `graphMesaures`. Hence, use:
```python
from graphMeasures import FeatureCalculator
```
## Calculating Features

There are two main methods to calculate features:
1. Using [FeatureCalculator](https://github.com/AmitKabya/graph-measures/blob/master/src/graphMeasures/features_for_any_graph.py) (**recommended**): \
A class for calculating any requested features on a given graph. \
The graph is input to this class as a text-like file of edges, with a comma delimiter, or a networkx _Graph_ object. 
For example, the graph [example_graph.txt](https://github.com/AmitKabya/graph-measures/blob/master/src/graphMeasures/measure_tests/example_graph.txt) is the following file: 
    ```
    0,1
    0,2
    1,3
    3,2
    ```
    Now, an implementation of feature calculations on this graph looks like this:
    ```python
   import os
   from graphMeasures import FeatureCalculator
   
   # set of features to be calculated
   feats = ["motif3", "louvain"]
   
   # path to the graph's edgelist or nx.Graph object
   graph = os.path.join("measure_tests", "example_graph.txt")
   
   # The path in which one would like to save the pickled features calculated in the process. 
   dir_path = "" 
   
   # More options are shown here. For information about them, refer to the file.
   ftr_calc = FeatureCalculator(path, feats, dir_path=dir_path, acc=True, directed=False,
                                 gpu=True, device=0, verbose=True)
   
   # Calculates the features. If one do not want the features to be saved,
   # one should set the parameter 'should_dump' to False (set to True by default).
   # If the features was already saved, you can set force_build to be True. 
   ftr_calc.calculate_features(force_build=True)
   features = ftr_calc.get_features() # return pandas Dataframe with the features 
    ``` 
   <!-- More information can be found in [features_for_any_graph.py](https://github.com/AmitKabya/graph-measures/blob/master/src/graphMeasures/features_for_any_graph.py). \-->
   **Note:** If one set `acc=True` without using a Linux+Conda machine, an exception will be thrown.\
   **Note:** If one set `gpu=True` without using a Linux+Conda machine that has cuda available on it, an exception will be thrown.
<br />
<br />
<br />
2. By the calculations as below **(not recommended)**: \
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
   in `graphMeasures.features_meta` or `graphMeasures.accelerated_features_meta`.
   
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

## Edges motifs:
For now, you can calculate only motifs for edges. Unfortunately, you will have to do it alone from the nodes features.
There are two options for motif calculation - python version, and accelerate version (in CPP).
The python version is always available, but the accelerate version available only on linux machine 
(the makefile targets linux, but the code should work for any os). Anyway, if you have a suitable machine,
the accelerated version is more recommended.

To run the accelerate version you should do:
1. Copy the graphMeasures directory to your project (available in this branch).
2. Open terminal in `graphMeasures/edges_features/acc_features/acc/`
3. Run the command `make`. If the makefile ends normally, a so file should be in a dir named bin.

Execution example:
```python
import networkx as nx
from graphMeasures.edges_features.feature_calculator import FeatureCalculator
from graphMeasures.edges_features.pickle_manager import PickleManager

pickle_manager = PickleManager("./pkl")

path = "./data/graph.txt"
gnx = nx.read_edgelist(path, delimiter=",", create_using=nx.DiGraph)
# acc signs if we will use the accelerate version.
calculator = FeatureCalculator(["motif3", "motif4"], gnx, acc=True)     
calculator.build()

# The result will be a pandas Dataframe named calculator.df.
# You can use the pickle_manager module to save it. 
pickle_manager.dump("graph", "motif3", calculator.df)
```


## Contact us
This package was written by [Yolo lab's](https://yolo.math.biu.ac.il/) team from Bar-Ilan University. \
For questions, comments or suggestions you can contact louzouy@math.biu.ac.il .

[//]: # (### Ouptput)

[//]: # (graphMeasures uses ```networkx```'s ```networkx.convert_node_labels_to_integers``` function, and then calculates the features. )

[//]: # (That's why the output matrix is ordered by the new nodes labels &#40;from 0 to n-1&#41; and not by the original labels.)

[//]: # ()
[//]: # (graphMeasures sometimes return the output columns not in the order they were inserted. The )
   
