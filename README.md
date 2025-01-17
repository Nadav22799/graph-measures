# Topological Graph Features

Topological feature calculators infrastructure.

## Calculating Features
This package helps one to calculate features for a given graph. All features are implemented in python codes, 
and some features have also an accelerated version written in C++. Among the accelerated features, one can find 
a code for calculating 3- and 4-motifs using VDMC, a distributed algorithm to calculate 3- and 4-motifs in a 
GPU-parallelized way.

## Versions
- Last version: 0.1.55 (most recommended)

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
1. Using [FeatureCalculator](https://github.com/louzounlab/graph-measures/blob/master/graphMeasures/features_for_any_graph.py) (**recommended**): \
A class for calculating any requested features on a given graph. \
The graph is input to this class as a text-like file of edges, with a comma delimiter, or a networkx _Graph_ object. 
For example, the graph [example_graph.txt](https://github.com/louzounlab/graph-measures/blob/master/graphMeasures/measure_tests/example_graph.txt) is the following file: 
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
2. Using graphMeasure <a href="https://github.com/louzounlab/graph-measures/blob/master/OTHERS.md">without FeatureCalculator</a> (**less recommended**).

[//]: # (2. Using graphMeasure [without FeatureCalculator]&#40;https://github.com/louzounlab/graph-measures/blob/master/OTHERS.md&#41; &#40;**less recommended**&#41;.)
<br />

## Example 
For the next directed graph and these features, FeatureCalculator should return this dataframe:

**The features:** ["out_degree", "k_core", "in_degree", "page_rank", "not-exist-feature"] <br>
**The graph (directed):**
```
5,1
0,1
0,3
1,3
2,5
2,4
3,2
4,0
```
**The result:**

| node | k_core | page_rank | out_degree | in_degree | not-exist-feature |
|------|--------|-----------|------------|-----------|-------------------|
| 0    | 2.0    | 0.123484  | 2.0        | 1.0       | NaN               |
| 1    | 2.0    | 0.179051  | 1.0        | 2.0       | NaN               |
| 2    | 2.0    | 0.226711  | 2.0        | 1.0       | NaN               |
| 3    | 2.0    | 0.226711  | 1.0        | 2.0       | NaN               |
| 4    | 2.0    | 0.118687  | 1.0        | 1.0       | NaN               |
| 5    | 2.0    | 0.118687  | 1.0        | 1.0       | NaN               |




## Edges motifs:
For now, you can calculate only motifs for edges. Unfortunately, you will have to do it separately from the nodes features.
There are two options for motif calculation - python version, and accelerated version (in CPP).
The python version is always available, but the accelerated version available only on linux machine 
(the makefile targets linux, but the code should work for any os). Anyway, if you have a suitable machine,
the accelerated version is more recommended.

To run the accelerated version you should do:
1. Copy the graphMeasures directory to your project (available in this branch).
2. Open terminal in `graphMeasures/edges_features/acc_features/acc/`
3. Run the command `make`. If the makefile ends normally, a so file should be in a dir named bin.

Execution example:
```python
import networkx as nx
from graphMeasures.edges_features.feature_calculator import FeatureCalculator

path = "./data/graph.txt"
gnx = nx.read_edgelist(path, delimiter=",", create_using=nx.DiGraph)
# acc signs if we will use the accelerated version.
calculator = FeatureCalculator(["motif3", "motif4"], gnx, acc=True)     
calculator.build()

# The result will be a pandas Dataframe named calculator.df.
print(calculator.df)
```


## Contact us
This package was written by [Yolo lab's](https://yolo.math.biu.ac.il/) team from Bar-Ilan University. \
For questions, comments or suggestions you can contact louzouy@math.biu.ac.il.

[//]: # (### Ouptput)

[//]: # (graphMeasures uses ```networkx```'s ```networkx.convert_node_labels_to_integers``` function, and then calculates the features. )

[//]: # (That's why the output matrix is ordered by the new nodes labels &#40;from 0 to n-1&#41; and not by the original labels.)

[//]: # ()
[//]: # (graphMeasures sometimes return the output columns not in the order they were inserted. The )
   
