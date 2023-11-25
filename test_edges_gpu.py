
import os
from graphMeasures import FeatureCalculator
import pickle
import pandas as pd
import numpy as np

def edge2motifDF(key,value,motif_n):
    df=pd.DataFrame()
    df["Edges"]=key
    num_motifs=len(list(value)[0])
    value=np.array(list(value)).T
    for i in range(num_motifs):
      motifName="motif"+str(motif_n)+"_"+str(i)
      df[motifName]=value[i]
    return df.set_index("Edges")
    
def order_nodes(order,dicti):
  feat={}
  for edge in dicti:
    new_edge=(order[int(edge[0])],order[int(edge[1])])
    feat[new_edge]=dicti[edge]
  return feat


# set of features to be calculated
feats = ["motif3_edges_gpu","motif4_edges_gpu"]


# The path in which one would like to save the pickled features calculated in the process. 
dir_path = "./pkl"
path = "graph.txt" 

# More options are shown here. For information about them, refer to the file.
ftr_calc = FeatureCalculator(path, feats, dir_path=dir_path, acc=True, directed=False,
                             gpu=True, device=0, verbose=True, should_zscore=False)
# Calculates the features. If one do not want the features to be saved,
# one should set the parameter 'should_dump' to False (set to True by default).
# If the features was already saved, you can set force_build to be True. 
ftr_calc.calculate_features(force_build=True)
with open(dir_path+str("/motif3_edges_gpu.pkl"), "rb") as f:
        motif3_edges_gpu = pickle.load(f)

with open(dir_path+str("/motif4_edges_gpu.pkl"), "rb") as f:
        motif4_edges_gpu = pickle.load(f)
        
original_node_order=ftr_calc.nodes_order
motif3_edges_gpu=order_nodes(original_node_order,motif3_edges_gpu.features)
motif4_edges_gpu=order_nodes(original_node_order,motif4_edges_gpu.features)

df_motif3_edges_gpu=edge2motifDF(motif3_edges_gpu.keys(),motif3_edges_gpu.values(),3)
df_motif4_edges_gpu=edge2motifDF(motif4_edges_gpu.keys(),motif4_edges_gpu.values(),4)

print(df_motif3_edges_gpu)
print(df_motif4_edges_gpu)

