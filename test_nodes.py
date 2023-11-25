
import os
from graphMeasures import FeatureCalculator
import pickle
# set of features to be calculated
feats = ["degree","motif3","motif4"] #


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

#Dataframe if there are only node features
features = ftr_calc.get_features() # return pandas Dataframe with the features 
print(features)
