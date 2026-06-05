# %%

import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_aligned_bouts_wIBI, get_bout_features, extract_bout_features_v5)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from plot_functions.plt_functions import (plt_categorical_grid, plt_network_graphs)
from plot_functions.get_bout_correlation import get_cluster_phaseSpace
import math
from plot_functions.plt_tools import round_half_up 
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter

import networkx as nx

set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'a_gtau'
which_ztime = 'day'
compare_which = 'cond1' # condition for separation None for treat as whole
nCluster = 10

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} cluster graphs_by{compare_which}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% 
all_around_peak_data, all_feature_cond, all_cond0, all_cond1, idxRANGE = get_aligned_bouts_wIBI(root, FRAME_RATE, ztime=which_ztime)

# %%
all_feature_clustered = get_cluster_phaseSpace(all_around_peak_data, all_feature_cond, idxRANGE, nCluster)
# get_cluster_features()
# %%
connected_bout_features = all_feature_clustered[all_feature_clustered['to_bout'].notna()]

to_cluster = connected_bout_features[['to_bout']].merge(all_feature_clustered[['cluster','bout_uid']], left_on='to_bout',right_on='bout_uid',how='left')

to_cluster.columns = ['to_bout', 'to_cluster', 'to_bout_uid']

connected_bout_features = connected_bout_features.merge(to_cluster[['to_bout','to_cluster']], on='to_bout', how='left')
connected_bout_features = connected_bout_features[connected_bout_features['to_cluster'].notna()]
connected_bout_features.reset_index(drop=True, inplace=True)
connected_bout_features = connected_bout_features.assign(
    to_cluster = connected_bout_features['to_cluster'].astype('int'),
)

# %%
if bool(compare_which):
    for this_condition in connected_bout_features[compare_which].unique():
        this_cond_features = connected_bout_features.loc[connected_bout_features[compare_which]==this_condition]
        # plot_network_graphs(extracted_features=this_cond_features, cond_sep=this_condition, sort_by_feature='ydispl_swim', total_features=connected_bout_features)
        extracted_features=this_cond_features
        cond_sep=this_condition
        sort_by_feature='ydispl_swim'
        total_features=connected_bout_features
        
        plt_network_graphs(connected_bout_features, 
                            fig_dir = fig_dir,
                            cond_sep=this_condition, 
                            extracted_features=this_cond_features)
else:
    sort_by_feature='ydispl_swim'
    plt_network_graphs(connected_bout_features, 
                        fig_dir = fig_dir,
                        sort_by_feature = sort_by_feature)
    
# %%
