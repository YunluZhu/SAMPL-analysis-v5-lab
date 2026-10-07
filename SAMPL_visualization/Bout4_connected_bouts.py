'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_connected_bouts
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.get_bout_consecutive_features import extract_consecutive_bout_features
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
#%%

##### Parameters to change #####
pick_data = 'nMLF' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
# my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
# my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)

# %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# tidy bout uid
all_features_df = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)

#%%
# NOTE: specify features to preserve for constructing a consecutive bout dataframe. you can pick any column in the "all_features" dataframe
list_of_features = [
    # 'WHM',
    'pre_IBI_time',
    'pitch_initial',
    'pitch_end',
    'rot_total',
    'y_initial',
    'y_end',
    'x_initial',
    'x_end',
                    ]

# %% associate consecutive bouts

#####################
# NOTE: specify the maximum lag to consider for consecutive bouts, aka the number of bouts to consider in a sequence
max_lag = 3
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features_df, list_of_features, max_lag)

#%%
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond1','cond0','id','lag','ztime']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    bouts = sel_consecutive_bouts['lag'] + 1
)

# Compare current y_initial with next bout's y_initial
sel_consecutive_bouts['bout_direction'] = sel_consecutive_bouts.apply(
    lambda row: 'climb' if row['y_initial'] < row['y_end'] else 'dive',
    axis=1
)

# NOTE add any additional features you want to compute for consecutive bouts here. 
selected_data = (
    sel_consecutive_bouts
    .groupby(["cond1",  "ztime", "expNum","id"], as_index=False)
    .apply(lambda group: group.assign(
        preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        preIBI_x_displ=np.abs(group["x_initial"]-group["x_end"].shift(1)) ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        # postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        preIBI_rot=group["pitch_initial"] - group["pitch_end"].shift(1),
        # postIBI_rot=group["pitch_initial"].shift(-1) - group["pitch_end"]
    ), include_groups=False)
    .reset_index(drop=True)  # Reset index after apply()
)

#%% The plot however you want
