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

import scipy.stats as st
import statsmodels.api as sm
import statsmodels.robust.norms as norms
from sklearn.metrics import r2_score

#%%

##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'night' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
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
all_features = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)

# select dataset
all_features = all_features.query('cond1 == "ld"')
#%%
list_of_features = [
    'WHM',
    'pre_IBI_time',
    'post_IBI_time',
    # 'pitch_initial',
    # 'pitch_end',
    # 'rot_total',
    'y_initial',
    'y_end',
    # 'x_initial',
    # 'x_end',
    'depth_chg_fullBout',
    'atk_ang',
    # 'lift_distance',
    'traj_peak',
    'pitch_peak',
    
                    ]

# %% associate consecutive bouts

#####################
max_lag = 1
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)

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

selected_data = (
    sel_consecutive_bouts
    .groupby(["cond1",  "ztime", "expNum","id"], as_index=False)
    .apply(lambda group: group.assign(
        preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        # preIBI_x_displ=np.abs(group["x_initial"]-group["x_end"].shift(1)) ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        # preIBI_rot=group["pitch_initial"] - group["pitch_end"].shift(1),
        # postIBI_rot=group["pitch_initial"].shift(-1) - group["pitch_end"]
    ), include_groups=False)
    .reset_index(drop=True)  # Reset index after apply()
)

#%% no correlation

data_to_plot = selected_data.query('bouts == 2')    

g = sns.relplot(
    kind='scatter',
    data=data_to_plot.groupby('cond0').sample(1000),
    x='preIBI_y_displ',
    y='depth_chg_fullBout',
    col='cond0',
    alpha=0.1,
    color='black'
)
g.set(xlim=(-2,2),
      ylim=(-2,2)
      )

plt.savefig(os.path.join(fig_dir, 'allIBI preYdispl vs depth change.pdf'), format='pdf')


data_to_plot = selected_data.query('bouts == 1')    

g = sns.relplot(
    kind='scatter',
    data=data_to_plot.groupby('cond0').sample(1000),
    x='postIBI_y_displ',
    y='depth_chg_fullBout',
    col='cond0',
    alpha=0.1,
    color='black'
)
g.set(xlim=(-2,2),
      ylim=(-2,2)
      )
plt.savefig(os.path.join(fig_dir, 'allIBI postYdispl vs depth change.pdf'), format='pdf')


# %%
# consecutive bouts vs depth change in total

max_lag = 4
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)

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

selected_data = (
    sel_consecutive_bouts
    .groupby(["cond1",  'cond0', "ztime", "expNum","id"])
    .apply(lambda group: group.assign(
        preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
    ), include_groups=False)
    .reset_index()  # Reset index after apply()
)

#%%
bout_series_df = (
    selected_data
    .groupby(["cond1", "cond0", "ztime", "expNum", "id"], as_index=False)
    .agg(
        avgIBI_y=("preIBI_y_displ", "mean"),
        avgIBITime=("pre_IBI_time", "mean"), 
        avgBout_y=("depth_chg_fullBout", lambda x: x.iloc[1:].mean()),
        avgTraj=("traj_peak", "mean"), 
        avgAtkAng=("atk_ang", "mean"), 
        avgWHM=("WHM", lambda x: x.iloc[1:].mean()),
    )
)

bout_series_df = bout_series_df.assign(
    IBI_duration = pd.cut(bout_series_df['avgIBITime'], 
                          bins=[0,np.quantile(bout_series_df['avgIBITime'],.25),np.inf], 
                          labels=['short IBI','long IBI'])
    )

bout_series_df = bout_series_df.assign(
    WHM_duration = pd.cut(bout_series_df['avgWHM'], 
                          bins=[0,0.1,np.inf], 
                          labels=['short bout','long bout']),
    y_residual = bout_series_df['avgIBI_y'] + bout_series_df['avgBout_y'] 
    )

#%%

# g = sns.relplot(
#     kind='scatter',
#     data=bout_series_df.groupby(['cond0']).sample(1000, replace=True),
#     x='avgWHM',
#     y='y_residual',
#     col='cond0',
#     # row='WHM_duration',
#     alpha=0.1,
#     color='black',
#     hue='avgTraj',
#     palette=sns.color_palette("viridis", as_cmap=True)
# )
# g.set(xlim=(-1,1.2),
#       ylim=(-1,1.2)
#       )
# plt.savefig(os.path.join(fig_dir, 'allIBI postYdispl vs depth change.pdf'), format='pdf')

