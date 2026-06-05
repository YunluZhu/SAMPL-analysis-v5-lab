# %% import sys
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_sum, distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid
import matplotlib as mpl

# %%
##### Parameters to change #####
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'B7_freq_{which_ztime}'
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
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond = all_feature_cond.sort_values(by=['cond0','cond1','expNum']).reset_index(drop=True)


# %%
hour_bins = np.arange(0,24+2,2)
all_feature_cond = all_feature_cond.assign(
    hour = all_feature_cond.bout_time.dt.hour
)
all_feature_cond = all_feature_cond.assign(
    binned_hour = pd.cut(all_feature_cond['hour'], hour_bins)
)

# %%
bout_num_time = all_feature_cond.groupby(['cond1', 'binned_hour','expNum']).size()
bout_num_time.name = 'num_of_bouts'
bout_num_time = bout_num_time.reset_index()
bout_num_time = bout_num_time.assign(
    mid_hour = list(hour_bins[:-1]+1) * len(all_feature_cond.groupby(['cond1','expNum']).size())
)

sns.lineplot(data=bout_num_time, x='mid_hour', hue='cond1', y='num_of_bouts')

# %%

feature_sel = 'pitch_initial'
all_res = pd.DataFrame()
for (cond0, cond1, expNum), group in all_feature_cond.groupby(['cond0', 'cond1', 'expNum']):
    
    this_cond_res = distribution_binned_average(
        df=group,
        by_col='hour',
        bin_col=feature_sel,
        bin=hour_bins,
        method='std'
    )
    this_cond_res = this_cond_res.assign(
        cond0 = cond0,
        cond1 = cond1,
        expNum = expNum
    )
    this_cond_res.index.names=['hour_bin']
    this_cond_res = this_cond_res.reset_index()
    all_res = pd.concat([all_res, this_cond_res])
all_res = all_res.assign(
    mid_hour = list(hour_bins[:-1]+1) * len(all_feature_cond.groupby(['cond0','cond1','expNum']).size())
)
# %%
sns.lineplot(
    data=all_res,
    hue='cond1',
    x='mid_hour',
    y=feature_sel
)
# %%
