'''
Plot bout features - UP DOWN separated by set point

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknnife = False # Whether to jackknife (bool)

- change the var DAY_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change DAY_RESAMPLE to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True

'''

#%%
# import sys
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.plt_tools import (jackknife_mean_by_col,set_font_type, defaultPlotting)
from plot_functions.plt_functions import plt_categorical_grid
import matplotlib as mpl

# %%
##### Parameters to change #####
pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknnife = False # Whether to jackknife (bool)

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'BF1_bySetPoint_z{which_ztime}_sample{DAY_RESAMPLE}'
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
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# %%
# assign up and down 

all_feature_UD = pd.DataFrame()
all_feature_cond = all_feature_cond.assign(direction=np.nan)
for (this_cond0,this_cond1,this_ztime), group in all_feature_cond.groupby(['cond0','cond1','ztime']):
    set_point = get_kinetics(group)['set_point']
    group['direction'] = pd.cut(group['pitch_pre_bout'],
                                bins=[-90,set_point,90],
                                labels=['dn','up'])
    group = group.assign(
        cond0 = this_cond0,
        cond1 = this_cond1,
        ztime = this_ztime
    )
    all_feature_UD = pd.concat([all_feature_UD,group],ignore_index=True) 

# %%
# Plots

# %%
#mean
cat_cols = ['cond1','expNum','direction','cond0','bout_time','ztime']
feature_to_plt = [c for c in all_feature_UD.columns if c not in cat_cols]
feature_for_comp = feature_to_plt + ['expNum']
# jackknife
all_feature_sampled = all_feature_UD

if DAY_RESAMPLE != 0:
    all_feature_sampled = all_feature_sampled.groupby(
            ['cond0','cond1','expNum','direction']
            ).sample(
                    n=DAY_RESAMPLE,
                    replace=True
                    )

cat_cols = ['cond0','cond1','direction','expNum']
mean_data = all_feature_sampled.groupby(cat_cols).mean()
mean_data = mean_data.reset_index()

cat_cols = ['cond0','cond1','direction']

mean_data_jackknife = all_feature_sampled.groupby(cat_cols)[feature_for_comp].apply(
    lambda x: jackknife_mean_by_col(x,'expNum')
 )
mean_data_jackknife = mean_data_jackknife.reset_index()
# %%

####################################
###### Plotting Starts Here ######
####################################

if if_jackknnife:
    toplt = mean_data_jackknife
    x_name = 'cond1'
    gridrow = 'direction'
    gridcol = 'cond0'
    units = 'jackknife_idx'
    prename = 'jackknifed__'
else: 
    toplt = mean_data
    x_name = 'cond1'
    gridrow = 'direction'
    gridcol = 'cond0'
    units = 'expNum'
    prename = ''

for feature in feature_to_plt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=False,
        height = 3,
        aspect = 1.2
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
