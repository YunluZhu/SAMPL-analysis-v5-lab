'''
Plot jackknifed and raw kinematic parameters

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
SAMPLE_NUM = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling

'''

#%%
# import sys
import os,glob
# import time
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_index import get_index
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.plt_functions import plt_categorical_grid

# %%
##### Parameters to change #####

pick_data = 'tmp' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
SAMPLE_NUM = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling

##### Parameters to change #####

# %% get data and constants

root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)

# make figure folder
folder_name = f'BK1_z{which_ztime}_sample{SAMPLE_NUM}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
defaultPlotting()
# %%
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond1 = get_bout_kinetics(root, FRAME_RATE, ztime=which_ztime, sample=SAMPLE_NUM)
all_cond0.sort()
all_cond1.sort()

# %%

####################################
###### Plotting Starts Here ######
####################################

# by speed bins

toplt = kinetics_bySpd_jackknife
cat_cols = ['jackknife_group','cond1','expNum','cond0','ztime']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        row = 'ztime',
        col = 'cond0',
        hue = 'cond1',
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"{feature_toplt}_z{which_ztime}_bySpd.pdf")
    plt.savefig(filename,format='PDF')

# by ztime

if which_ztime == 'all': # if there's both day and night data
    toplt = kinetics_jackknife
    cat_cols = ['jackknife_group','cond1','jackknife_group','cond0','ztime']
    all_features = [c for c in toplt.columns if c not in cat_cols]
    # plot ztime on x
    plt_categorical_grid(
        data = toplt,
        x_name = 'ztime',
        y_name = all_features,
        gridrow = 'cond0',
        gridcol = 'cond1',
        units = 'jackknife_group',
        fig_dir = fig_dir,
        )
    
# %% 
# by condition

toplt = kinetics_jackknife
cat_cols = ['jackknife_group','cond1','expNum','cond0','ztime']
columns_toplt = [c for c in toplt.columns if c not in cat_cols]

x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'
units = 'jackknife_group'

for feature in columns_toplt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        )
    filename = os.path.join(fig_dir,f"{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
# by condition, plot un-jackknifed results. each line represnets a repeat

toplt = all_kinetic_cond
toplt.sort_values(by='cond1',inplace=True)
cat_cols = ['expNum','cond1','cond0','ztime']
columns_toplt = [col for col in toplt.columns if col not in cat_cols]

x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'
units = 'expNum'

for feature in columns_toplt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        )
    filename = os.path.join(fig_dir,f"no_jackknife__{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
