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
from scipy.stats import ttest_rel, ttest_ind


# %%
##### Parameters to change #####
pick_data = 'a_rtau_box' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
if_jackknnife = True # Whether to jackknife (bool)

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'astro_BF1_{which_ztime}'
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

all_feature_toplt = all_feature_cond.copy()

# %%
# Plots

# %%
#mean
cat_cols = ['cond1','expNum','ztime','cond0','bout_time', 'exp']
feature_to_plt = [c for c in all_feature_toplt.columns if c not in cat_cols]
feature_for_comp = feature_to_plt + ['expNum']
# jackknife
all_feature_sampled = all_feature_toplt


cat_cols = ['cond0','cond1','ztime','expNum']
mean_data = all_feature_sampled.groupby(cat_cols).mean()
mean_data = mean_data.reset_index()

cat_cols = ['cond0','cond1','ztime']

mean_data_jackknife = all_feature_sampled.groupby(cat_cols)[feature_for_comp].apply(
    lambda x: jackknife_mean_by_col(x,'expNum','median')
 )
mean_data_jackknife = mean_data_jackknife.reset_index()
# %%

####################################
###### Plotting Starts Here ######
####################################

if if_jackknnife:
    toplt = mean_data_jackknife
    x_name = 'cond1'
    gridrow = 'cond0'
    gridcol = 'ztime'
    units = 'jackknife_idx'
    prename = 'jackknifed__'
else: 
    toplt = mean_data
    x_name = 'cond1'
    gridrow = 'cond0'
    gridcol = 'ztime'
    units = 'expNum'
    prename = ''


feature_to_plt = ['spd_peak']

for feature in feature_to_plt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=True,
        height = 3,
        aspect = 1.2
        )
    filename = os.path.join(fig_dir,f"{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
    for ztime in ['day', 'night']:
        this_compare = toplt.query('ztime == @ztime')
        print(ztime)
        ttest_res, ttest_p = ttest_rel(this_compare.loc[this_compare['cond1']==all_cond1[0],feature],
                                this_compare.loc[this_compare['cond1']==all_cond1[1],feature])
        print(f'{feature} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')


# %%
for feature in feature_to_plt:
    g = sns.catplot(
        data=all_feature_toplt,
        x='cond1',
        col='ztime',
        y=feature,
        kind='point',
    )
    for ztime in ['day', 'night']:
        this_compare = all_feature_toplt.query('ztime == @ztime')
        print(ztime)
        ttest_res, ttest_p = ttest_ind(this_compare.loc[this_compare['cond1']==all_cond1[0],feature],
                                this_compare.loc[this_compare['cond1']==all_cond1[1],feature])
        print(f'{feature} Sibs v.s. Tau: paired t-test p-value = {ttest_p}')
    plt.savefig(fig_dir+f"/{feature} point.pdf",format='PDF')

# %%
