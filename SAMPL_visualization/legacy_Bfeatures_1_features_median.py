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
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean_by_col,set_font_type)
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl

# %%
##### Parameters to change #####
pick_data = 'sldp2025' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknnife = False # Whether to jackknife (bool)

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'BF1_median_z{which_ztime}_sample{DAY_RESAMPLE}'
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

all_feature_UD = all_feature_cond.copy()
# for (this_cond0,this_cond1), group in all_feature_cond.groupby(['cond0','cond1']):
#     # group['direction'] = pd.cut(group['rot_l_decel'],
#     #                             bins=[-90,0,90],
#     #                             labels=['dn','up'])
#     group = group.assign(
#         cond0 = this_cond0,
#         cond1 = this_cond1,
#     )
#     all_feature_UD = pd.concat([all_feature_UD,group],ignore_index=True) 

# %%
# Plots

# %%
#mean
cat_cols = ['cond1','expNum','cond0','bout_time','ztime','exp','direction']
feature_to_plt = [c for c in all_feature_UD.columns if c not in cat_cols]
feature_for_comp = feature_to_plt
# jackknife
all_feature_sampled = all_feature_UD
cat_cols = ['cond1','expNum','cond0','bout_time','ztime','exp']

if DAY_RESAMPLE != 0:
    all_feature_sampled = all_feature_sampled.groupby(
            ['cond0','cond1','expNum']
            ).sample(
                    n=DAY_RESAMPLE,
                    replace=True
                    )

cat_cols = ['cond0','cond1','expNum']
mean_data = all_feature_sampled.groupby(cat_cols)[feature_to_plt].median()
mean_data = mean_data.reset_index()

cat_cols = ['cond0','cond1']

mean_data_jackknife = all_feature_sampled.groupby(cat_cols)[feature_for_comp + ['expNum']].apply(
    lambda x: jackknife_mean_by_col(x,'expNum', method='median')
 )
mean_data_jackknife = mean_data_jackknife.reset_index()
# %%
feature_to_plt.sort()
####################################
###### Plotting Starts Here ######
####################################

if if_jackknnife:
    toplt = mean_data_jackknife
    x_name = 'cond1'
    # gridrow = 'direction'
    gridcol = 'cond0'
    units = 'jackknife_idx'
    prename = 'jackknifed__'
else: 
    toplt = mean_data
    x_name = 'cond1'
    # gridrow = 'direction'
    gridcol = 'cond0'
    units = 'expNum'
    prename = ''

for feature in feature_for_comp:
    g = plt_categorical_grid2(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = None,
        gridcol = gridcol,
        units = units,
        sharey=True,
        height = 2.5,
        aspect = 1
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
