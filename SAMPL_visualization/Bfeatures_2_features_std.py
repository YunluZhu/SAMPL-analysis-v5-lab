'''
Plot standard deviation of bout features

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
NIGHT_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
if_jackknnife = False # Whether to jackknife (bool)

- change the var DAY_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change DAY_RESAMPLE to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_mean, distribution_binned_average)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_functions import plt_categorical_grid


##### Parameters to change #####

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
NIGHT_RESAMPLE = 0
if_jackknnife = False # Whether to jackknife (bool)

##### Parameters to change #####

# %%
def jackknife_std(df,all_features):
    jackknife_df_std = pd.DataFrame()
    cat_cols = ['cond1','cond0','ztime']
    for (this_cond, this_dpf, this_ztime), group in df.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(group['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),all_features].std().to_frame().T
            # this_mean = group.loc[group['expNum'].isin(idx_group),all_features].mean().to_frame().T
            jackknife_df_std = pd.concat([jackknife_df_std, this_std.assign(dpf=this_dpf,
                                                                    cond1=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
    jackknife_df_std = jackknife_df_std.reset_index(drop=True)
    return jackknife_df_std

# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'BF2_std_z{which_ztime}_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'B2_std_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

defaultPlotting()
set_font_type()
# %%
# main function
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

bins = list(range(-90,95,5))

all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(all_feature_cond.ztime))
all_ztime.sort()

all_features = ['pitch_initial', 
                'pitch_pre_bout', 
                'pitch_peak', 
                'pitch_post_bout',
                'pitch_end', 
                'traj_initial', 
                'traj_pre_bout', 
                'traj_peak',
                'traj_post_bout', 
                'traj_end', 
                'spd_peak', 
                'angvel_initial_phase',
                'angvel_prep_phase',  
                'angvel_post_phase',
                'rot_total', 
                'rot_bout', 
                'rot_pre_bout', 
                'rot_l_accel', 
                'rot_l_decel',
                'bout_displ', 
                'atk_ang', 
                'angvel_chg',
                'depth_chg',
                ]

# %% calculate std
std_by_exp = all_feature_cond.groupby(cond_cols+['expNum'])[all_features].std().reset_index()

# %% jackknife 
if if_jackknnife:
    if which_ztime != 'night':
        all_feature_day = all_feature_cond.loc[
            all_feature_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            all_feature_day = all_feature_day.groupby(
                    ['cond0','cond1','expNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        jackknifed_day_std = jackknife_std(all_feature_day,all_features)

    jackknifed_night_std = pd.DataFrame()

    if which_ztime != 'day':
        all_feature_night = all_feature_cond.loc[
            all_feature_cond['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            all_feature_night = all_feature_night.groupby(
                    ['cond0','cond1','expNum']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True
                            )
        jackknifed_night_std = jackknife_std(all_feature_night,all_features)

    jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)

# %%  

####################################
###### Plotting Starts Here ######
####################################

x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'

if if_jackknnife:
    toplt = jackknifed_std
    units = 'excluded_exp'
    prename = 'jackknifed__'
else: 
    toplt = std_by_exp
    units = 'expNum'
    prename = ''

for feature in all_features:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=False,
        height = 3,
        aspect = 1,
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

if which_ztime == 'all':
    x_name = 'ztime'
    gridrow = 'cond1'
    gridcol = 'cond0'
    
    if if_jackknnife:
        toplt = jackknifed_std
        units = 'excluded_exp'
        prename = 'jackknifed__'
    else: 
        toplt = std_by_exp
        units = 'expNum'
        prename = ''

    for feature in all_features:
        g = plt_categorical_grid(
            data = toplt,
            x_name = x_name,
            y_name = feature,
            gridrow = gridrow,
            gridcol = gridcol,
            units = units,
            sharey=False,
            height = 3,
            aspect = 1,
            )
        filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
        plt.savefig(filename,format='PDF')
        plt.show()
