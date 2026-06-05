'''
calculate std or median per box
plot median of calculated values

'''

#%%
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,jackknife_list,defaultPlotting, set_font_type, distribution_binned_average)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_functions import plt_categorical_grid2


##### Parameters to change #####

pick_data = 'creTau8' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
DAY_RESAMPLE = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling
NIGHT_RESAMPLE = 0
if_jackknnife = True # leave at false

##### Parameters to change #####

# %%
def jackknife_std(df,all_features,which_col):
    jackknife_df_std = pd.DataFrame()
    cat_cols = ['cond1','cond0','ztime']
    for (this_cond, this_dpf, this_ztime), group in df.groupby(cat_cols):
        jackknife_idx = jackknife_list(group[which_col].unique())
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group[which_col].isin(idx_group),all_features].std().to_frame().T
            # this_mean = group.loc[group['expNum'].isin(idx_group),all_features].mean().to_frame().T
            jackknife_df_std = pd.concat([jackknife_df_std, this_std.assign(cond0=this_dpf,
                                                                    cond1=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
    jackknife_df_std = jackknife_df_std.reset_index(drop=True)
    return jackknife_df_std


def jackknife_avg(df,all_features, which_col):
    jackknife_df_std = pd.DataFrame()
    cat_cols = ['cond1','cond0','ztime']
    for (this_cond, this_dpf, this_ztime), group in df.groupby(cat_cols):
        jackknife_idx = jackknife_list(group[which_col].unique())
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group[which_col].isin(idx_group),all_features].median().to_frame().T
            # this_mean = group.loc[group['expNum'].isin(idx_group),all_features].mean().to_frame().T
            jackknife_df_std = pd.concat([jackknife_df_std, this_std.assign(cond0=this_dpf,
                                                                    cond1=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
    jackknife_df_std = jackknife_df_std.reset_index(drop=True)
    return jackknife_df_std
# %%
root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'BF2_ByBox_z{which_ztime}_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'BF2_ByBox_z{which_ztime}'
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

all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','cond0','cond1']
all_ztime = list(set(all_feature_cond.ztime))
all_ztime.sort()
all_feature_cond = all_feature_cond.assign(
    expBoxNum = all_feature_cond['expNum'].astype(str) + '_' + all_feature_cond['boxNum'].astype(str) 
)

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
                'fish_length'
                ]


# %% calculate std
std_by_exp = all_feature_cond.groupby(cond_cols+['expBoxNum'])[all_features].std().reset_index()
std_by_exp = std_by_exp.assign(expNum = std_by_exp['expBoxNum'].str.split('_', expand=True).iloc[:,0].values)
# % jackknife 
if if_jackknnife:
    if which_ztime != 'night':
        all_feature_day = all_feature_cond.loc[
            all_feature_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            all_feature_day = all_feature_day.groupby(
                    ['cond0','cond1','expBoxNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        jackknifed_day_std = jackknife_std(all_feature_day,all_features, 'expBoxNum')

    jackknifed_night_std = pd.DataFrame()

    if which_ztime != 'day':
        all_feature_night = all_feature_cond.loc[
            all_feature_cond['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            all_feature_night = all_feature_night.groupby(
                    ['cond0','cond1','expBoxNum']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True
                            )
        jackknifed_night_std = jackknife_std(all_feature_night,all_features, 'expBoxNum')

    jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)

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
    g = plt_categorical_grid2(
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
    g.fig.suptitle('std')
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
        units = 'expBoxNum'
        prename = ''

    for feature in all_features:
        g = plt_categorical_grid2(
            data = toplt,
            x_name = x_name,
            y_name = feature,
            gridrow = gridrow,
            gridcol = gridcol,
            units = units,
            sharey=False,
            height = 3,
            aspect = 1,
            method = 'mean'
            )
        filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
        g.fig.suptitle('std')
        plt.savefig(filename,format='PDF')
        plt.show()

###################
###################
###################
###################
###################

#%%
x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'

if if_jackknnife:
    toplt = jackknifed_std
    units = 'excluded_exp'
    prename = 'jackknifed__'
else: 
    toplt = std_by_exp
    units = 'expBoxNum'
    prename = ''

for feature in all_features:
    g = sns.catplot(
        data=toplt.query("ztime=='day'"),
        x=x_name,
        y=feature,
        units=units,
        estimator=None,
        hue=units,
        height=3,
        col=gridcol
    )
    

# %% calculate average
avg_by_exp = all_feature_cond.groupby(cond_cols+['expBoxNum'])[all_features].median().reset_index()
avg_by_exp = avg_by_exp.assign(expNum = avg_by_exp['expBoxNum'].str.split('_', expand=True).iloc[:,0].values)

# all_features = ['angvel_post_phase', 'traj_initial']
# % jackknife 
if if_jackknnife:
    if which_ztime != 'night':
        all_feature_day = all_feature_cond.loc[
            all_feature_cond['ztime']=='day',:
                ]
        if DAY_RESAMPLE != 0:  # if resampled
            all_feature_day = all_feature_day.groupby(
                    ['cond0','cond1','expBoxNum']
                    ).sample(
                            n=DAY_RESAMPLE,
                            replace=True
                            )
        jackknifed_day_std = jackknife_avg(all_feature_day,all_features, 'expBoxNum')

    jackknifed_night_std = pd.DataFrame()

    if which_ztime != 'day':
        all_feature_night = all_feature_cond.loc[
            all_feature_cond['ztime']=='night',:
                ]
        if NIGHT_RESAMPLE != 0:  # if resampled
            all_feature_night = all_feature_night.groupby(
                    ['cond0','cond1','expBoxNum']
                    ).sample(
                            n=NIGHT_RESAMPLE,
                            replace=True
                            )
        jackknifed_night_std = jackknife_avg(all_feature_night,all_features, 'expBoxNum')

    jackknifed_avg = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)

# %

x_name = 'cond1'
gridrow = 'ztime'
gridcol = 'cond0'

if if_jackknnife:
    toplt = jackknifed_avg
    units = 'excluded_exp'
    prename = 'jackknifed__'
else: 
    toplt = avg_by_exp
    units = 'expNum' # or expBoxNum
    prename = ''

for feature in all_features:
    g = plt_categorical_grid2(
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
    g.fig.suptitle('median')
    plt.savefig(filename,format='PDF')
    plt.show()

if which_ztime == 'all':
    x_name = 'ztime'
    gridrow = 'cond1'
    gridcol = 'cond0'
    
    if if_jackknnife:
        toplt = jackknifed_avg
        units = 'excluded_exp'
        prename = 'jackknifed__'
    else: 
        toplt = avg_by_exp
        units = 'expBoxNum'
        prename = ''

    for feature in all_features:
        g = plt_categorical_grid2(
            data = toplt,
            x_name = x_name,
            y_name = feature,
            gridrow = gridrow,
            gridcol = gridcol,
            units = units,
            sharey=False,
            height = 3,
            aspect = 1,
            method = 'median'
            )
        filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
        g.fig.suptitle('median')
        plt.savefig(filename,format='PDF')
        plt.show()
# %%
