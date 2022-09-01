'''
plot mean IBI body angle distribution and standard deviation.

zeitgeber time? Yes
Jackknife? Yes
Sampled? Yes - separated sample number for day and night
- change the var DAY_RESAMPLE & NIGHT_RESAMPLE to select the number of bouts sampled per condition per repeat. 
- to disable sampling, change them to 0 
- If ztime == all, day and night count as 2 conditions
- for the pd.sample function, replace = True
'''

#%%
import sys
import os,glob
from matplotlib import style
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling

from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_mean, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_bout_features import get_bout_features

defaultPlotting()
set_font_type()

# %%
def jackknife_std(df,all_features):
    jackknife_df_std = pd.DataFrame()
    cat_cols = ['condition','dpf','ztime']
    for (this_cond, this_dpf, this_ztime), group in df.groupby(cat_cols):
        jackknife_idx = jackknife_resampling(np.array(list(range(df['expNum'].max()+1))))
        for excluded_exp, idx_group in enumerate(jackknife_idx):
            this_std = group.loc[group['expNum'].isin(idx_group),all_features].std().to_frame().T
            # this_mean = group.loc[group['expNum'].isin(idx_group),all_features].mean().to_frame().T
            jackknife_df_std = pd.concat([jackknife_df_std, this_std.assign(dpf=this_dpf,
                                                                    condition=this_cond,
                                                                    excluded_exp=excluded_exp,
                                                                    ztime=this_ztime)])
    jackknife_df_std = jackknife_df_std.reset_index(drop=True)
    return jackknife_df_std
# %%
# Paste root directory here
pick_data = 'tau_bkg'
which_zeitgeber = 'all'
DAY_RESAMPLE = 1000
NIGHT_RESAMPLE = 500

# %%
# ztime_dict = {}

root, FRAME_RATE = get_data_dir(pick_data)
if DAY_RESAMPLE+NIGHT_RESAMPLE > 0:
    folder_name = f'B2_std_z{which_zeitgeber}_resample_zD{DAY_RESAMPLE}_zN{NIGHT_RESAMPLE}'
else:
    folder_name = f'B2_std_z{which_zeitgeber}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

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

all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','dpf','condition']
all_ztime = list(set(all_feature_cond.ztime))
all_ztime.sort()

# %% jackknife for day bouts
# not the best code - jackknife and resample to be wrapped into a function
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
                'pitch_prep_phase', 
                'angvel_post_phase',
                'rot_total', 
                'rot_bout', 
                'rot_pre_bout', 
                'rot_l_accel', 
                'rot_l_decel',
                # 'rot_early_accel', 
                # 'rot_late_accel', 
                # 'rot_early_decel',
                # 'rot_late_decel', 
                'bout_traj', 
                'bout_displ', 
                'atk_ang', 
                'tsp_peak',
                'angvel_chg']

if which_zeitgeber != 'night':
    all_feature_day = all_feature_cond.loc[
        all_feature_cond['ztime']=='day',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        all_feature_day = all_feature_day.groupby(
                ['dpf','condition','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )
    jackknifed_day_std = jackknife_std(all_feature_day,all_features)

if which_zeitgeber != 'day':
    all_feature_night = all_feature_cond.loc[
        all_feature_cond['ztime']=='night',:
            ]
    if DAY_RESAMPLE != 0:  # if resampled
        all_feature_night = all_feature_night.groupby(
                ['dpf','condition','expNum']
                ).sample(
                        n=DAY_RESAMPLE,
                        replace=True
                        )
    jackknifed_night_std = jackknife_std(all_feature_night,all_features)

jackknifed_std = pd.concat([jackknifed_day_std,jackknifed_night_std]).reset_index(drop=True)
std_by_exp = all_feature_cond.groupby(cond_cols+['expNum'])[all_features].std().reset_index()
# %% ignore this

# # %%
# # plot kde of all
# g = sns.FacetGrid(IBI_angles_cond, 
#                   row="ztime", row_order=all_ztime,
#                   col='dpf', col_order=cond1,
#                   hue='condition', hue_order=cond2,
#                   )
# g.map(sns.kdeplot, "IBI_pitch",alpha=0.5,)
# g.add_legend()
# filename = os.path.join(fig_dir,"IBI pitch kde.pdf")
# plt.savefig(filename,format='PDF')

# %% 
# raw pitch by exp day vs night

if which_zeitgeber == 'all':
    for feature in all_features:
        g = sns.catplot(data=std_by_exp,
                        col='dpf',row='condition',
                        x='ztime', y=feature,
                        hue='dpf',
                        ci='sd',
                        kind='point',
                        aspect=0.6)
        g.map(sns.lineplot,'ztime',feature,estimator=None,
        units='expNum',
        data = std_by_exp,
        alpha=0.2,)
        filename = os.path.join(fig_dir,f"by_exp {feature} std day-night.pdf")
        plt.savefig(filename,format='PDF')
        plt.close()

# pitch cond vs ctrl
for feature in all_features:
    g = sns.catplot(data=std_by_exp,
                    col='dpf',
                    row='ztime',
                    x='condition', y=feature,
                    hue='expNum',
                    ci=None,
                    # markers=['d','d'],
                    sharey=False,
                    kind='point',
                    aspect=.6
                    )
    g.map(sns.lineplot,'condition',feature,estimator=None,
        units='expNum',
        data = std_by_exp,
        alpha=0.2,)
    filename = os.path.join(fig_dir,f"by_exp {feature} std cond individual.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

    g = sns.catplot(data=std_by_exp,
                    col='dpf',row='ztime',
                    x='condition', y=feature,
                    hue='dpf',
                    ci='sd',
                    kind='point',
                    aspect=.6)
    g.map(sns.lineplot,'condition',feature,estimator=None,
        units='expNum',
        data = std_by_exp,
        alpha=0.2,)
    filename = os.path.join(fig_dir,f"by_exp {feature} std cond mean.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

# %%
# jackknifed resampled  std
for feature in all_features:
    g = sns.catplot(data=jackknifed_std,
                    col='dpf',
                    row='ztime',
                    x='condition', y=feature,
                    hue='condition',
                    ci='sd', 
                    # markers=['d','d'],
                    sharey=False,
                    kind='point',
                    aspect=.8
                    )
    g.map(sns.lineplot,'condition',feature,estimator=None,
        units='excluded_exp',
        data = jackknifed_std,
        color='grey',
        alpha=0.2,)
    g.add_legend()
    sns.despine(offset=10)
    filename = os.path.join(fig_dir,f"jackknifed {feature} std.pdf")
    plt.savefig(filename,format='PDF')
    plt.close()

# jackknifed resampled per repeat
# mean cond vs ctrl
# plot on same scale
    g = sns.catplot(data=jackknifed_std,
                    col='dpf',
                    row='ztime',
                    x='condition', y=feature,
                    hue='condition',
                    ci='sd', 
                    # markers=['d','d'],
                    sharey='row',
                    kind='point',
                    aspect=.8
                    )
    g.map(sns.lineplot,'condition',feature,estimator=None,
        units='excluded_exp',
        data = jackknifed_std,
        color='grey',
        alpha=0.2,)
    g.add_legend()
    sns.despine(offset=10)
    filename = os.path.join(fig_dir,f"jackknifed {feature} std sharey.pdf")
    plt.savefig(filename,format='PDF')


# %%
