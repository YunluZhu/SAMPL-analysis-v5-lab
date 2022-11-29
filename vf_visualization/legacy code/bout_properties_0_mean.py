'''
This version uses "propBoutAligned_angVel" in the "prop_bout_aligned" key of bout_data file
which includes angular velocity during bouts
Output: plots of distribution of angular velocity 31 frames around peak speed during bouts
Conditions are soft-coded.
'''

#%%
import sys
import os,glob
import time
import math
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir
from plot_functions.get_index import get_index

from plot_functions.plt_tools import (set_font_type, defaultPlotting)

# %%
# Paste root directory here
pick_data = 'hc4'
root,FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data}_boutPropertiesV4'
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print('fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
# FRAME_RATE = 166
# %%
# CONSTANTS
HEADING_LIM = 90
peak_idx , total_aligned = get_index(FRAME_RATE)
# %%
set_font_type()
defaultPlotting()

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    return df_day
# %%
# main function
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs())
                # assign frame number, total_aligned frames per bout
                diff_res = np.diff(exp_data['oriIndex'].values - exp_data.index)
                total_aligned = np.nonzero(diff_res)
                total_aligned = total_aligned[0][0] + 1
                exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+round_half_up(15*FRAME_RATE/40),i*total_aligned+round_half_up(46*FRAME_RATE/40))))
                exp_data = exp_data.assign(expNum = expNum)
                around_peak_data = pd.concat([around_peak_data,exp_data.loc[rows,:]])
            # combine data from different conditions
            all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(dpf=all_conditions[condition_idx].split("_")[0],
                                                                                            condition=all_conditions[condition_idx].split("_")[1])])
            
            
# %%
all_around_peak_data = all_around_peak_data.reset_index(drop=True)
all_around_peak_data = all_around_peak_data.assign(
    linear_accel = all_around_peak_data['propBoutAligned_speed'].diff(),
    ang_accel = all_around_peak_data['ang_speed'].diff()
)

# Peak data 
peak_data = all_around_peak_data.loc[all_around_peak_data['idx']==peak_idx].reset_index(drop=True)
peak_grp = peak_data.groupby(['expNum','condition'],as_index=False)


# %%

x_ticks = np.arange(min(all_around_peak_data['idx'])-1,max(all_around_peak_data['idx'])+1,2)


# # plot speed as a function of time during bouts
# defaultPlotting()
# g = sns.relplot(x='idx',y='propBoutAligned_speed', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#                 aspect=2)
# plt.xticks(x_ticks)
# plt.show()

# g = sns.relplot(x='idx',y='propBoutAligned_speed_hDn', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#                 aspect=2)
# plt.xticks(x_ticks)
# plt.show()

# g = sns.relplot(x='idx',y='propBoutAligned_speed_hUp', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#                 aspect=2)
# plt.xticks(x_ticks)
# plt.show()

# g = sns.relplot(x='idx',y='propBoutAligned_speed_flat', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#                 aspect=2)
# plt.xticks(x_ticks)
# plt.show()

# %%
# # plot linear and angular acceleration during swim bouts


# # all_around_peak_data = all_around_peak_data.loc[all_around_peak_data['idx'] != 14]
# sns.relplot(x='idx',y='linear_accel', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#             aspect=2)
# plt.xticks(x_ticks)
# plt.show()
# sns.relplot(x='idx',y='ang_accel', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
#             aspect=2)
# plt.xticks(x_ticks)
# plt.show()

# %%
# plot Angular speed as a function of time during bouts
g = sns.relplot(x='idx',y='ang_speed', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
                aspect=2)
plt.xticks(x_ticks)
plt.show()

g = sns.relplot(x='idx',y='propBoutAligned_angVel_hDn', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
                aspect=2)
plt.xticks(x_ticks)
plt.show()

g = sns.relplot(x='idx',y='propBoutAligned_angVel_hUp', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
                aspect=2)
plt.xticks(x_ticks)
plt.show()

g = sns.relplot(x='idx',y='propBoutAligned_angVel_flat', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
                aspect=2)
plt.xticks(x_ticks)
plt.show()


# %%
# plot pitch change during bouts
defaultPlotting()
sns.relplot(x='idx',y='propBoutAligned_pitch', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
            aspect=2)
plt.xticks(x_ticks)

plt.show()
sns.relplot(x='idx',y='propBoutAligned_pitch_hDn', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
            aspect=2)
plt.xticks(x_ticks)

plt.show()
sns.relplot(x='idx',y='propBoutAligned_pitch_hUp', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
            aspect=2)
plt.xticks(x_ticks)
plt.show()

sns.relplot(x='idx',y='propBoutAligned_pitch_flat', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band',
            aspect=2)
plt.xticks(x_ticks)
plt.show()

# %%
# plot y change during bouts

defaultPlotting()
sns.relplot(x='idx',y='propBoutAligned_y', hue='condition', col='dpf', data=all_around_peak_data, kind="line",ci='sd', err_style='band')
plt.xticks(x_ticks)

plt.show()

 
# %%
# Peak speed

# defaultPlotting()
# g = sns.boxplot(x='condition',y='propBoutAligned_speed', hue='condition', data=peak_data, dodge=True)
# plt.show()

defaultPlotting()
data = peak_grp['propBoutAligned_speed'].aggregate(np.average)

flatui = ["#D0D0D0"] * (data.groupby('condition').size()[0])

g = sns.pointplot(x='condition',y='propBoutAligned_speed', hue='expNum', data=data,
                  palette=sns.color_palette(flatui), scale=0.5)
p = sns.pointplot(data=data.groupby('condition',as_index=False).aggregate(np.average), x='condition',y='propBoutAligned_speed', hue='condition',
              linewidth=0,
              alpha=0.9,
              ci=None,
              markers='d',)
p.legend_.remove()
plt.show()

# %%
# some Peak parameters 
defaultPlotting()
g = sns.boxplot(x='dpf',y='linear_accel', hue='condition', data=peak_data, dodge=True)
plt.show()
g = sns.boxplot(x='dpf',y='ang_speed', hue='condition', data=peak_data, dodge=True)
plt.show()
# %%
# # determine the range to use for pre bout rotation (posture change) calculation
# posture_chg_a = all_around_peak_data.loc[all_around_peak_data['idx']==29,'propBoutAligned_pitch'].values - all_around_peak_data.loc[all_around_peak_data['idx']==27,'propBoutAligned_pitch'].values 
# posture_chg_b = all_around_peak_data.loc[all_around_peak_data['idx']==28,'propBoutAligned_pitch'].values - all_around_peak_data.loc[all_around_peak_data['idx']==26,'propBoutAligned_pitch'].values 

# sns.distplot(posture_chg_a)
# sns.distplot(posture_chg_b)

# %%
# plot a 2d-histogram
# p = sns.jointplot(posture_chg_b,peak_data['atk_ang'].values, kind="kde", height=7, space=0, xlim=(-12, 12), ylim=(-15, 25))

# print("for distribution under different conditions, see atk_angle_fin_body_ratio")
# %%
# Conditions below are hard-coded
# # Separate conditions
# aligned_4s = all_around_peak_data.loc[(all_around_peak_data['dpf']=='4') & (all_around_peak_data['condition']=='Sibs')]
# aligned_4t = all_around_peak_data.loc[(all_around_peak_data['dpf']=='4') & (all_around_peak_data['condition']=='Tau')]
# aligned_7s = all_around_peak_data.loc[(all_around_peak_data['dpf']=='7') & (all_around_peak_data['condition']=='Sibs')]
# aligned_7t = all_around_peak_data.loc[(all_around_peak_data['dpf']=='7') & (all_around_peak_data['condition']=='Tau')]

# # %%
# sns.distplot(aligned_4s['propBoutAligned_pitch'])
# sns.distplot(aligned_4t['propBoutAligned_pitch'])
# plt.show()

# sns.distplot(aligned_7s['propBoutAligned_pitch'])
# sns.distplot(aligned_7t['propBoutAligned_pitch'])
# plt.show()
