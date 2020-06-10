'''
This version uses "prop_bout_aligned" in the "propBoutAligned_speed" file
which includes swim speed during bouts
Output: plots of distribution of swim speed across different conditioins
Folder structure:
    root -|
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- abb_condition
          |- ...
Notes 
    - a: is the dpf, only supports digits for now
    - bb: is the light-dark condition. Ignored in plotting.
    - condition: is the condition of the exp, such as control/leasion/tau... Flexible length
    - the number of folders is also flexible 
'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
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


# %%
# CONSTANTS
root = "/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/test_data"
HEADING_LIM = 90
# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="whitegrid")

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    return df_day
# %%
# main
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_angVel = pd.DataFrame()
all_around_peak_angVel_cond = pd.DataFrame()
all_average_atk_ang = pd.DataFrame()
all_average_atk_ang_cond = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                angVel = pd.read_pickle(f"{exp_path}/prop_bout_aligned.pkl").loc[:,['propBoutAligned_angVel']].abs()
                # assign frame number, 51 frames per bout
                angVel = angVel.assign(idx=int(len(angVel)/51)*list(range(0,51)))
                
                # - get the index of the rows in angVel to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_pickle(f"{exp_path}/prop_bout2.pkl").loc[:,['aligned_time']]
                for i in bout_time.index:
                # # if only need day or night bouts:
                # for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*51+15,i*51+46)))
                all_around_peak_angVel = pd.concat([all_around_peak_angVel,angVel.loc[rows,:]])
                
                # - attack angle calculation
                angles = pd.read_pickle(f"{exp_path}/prop_bout_aligned.pkl").loc[:,['propBoutAligned_heading','propBoutAligned_pitch']]
                angles = angles.assign(idx=int(len(angles)/51)*list(range(0,51)))
                peak_angles = angles.loc[angles['idx']==30]
                peak_angles.dropna(inplace=True)
                peak_angles = peak_angles.loc[(peak_angles['propBoutAligned_heading']<HEADING_LIM) & (peak_angles['propBoutAligned_heading']>-HEADING_LIM)]
                
                average_atk_ang = np.nanmean(peak_angles['propBoutAligned_heading'] - peak_angles['propBoutAligned_pitch'])
                all_average_atk_ang = pd.concat([all_average_atk_ang, pd.DataFrame(data={'atkAng':average_atk_ang},index=[expNum])])
                
            all_around_peak_angVel_cond = pd.concat([all_around_peak_angVel_cond, all_around_peak_angVel.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])
            all_average_atk_ang_cond = pd.concat([all_average_atk_ang_cond, all_average_atk_ang.assign(dpf=all_conditions[condition_idx][0],condition=all_conditions[condition_idx][4:])])

# %%
# plot Angular speed as a function of time during bouts
defaultPlotting()
g = sns.relplot(x='idx',y='propBoutAligned_angVel', hue='condition', col='dpf', data=all_around_peak_angVel_cond, kind="line",ci='sd', err_style='band')
g.axes[0,0].set_ylim(0,)
plt.show()

# %%
defaultPlotting()
g = sns.violinplot(x='dpf',y='atkAng', hue='condition', data=all_average_atk_ang_cond)
plt.show()

# %%
