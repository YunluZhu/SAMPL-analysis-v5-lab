'''
This version uses "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" file
which includes mean of body angles during IEI
Output: plots of distribution of body angles across different conditioins

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

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="whitegrid")


# %%
# main
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

bins = list(range(-90,95,5))

jack_y_all = pd.DataFrame()
ang_std_all = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_angles = pd.DataFrame()
            ang_std = []
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                df = pd.read_pickle(f"{exp_path}/prop_bout_IEI2.pkl")
                # get pitch
                body_angles = df.loc[:,['propBoutIEI_pitch']].rename(columns={'propBoutIEI_pitch':f'exp{expNum}'}).transpose()
                all_angles = pd.concat([all_angles, body_angles])
            # jackknife for the index
            jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
            # get the distribution of every jackknifed sample
            jack_y = pd.concat([pd.DataFrame(np.histogram(all_angles.iloc[idx_group].to_numpy().flatten(), bins=bins, density=True)) for idx_group in jackknife_idx], axis=1).transpose()
            jack_y_all = pd.concat([jack_y_all, jack_y.assign(age=all_conditions[condition_idx][0], condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)
            # get the std of every jackknifed sample
            for idx_group in jackknife_idx:
                ang_std.append(np.nanstd(all_angles.iloc[idx_group].to_numpy().flatten())) 
            ang_std = pd.DataFrame(ang_std)
            ang_std_all = pd.concat([ang_std_all, ang_std.assign(age=all_conditions[condition_idx][0], condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)

jack_y_all.columns = ['Probability','Posture (deg)','dpf','condition']
jack_y_all.sort_values(by=['condition'],inplace=True)
ang_std_all.columns = ['std(posture)','dpf','condition']                
ang_std_all.sort_values(by=['condition'],inplace=True)

# %%
# Stats
multi_comp = MultiComparison(ang_std_all['std(posture)'], ang_std_all['dpf']+ang_std_all['condition'])
print(multi_comp.tukeyhsd().summary())


# %%
defaultPlotting()

# setup 2 panels
f, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=0.3, hspace=None)
# Plot distribution
g = sns.lineplot(x='Posture (deg)',y='Probability', hue='condition', style='dpf', data=jack_y_all, ci='sd', err_style='band', ax=axes[0])
g.set_xlim([-100,100])

# Plot std of body angles
sns.swarmplot(x='dpf', y='std(posture)', hue='condition', data=ang_std_all,  ax=axes[1])
# sns.violinplot(x='dpf', y='std(posture)', hue='condition', data=ang_std_all, scale='area', cut=True, ax=axes[1])

plt.show()

# %%
