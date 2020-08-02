'''
This version uses "prop_bout_aligned" in the "propBoutAligned_speed" file
which includes swim speed during bouts
Output: plots of distribution of swim speed across different conditioins
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
# root = "/Users/yunluzhu/Lab/Lab2/Data/VF/vf_data/combined_TTau_data"
bins = list(range(1,30,1))

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")


# %%
# main function
def bout_speed_aligned_jacknife(root):
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    jack_y_all = pd.DataFrame()
    ang_std_all = pd.DataFrame()
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                all_speed = pd.DataFrame()
                ang_std = []
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    df = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                    # get pitch
                    swim_speed = df.loc[:,['propBoutAligned_speed']].rename(columns={'propBoutAligned_speed':f'exp{expNum}'}).transpose()
                    all_speed = pd.concat([all_speed, swim_speed])
                # jackknife for the index
                jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
                # get the distribution of every jackknifed sample
                jack_y = pd.concat([pd.DataFrame(np.histogram(all_speed.iloc[idx_group].to_numpy().flatten(), bins=bins, density=True)) for idx_group in jackknife_idx], axis=1).transpose()
                jack_y_all = pd.concat([jack_y_all, jack_y.assign(age=all_conditions[condition_idx][0], condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)
                # # get the std of every jackknifed sample
                # for idx_group in jackknife_idx:
                #     ang_std.append(np.nanstd(all_speed.iloc[idx_group].to_numpy().flatten())) 
                # ang_std = pd.DataFrame(ang_std)
                # ang_std_all = pd.concat([ang_std_all, ang_std.assign(age=all_conditions[condition_idx][0], condition=all_conditions[condition_idx][4:])], axis=0, ignore_index=True)

    jack_y_all.columns = ['Probability','swim_speed','dpf','condition']
    jack_y_all.sort_values(by=['dpf','condition'],inplace=True)

    # %%
    defaultPlotting()
    g = sns.lineplot(x='swim_speed',y='Probability', hue='condition', style='dpf', data=jack_y_all, ci='sd', err_style='band')

    plt.show()
