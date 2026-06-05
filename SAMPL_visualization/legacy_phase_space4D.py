'''
Generate 4D phase space using 'xvel_adj', 'yvel', 'propBoutAligned_pitch', 'propBoutAligned_angVel'
Save phase parameters for one bout
which can be view using ManyLands: https://amirkhanov.net/manylands/
'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from plot_functions.plt_functions import plt_categorical_grid
from plot_functions.get_bout_kinetics import get_bout_kinetics
import math
from plot_functions.plt_tools import round_half_up 
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
# from statsmodels.stats.multicomp import MultiComparison
# from sklearn import metrics
# import scipy.stats as st
# from tqdm import tqdm
# import matplotlib as mpl

# from pymer4.models import Lm
# from pymer4.models import Lmer

set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'depth_7d'
which_ztime = 'all'
spd_bins = np.arange(5,30,5)

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} phase space'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
# Paste root directory here
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.2*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]

# %% features for plotting
all_features = [
    'propBoutAligned_speed', 
    'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    # 'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    # # 'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    # 'heading_sub_pitch',
            'propBoutAligned_x',
            'propBoutAligned_y', 
    #         # 'propBoutInflAligned_angVel',
    #         # 'propBoutInflAligned_speed', 
    #         # 'propBoutAligned_angVel_hDn',
    #         # # 'propBoutAligned_speed_hDn', 
    #         # 'propBoutAligned_pitch_hDn',
    #         # # 'propBoutAligned_angVel_flat', 
    #         # # 'propBoutAligned_speed_flat',
    #         # # 'propBoutAligned_pitch_flat', 
    #         # 'propBoutAligned_angVel_hUp',
    #         # 'propBoutAligned_speed_hUp', 
    #         # 'propBoutAligned_pitch_hUp', 
    # # 'ang_speed',
    'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    # 'xvel', 'yvel',
    'fish_length',
    'traj_cur',

]

# %%
# CONSTANTS
BIN_NUM = 4  # number of speed bins
SMOOTH = 11
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()
all_cond0 = []
all_cond1 = []

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
                raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                raw = raw.assign(ang_speed=raw['propBoutAligned_angVel'].abs(),
                                            yvel = raw['propBoutAligned_y'].diff()*FRAME_RATE,
                                            xvel = raw['propBoutAligned_x'].diff()*FRAME_RATE,
                                            linear_accel = raw['propBoutAligned_speed'].diff(),
                                            ang_accel_of_SMangVel = raw['propBoutAligned_angVel'].diff(),
                                           )
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(idx=round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time',ztime=which_ztime).index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                exp_data = raw.loc[rows,:]
                exp_data = exp_data.assign(expNum = expNum,
                                      exp_id = condition_idx*100+expNum)
                
                grp = exp_data.groupby(np.arange(len(exp_data))//(idxRANGE[1]-idxRANGE[0]))
                
                # angvel_smoothed = grp['propBoutAligned_angVel'].apply(
                #     lambda x: savgol_filter(x, 11, 3)
                # )
                # angvel_smoothed = angvel_smoothed.explode().values
                
                xvel_adj = grp['xvel'].apply(
                    lambda x: x * (np.absolute(x.mean())/x.mean())  # adjust sign of x velocity, make it positive
                )
                
                exp_data = exp_data.assign(
                    # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                    # traj_cur = angvel_smoothed/exp_data['propBoutAligned_speed'] * math.pi / 180,
                    xvel_adj = xvel_adj
                )
                around_peak_data = pd.concat([around_peak_data,exp_data])
    # combine data from different conditions
    cond0 = all_conditions[condition_idx].split("_")[0]
    all_cond0.append(cond0)
    cond1 = all_conditions[condition_idx].split("_")[1]
    all_cond1.append(cond1)
    all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(cond0=cond0,
                                                                                            cond1=cond1)])
all_around_peak_data = all_around_peak_data.assign(time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000)
# %% tidy data
all_cond0 = list(set(all_cond0))
all_cond0.sort()
all_cond1 = list(set(all_cond1))
all_cond1.sort()

all_around_peak_data = all_around_peak_data.assign(
    velocity = all_around_peak_data['propBoutAligned_speed'] * (all_around_peak_data['propBoutAligned_instHeading'].abs()/all_around_peak_data['propBoutAligned_instHeading']),
    bout_num = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0])).ngroup(),
)
# %%
# parameters to describe state of fish: "propBoutAligned_pitch", "propBoutAligned_angVel", "yvel", "xvel"

sample_number = 1

grouped = all_around_peak_data.groupby('bout_num')
gnum = np.arange(grouped.ngroups)
np.random.shuffle(gnum)
df_toplt = all_around_peak_data[grouped.ngroup().isin(gnum[:sample_number])]


# x_feature, y_feature = 'xvel_adj','yvel'
x_feature, y_feature = 'propBoutAligned_angVel','propBoutAligned_pitch'

g = sns.lineplot(
    data = df_toplt,
    x = x_feature,
    y = y_feature,
    hue = 'bout_num',
    # kind='line',
    palette=sns.color_palette("muted", sample_number),
    alpha = 1,
    sort=False)
# g.set_aspect('equal', adjustable=None)

# plt.figure()
# g = sns.lineplot(
#     data = df_toplt,
#     x = 'propBoutAligned_x',
#     y = 'propBoutAligned_y',
#     hue = 'bout_num',
#     # kind='line',
#     palette=sns.color_palette("muted", sample_number),
#     alpha = 1,
#     sort=False,
#     )
# g.set_xlim(-18, 18)
# g.set_ylim(-18, 18)
# g.set_aspect('equal', adjustable=None)

df_toplt = df_toplt.assign(time_adj = df_toplt['time_ms'].values+200)
np.savetxt(f'{fig_dir}\phase_space.txt',df_toplt[['time_adj', 'xvel_adj', 'yvel', 'propBoutAligned_pitch', 'propBoutAligned_angVel']].values)
# which can be viewed using: https://amirkhanov.net/manylands-old/full-version/
# DOI: https://doi.org/10.1111/cgf.13828


# %% overlay multiple
# sample_number = 5000

# grouped = all_around_peak_data.groupby('bout_num')
# gnum = np.arange(grouped.ngroups)
# np.random.shuffle(gnum)
# df_toplt = all_around_peak_data[grouped.ngroup().isin(gnum[:sample_number])]


# x_feature, y_feature = 'propBoutAligned_angVel','propBoutAligned_pitch'

# g = sns.lineplot(
#     data = df_toplt,
#     x = x_feature,
#     y = y_feature,
#     hue = 'bout_num',
#     # kind='line',
#     # palette=sns.color_palette("muted", sample_number),
#     alpha = 0.01,
#     sort=False)
# # g.set_aspect('equal', adjustable=None)
