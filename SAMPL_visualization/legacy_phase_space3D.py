'''
Based on dimRed_phaseSpace, omit yvel

top feature names
      feature1     feature2     feature3     feature4     feature5
0     pitch_79     pitch_78     pitch_80     pitch_77     pitch_81
1  xvel_adj_90  xvel_adj_91  xvel_adj_89  xvel_adj_92  xvel_adj_88
2   angvel_107   angvel_108   angvel_106   angvel_109   angvel_105
3  xvel_adj_61  xvel_adj_60  xvel_adj_64  xvel_adj_63  xvel_adj_62
4      yvel_57      yvel_56      yvel_58      yvel_55      yvel_59
5  xvel_adj_57  xvel_adj_58  xvel_adj_56  xvel_adj_59  xvel_adj_55
6  xvel_adj_73  xvel_adj_72  xvel_adj_71  xvel_adj_74  xvel_adj_70
7     yvel_111     yvel_112     yvel_110     yvel_113     yvel_109
8      yvel_50    angvel_85    angvel_84      yvel_51    angvel_86

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
import itertools
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

sample_number = 3

grouped = all_around_peak_data.groupby('bout_num')
gnum = np.arange(grouped.ngroups)
np.random.shuffle(gnum)
df_toplt = all_around_peak_data[grouped.ngroup().isin(gnum[:sample_number])]

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
for i, (x, y) in enumerate(itertools.combinations(['propBoutAligned_pitch', 'xvel_adj', 'propBoutAligned_angVel'], 2)):
    sns.lineplot(
        data = df_toplt, 
        ax = axes[i], 
        x = x, 
        y = y, 
        hue = 'bout_num',
        sort = False,
        palette=sns.color_palette("muted", sample_number),
        )
plt.show()


# %%
# x_feature, y_feature = 'xvel_adj','yvel'
# x_feature, y_feature = 'propBoutAligned_angVel','propBoutAligned_pitch'
# x_feature, y_feature = 'xvel_adj','propBoutAligned_pitch'

# g = sns.lineplot(
#     data = df_toplt,
#     x = x_feature,
#     y = y_feature,
#     hue = 'bout_num',
#     # kind='line',
#     palette=sns.color_palette("muted", sample_number),
#     alpha = 1,
#     sort=False)
# g.set_aspect('equal', adjustable=None)

# # %% overlay multiple
# sample_number = 5000

# grouped = all_around_peak_data.groupby('bout_num')
# gnum = np.arange(grouped.ngroups)
# np.random.shuffle(gnum)
# df_toplt = all_around_peak_data[grouped.ngroup().isin(gnum[:sample_number])]

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

# %% line in 3d
# sns.set_style("whitegrid")
sample_number = 1

grouped = all_around_peak_data.groupby('bout_num')
gnum = np.arange(grouped.ngroups)
np.random.shuffle(gnum)
df_toplt = all_around_peak_data[grouped.ngroup().isin(gnum[:sample_number])]

Z1 = df_toplt['propBoutAligned_pitch'].values
X1 = df_toplt['xvel_adj'].values
Y1 = df_toplt['propBoutAligned_angVel'].values

plt.figure(figsize=(6,5))
axes = plt.axes(projection='3d')
axes.plot3D(X1,Y1,Z1)
# keeps padding between figure elements
plt.tight_layout()
plt.show()
# %%
