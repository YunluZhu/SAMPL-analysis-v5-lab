'''
Plot bout features at different peak speeds (use BIN_NUM to specify bin numbers) as a function of time (index range specified by idxRANGE)
with posture neg and posture pos bouts separated
Change all_features for the features to plot
'''

#%%
# import sys
import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from tqdm import tqdm
import matplotlib as mpl

##### Parameters to change #####
pick_data = 'nMLF' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
if_strict_DayNightSplit = True # if True, only include bouts that are fully within the day or night period. If False, include bouts that may cross the day/night boundary.
# my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
# my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

    
#%%
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]

all_features = [
    'propBoutAligned_speed', 
    # 'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    'heading_sub_pitch',
    'propBoutAligned_x',
    'propBoutAligned_y', 
    'propBoutAligned_headx', 
    'propBoutAligned_heady',
    # 'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    'xvel', 
    'yvel',
    'fish_length',

    # 'boxNum',
    # 'bout_i',
    # 'frame_i',
]

# %%
# CONSTANTS
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
                raw = raw.assign(
                    # ang_speed=raw['propBoutAligned_angVel'].abs(),
                    yvel = raw['propBoutAligned_y'].diff()*FRAME_RATE,
                    xvel = raw['propBoutAligned_x'].diff()*FRAME_RATE,
                    # linear_accel = raw['propBoutAligned_speed'].diff(),
                    # ang_accel_of_SMangVel = raw['propBoutAligned_angVel'].diff(),
                                           )
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(idx=round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                day_night_split_res = day_night_split(bout_time, 'aligned_time', narrow_bin = if_strict_DayNightSplit, ztime=which_ztime)
                for i in day_night_split_res.index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                exp_data = raw.loc[rows,:]
                exp_data = exp_data.assign(
                    expNum = expNum,
                    ztime = np.repeat(day_night_split_res.ztime,(idxRANGE[1]-idxRANGE[0])).values,
                    exp_id = exp,
                    )
                grp = exp_data.groupby(np.arange(len(exp_data))//(idxRANGE[1]-idxRANGE[0]))
                angvel_smoothed = grp['propBoutAligned_angVel'].apply(
                    lambda x: savgol_filter(x, SMOOTH, 3)
                )
                # exp_data = exp_data.assign(
                    # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                    # traj_cur = angvel_smoothed/exp_data['propBoutAligned_speed'] * math.pi / 180
                # )
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

all_around_peak_data = all_around_peak_data.reset_index(drop=True)
peak_speed = all_around_peak_data.loc[all_around_peak_data.idx==peak_idx,'propBoutAligned_speed'],
grp = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))
all_around_peak_data = all_around_peak_data.assign(
                                    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])),
                                    bout_number = grp.ngroup(),
                                )

#%%
#NOTE now we end up with "all_around_peak_data" dataframe that contains all the data we need for plotting
#NOTE the bout_number column is the unique bout number for each bout, and the peak_speed column is the peak speed of that bout
# %%
# normalize fish length by the beginning of each bout
#define a function to pull out fish length by the beginning of each bout so we can run groupby apply
def get_fish_length_by_bout(df):
    return df.loc[(df.time_ms < -280),'fish_length'].median()
fish_length_by_bout = all_around_peak_data.groupby('bout_number').apply(get_fish_length_by_bout, include_groups=False).reset_index().rename(columns={0:'fish_length_by_bout'})
# assign the fish length by bout to the all_around_peak_data dataframe
df_timeseries_toana = pd.DataFrame()
df_timeseries_toana = all_around_peak_data.merge(fish_length_by_bout, on='bout_number', how='left', )
df_timeseries_toana = df_timeseries_toana.assign(fish_length_norm = df_timeseries_toana['fish_length']/df_timeseries_toana['fish_length_by_bout'])
# %%

# select 20 bouts and plot normalized fihs length for each bout

average_df = df_timeseries_toana.groupby(['time_ms','cond0','cond1','ztime'])['fish_length_norm'].mean().reset_index()
sns.lineplot(data=average_df, x='time_ms', y='fish_length_norm', hue='cond1')
# %%
