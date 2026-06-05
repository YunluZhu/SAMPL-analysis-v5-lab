'''
Plot bout features as a function of time separated by steering righting rotation (index range specified by idxRANGE)
Bouts are categorized into 4 types by direction (positive negative) of steering and righting rotation
'''

#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from tqdm import tqdm
import math
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
import matplotlib as mpl

##### Parameters to change #####
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)

# %%
root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)


# %%
# Paste root directory here
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]

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
    #         # 'propBoutAligned_x',
    #         # 'propBoutAligned_y', 
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
    # 'ang_speed',
    'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    # 'xvel', 'yvel',
    'fish_length',
    # 'traj_cur',

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
    cond0 = all_conditions[condition_idx].split("_")[0]
    cond1 = all_conditions[condition_idx].split("_")[1]
    
    if cond1 not in ['ld']:
        continue
    else:
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
                    for i in day_night_split(bout_time,'aligned_time',ztime=which_ztime, narrow_bin=True).index:
                        rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    exp_data = raw.loc[rows,:]
                    exp_data = exp_data.assign(expNum = expNum,
                                        exp_id = condition_idx*100+expNum)
                    grp = exp_data.groupby(np.arange(len(exp_data))//(idxRANGE[1]-idxRANGE[0]))
                    angvel_smoothed = grp['propBoutAligned_angVel'].apply(
                        lambda x: savgol_filter(x, 11, 3)
                    )
                    angvel_smoothed = angvel_smoothed.explode().values
                    exp_data = exp_data.assign(
                        # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                        traj_cur = angvel_smoothed/exp_data['propBoutAligned_speed'] * math.pi / 180
                    )
                    around_peak_data = pd.concat([around_peak_data,exp_data])
        # combine data from different conditions
        all_cond0.append(cond0)
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

all_around_peak_data = all_around_peak_data.assign(
    heading_sub_pitch = all_around_peak_data['propBoutAligned_instHeading']-all_around_peak_data['propBoutAligned_pitch'],
)

grp = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))
all_around_peak_data = all_around_peak_data.assign(
                                    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])),
                                    bout_number = grp.ngroup(),
                                )
all_around_peak_data = all_around_peak_data.assign(
                                    speed_bin = pd.cut(all_around_peak_data['peak_speed'],BIN_NUM,labels = np.arange(BIN_NUM))
                                )
print("speed buckets:")
print('--mean')
print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('mean'))
print('--min')
print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('min'))
print('--max')
print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('max'))


# %%
####################################
###### Plotting Starts Here ######
####################################

all_features = [
    'propBoutAligned_speed',
]
all_around_peak_data['abstime'] = all_around_peak_data.groupby(['cond0','cond1','expNum','bout_i','bout_number'])['propBoutAligned_time'].transform('first') 
all_around_peak_data_timed = day_night_split(all_around_peak_data,'abstime',ztime=which_ztime,narrow_bin=True)

median_df = all_around_peak_data_timed.groupby(['cond0','cond1','expNum','time_ms','ztime'])[all_features].median().reset_index()

#%%
toplt = median_df.loc[median_df['ztime'].isin(['day','night']), :]

toplt = toplt.loc[toplt['time_ms'].between(-250,200), :]
print('Plotting features binned by speed...')


for feature_toplt in tqdm(all_features):
    p = sns.relplot(
    data = toplt, x = 'time_ms', y = feature_toplt, 
    col='cond0',
    hue = 'ztime',
    errorbar = 'se',
    kind = 'line',aspect=1.6, height=2
    )
    p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
    plt.savefig(fig_dir+f"/{pick_data}_byCat_{feature_toplt}.pdf",format='PDF')

# %%
