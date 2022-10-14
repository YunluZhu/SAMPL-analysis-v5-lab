'''
Plot bout features at different peak speeds (use BIN_NUM to specify bin numbers) as a function of time (index range specified by idxRANGE)
with posture neg and posture pos bouts separated
Change all_features for the features to plot
zeitgeber time? No
'''

#%%
# import sys
import os,glob
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from tqdm import tqdm
set_font_type()

# %%
# Paste root directory here
if_plot_by_speed = False
pick_data = 'tmp'
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'BT1_features_Tseries'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
    
BIN_NUM = 4
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-int(0.3*FRAME_RATE),peak_idx+int(0.2*FRAME_RATE)]

# %% features for plotting
all_features = [
    'propBoutAligned_speed', 
    # 'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    'heading_sub_pitch',
            # 'propBoutAligned_x',
            # 'propBoutAligned_y', 
            # 'propBoutInflAligned_angVel',
            # 'propBoutInflAligned_speed', 
            # 'propBoutAligned_angVel_hDn',
            # # 'propBoutAligned_speed_hDn', 
            # 'propBoutAligned_pitch_hDn',
            # # 'propBoutAligned_angVel_flat', 
            # # 'propBoutAligned_speed_flat',
            # # 'propBoutAligned_pitch_flat', 
            # 'propBoutAligned_angVel_hUp',
            # 'propBoutAligned_speed_hUp', 
            # 'propBoutAligned_pitch_hUp', 
    'ang_speed',
    'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    # 'xvel', 'yvel',
    'traj_cur',

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
all_cond1 = []
all_cond2 = []

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
                raw = raw.assign(idx=int(len(raw)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                exp_data = raw.loc[rows,:]
                exp_data = exp_data.assign(expNum = expNum,
                                      exp_id = condition_idx*100+expNum)
                grp = exp_data.groupby(np.arange(len(exp_data))//(idxRANGE[1]-idxRANGE[0]))
                angvel_smoothed = grp['propBoutAligned_angVel'].apply(
                    lambda x: savgol_filter(x, 11, 3)
                )
                exp_data = exp_data.assign(
                    # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                    traj_cur = angvel_smoothed/exp_data['propBoutAligned_speed'] * math.pi / 180
                )
                around_peak_data = pd.concat([around_peak_data,exp_data])
    # combine data from different conditions
    cond1 = all_conditions[condition_idx].split("_")[0]
    all_cond1.append(cond1)
    cond2 = all_conditions[condition_idx].split("_")[1]
    all_cond2.append(cond2)
    all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(dpf=cond1,
                                                                                            condition=cond2)])
all_around_peak_data = all_around_peak_data.assign(time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000)
# %% tidy data
all_cond1 = list(set(all_cond1))
all_cond1.sort()
all_cond2 = list(set(all_cond2))
all_cond2.sort()

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
# Peak data and pitch segmentation
T_INITIAL = -0.25 #s
T_PREP_200 = -0.2
T_PREP_150 = -0.15
T_PRE_BOUT = -0.10 #s
T_POST_BOUT = 0.1 #s
T_END = 0.2
T_MID_ACCEL = -0.05
T_MID_DECEL = 0.05
idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)


peak_data = all_around_peak_data.loc[all_around_peak_data['idx']==peak_idx].reset_index(drop=True)
peak_data = peak_data.assign(
    pitch_pre_bout = all_around_peak_data.loc[all_around_peak_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values,
)

yy = (all_around_peak_data.loc[all_around_peak_data['idx']==idx_post_bout,'propBoutAligned_y'].values - all_around_peak_data.loc[all_around_peak_data['idx']==idx_pre_bout,'propBoutAligned_y'].values)
absxx = np.absolute((all_around_peak_data.loc[all_around_peak_data['idx']==idx_post_bout,'propBoutAligned_x'].values - all_around_peak_data.loc[all_around_peak_data['idx']==idx_pre_bout,'propBoutAligned_x'].values))
epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
peak_data = peak_data.assign(
    traj_deviation = epochBouts_trajectory - peak_data['pitch_pre_bout'].values
)


peak_grp = peak_data.groupby(['expNum','condition'],as_index=False)

# assign by pitch
neg_pitch_bout_num = peak_data.loc[peak_data['pitch_pre_bout']<10,'bout_number']
pos_pitch_bout_num = peak_data.loc[peak_data['pitch_pre_bout']>10,'bout_number']
all_around_peak_data = all_around_peak_data.assign(
    pitch_dir = 'neg_pitch' 
)
all_around_peak_data.loc[all_around_peak_data['bout_number'].isin(pos_pitch_bout_num.values),'pitch_dir'] = 'pos_pitch'

# assign by traj deviation
neg_trajDev_bout_num = peak_data.loc[peak_data['traj_deviation']<0,'bout_number']
pos_trajDev_bout_num = peak_data.loc[peak_data['traj_deviation']>0,'bout_number']
all_around_peak_data = all_around_peak_data.assign(
    traj_deviation_dir = 'neg_traj_deviation' 
)
all_around_peak_data.loc[all_around_peak_data['bout_number'].isin(pos_trajDev_bout_num.values),'traj_deviation_dir'] = 'pos_traj_deviation'

# %% speed binned plots
# leave out the fastest bin
toplt = all_around_peak_data.loc[all_around_peak_data['speed_bin']<BIN_NUM-1,:]
# toplt = all_around_peak_data

if if_plot_by_speed:
    print('Plotting features binned by speed...')
    for feature_toplt in tqdm(all_features):
        p = sns.relplot(
        data = toplt, x = 'time_ms', y = feature_toplt, row = 'speed_bin',
        col='dpf',
        hue = 'condition',
        hue_order=all_cond2,
        style = 'dpf',
        style_order=all_cond1,
        # ci='sd',
        kind = 'line',aspect=3, height=2
        )
        p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
        plt.savefig(fig_dir+f"/{pick_data}_bySpd_{feature_toplt}.pdf",format='PDF')

# %% pitch neg and pos only
toplt = all_around_peak_data

print('Plotting features binned by pitch dir...')
for feature_toplt in tqdm(all_features):
    p = sns.relplot(
        data = toplt, x = 'time_ms', y = feature_toplt, row = 'pitch_dir',
        col='dpf',
        hue = 'condition',
        hue_order=all_cond2,
        style = 'dpf',
        style_order=all_cond1,
        # ci='sd',
        kind = 'line',aspect=3, height=2
    )
    p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
    plt.savefig(fig_dir+f"/{pick_data}_byPitch_{feature_toplt}.pdf",format='PDF')
    plt.close()
    
# %% pitch neg and pos longitudinally separate by condition
toplt = all_around_peak_data

print('Plotting features binned by pitch dir...')
for feature_toplt in tqdm(all_features):
    p = sns.relplot(
        data = toplt, x = 'time_ms', y = feature_toplt, 
        row = 'pitch_dir',
        col='condition', 
        hue = 'dpf',
        hue_order=all_cond1,
        style = 'dpf',
        style_order=all_cond1,
        # ci='sd',
        kind = 'line',aspect=3, height=2,
        facet_kws={'sharey': False, 'sharex': True}
    )
    
    p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
    plt.savefig(fig_dir+f"/{pick_data}_lgtdn_byPitch_{feature_toplt}.pdf",format='PDF')
    plt.close()
# %% traj neg and pos only
# toplt = all_around_peak_data

# print('Plotting features binned by traj deviation dir...')
# for feature_toplt in tqdm(all_features):
#     p = sns.relplot(
#     data = toplt, x = 'time_ms', y = feature_toplt, row = 'traj_deviation_dir',
#     col='dpf',
#     hue = 'condition',
#     hue_order=all_cond2,
#     style = 'dpf',
#     style_order=all_cond1,
#     ci=None,
#     kind = 'line',aspect=3, height=2
#     )
#     p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
#     plt.savefig(fig_dir+f"/{pick_data}_byTrajDeviation_{feature_toplt}.pdf",format='PDF')
#     plt.close()

# %% pitch neg and pos and Speed
if if_plot_by_speed:
    # # leave out the fastest bin
    toplt = all_around_peak_data.loc[all_around_peak_data['speed_bin']<BIN_NUM-1,:]

    print('Plotting features binned by speed with neg and pos pitch separated...')
    for feature_toplt in tqdm(all_features):
        g = sns.relplot(
        data = toplt, x = 'time_ms', y = feature_toplt, 
        row = 'speed_bin', col = 'pitch_dir',
        
        hue = 'condition',
        hue_order=all_cond2,
        style = 'dpf',
        style_order=all_cond1,
        kind = 'line',aspect=3, height=2
        )
        g.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
        plt.savefig(fig_dir+f"/{pick_data}_bySpdPitch   _{feature_toplt}.pdf",format='PDF')
        plt.close()