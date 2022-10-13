'''
plot correlation of features by time
1. corr of angular vel at each timpoint with preBoutPitch / atkAngle / trajectory deviation
2. corr of ins. trajectory at each timepoint with bout trajectory
trajectory deviation (trajecgtory residual) is defined as (bout_trajecgtory - pitch_pre_bout)

'''

#%%
# import sys
import os,glob
from pickle import FRAME
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_average, day_night_split)
from plot_functions.get_bout_kinetics import get_set_point
from tqdm import tqdm
import matplotlib as mpl
set_font_type()

# %%
def corr_calc(df, grp_cols, col1, col2, name):
    corr_calc = df.groupby(grp_cols).apply(
            lambda y: stats.pearsonr(
                y[col1].values,y[col2].values)[0]
                )
    corr_calc.name = name
    output = corr_calc.to_frame()
    return output
# %%
# Paste root directory here
# if_plot_by_speed = True
pick_data = 'wt_daylight'
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'BT2_corr_features'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
    
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-int(0.30*FRAME_RATE),peak_idx+int(0.22*FRAME_RATE)]
spd_bins = np.arange(5,25,4)

# %% features for plotting
all_features = [
    'propBoutAligned_speed', 
    'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    # 'heading_sub_pitch',
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

]

# %%
# CONSTANTS
# %%
T_INITIAL = -0.25 #s
T_PREP_200 = -0.2
T_PREP_150 = -0.15
T_PRE_BOUT = -0.10 #s
T_POST_BOUT = 0.1 #s
T_post_150 = 0.15
T_END = 0.2
T_MID_ACCEL = -0.05
T_MID_DECEL = 0.05


idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
idx_mid_accel = int(peak_idx + T_MID_ACCEL * FRAME_RATE)
idx_mid_decel = int(peak_idx + T_MID_DECEL * FRAME_RATE)
idx_end = int(peak_idx + T_END * FRAME_RATE)

idx_dur250ms = int(250/1000*FRAME_RATE)
idx_dur275ms = int(275/1000*FRAME_RATE)
# %%
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
                exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs(),
                                            yvel = exp_data['propBoutAligned_y'].diff()*FRAME_RATE,
                                            xvel = exp_data['propBoutAligned_x'].diff()*FRAME_RATE,
                                            linear_accel = exp_data['propBoutAligned_speed'].diff(),
                                            ang_accel_of_SMangVel = exp_data['propBoutAligned_angVel'].diff(),
                                           )
                # assign frame number, total_aligned frames per bout
                exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                exp_data = exp_data.assign(expNum = expNum,
                                           exp_id = condition_idx*100+expNum)
                around_peak_data = pd.concat([around_peak_data,exp_data.loc[rows,:]])
            # combine data from different conditions
            cond1 = all_conditions[condition_idx].split("_")[0]
            all_cond1.append(cond1)
            cond2 = all_conditions[condition_idx].split("_")[1]
            all_cond2.append(cond2)
            all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(dpf=cond1,
                                                                                            condition=cond2)])
all_around_peak_data = all_around_peak_data.assign(
    time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000,
)
# %% tidy data
all_cond1 = list(set(all_cond1))
all_cond1.sort()
all_cond2 = list(set(all_cond2))
all_cond2.sort()

all_around_peak_data = all_around_peak_data.reset_index(drop=True)
peak_speed = all_around_peak_data.loc[all_around_peak_data.idx==peak_idx,'propBoutAligned_speed'],

grp = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))
all_around_peak_data = all_around_peak_data.assign(
    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])),
    bout_number = grp.ngroup(),
                                )
all_around_peak_data = all_around_peak_data.assign(
                                    speed_bin = pd.cut(all_around_peak_data['peak_speed'],spd_bins,labels = np.arange(len(spd_bins)-1))
                                )
# %%
# cal bout features
corr_all = pd.DataFrame()
corr_bySpd = pd.DataFrame()
features_all = pd.DataFrame()
expNum = all_around_peak_data['expNum'].max()
# jackknife_idx = jackknife_resampling(np.array(list(range(expNum+1))))
idx_list = np.array(list(range(expNum+1)))
for excluded_exp, idx_group in enumerate(idx_list):
    group = all_around_peak_data.loc[all_around_peak_data['expNum'].isin([idx_group])]
    yy = (group.loc[group['idx']==idx_post_bout,'propBoutAligned_y'].values - group.loc[group['idx']==idx_pre_bout,'propBoutAligned_y'].values)
    absxx = np.absolute((group.loc[group['idx']==idx_post_bout,'propBoutAligned_x'].values - group.loc[group['idx']==idx_pre_bout,'propBoutAligned_x'].values))
    epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
    pitch_pre_bout = group.loc[group.idx==idx_pre_bout,'propBoutAligned_pitch'].values
    pitch_initial = group.loc[group.idx==idx_initial,'propBoutAligned_pitch'].values

    pitch_peak = group.loc[group.idx==int(peak_idx),'propBoutAligned_pitch'].values
    pitch_mid_accel = group.loc[group.idx==int(idx_mid_accel),'propBoutAligned_pitch'].values
    pitch_post_bout = group.loc[group.idx==idx_post_bout,'propBoutAligned_pitch'].values
    traj_peak = group.loc[group['idx']==peak_idx,'propBoutAligned_instHeading'].values
    rot_l_decel = pitch_post_bout - pitch_peak
    rot_l_accel = pitch_peak - pitch_pre_bout
    rot_early_accel = pitch_mid_accel - pitch_pre_bout
    bout_features = pd.DataFrame(data={'pitch_pre_bout':pitch_pre_bout,
                                       'rot_l_accel':rot_l_accel,
                                       'rot_l_decel':rot_l_decel,
                                       'rot_pre_bout':pitch_pre_bout - pitch_initial,
                                       'rot_early_accel':rot_early_accel,
                                       'pitch_initial':pitch_initial,
                                       'bout_traj':epochBouts_trajectory,
                                       'traj_peak':traj_peak, 
                                       'traj_deviation':epochBouts_trajectory-pitch_pre_bout,
                                       'atk_ang':traj_peak-pitch_peak,
                                       'spd_peak': group.loc[group.idx==int(peak_idx),'propBoutAligned_speed'].values,
                                       })
    features_all = pd.concat([features_all,bout_features],ignore_index=True)


    grp = group.groupby(np.arange(len(group))//(idxRANGE[1]-idxRANGE[0]))
    this_dpf_res = group.assign(
                                pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])),
                                pitch_initial = np.repeat(pitch_initial,(idxRANGE[1]-idxRANGE[0])),
                                bout_traj = np.repeat(epochBouts_trajectory,(idxRANGE[1]-idxRANGE[0])),
                                traj_peak = np.repeat(traj_peak,(idxRANGE[1]-idxRANGE[0])),
                                pitch_peak = np.repeat(pitch_peak,(idxRANGE[1]-idxRANGE[0])),
                                bout_number = grp.ngroup(),
                                )
    this_dpf_res = this_dpf_res.assign(
                                atk_ang = this_dpf_res['traj_peak']-this_dpf_res['pitch_peak'],
                                traj_deviation = this_dpf_res['bout_traj']-this_dpf_res['pitch_pre_bout'],
                                )
    
    null_initial_pitch = grp.apply(
        lambda group: group.loc[(group['idx']>(peak_idx-idx_dur275ms))&(group['idx']<(peak_idx-idx_dur250ms)), 
                                'propBoutAligned_pitch'].mean()
    )
    this_dpf_res = this_dpf_res.assign(
        relative_pitch_change = this_dpf_res['propBoutAligned_pitch'].values - np.repeat(null_initial_pitch,(idxRANGE[1]-idxRANGE[0])).values
    )
    
    # correlation calculation

    
    # Make a dictionary for correlation to be calculated
    corr_dict = {
        "angVel_corr_preBoutPitch":['pitch_pre_bout','propBoutAligned_angVel'],
        'angVel_corr_atkAng':['atk_ang','propBoutAligned_angVel'],
        'angVel_corr_trajDeviation':['traj_deviation','propBoutAligned_angVel'],
        'pitch_corr_traj':['propBoutAligned_pitch','propBoutAligned_instHeading'],
        'rotFromInitial_corr_trajDeviation':['relative_pitch_change','traj_deviation'],
        'rotFromInitial_corr_atkAng':['relative_pitch_change','atk_ang'],
    }
    
    cat_cols = ['condition','dpf']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_thisName = corr_calc(this_dpf_res, grp_cols, col1, col2, name)
        if i == 0:
            corr_res = corr_thisName
        else:
            corr_res = corr_res.join(corr_thisName)
    corr_all = pd.concat([corr_all, corr_res])
    corr_all = corr_all.assign(
        exp_num = excluded_exp,
    )
# # repeat by speed if needed
#     cat_cols = ['condition','dpf','spd_peak']
#     grp_cols = cat_cols + ['time_ms']
#     for i, name in enumerate(corr_dict):
#         [col1, col2] = corr_dict[name]
#         corr_thisName = corr_calc(this_dpf_res, grp_cols, col1, col2, name)
#         if i == 0:
#             corr_res = corr_thisName
#         else:
#             corr_res = corr_res.join(corr_thisName)
#     corr_bySpd = pd.concat([corr_bySpd, corr_res])
corr_all = corr_all.reset_index(drop=True)
corr_bySpd = corr_bySpd.reset_index(drop=True)

# %%
# ignore dir, ignore speed

# plot angvel corr w. preBoutPitch/atkAngle/trajectory residual
# plot inst traj. corr w. bout trajectory
for corr_which in ['corr_preBoutPitch','corr_atkAng','corr_traj_deviation','corr_instTraj']:
    g = sns.relplot(
        # col='condition',
        x='time_ms',
        y=corr_which,
        data=corr_all,
        kind='line',
        # col='dpf',
        hue='condition',
        ci='sd',
        aspect=1.2,
        height=3
        )
    g.set(xlim=(-250,200))
    if corr_which=='corr_instTraj':
        plt.savefig(fig_dir+f"/bout traj {corr_which}.pdf",format='PDF')
    else:
        plt.savefig(fig_dir+f"/angvel {corr_which}.pdf",format='PDF')

# %%
# by speed

for corr_which in ['corr_preBoutPitch','corr_atkAng','corr_traj_deviation','corr_instTraj']:
    g = sns.relplot(
        # hue='condition',
        x='time_ms',
        y=corr_which,
        data=corr_bySpd,
        col='speed_bin',
        kind='line',
        # col='dpf',
        hue='condition',
        ci='sd',
        aspect=1.2,
        height=3
        )
    g.set(xlim=(-250,200))
    if corr_which=='corr_instTraj':
        plt.savefig(fig_dir+f"/bySpd__bout traj {corr_which}.pdf",format='PDF')
    else:
        plt.savefig(fig_dir+f"/bySpd__angvel {corr_which}.pdf",format='PDF')
