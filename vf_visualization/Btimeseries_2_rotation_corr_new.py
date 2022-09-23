'''

'''

#%%
# import sys
import os,glob
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
# Paste root directory here
# if_plot_by_speed = True
pick_data = 'wt_daylight'
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'corr_timeSeries_newAtk'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
    
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-int(0.27*FRAME_RATE),peak_idx+int(0.22*FRAME_RATE)]
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


HEADING_LIM = 90

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
    
    cat_cols = ['condition','dpf']
    grp_cols = cat_cols + ['time_ms']
    
    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['pitch_pre_bout'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_preBoutPitch'
    corr_angvel = corr_calc.to_frame()
    
    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['atk_ang'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_atkAng'
    corr_atkAng = corr_calc

    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['traj_deviation'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_trajResidual'
    corr_trajResidual = corr_calc
      
    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['bout_traj'].values,y['propBoutAligned_instHeading'].values)[0]
            )
    corr_calc.name = 'corr_instTraj'
    corr_instTraj = corr_calc
    
    corr_angvel = corr_angvel.join(corr_atkAng)
    corr_angvel = corr_angvel.join(corr_trajResidual)
    corr_angvel = corr_angvel.join(corr_instTraj)
    corr_angvel = corr_angvel.reset_index()
    corr_angvel = corr_angvel.assign(
                                    jackknife_num = np.mean(idx_group),
                                    )
    corr_all = pd.concat([corr_all,corr_angvel])
    
    
    
    
    
    
    # repeat for speed
    cat_cols = ['condition','dpf','speed_bin']
    grp_cols = cat_cols + ['time_ms']
    
    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['pitch_pre_bout'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_preBoutPitch'
    corr_angvel = corr_calc.to_frame()
    
    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['atk_ang'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_atkAng'
    corr_atkAng = corr_calc

    corr_calc = this_dpf_res.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(
            y['traj_deviation'].values,y['propBoutAligned_angVel'].values)[0]
            )
    corr_calc.name = 'corr_trajResidual'
    
    corr_trajResidual = corr_calc
      
    corr_angvel = corr_angvel.join(corr_atkAng)
    corr_angvel = corr_angvel.join(corr_trajResidual)
    corr_angvel = corr_angvel.reset_index()
    corr_angvel = corr_angvel.assign(
                                    jackknife_num = np.mean(idx_group),
                                    )
    corr_bySpd = pd.concat([corr_bySpd,corr_angvel])
corr_all = corr_all.reset_index(drop=True)
corr_bySpd = corr_bySpd.reset_index(drop=True)

# %%
# ignore dir, ignore speed

# righting corr
for corr_which in ['corr_preBoutPitch','corr_atkAng','corr_trajResidual','corr_instTraj']:
    g = sns.relplot(
        # hue='condition',
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

for corr_which in ['corr_preBoutPitch','corr_atkAng','corr_trajResidual']:
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
    plt.savefig(fig_dir+f"/bySpd__angvel {corr_which}.pdf",format='PDF')



# # %%
# print("Figure 5: Distribution of acceleration rotation")
# feature_to_plt = ['rot_l_accel']
# toplt = features_all

# for feature in feature_to_plt:
#     # let's add unit
#     if 'spd' in feature:
#         xlabel = feature + " (mm*s^-1)"
#     elif 'dis' in feature:
#         xlabel = feature + " (mm)"
#     else:
#         xlabel = feature + " (deg)"
#     plt.figure(figsize=(3,2))
#     upper = np.percentile(toplt[feature], 99.5)
#     lower = np.percentile(toplt[feature], 1)
    
#     g = sns.histplot(data=toplt, x=feature, 
#                         bins = 20, 
#                         element="poly",
#                         #  kde=True, 
#                         stat="density",
#                         pthresh=0.05,
#                         binrange=(lower,upper),
#                         color='grey'
#                         )
#     g.set_xlabel(xlabel)
#     sns.despine()
#     plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')
#     plt.close()
# # %%
# to_plt = features_all.loc[features_all['spd_peak']>7]
# BIN_WIDTH = 0.5
# AVERAGE_BIN = np.arange(int(lower),int(upper),BIN_WIDTH)
# binned_df, _ = distribution_binned_average(to_plt,by_col='rot_l_accel',bin_col='atk_ang',bin=AVERAGE_BIN)
# binned_df.columns=['Accel rotation','atk_ang']

# # %%
# # regression: attack angle vs accel rotation
# xlabel = "Acceleration rot (deg)"
# ylabel = 'Attack angle (deg)'
# plt.figure(figsize=(4,4))
 
# g = sns.lmplot(
#     data = to_plt,
#     x = 'rot_l_accel',
#     y = 'atk_ang',
#     x_bins=np.arange(int(lower),int(upper),3),
#     x_ci=95,
#     markers='+'
# )
# g.map(sns.lineplot,data=binned_df,
#       x='Accel rotation',
#       y='atk_ang')

# # g.set_xlabel(xlabel)
# # g.set_ylabel(ylabel)
# sns.despine()
# plt.savefig(fig_dir+f"/atkAng vs accelRot.pdf",format='PDF')
# stats.pearsonr(to_plt['rot_l_accel'],to_plt['atk_ang'])[0]
# # %%

# %%
