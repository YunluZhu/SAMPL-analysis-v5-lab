'''
a copy of Fig2_4_parameters_alignedAverage
plots parameters of bouts aligned at the time of the peak speed.
Input directory needs to be a folder containing analyzed dlm data.
'''

#%%
from cmath import exp
import os
from pickle import FRAME
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.get_index import (get_index)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from scipy.signal import savgol_filter
from scipy import stats
from tqdm import tqdm

# %%
set_font_type()
print('- Figure 5: angvel time series and time of peak angvel')

# choose the time duration to plot. 
# total aligned duration = [-0.5, 0.4] (sec) around time of peak speed
# [-0.3,0.2] (sec) around peak speed is recommended 

BEFORE_PEAK = 0.3 # s
AFTER_PEAK = 0.2 #s

# %% features for plotting
# below are all the properties can be plotted. 
all_features = {
    'propBoutAligned_speed':'speed (mm*s-1)', 
    # 'propBoutAligned_linearAccel':'linear accel (mm*s-2)',
    'propBoutAligned_pitch':'pitch (deg)', 
    'propBoutAligned_angVel':'ang vel (deg*s-1)',   # smoothed angular velocity
    'propBoutAligned_angSpeed': 'ang speed (deg*s-1)', 
    'propBoutAligned_accel':'ang accel (deg*s-2)',    # angular accel calculated using raw angular vel
    'abs_ang_accel_of_angvel':'abs_ang_accel_of_angvel',
    'propBoutAligned_angVel_sm':'propBoutAligned_angVel_sm',
    'adj_ang_accel':'adj_ang_accel',
    'adj_angvel':'adj_angvel',
    
    # 'propBoutInflAligned_accel',
    # 'propBoutAligned_instHeading':'propBoutAligned_instHeading',
    # 'propBoutAligned_x':'x position (mm)',
    # 'propBoutAligned_y':'y position (mm)', 
    # 'propBoutInflAligned_angVel',
    # 'propBoutInflAligned_speed': 'ang speed (deg*s-1)',   #
    # 'propBoutAligned_angVel_hDn',
    # # 'propBoutAligned_speed_hDn', 
    # 'propBoutAligned_pitch_hDn',
    # # 'propBoutAligned_angVel_flat', 
    # # 'propBoutAligned_speed_flat',
    # # 'propBoutAligned_pitch_flat', 
    # 'propBoutAligned_angVel_hUp',
    # 'propBoutAligned_speed_hUp', 
    # 'propBoutAligned_pitch_hUp', 
}
# %%
# Select data and create figure folder
pick_data = 'wt_daylight'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} atk_ang fin_body_ratio'
folder_dir5 = get_figure_dir('Fig_5')
fig_dir5 = os.path.join(folder_dir5, folder_name)
try:
    os.makedirs(fig_dir5)
except:
    pass
# %%

# get the index for the time of peak speed, and total time points for each aligned bout
peak_idx, total_aligned = get_index(FRAME_RATE)
idx_dur300ms = round_half_up(0.3*FRAME_RATE)
idx_dur250ms = round_half_up(0.25*FRAME_RATE)
idx_pre_bout = round_half_up(peak_idx - 0.1 * FRAME_RATE)
idx_mid_accel = round_half_up(peak_idx - 0.05 * FRAME_RATE)
idx_initial = round_half_up(peak_idx - 0.25 * FRAME_RATE)

all_conditions = []
folder_paths = []
all_cond0 = []
all_cond0 = []
exp_data_all = pd.DataFrame()



for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)
# calculate indicies
idxRANGE = [peak_idx-round_half_up(BEFORE_PEAK*FRAME_RATE),peak_idx+round_half_up(AFTER_PEAK*FRAME_RATE)]

for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            this_cond_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                rows = []
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                ang_accel_of_angvel = np.diff(savgol_filter(raw['propBoutAligned_angVel'].values, 11, 3),prepend=np.array([np.nan]))*FRAME_RATE
                abs_ang_accel_of_angvel = np.absolute(ang_accel_of_angvel)
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(
                    idx = round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)),
                    ang_accel_of_angvel = ang_accel_of_angvel,
                    abs_ang_accel_of_angvel = abs_ang_accel_of_angvel,
                    )
                # - get the index of the rows in exp_data to keep
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                selected_range = raw.loc[rows,:]
                # calculate angular speed (smoothed)
                grp = selected_range.groupby(np.arange(len(selected_range))//(idxRANGE[1]-idxRANGE[0]))
                propBoutAligned_angVel = grp['propBoutAligned_pitch'].apply(
                    lambda grp_pitch: np.diff(savgol_filter(grp_pitch, 7, 3),prepend=np.array([np.nan]))*FRAME_RATE,
                )
                propBoutAligned_angVel = propBoutAligned_angVel.apply(pd.Series).T.melt()
                # assign angvel and ang speed
                selected_range = selected_range.assign(
                    propBoutAligned_angVel_sm = propBoutAligned_angVel['value'].values,
                    propBoutAligned_angSpeed = np.absolute(propBoutAligned_angVel['value'].values),
                )
                # calculate angspeed with sign adjusted by sign of trajectory deviation from posture
                traj_deviation = grp.apply(
                    lambda group: group.loc[group['idx']==peak_idx,'propBoutAligned_instHeading'].values - \
                        group.loc[group['idx']==idx_initial,'propBoutAligned_pitch'].values
                ).apply(pd.Series).melt()
                traj_deviation = traj_deviation['value']
                adj_by_traj_deviation = traj_deviation/np.absolute(traj_deviation)

                bout_traj = selected_range.loc[selected_range['idx']==peak_idx,'propBoutAligned_instHeading']
                adj_by_bout_traj =  bout_traj/np.absolute(bout_traj)
                
                accel_angvel_mean = grp.apply(
                    lambda group: group.loc[(group['idx']>idx_pre_bout)&(group['idx']<idx_mid_accel), 
                                            'propBoutAligned_angVel'].mean()
                )
                adj_by_angvel = accel_angvel_mean/np.absolute(accel_angvel_mean)
                #|||||||||||||||||||||||||
                adj_by_which = adj_by_traj_deviation #adj_by_angvel#adj_by_traj_deviation #  #
                #|||||||||||||||||||||||||
                
                adj_angvel = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)
                
                # calculate angaccel with sign adjusted (positive value before 0ms, negative value after)
                # angAccel_adj_sign = grp.apply(
                #     lambda group: group.loc[(group['idx']>idx_pre_bout)&(group['idx']<idx_mid_accel), 
                #                             'ang_accel_of_angvel'].mean()
                # )
                # angAccel_adj_sign = angAccel_adj_sign/np.absolute(angAccel_adj_sign)
                adj_ang_accel = selected_range['ang_accel_of_angvel'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)
                
                
                selected_range = selected_range.assign(
                    adj_ang_accel = adj_ang_accel,
                    adj_angvel = adj_angvel,
                )

                columns_to_pass = list(all_features.keys()) + ['idx']
                exp_data = selected_range.loc[:,columns_to_pass]
                exp_data = exp_data.rename(columns=all_features)

                exp_data = exp_data.assign(
                    time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                    expNum = expNum)
                this_cond_data = pd.concat([this_cond_data,exp_data.loc[rows,:]])
            
    cond1 = all_conditions[condition_idx].split("_")[0]
    cond1 = all_conditions[condition_idx].split("_")[1]
    all_cond0.append(cond1)
    all_cond0.append(cond1)
    
    this_cond_data = this_cond_data.reset_index(drop=True)
    this_cond_data = this_cond_data.assign(
        cond1 = cond1,
        cond1 = cond1,
    )
    exp_data_all = pd.concat([exp_data_all,this_cond_data], ignore_index=True)
            
# %%
separation_posture = 10

peak_speed = exp_data_all.loc[exp_data_all.idx==peak_idx,'speed (mm*s-1)']
pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==idx_pre_bout,'pitch (deg)']

grp = exp_data_all.groupby(np.arange(len(exp_data_all))//(idxRANGE[1]-idxRANGE[0]))
exp_data_all = exp_data_all.assign(
                                    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])).values,
                                    pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])).values,
                                    bout_number = grp.ngroup(),
                                )
exp_data_all = exp_data_all.assign(
                                    direction = pd.cut(exp_data_all['pitch_pre_bout'],[-90,separation_posture,90],labels = ['Nose-down', 'Nose-up'])
                                )
# %%
# plot timeseries
all_features_toplt = {
    # 'abs_ang_accel_of_angvel':'abs_ang_accel_of_angvel',
    # 'adj_ang_accel':'adj_ang_accel',
    # 'propBoutAligned_accel':'ang accel (deg*s-2)',
    'adj_angvel':'adj_angvel',

    }

for feature_toplt in tqdm(list(all_features_toplt.values())):
    p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt,
            kind = 'line',aspect=3, height=2, 
            ci='sd',
            row = 'cond1', col='cond1',
            # hue='direction',
            # ci=None
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    plt.savefig(os.path.join(fig_dir5, f"angvel_timeSeries ± SD.pdf"),format='PDF')
    plt.show()

# %%
# calculate time of max adj ang speed, mean of each exp, then average
initial_bound = -0.25 #ms
df_tocalc = exp_data_all.loc[exp_data_all['time_ms']<200]
idx_mean_max = df_tocalc.groupby(['bout_number'])['adj_angvel'].apply(
    lambda y: np.argmax(savgol_filter(y,7,3)[round_half_up((0.3+initial_bound)*FRAME_RATE):])
)
time_by_bout_max = ((idx_mean_max/166 + initial_bound)*1000).reset_index()
condition_match = exp_data_all.groupby(['bout_number'])[['cond1','cond1']].head(1)

time_by_bout_max.columns = ['bout_number','time_of_peak_adjAngVel (ms)']
time_by_bout_max = time_by_bout_max.assign(
    cond1 = condition_match['cond1'].values,
    cond1 = condition_match['cond1'].values,
)
time_of_peak__byBout_mean = time_by_bout_max['time_of_peak_adjAngVel (ms)'].values.mean()

plt.figure(figsize=(3,2))
g = sns.pointplot(
    data = time_by_bout_max,
    y = 'cond1',
    x = 'time_of_peak_adjAngVel (ms)',
    ci='sd', 
)
sns.despine()
# g.set_xlim(None,0)
plt.savefig(os.path.join(fig_dir5,f"time_of_peak_Angvel.pdf"),format='PDF')
plt.show()
print(f"Time of the peak angular velocity by Exp mean = {time_of_peak__byBout_mean}±{time_by_bout_max['time_of_peak_adjAngVel (ms)'].values.std()} ms")

# %%
toplt = time_by_bout_max
feature = 'time_of_peak_adjAngVel (ms)'
upper = np.percentile(toplt[feature], 99)
lower = np.percentile(toplt[feature], 1)
plt.figure(figsize=(3,2))
p = sns.histplot(data=toplt, x=feature, 
                    bins = 18, 
                    element="poly",
                    #  kde=True, 
                    stat="probability",
                    pthresh=0.05,
                    binrange=(lower,upper),
                    color='grey'
                    )
sns.despine()
plt.savefig(os.path.join(fig_dir5,f"time_of_peak_angvel_bybout_hist.pdf"),format='PDF')
plt.show()

# %%

# %%
# calculate time of max angaccel, mean of each exp, then average
mean_angAccel = exp_data_all.groupby(['time_ms','expNum','cond1','cond1'])['adj_angvel'].mean().reset_index()
mean_angAccel = mean_angAccel.loc[mean_angAccel['time_ms']<0]
idx_mean_max = mean_angAccel.groupby(['expNum','cond1','cond1'])['adj_angvel'].apply(
    lambda y: np.argmax(y)
)
time_by_bout_max = ((idx_mean_max/166 - 0.3)*1000).reset_index()
# condition_match = exp_data_all.groupby(['expNum','cond1'])['cond1','cond1'].head(1)

time_by_bout_max.columns = ['expNum','cond1','cond1','time_adj_angvel (ms)']
# time_by_bout_max = time_by_bout_max.assign(
#     cond1 = condition_match['cond1'].values,
#     cond1 = condition_match['cond1'].values,
# )
time_of_peak__byBout_mean = time_by_bout_max['time_adj_angvel (ms)'].values.mean()

plt.figure(figsize=(3,2))
g = sns.pointplot(
    data = time_by_bout_max,
    y = 'cond1',
    x = 'time_adj_angvel (ms)',
    ci='sd', 
)
sns.despine()
g.set_xlim(None,0)
plt.savefig(os.path.join(fig_dir5,f"time_adj_angvel.pdf"),format='PDF')
plt.show()
print(f"Time of the peak angular accel by Exp mean = {time_of_peak__byBout_mean}±{time_by_bout_max['time_adj_angvel (ms)'].values.std()} ms")


# %%
