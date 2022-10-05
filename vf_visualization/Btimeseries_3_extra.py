'''
a copy of Fig2_4_parameters_alignedAverage
plots parameters of bouts aligned at the time of the peak speed.
Input directory needs to be a folder containing analyzed dlm data.
'''

#%%
from cmath import exp
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.get_index import (get_index)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from scipy.signal import savgol_filter

from tqdm import tqdm

# %%

print('- Figure 2: Bout parameter time series ± SD')

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
    'absolute_pitch_change':'absolute pitch chg (deg)',
    # 'propBoutAligned_accel':'ang accel (deg*s-2)',    # angular accel calculated using raw angular vel
    # 'propBoutInflAligned_accel',
    # 'propBoutAligned_instHeading', 
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
pick_data = 'all_7dd'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} parameter time series'
fig_dir2 = os.path.join(get_figure_dir('Fig_2'), folder_name)
fig_dir4 = os.path.join(get_figure_dir('Fig_4'), folder_name)

try:
    os.makedirs(fig_dir2)
except:
    pass
try:
    os.makedirs(fig_dir4)
except:
    pass
# %%

# get the index for the time of peak speed, and total time points for each aligned bout
peak_idx, total_aligned = get_index(FRAME_RATE)
idx_dur300ms = int(0.3*FRAME_RATE)
idx_dur250ms = int(0.25*FRAME_RATE)
all_conditions = []
folder_paths = []
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)
# calculate indicies
idxRANGE = [peak_idx-int(BEFORE_PEAK*FRAME_RATE),peak_idx+int(AFTER_PEAK*FRAME_RATE)]

for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            exp_data_all = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                rows = []
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(
                    idx = int(len(raw)/total_aligned)*list(range(0,total_aligned)),
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
                propBoutAligned_angSpeed = grp['propBoutAligned_pitch'].apply(
                    lambda grp_pitch: np.absolute(np.diff(savgol_filter(grp_pitch, 7, 3),prepend=np.array([np.nan]))*FRAME_RATE),
                )
                propBoutAligned_angSpeed = propBoutAligned_angSpeed.apply(pd.Series).T.melt()
                selected_range = selected_range.assign(
                    propBoutAligned_angSpeed = propBoutAligned_angSpeed['value'].values,
                )

                # calculate absolute pitch change, null pitch = mean pitch between -250 to -200 ms
                null_initial_pitch = grp.apply(
                    lambda group: group.loc[(group['idx']>(peak_idx-idx_dur300ms))&(group['idx']<(peak_idx-idx_dur250ms)), 
                                            'propBoutAligned_pitch'].mean()
                )
                selected_range = selected_range.assign(
                    absolute_pitch_change = selected_range['propBoutAligned_pitch'] - np.repeat(null_initial_pitch,(idxRANGE[1]-idxRANGE[0])).values
                )
                columns_to_pass = list(all_features.keys()) + ['idx']
                exp_data = selected_range.loc[:,columns_to_pass]
                exp_data = exp_data.rename(columns=all_features)

                exp_data = exp_data.assign(
                    time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                    expNum = expNum)
                exp_data_all = pd.concat([exp_data_all,exp_data.loc[rows,:]])
            exp_data_all = exp_data_all.reset_index(drop=True)

# %%
# get bout features
# all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# assign up and down
separation_posture = 10

peak_speed = exp_data_all.loc[exp_data_all.idx==peak_idx,'speed (mm*s-1)']
pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==int(peak_idx - 0.1 * FRAME_RATE),'pitch (deg)']

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
print("Figure 2: time series")
set_font_type()
for feature_toplt in tqdm(list(all_features.values())):
    p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt,
            hue='direction',
            kind = 'line',aspect=3, height=2, ci='sd',
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    plt.savefig(os.path.join(fig_dir2, f"{feature_toplt}_timeSeries_up_dn ± SD.pdf"),format='PDF')

for feature_toplt in tqdm(list(all_features.values())):
    p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt,
            kind = 'line',aspect=3, height=2, ci='sd'
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    plt.savefig(os.path.join(fig_dir2, f"{feature_toplt}_timeSeries ± SD.pdf"),format='PDF')

# # %%
# # calculate time of the max ang speed by dir
# mean_angSpd = exp_data_all.groupby(['time_ms','direction','expNum'])['ang speed (deg*s-1)'].mean().reset_index()
# idx_mean_maxAngSpd = mean_angSpd.groupby(['expNum','direction'])['ang speed (deg*s-1)'].apply(
#     lambda y: np.argmax(y)
# )
# time_by_dir_maxAngSpd = ((idx_mean_maxAngSpd/166 - 0.3)*1000).reset_index()

# time_by_dir_maxAngSpd.columns = ['expNum','direction','time_of_peak_angSpd (ms)']
# time_of_peak_angSpd_byDir = time_by_dir_maxAngSpd.groupby('direction')['time_of_peak_angSpd (ms)'].mean()
# time_of_peak_angSpd_byDirmean = time_by_dir_maxAngSpd['time_of_peak_angSpd (ms)'].values.mean()
# plt.figure(figsize=(3,2))
# g = sns.pointplot(
#     data = time_by_dir_maxAngSpd,
#     x = 'time_of_peak_angSpd (ms)',
#     y = 'direction',
#     hue='direction',
# )
# sns.despine()
# plt.savefig(os.path.join(fig_dir4,f"time_of_peak_byExp_byDirection_angSpd.pdf"),format='PDF')

# print(f"Time of the peak angular speed by dir = {time_of_peak_angSpd_byDir}")
# print(f"Time of the peak angular speed by Dir mean = {time_of_peak_angSpd_byDirmean} ms")

# # %%
# # another way to calculate time of the max ang speed
# mean_angSpd = exp_data_all.groupby(['time_ms','expNum'])['ang speed (deg*s-1)'].mean().reset_index()
# idx_mean_maxAngSpd = mean_angSpd.groupby(['expNum'])['ang speed (deg*s-1)'].apply(
#     lambda y: np.argmax(y)
# )
# time_by_exp_maxAngSpd = ((idx_mean_maxAngSpd/166 - 0.3)*1000).reset_index()

# time_by_exp_maxAngSpd.columns = ['expNum','time_of_peak_angSpd (ms)']
# time_of_peak_angSpd_byExp = time_by_exp_maxAngSpd.groupby('expNum')['time_of_peak_angSpd (ms)'].mean()
# time_of_peak_angSpd_byExp_mean = time_by_exp_maxAngSpd['time_of_peak_angSpd (ms)'].values.mean()
# plt.figure(figsize=(3,2))
# g = sns.pointplot(
#     data = time_by_exp_maxAngSpd,
#     x = 'time_of_peak_angSpd (ms)',
# )
# sns.despine()
# g.set_xlim(None,0)
# plt.savefig(os.path.join(fig_dir4,f"time_of_peak_byExp_byExp_angSpd.pdf"),format='PDF')

# print(f"Time of the peak angular speed by Exp mean = {time_of_peak_angSpd_byExp_mean} ms")

# %%
# # plot average angular speed
# plt_features = {'propBoutAligned_angSpeed': 'ang speed (deg*s-1)'}

# for feature_toplt in tqdm(list(plt_features.values())):
#     p = sns.relplot(
#             data = exp_data_all, x = 'time_ms', y = feature_toplt,
#             kind = 'line',aspect=3, height=2, ci='sd'
#             )
#     p.map(
#         plt.axvline, x=0, linewidth=1, color=".3", zorder=0
#         )
#     # p.map(
#     #     plt.axvline, x=time_of_peak_angSpd_mean, linewidth=2, color="green", zorder=0
#     #     )
#     plt.savefig(os.path.join(fig_dir4, f"{feature_toplt}_timeSeries ± SD.pdf"),format='PDF')

# %%
# df = exp_data_all.loc[exp_data_all['bout_number']==2000]
# sns.lineplot(data = df,
#              x = 'time_ms',
#              y = savgol_filter(df['ang vel (deg*s-1)'],15,3)
# )
# %%
# plot average angular speed
plt_features = {'absolute_pitch_change':'absolute pitch chg (deg)'}
# sample bout groups
sample_bouts = np.random.choice(exp_data_all["bout_number"].unique(), 500)
df_sampled = exp_data_all.query('bout_number in @sample_bouts')

for feature_toplt in tqdm(list(plt_features.values())):
    p = sns.relplot(
            data = df_sampled, x = 'time_ms', y = feature_toplt,
            kind = 'line',
            aspect=2, height=2, 
            # ci='sd',
            estimator=None,
            units = 'bout_number',
            alpha = 0.05
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    p.set(xlim=(-250,200))
    # p.map(
    #     plt.axvline, x=time_of_peak_angSpd_mean, linewidth=2, color="green", zorder=0
    #     )
    plt.savefig(os.path.join(fig_dir4, f"{feature_toplt}_timeSeries sample.pdf"),format='PDF')
# %%
