'''

'''

#%%
# import sys
import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from tqdm import tqdm
import matplotlib as mpl
set_font_type()

# %%
# Paste root directory here
# if_plot_by_speed = True
pick_data = 'tau_long'
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'angvel_corr_timeSeries'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
    
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]
spd_bins = np.arange(5,25,4)

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


idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = round_half_up(peak_idx + T_POST_BOUT * FRAME_RATE)
idx_mid_accel = round_half_up(peak_idx + T_MID_ACCEL * FRAME_RATE)
idx_mid_decel = round_half_up(peak_idx + T_MID_DECEL * FRAME_RATE)
idx_end = round_half_up(peak_idx + T_END * FRAME_RATE)


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
                exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                
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
                                    speed_bin = pd.cut(all_around_peak_data['peak_speed'],spd_bins,labels = np.arange(len(spd_bins)-1))
                                )

# %%
all_pre_bout_angles = all_around_peak_data.loc[all_around_peak_data['idx']==idx_pre_bout,'propBoutAligned_pitch']
initial_pitch = all_around_peak_data.loc[all_around_peak_data['idx']==idx_initial,'propBoutAligned_pitch']
# %%
all_around_peak_data = all_around_peak_data.assign(
    pre_bout_angle = np.repeat(all_pre_bout_angles,(idxRANGE[1]-idxRANGE[0])).values,
    initial_pitch = np.repeat(initial_pitch,(idxRANGE[1]-idxRANGE[0])).values,
)
all_around_peak_data = all_around_peak_data.assign(
    initial_posture = pd.cut(all_around_peak_data['initial_pitch'],bins=[-90,10,90],labels=['dn','up'])
)

# print("speed buckets:")
# print('--mean')
# print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('mean'))
# print('--min')
# print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('min'))
# print('--max')
# print(all_around_peak_data.groupby('speed_bin')['peak_speed'].agg('max'))

# %%
# correlation with pre bout pitch
# which to corr
# which_to_corr = 'initial_pitch' # initial_pitch or pre_bout_angle
for which_to_corr in ['initial_pitch', 'pre_bout_angle']:
    # cat_cols = ['speed_bin','condition','initial_posture','dpf']
    # # cat_cols = ['condition','initial_posture']
    # grp_cols = cat_cols + ['time_ms']

    # corr_angvel = all_around_peak_data.groupby(grp_cols).apply(
    #     lambda y: stats.pearsonr(y[which_to_corr].values,y['propBoutAligned_angVel'].values)[0]
    # )
    # corr_angvel.name = 'corr'
    # corr_angvel = corr_angvel.reset_index()

    # palette = sns.color_palette("mako_r", 4)

    # g = sns.relplot(
    #     style='condition',
    #     row='initial_posture',
    #     # hue_order=[0,2,4],
    #     hue='speed_bin',
    #     col='dpf',
    #     # size='speed_bin', size_order=[3,2,1,0],

    #     x='time_ms',y='corr',
    #     data=corr_angvel,
    #     kind='line',
    #     palette=palette, 
    #     )
    # g.set(xlim=(-200,200))
    # plt.savefig(fig_dir+f"/{which_to_corr}_by dir and spd.pdf",format='PDF')

    cat_cols = ['condition','initial_posture','dpf']
    grp_cols = cat_cols + ['time_ms']

    corr_angvel = all_around_peak_data.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(y['pre_bout_angle'].values,y['propBoutAligned_angVel'].values)[0]
    )
    corr_angvel.name = 'corr'
    corr_angvel = corr_angvel.reset_index()

    g = sns.relplot(
        row='initial_posture',
        col='dpf',
        hue='condition',
        x='time_ms',y='corr',
        data=corr_angvel,
        kind='line',
        # palette="flare", 
        # hue_norm=mpl.colors.LogNorm()
        )
    g.set(xlim=(-200,200))
    plt.savefig(fig_dir+f"/{which_to_corr}_by dir and cond.pdf",format='PDF')

    # %%
    # ignore dir
    cat_cols = ['condition','dpf']
    grp_cols = cat_cols + ['time_ms']

    corr_angvel = all_around_peak_data.groupby(grp_cols).apply(
        lambda y: stats.pearsonr(y['pre_bout_angle'].values,y['propBoutAligned_angVel'].values)[0]
    )
    corr_angvel.name = 'corr'
    corr_angvel = corr_angvel.reset_index()

    g = sns.relplot(
        hue='condition',
        x='time_ms',y='corr',
        data=corr_angvel,
        kind='line',
        col='dpf',
        # palette="flare", 
        # hue_norm=mpl.colors.LogNorm()
        )
    g.set(xlim=(-200,200))
    plt.savefig(fig_dir+f"/{which_to_corr}_by cond.pdf",format='PDF')
# %%
