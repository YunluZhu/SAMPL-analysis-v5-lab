#%%
from cmath import exp
from plot_functions.plt_tools import round_half_up
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.get_index import (get_index)
from scipy.signal import savgol_filter

from tqdm import tqdm

# %%
def Fig2_parameters_alignedAverage(root):
    print('- Figure 2: Bout parameter time series ± SD')

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s
    # %%
    # Select data and create figure folder
    FRAME_RATE = 166

    folder_name = f'Parameter time series'
    fig_dir2 = os.path.join(get_figure_dir('Fig_2'), folder_name)

    try:
        os.makedirs(fig_dir2)
    except:
        pass

    # %% features for plotting
    # below are all the properties can be plotted. 
    all_features = {
        'propBoutAligned_speed':'speed (mm*s-1)', 
        # 'propBoutAligned_linearAccel':'linear accel (mm*s-2)',
        'propBoutAligned_pitch':'pitch (deg)', 
        # 'propBoutAligned_angVel':'ang vel (deg*s-1)',   # smoothed angular velocity
        # 'propBoutAligned_angSpeed': 'ang speed (deg*s-1)', 
        # 'relative_pitch_change':'absolute pitch chg (deg)',
        # 'propBoutAligned_accel':'ang accel (deg*s-2)',    # angular accel calculated using raw angular vel
        # 'propBoutAligned_instHeading': 'instantaneous trajectory (deg)',  
        # 'propBoutAligned_x':'x position (mm)',
        # 'propBoutAligned_y':'y position (mm)', 
        # 'propBoutInflAligned_speed': 'ang speed (deg*s-1)',   #
    }
    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)
    idx_dur300ms = round_half_up(0.3*FRAME_RATE)
    idx_dur250ms = round_half_up(0.25*FRAME_RATE)
    all_conditions = []
    folder_paths = []
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
                exp_data_all = pd.DataFrame()
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    rows = []
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                    # assign frame number, total_aligned frames per bout
                    raw = raw.assign(
                        idx = round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)),
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
                        relative_pitch_change = selected_range['propBoutAligned_pitch'] - np.repeat(null_initial_pitch,(idxRANGE[1]-idxRANGE[0])).values
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
    pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==round_half_up(peak_idx - 0.1 * FRAME_RATE),'pitch (deg)']

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
                kind = 'line',aspect=3, height=2, errorbar='sd',
                )
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", zorder=0
            )
        plt.savefig(os.path.join(fig_dir2, f"{feature_toplt}_timeSeries_up_dn ± SD.pdf"),format='PDF')

    for feature_toplt in tqdm(list(all_features.values())):
        p = sns.relplot(
                data = exp_data_all, x = 'time_ms', y = feature_toplt,
                kind = 'line',aspect=3, height=2, errorbar='sd'
                )
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", zorder=0
            )
        plt.savefig(os.path.join(fig_dir2, f"{feature_toplt}_timeSeries ± SD.pdf"),format='PDF')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig2_parameters_alignedAverage(root)