#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from plot_functions.get_index import (get_index)
from scipy.signal import savgol_filter
from tqdm import tqdm

def Fig5_time_of_maxAngvel(root):
    set_font_type()
    print('- Figure 5: angvel time series and time of peak angvel')

    # choose the time duration to plot. 
    # total aligned duration = [-0.5, 0.4] (sec) around time of peak speed
    # [-0.3,0.2] (sec) around peak speed is recommended 

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s

    # %%
    # Select data and create figure folder
    FRAME_RATE = 166

    folder_name = f'Time of max angvel'
    folder_dir5 = get_figure_dir('Fig_5')
    fig_dir5 = os.path.join(folder_dir5, folder_name)
    try:
        os.makedirs(fig_dir5)
    except:
        pass
    # %%

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx - 0.1 * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx - 0.05 * FRAME_RATE)

    all_conditions = []
    folder_paths = []
    all_cond1 = []
    all_cond2 = []
    exp_data_all = pd.DataFrame()

    all_features = {
        'propBoutAligned_speed':'speed (mm*s-1)', 
        'propBoutAligned_pitch':'pitch (deg)', 
        'propBoutAligned_angVel':'ang vel (deg*s-1)',   # smoothed angular velocity
        'propBoutAligned_angVel_sm':'propBoutAligned_angVel_sm',
        'adj_ang_speed':'angvel (deg/s)',
    }
    
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
                    # ang_accel_of_angvel = np.diff(savgol_filter(raw['propBoutAligned_angVel'].values, 15, 3),prepend=np.array([np.nan]))*FRAME_RATE
                    # abs_ang_accel_of_angvel = np.absolute(ang_accel_of_angvel)
                    # assign frame number, total_aligned frames per bout
                    raw = raw.assign(
                        idx = round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)),
                        # ang_accel_of_angvel = ang_accel_of_angvel,
                        # abs_ang_accel_of_angvel = abs_ang_accel_of_angvel,
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
                        lambda grp_pitch: np.diff(savgol_filter(grp_pitch, 11, 3),prepend=np.array([np.nan]))*FRAME_RATE,
                    )
                    propBoutAligned_angVel = propBoutAligned_angVel.apply(pd.Series).T.melt()
                    # assign angvel and ang speed
                    selected_range = selected_range.assign(
                        propBoutAligned_angVel_sm = propBoutAligned_angVel['value'].values,
                        # propBoutAligned_angSpeed = np.absolute(propBoutAligned_angVel['value'].values),
                    )
                    grp = selected_range.groupby(np.arange(len(selected_range))//(idxRANGE[1]-idxRANGE[0]))
                    accel_angvel_mean = grp.apply(
                        lambda group: group.loc[(group['idx']>idx_pre_bout)&(group['idx']<idx_mid_accel), 
                                                'propBoutAligned_angVel_sm'].mean()
                    )
                    adj_by_angvel = accel_angvel_mean/np.absolute(accel_angvel_mean)
                    #|||||||||||||||||||||||||
                    adj_by_which = adj_by_angvel #adj_by_traj_deviation #  #
                    #|||||||||||||||||||||||||
                    
                    adj_ang_speed = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)
                    
                    # calculate angaccel with sign adjusted (positive value before 0ms, negative value after)
                    # angAccel_adj_sign = grp.apply(
                    #     lambda group: group.loc[(group['idx']>idx_pre_bout)&(group['idx']<idx_mid_accel), 
                    #                             'ang_accel_of_angvel'].mean()
                    # )
                    # angAccel_adj_sign = angAccel_adj_sign/np.absolute(angAccel_adj_sign)
                    # adj_ang_accel = selected_range['ang_accel_of_angvel'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)
                    
                    
                    selected_range = selected_range.assign(
                        adj_ang_speed = adj_ang_speed,
                    )

                    columns_to_pass = list(all_features.keys()) + ['idx']
                    this_exp_data = selected_range.loc[:,columns_to_pass]
                    this_exp_data = this_exp_data.rename(columns=all_features)

                    this_exp_data = this_exp_data.assign(
                        time_ms = (this_exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                        expNum = expNum)
                    this_cond_data = pd.concat([this_cond_data,this_exp_data])
                
        cond1 = all_conditions[condition_idx].split("_")[0]
        cond2 = all_conditions[condition_idx].split("_")[1]
        all_cond1.append(cond1)
        all_cond2.append(cond2)
        
        this_cond_data = this_cond_data.reset_index(drop=True)
        this_cond_data = this_cond_data.assign(
            cond1 = cond1,
            cond2 = cond2,
        )
        exp_data_all = pd.concat([exp_data_all,this_cond_data], ignore_index=True)

    exp_data_all['expNum'] = exp_data_all.groupby(['cond1','cond2','expNum']).ngroup()
    exp_data_all['cond1'] = '7DD'
    exp_data_all['cond2'] = 'WT'
    all_cond1 = exp_data_all['cond1'].unique()
    all_cond2 = exp_data_all['cond2'].unique()
    # %%
    # separation_posture = 10

    peak_speed = exp_data_all.loc[exp_data_all.idx==peak_idx,'speed (mm*s-1)']
    pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==idx_pre_bout,'pitch (deg)']

    grp = exp_data_all.groupby(np.arange(len(exp_data_all))//(idxRANGE[1]-idxRANGE[0]))
    exp_data_all = exp_data_all.assign(
                                        peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])).values,
                                        pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])).values,
                                        bout_number = grp.ngroup(),
                                    )
    # exp_data_all = exp_data_all.assign(
    #                                     direction = pd.cut(exp_data_all['pitch_pre_bout'],[-90,separation_posture,90],labels = ['Nose-down', 'Nose-up'])
    #                                 )
    # exp_data_all.drop(exp_data_all[exp_data_all['peak_speed']<7].index, inplace=True)
    # %%
    # plot timeseries
    all_features_toplt = {
        'adj_ang_speed':'angvel (deg/s)',
        }
    toplt = exp_data_all.groupby(['expNum','cond1','cond2','time_ms']).median().reset_index()
    for feature_toplt in tqdm(list(all_features_toplt.values())):
        p = sns.relplot(
                data = toplt, x = 'time_ms', y = feature_toplt,
                kind = 'line',aspect=3, height=2, 
                errorbar='sd',
                row = 'cond2', col='cond1',
                # hue='direction',
                # errorbar=None
                )
        p.set(xlim=(-250,0))
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", zorder=0
            )
        plt.savefig(os.path.join(fig_dir5, f"angvel_timeSeries ± SD.pdf"),format='PDF')
        # 

    # %%
    # calculate time of max angaccel from mean time series of each exp, then average
    mean_angvel = exp_data_all.groupby(['time_ms','expNum','cond1','cond2'])['angvel (deg/s)'].median().reset_index()
    mean_angvel = mean_angvel.loc[mean_angvel['time_ms']<0]
    idx_mean_max = mean_angvel.groupby(['expNum','cond1','cond2'])['angvel (deg/s)'].apply(
        lambda y: np.argmax(y)
    )
    time_by_expNum_max = ((idx_mean_max/166 - BEFORE_PEAK)*1000).reset_index()
    # condition_match = exp_data_all.groupby(['expNum','cond2'])['cond2','cond1'].head(1)

    time_by_expNum_max.columns = ['expNum','cond1','cond2','time_adj_angvel (ms)']

    time_of_peak__byBout_mean = time_by_expNum_max['time_adj_angvel (ms)'].mean()

    toplt = time_by_expNum_max
    feature = 'time_adj_angvel (ms)'
    upper = np.percentile(toplt[feature], 99)
    lower = np.percentile(toplt[feature], 1)
    plt.figure(figsize=(3,2))
    p = sns.histplot(data=toplt, x=feature, 
                        bins = 5,  # satisfies Sturge's
                        element="poly",
                        #  kde=True, 
                        stat="probability",
                        pthresh=0.05,
                        binrange=(lower,upper),
                        color='grey'
                        )
    sns.despine()
    plt.savefig(os.path.join(fig_dir5,f"time_of_peak_angvel_byexp_hist.pdf"),format='PDF')
    # 

    plt.figure(figsize=(3,2))
    g = sns.pointplot(
        data = time_by_expNum_max,
        y = 'cond2',
        x = 'time_adj_angvel (ms)',
        errorbar='sd', 
    )
    sns.despine()
    g.set_xlim(lower,upper)
    plt.savefig(os.path.join(fig_dir5,f"time_adj_angvel_byexp.pdf"),format='PDF')
    # 
    print(f"Time of the peak angular accel by Exp mean = {time_of_peak__byBout_mean}±{time_by_expNum_max['time_adj_angvel (ms)'].values.std()} ms")


# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig5_time_of_maxAngvel(root)