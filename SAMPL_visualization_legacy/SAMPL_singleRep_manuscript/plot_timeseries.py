'''
Plot basic parameters as a function of time. Modify "all_features" to choose parameters to plot.
This script contains two functions:
    plot_aligned
    plot_raw

plt_timeseries.plot_aligned(dir)
    plots parameters of aligned bouts as a function of time.

        
plt_timeseries.plot_raw(dir) 
    plots raw data from a single epoch that contains one or more bouts as a function of time


This script takes two types of data structures:
1. Input directory being a folder containing analyzed dlm data
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. Input directory being a folder with subfolders containing dlm data
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
'''

#%%
from cmath import exp
import os
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import (set_font_type, day_night_split, round_half_up, setup_vis_parameter, defaultPlotting)
from plot_functions.get_index import (get_index, get_frame_rate)

from tqdm import tqdm

# %%
def plot_aligned(root, **kwargs):
    """plots parameters of swim bouts aligned at the time of the peak speed with and without nose-up bouts and nose-down bouts separated.

    Args:
        root (str): a directory containing analyzed dlm data.
        ---kwargs---
        figure_dir (str): directory to save figures. If not defined, figures will be saved to folder "figures"

    """
    print('------\n+ Plotting parameter time series (mean ± SD).')

    # %%
    # choose the time duration to plot. 
    # total aligned duration = [-0.5, 0.4] (sec) around time of peak speed
    # [-0.3,0.2] (sec) around peak speed is recommended 

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s

    # %% features for plotting
    # below are all the properties can be plotted. 
    all_features = {
        'propBoutAligned_speed':'speed (mm*s-1)', 
        'propBoutAligned_linearAccel':'linear accel (mm*s-2)',
        'propBoutAligned_pitch':'pitch (deg)', 
        'propBoutAligned_angVel':'ang vel (deg*s-1)',   # smoothed angular velocity
        'propBoutAligned_accel':'ang accel (deg*s-2)',    # angular accel calculated using raw angular vel
        # 'propBoutInflAligned_accel',
        # 'propBoutAligned_instHeading', 
        'propBoutAligned_x':'x position (mm)',
        'propBoutAligned_y':'y position (mm)', 
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
    }
    # %%
    # generate figure folder
    folder_name = 'timeseries_aligned'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    root, all_dir, fig_dir, if_sample, SAMPLE_N, if_multiple_repeats = setup_vis_parameter(root, fig_dir, if_sample=False, SAMPLE_N=0, if_multiple_repeats=False, **kwargs)
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = round_half_up(input("Frame rate? "))

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)

    # calculate indicies
    idxRANGE = [peak_idx-round_half_up(BEFORE_PEAK*FRAME_RATE),peak_idx+round_half_up(AFTER_PEAK*FRAME_RATE)]
    
    exp_data_all = pd.DataFrame()
    for expNum, exp_path in enumerate(all_dir):
        rows = []
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        exp_data = exp_data.assign(
            propBoutAligned_linearAccel = exp_data['propBoutAligned_speed'].diff()
        )
        exp_data = exp_data.loc[:,all_features.keys()]
        exp_data = exp_data.rename(columns=all_features)
        # assign frame number, total_aligned frames per bout
        exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))

        # - get the index of the rows in exp_data to keep
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
        # for i in bout_time.index:
        # # if only need day or night bouts:
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))

        exp_data = exp_data.assign(time_s = (exp_data['idx']-peak_idx)/FRAME_RATE*1000)
        exp_data_all = pd.concat([exp_data_all,exp_data.loc[rows,:]])
    exp_data_all = exp_data_all.reset_index(drop=True)
    # %%
    # separate up and down by set_point
    exp_data_all
    pitch_pre_bout = exp_data_all.loc[exp_data_all.idx==round_half_up(peak_idx - 0.1 * FRAME_RATE),'pitch (deg)'].values
    pitch_peak = exp_data_all.loc[exp_data_all.idx==round_half_up(peak_idx),'pitch (deg)']
    pitch_post_bout = exp_data_all.loc[exp_data_all.idx==round_half_up(peak_idx + 0.1 * FRAME_RATE),'pitch (deg)'].values
    rot_righting = pitch_post_bout - pitch_peak
    bout_features = pd.DataFrame(data={'pitch_pre_bout':pitch_pre_bout,'rot_righting':rot_righting})
    separation_pitch = 10
    
    grp = exp_data_all.groupby(np.arange(len(exp_data_all))//(idxRANGE[1]-idxRANGE[0]))
    exp_data_all = exp_data_all.assign(
                                    pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])),
                                    bout_number = grp.ngroup(),
                                )
    exp_data_all = exp_data_all.assign(
                                    direction = pd.cut(exp_data_all['pitch_pre_bout'],[-90,separation_pitch,90],labels = ['Nose-down', 'Nose-up'])
                                )
    # %%
    # plot average
    set_font_type()
    print("Mean bout parameters separated by set point, labeled as nose-up & nose-down")
    for feature_toplt in tqdm(list(all_features.values())):
        p = sns.relplot(
                data = exp_data_all, x = 'time_s', y = feature_toplt,
                col='direction',
                kind = 'line',aspect=3, height=2, errorbar='sd'
                )
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", 
            )
        plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_timeSeries_up_dn.pdf"),format='PDF')
    print("Mean bout parameters")
    for feature_toplt in tqdm(list(all_features.values())):
        p = sns.relplot(
                data = exp_data_all, x = 'time_s', y = feature_toplt,
                kind = 'line',aspect=3, height=2, errorbar='sd'
                )
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", 
            )
        plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_timeSeries.pdf"),format='PDF')

# %%
def plot_raw(root):
    """Plots single epoch that contains one or more bouts

    Args:
        root (string): a directory containing analyzed dlm data.

    """
    print('------\n+ Plotting time series (raw)')

    # %% features for plotting
    # below are all the properties can be plotted. 
    all_features = {
        'ang':'pitch (deg)',
        # 'absy':'y position (mm)'
        # 'deltaT', 
        'x':'x',
        'y':'y',
        'headx':'head x (mm)',
        'heady':'head y (mm)',
        # 'centeredAng':'centered angle (deg)',
        # 'xvel', 
        # 'yvel', 
        'dist':'distance (mm)',
        # 'displ':'displacement (mm)',
        'angVel':'ang vel (deg*s-1)',
        # 'angVelSmoothed', 
        # 'angAccel':'ang accel (deg*s-2)',
        'swimSpeed':'speed (mm*s-1)',
        'velocity':'velocity (mm*s-1)'
    }
    # %%
    # generate figure folder
    folder_name = 'timeseries_raw'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig folder created: {folder_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced.')
        print(folder_dir)

    # %%
    # for each sub-folder, get the path
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        all_dir = all_dir[1:]
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = round_half_up(input("Frame rate? "))
    
    epoch_info_all = pd.DataFrame()
    epoch_data_all = pd.DataFrame()
    for expNum, exp_path in enumerate(all_dir):
        # get pitch                
        all_data = pd.read_hdf(f"{exp_path}/all_data.h5", key='grabbed_all')

        exp_data = all_data.loc[:,all_features.keys()]
        exp_data = exp_data.rename(columns=all_features)
        exp_data = exp_data.assign(
            expNum = expNum,
            epochNum = all_data['epochNum'].values,
            deltaT = all_data['deltaT'].values
        )
        
        epoch_info = exp_data.groupby('epochNum').size().reset_index()

        epoch_info = epoch_info.rename(columns={
            'epochNum':'epoch_num',
            0:'frame_num',
        })
        epoch_info.reset_index(drop=True)
        epoch_info = epoch_info.assign(
            idx = np.arange(0,len(epoch_info))+1,
            duration = epoch_info['frame_num']/FRAME_RATE,
            expNum = expNum,
        )
        epoch_info_all = pd.concat([epoch_info_all,epoch_info], ignore_index=True)
        epoch_data_all = pd.concat([epoch_data_all,exp_data], ignore_index=True)
        
    epoch_info_all = epoch_info_all.sort_values(by='duration',ascending=False)
    epoch_info_all = epoch_info_all.reset_index(drop=True)
    print(f'{len(epoch_info_all)} epochs detected. Sorted from long to short.')
    # %%
    if_plot_others = 'y'
    while if_plot_others != 'n':
        which_toplt = round_half_up(input(f'Which epoch to plot? (1-{len(epoch_info_all)}) \n'))
        toplt = epoch_info_all.loc[which_toplt-1,:]
        data_toplt = epoch_data_all.loc[(epoch_data_all['expNum']==toplt['expNum']) & (epoch_data_all['epochNum']==round_half_up(toplt['epoch_num'])), :]

        # %%
        data_toplt = data_toplt.assign(
            time_s = np.cumsum(data_toplt['deltaT'])
        )

        set_font_type()

        for feature_toplt in tqdm(list(all_features.values())):
            p = sns.relplot(
                data = data_toplt, x = 'time_s', y = feature_toplt,
                kind = 'line',aspect=3, height=2
                )
            plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_raw.pdf"),format='PDF')
            plt.close()
        if_plot_others = input(f'Plot another epoch? Previous plots will be overwritten: (y/n) ')
# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_aligned(root)
    plot_raw(root)