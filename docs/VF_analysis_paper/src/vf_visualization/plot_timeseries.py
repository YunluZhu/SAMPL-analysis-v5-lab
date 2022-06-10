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
from plot_functions.get_index import (get_index, get_frame_rate)
from plot_functions.plt_tools import (set_font_type, day_night_split)
from tqdm import tqdm

# %%
def plot_aligned(root):
    '''
    plots parameters of bouts aligned at the time of the peak speed.
    Input directory needs to be a folder containing analyzed dlm data.
    '''
    print('\n- Plotting time series (aligned swim bouts)')

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

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced.')
        print(fig_dir)

    # %%
    # get path and frame rate
    # for each sub-folder, get the path
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        all_dir = all_dir[1:]
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = int(input("Frame rate? "))

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)

    # calculate indicies
    idxRANGE = [peak_idx-int(BEFORE_PEAK*FRAME_RATE),peak_idx+int(AFTER_PEAK*FRAME_RATE)]
    
    exp_data_all = pd.DataFrame()
    for exp_num, exp_path in enumerate(all_dir):
        rows = []
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        exp_data = exp_data.assign(
            propBoutAligned_linearAccel = exp_data['propBoutAligned_speed'].diff()
        )
        exp_data = exp_data.loc[:,all_features.keys()]
        exp_data = exp_data.rename(columns=all_features)
        # assign frame number, total_aligned frames per bout
        exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))

        # - get the index of the rows in exp_data to keep
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
        # for i in bout_time.index:
        # # if only need day or night bouts:
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))

        exp_data = exp_data.assign(time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000)
        exp_data_all = pd.concat([exp_data_all,exp_data.loc[rows,:]])
    exp_data_all = exp_data_all.reset_index(drop=True)
    exp_data_all = exp_data_all.assign(
        pitch_dir = pd.cut(exp_data_all['pitch (deg)'],[-90,0,90],labels=['dive','climb'])
    )
    # %%
    # plot average
    set_font_type()

    for feature_toplt in tqdm(list(all_features.values())):
        p = sns.relplot(
            data = exp_data_all, x = 'time_ms', y = feature_toplt, col='pitch_dir',
            kind = 'line',aspect=3, height=2
            )
        p.map(
            plt.axvline, x=0, linewidth=1, color=".3", zorder=0
            )
        plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_aligned.pdf"),format='PDF')

# %%
def plot_raw(root):
    '''
    Plots single epoch that contains one or more bouts
    Input directory needs to be a folder containing analyzed dlm data.
    '''
    print('\n- Plotting time series (raw)')

    # %% features for plotting
    # below are all the properties can be plotted. 
    all_features = [
        # 'ang', 
        'absy', 
        # 'deltaT', 
        # 'x', 'y',
        'headx', 
        'heady', 
        'centeredAng', 
        # 'xvel', 
        # 'yvel', 
        'dist', 
        'displ',
        'angVel', 
        # 'angVelSmoothed', 
        # 'angAccel', 
        # 'swimSpeed',
        'velocity'
    ]
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
        # if yes, calculate jackknifed std(pitch)
        all_dir = all_dir[1:]
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = int(input("Frame rate? "))
    
    epoch_info_all = pd.DataFrame()
    epoch_data_all = pd.DataFrame()
    for exp_num, exp_path in enumerate(all_dir):
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/all_data.h5", key='grabbed_all')
        
        exp_data = exp_data.loc[:,all_features +['epochNum']]
        exp_data = exp_data.assign(
            exp_num = exp_num
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
            exp_num = exp_num
        )
        epoch_info_all = pd.concat([epoch_info_all,epoch_info], ignore_index=True)
        epoch_data_all = pd.concat([epoch_data_all,exp_data], ignore_index=True)
        
    epoch_info_all = epoch_info_all.sort_values(by='duration',ascending=False)
    epoch_info_all.reset_index(drop=True)
    print(f'{len(epoch_info_all)} epochs detected. Sorted from long to short.')
    # %%
    which_toplt = int(input(f'Which epoch to plot? (1-{len(epoch_info)}) \n'))
    toplt = epoch_info.loc[which_toplt-1,:]
    data_toplt = epoch_data_all.loc[(epoch_data_all['exp_num']==toplt['exp_num']) & (epoch_data_all['epochNum']==int(toplt['epoch_num'])), :]

    # %%
    data_toplt = data_toplt.assign(
        time_ms = np.arange(0,len(data_toplt))/FRAME_RATE*1000
    )

    set_font_type()

    for feature_toplt in tqdm(all_features):
        p = sns.relplot(
            data = data_toplt, x = 'time_ms', y = feature_toplt,
            kind = 'line',aspect=3, height=2
            )
        plt.savefig(os.path.join(fig_dir, f"{feature_toplt}_raw.pdf"),format='PDF')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_aligned(root)
    plot_raw(root)