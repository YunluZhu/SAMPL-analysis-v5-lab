'''
Plot bout kinetics:
    righting gain
    set point
    steering gain
    
This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, an averaged value for each parameter will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. if input directory is a folder with subfolders containing dlm data, an averaged value and error bars generated using jackknife resampling will be plotted
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
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.plt_v4 import (jackknife_kinetics, extract_bout_features_v4, get_kinetics)
from plot_functions.get_index import (get_index, get_frame_rate)
from plot_functions.plt_tools import (set_font_type, day_night_split)

# %%
def plot_kinetics(root):
    print('\n- Plotting bout kinetics')
    # generate figure folder
    folder_name = 'kinetics'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)

    try:
        os.makedirs(fig_dir)
        print(f'fig directory created: {fig_dir}')
    except:
        print('Figure folder already exist! Old figures will be replaced.')
        print(fig_dir)

    # Check if there are subfolders in the current directory
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        # if yes, calculate jackknifed std(pitch)
        if_jackknife = True
        all_dir = all_dir[1:]
    else:
        # if no, only calculate one std(pitch) for current experiment
        if_jackknife = False
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = int(input("Frame rate? "))

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)

    # %%
    # defaultPlotting()
    T_start = -0.3
    T_end = 0.3
    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.15 #s
    idx_start = int(peak_idx + T_start * FRAME_RATE)
    idx_end = int(peak_idx + T_end * FRAME_RATE)

    # idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
    # idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
    # idx_end_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

    # %%
    # main function

    bout_features = pd.DataFrame()
    bout_kinetics = pd.DataFrame()

    # go through each condition folders under the root
    for expNum, exp in enumerate(all_dir):
        # angular velocity (angVel) calculation
        rows = []
        # for each sub-folder, get the path
        exp_path = exp
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        # assign frame number, total_aligned frames per bout
        exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
        
        # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
        # # if only need day or night bouts:
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+int(idxRANGE[0]),i*total_aligned+int(idxRANGE[1]))))
        exp_data = exp_data.assign(expNum = exp)
        trunc_day_exp_data = exp_data.loc[rows,:]
        trunc_day_exp_data = trunc_day_exp_data.assign(
            bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
            )
        num_of_bouts = len(trunc_day_exp_data.loc[trunc_day_exp_data['idx'] == peak_idx])
        
        this_exp_features = extract_bout_features_v4(trunc_day_exp_data,peak_idx,FRAME_RATE)
        this_exp_features = this_exp_features.assign(
            exp_num = [expNum]*num_of_bouts,
            )
        this_exp_features = this_exp_features.reset_index(drop=True)
        
        this_exp_kinetics = get_kinetics(this_exp_features)
        this_exp_kinetics = this_exp_kinetics.append(pd.Series(data={'exp_num': expNum}))
        
        bout_features = pd.concat([bout_features,this_exp_features])
        bout_kinetics = pd.concat([bout_kinetics,this_exp_kinetics.to_frame().T], ignore_index=True)

    # %% Jackknife resampling
    if if_jackknife:
    # calculate kinetics
        kinetics_jackknife = jackknife_kinetics(bout_features,'exp_num')
        cat_cols = ['jackknife_group']
        kinetics_jackknife.rename(columns={c:c+'_jack' for c in kinetics_jackknife.columns if c not in cat_cols},inplace=True)
        kinetics_jackknife = kinetics_jackknife.sort_values(by=['jackknife_group']).reset_index(drop=True)

    defaultPlotting()

    if if_jackknife:
        toplt = kinetics_jackknife
        cat_cols = ['jackknife_group']
    else:
        toplt = bout_kinetics
        cat_cols = ['exp_num']
        print(bout_kinetics)

    all_features = [c for c in toplt.columns if c not in cat_cols]
    for feature_toplt in (all_features):
        g = sns.catplot(data = toplt,y = feature_toplt,
                        height=4, aspect=0.8, kind='point',
                        markers='d',sharey=False,
                        zorder=10
                        )
        filename = os.path.join(fig_dir,f"{feature_toplt}.pdf")
        plt.savefig(filename,format='PDF')


# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_kinetics(root)
