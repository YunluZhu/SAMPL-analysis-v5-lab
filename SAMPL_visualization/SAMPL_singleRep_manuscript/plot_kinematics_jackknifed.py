'''
Plot bout kinematics:
    righting gain
    set point
    steering gain

This script takes two types of directory:
1. if input directory is a folder containing analyzed dlm data, an averaged value for each parameter will be plotted
    dir/
    ├── all_data.h5
    ├── bout_data.h5
    └── IEI_data.h5
2. if input directory is a folder with subfolders containing dlm data, an averaged value and error bars will be plotted
    dir/
    ├── experiment 1/  
    │   ├── all_data.h5
    │   ├── bout_data.h5
    │   └── IEI_data.h5
    └── experiment 2/
        ├── all_data.h5
        ├── bout_data.h5
        └── IEI_data.h5
        
NOTE
User may define the number of bouts sampled from each experimental repeat by defining the argument "sample_bout"
Default is off (sample_bout = -1)
'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import (set_font_type, day_night_split, round_half_up, setup_vis_parameter, defaultPlotting)
from plot_functions.plt_v5 import (jackknife_kinematics, extract_bout_features_v5, get_kinematics)
from plot_functions.get_index import (get_index, get_frame_rate)

# %%
def plot_kinematics_jackknifed(root, **kwargs):
    """plot kinematics, if there are multiple repeats, jackknife to estimate errors. 

    Args:
        root (string): directory
        ---kwargs---
        sample_bout (int): number of bouts to sample from each experimental repeat. default is off
        figure_dir (str): directory to save figures. If not defined, figures will be saved to folder "figures"

    """
    print('------\n+ Plotting bout kinematics (jackknife across repeats)')
    # generate figure folder
    folder_name = 'kinematics'
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'figures', folder_name)
    
    root, all_dir, fig_dir, if_sample, SAMPLE_N, if_multiple_repeats = setup_vis_parameter(root, fig_dir, if_sample=False, SAMPLE_N=-1, if_multiple_repeats=False, **kwargs)
        
    # get frame rate
    try:
        FRAME_RATE = get_frame_rate(all_dir[0])
    except:
        print("No info file found!\n")
        FRAME_RATE = round_half_up(input("Frame rate? "))

    # get the index for the time of peak speed, and total time points for each aligned bout
    peak_idx, total_aligned = get_index(FRAME_RATE)

    # %%
    # defaultPlotting()
    T_start = -0.3
    T_end = 0.3
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

    # %%
    # main function

    bout_features = pd.DataFrame()
    bout_kinematics = pd.DataFrame()

    all_dir.sort()
    # go through each condition folders under the root
    for expNum, exp in enumerate(all_dir):
        # angular velocity (angVel) calculation
        rows = []
        # for each sub-folder, get the path
        exp_path = exp
        # get pitch                
        exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
        # assign frame number, total_aligned frames per bout
        exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
        
        # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
        bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
        # # if only need day or night bouts:
        for i in day_night_split(bout_time,'aligned_time').index:
            rows.extend(list(range(i*total_aligned+round_half_up(idxRANGE[0]),i*total_aligned+round_half_up(idxRANGE[1]))))
        exp_data = exp_data.assign(expNum = exp)
        trunc_day_exp_data = exp_data.loc[rows,:]
        trunc_day_exp_data = trunc_day_exp_data.assign(
            bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
            )
        num_of_bouts = len(trunc_day_exp_data.loc[trunc_day_exp_data['idx'] == peak_idx])
        
        this_exp_features = extract_bout_features_v5(trunc_day_exp_data,peak_idx,FRAME_RATE)
        this_exp_features = this_exp_features.assign(
            expNum = [expNum]*num_of_bouts,
            )
        this_exp_features = this_exp_features.reset_index(drop=True)
        
        if if_multiple_repeats == True:
            if if_sample == True:
                try:
                    this_exp_features = this_exp_features.sample(n=SAMPLE_N)
                except:
                    this_exp_features = this_exp_features.sample(n=SAMPLE_N,replace=True)
        
        this_exp_kinematics = get_kinematics(this_exp_features)
        this_exp_kinematics = pd.concat([this_exp_kinematics, pd.Series(data={'expNum': expNum})])
        
        bout_features = pd.concat([bout_features,this_exp_features])
        bout_kinematics = pd.concat([bout_kinematics,this_exp_kinematics.to_frame().T], ignore_index=True)

    mean_val = bout_kinematics.mean()
    table_filename = os.path.join(fig_dir,f"kinematics mean values individual exp.csv")
    mean_val.T.to_csv(table_filename)
    # %% Jackknife resampling
    if if_multiple_repeats:
    # calculate kinematics
        kinematics_jackknife = jackknife_kinematics(bout_features,'expNum')
        cat_cols = ['jackknife_group']
        kinematics_jackknife.rename(columns={c:c+'_jack' for c in kinematics_jackknife.columns if c not in cat_cols},inplace=True)
        kinematics_jackknife = kinematics_jackknife.sort_values(by=['jackknife_group']).reset_index(drop=True)

    defaultPlotting()

    if if_multiple_repeats:
        toplt = kinematics_jackknife
        cat_cols = ['jackknife_group']
        table_filename = os.path.join(fig_dir,f"jackknife kinematics mean values across repeats.csv")

    else:
        toplt = bout_kinematics
        cat_cols = ['expNum']
        print(bout_kinematics)
        table_filename = os.path.join(fig_dir,f"kinematics values.csv")

    output_par = toplt.iloc[:,[0,1,2]]
    output_par.describe().to_csv(table_filename)
    
    all_features = [c for c in toplt.columns if c not in cat_cols]
    for feature_toplt in (all_features):
        g = sns.catplot(data = toplt,y = feature_toplt,
                        height=4, aspect=0.8, kind='point',
                        markers='d',sharey=False,
                        )
        filename = os.path.join(fig_dir,f"{feature_toplt}.pdf")
        plt.savefig(filename,format='PDF')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    root = input("- Data directory? \n")
    plot_kinematics_jackknifed(root)
