import os,glob
import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import (day_night_split)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import round_half_up

def get_IBIangles(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

    # for day night split
    which_zeitgeber = 'day'
    for key, value in kwargs.items():
        if key == 'ztime':
            which_zeitgeber = value
            
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    all_feature_cond = pd.DataFrame()
    all_cond1 = []
    all_cond2 = []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                subdir_list.sort()
                # reset for each condition
                ibi_features = pd.DataFrame()
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # angular velocity (angVel) calculation
                    rows = []
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    exp_data = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')
                    exp_data_ztime = day_night_split(exp_data,'propBoutIEItime',ztime=which_zeitgeber)
                    exp_data_ztime = exp_data_ztime.assign(
                        expNum = expNum,
                        exp = exp,
                    )                 
                    ibi_features = pd.concat([ibi_features,exp_data_ztime])
        # combine data from different conditions
        cond1 = all_conditions[condition_idx].split("_")[0]
        cond2 = all_conditions[condition_idx].split("_")[1]
        all_cond1.append(cond1)
        all_cond2.append(cond2)
        all_feature_cond = pd.concat([all_feature_cond, ibi_features.assign(
            condition0=cond1,
            condition=cond2
            )],ignore_index=True)
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    return all_feature_cond, all_cond1, all_cond2

