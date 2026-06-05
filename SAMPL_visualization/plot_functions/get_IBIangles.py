import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
from plot_functions.plt_tools import (day_night_split)
from plot_functions.get_index import get_index
from tqdm import tqdm
#%%

def get_IBIangles(root, FRAME_RATE, ztime='day', if_strict_DayNightSplit=False,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

    # for day night split
    which_zeitgeber = ztime
            
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    ibi_features_ = []
    all_cond0 = []
    all_cond1 = []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        cond0 = all_conditions[condition_idx].split("_")[0]
        cond1 = all_conditions[condition_idx].split("_")[1]
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                subdir_list.sort()
                # reset for each condition
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # angular velocity (angVel) calculation
                    rows = []
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    exp_data = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2')
                    try:
                        exp_data_ztime = day_night_split(exp_data,'propBoutIEItime',ztime=which_zeitgeber, narrow_bin=if_strict_DayNightSplit)
                    except:
                        exp_data_ztime = day_night_split(exp_data,'propBout_time',ztime=which_zeitgeber, narrow_bin=if_strict_DayNightSplit)

                    exp_data_ztime = exp_data_ztime.assign(
                        expNum = expNum,
                        exp = exp,
                        cond0=cond0,
                        cond1=cond1,
                    )                 
                    ibi_features_.append(exp_data_ztime)
        # combine data from different conditions
        all_cond0.append(cond0)
        all_cond1.append(cond1)
        all_feature_cond = pd.concat(ibi_features_,ignore_index=True)
    all_cond0 = list(set(all_cond0))
    all_cond0.sort()
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    
    return all_feature_cond, all_cond0, all_cond1


def get_timeseriesIBIangles(root, FRAME_RATE,**kwargs):

    which_zeitgeber = kwargs.get('ztime', 'day')

    folder_paths = sorted([
        os.path.join(root, folder)
        for folder in os.listdir(root)
        if not folder.startswith('.')
    ])
    all_conditions = [os.path.basename(p) for p in folder_paths]

    ibi_features_ = []
    all_cond0, all_cond1 = [], []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(tqdm(folder_paths, desc="Conditions", position=0)):
        cond0 = all_conditions[condition_idx].split("_")[0]
        cond1 = all_conditions[condition_idx].split("_")[1]
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                subdir_list.sort()
                # reset for each condition
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(tqdm(subdir_list, desc=f"{all_conditions[condition_idx]} exps", position=1, leave=False)):
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    exp_data = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI_aligned')

                    exp_data_ztime = day_night_split(exp_data,'absTime',ztime=which_zeitgeber)

                    exp_data_ztime = exp_data_ztime.assign(
                        expNum = expNum,
                        exp = exp,
                        cond0=cond0,
                        cond1=cond1,
                    )                 
                    ibi_features_.append(exp_data_ztime)
        # combine data from different conditions
        all_cond0.append(cond0)
        all_cond1.append(cond1)

    all_feature_cond = pd.concat(ibi_features_, ignore_index=True)
    all_cond0 = list(set(all_cond0))
    all_cond0.sort()
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    
    return all_feature_cond, all_cond0, all_cond1