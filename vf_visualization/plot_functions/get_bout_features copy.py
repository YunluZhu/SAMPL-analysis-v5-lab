import os,glob
import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import (day_night_split)
from plot_functions.get_index import get_index

def get_bout_features(root, FRAME_RATE):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.2
    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.15 #s
    idx_start = int(peak_idx + T_start * FRAME_RATE)
    idx_end = int(peak_idx + T_end * FRAME_RATE)

    idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_end_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]

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
                # reset for each condition
                bout_features = pd.DataFrame()
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # angular velocity (angVel) calculation
                    rows = []
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                    exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs())
                    # assign frame number, total_aligned frames per bout
                    exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                    
                    # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                    bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                    # for i in bout_time.index:
                    # # if only need day or night bouts:
                    for i in day_night_split(bout_time,'aligned_time').index:
                        rows.extend(list(range(i*total_aligned+int(idxRANGE[0]),i*total_aligned+int(idxRANGE[1]))))
                    exp_data = exp_data.assign(expNum = expNum)
                    trunc_day_exp_data = exp_data.loc[rows,:]
                    trunc_day_exp_data = trunc_day_exp_data.assign(
                        bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
                        )
                    num_of_bouts = len(trunc_day_exp_data.loc[trunc_day_exp_data['idx'] == peak_idx])
                    this_exp_features = pd.DataFrame(data={
                                                        'pitch_initial':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_initial,'propBoutAligned_pitch'].values, 
                                                        'pitch_pre_bout':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values, 
                                                        'pitch_peak':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'propBoutAligned_pitch'].values, 
                                                        'pitch_end':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_end_bout,'propBoutAligned_pitch'].values, 

                                                        'traj_initial':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_initial,'propBoutAligned_instHeading'].values, 
                                                        'traj_pre_bout':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_pre_bout,'propBoutAligned_instHeading'].values, 
                                                        'traj_peak':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'propBoutAligned_instHeading'].values, 
                                                        'traj_end':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_end_bout,'propBoutAligned_instHeading'].values, 

                                                        'spd_peak':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'propBoutAligned_speed'].values, 

                                                        'bout_num':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'bout_num'].values, 
                                                        'expNum':[expNum]*num_of_bouts,
                                                        })
                    
                    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_end']-this_exp_features['pitch_initial'],
                                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                                rot_l_decel=this_exp_features['pitch_end']-this_exp_features['pitch_peak'],
                                                                tsp=this_exp_features['traj_peak']-this_exp_features['pitch_peak'],
                                                                )        

                    bout_features = pd.concat([bout_features,this_exp_features])
                # combine data from different conditions
                cond1 = all_conditions[condition_idx].split("_")[0]
                cond2 = all_conditions[condition_idx].split("_")[1]
                all_cond1.append(cond1)
                all_cond2.append(cond2)
                all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
                    dpf=cond1,
                    condition=cond2
                    )])
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    return all_feature_cond, all_cond1, all_cond2