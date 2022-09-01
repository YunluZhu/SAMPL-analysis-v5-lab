import os,glob
import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import (day_night_split)
from plot_functions.get_index import get_index


def extract_bout_features_v4(bout_data,peak_idx, FRAME_RATE):
    """_summary_

    Args:
        bout_data (_type_): _description_
        peak_idx (_type_): _description_
        FRAME_RATE (_type_): _description_

    Returns:
        _type_: _description_
    """
    T_INITIAL = -0.25 #s
    T_PREP_200 = -0.2
    T_PREP_150 = -0.15
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_post_150 = 0.15
    T_END = 0.2
    T_MID_ACCEL = -0.05
    T_MID_DECEL = 0.05

    
    idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
    idx_mid_accel = int(peak_idx + T_MID_ACCEL * FRAME_RATE)
    idx_mid_decel = int(peak_idx + T_MID_DECEL * FRAME_RATE)
    idx_end = int(peak_idx + T_END * FRAME_RATE)
    
    idx_prep_200 = int(peak_idx + T_PREP_200 * FRAME_RATE)
    idx_prep_150 = int(peak_idx + T_PREP_150 * FRAME_RATE)
    idx_post_150 = int(peak_idx + T_post_150 * FRAME_RATE)
    
    idx_initial_phase = np.arange(idx_initial,idx_pre_bout)
    idx_prep_phase = np.arange(idx_prep_200,idx_prep_150)
    idx_accel_phase = np.arange(idx_pre_bout,peak_idx)
    idx_decel_phase = np.arange(peak_idx,idx_post_bout)
    idx_post_phase = np.arange(idx_post_150,idx_end)
    
    this_exp_features = pd.DataFrame(data={
        'pitch_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_pitch'].values, 
        'pitch_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values, 
        'pitch_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_pitch'].values, 
        'pitch_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_pitch'].values, 
        'pitch_end': bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_pitch'].values, 
        
        'traj_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_instHeading'].values, 
        'traj_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_instHeading'].values, 
        'traj_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_instHeading'].values, 
        'traj_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_instHeading'].values, 
        'traj_end':bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_instHeading'].values, 

        'spd_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_speed'].values, 
        
        'angvel_initial_phase': bout_data.loc[bout_data['idx'].isin(idx_initial_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 
        'angvel_prep_phase': bout_data.loc[bout_data['idx'].isin(idx_prep_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 
        'pitch_prep_phase': bout_data.loc[bout_data['idx'].isin(idx_prep_phase),:].groupby('bout_num')['propBoutAligned_pitch'].mean().values, 
        'angvel_post_phase': bout_data.loc[bout_data['idx'].isin(idx_post_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 

        # 'angvel_accel_phase':
        # 'angvel_decel_phase':
        # 'angvel_post_phase':
        
        # 'spd_mid_decel':bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_speed'].values, 
        # 'bout_num':bout_data.loc[bout_data['idx']==peak_idx,'bout_num'].values, 
    })
    
    # calculate attack angles
    # bout trajectory is the same as (bout_data.h5, key='prop_bout2')['epochBouts_trajectory']
    yy = (bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_y'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_y'].values)
    absxx = np.absolute((bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_x'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_x'].values))
    epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
    displ = np.sqrt(np.square(yy) + np.square(absxx))

    pitch_mid_accel = bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].reset_index(drop=True)
    pitch_mid_decel = bout_data.loc[bout_data['idx']==idx_mid_decel,'propBoutAligned_pitch'].reset_index(drop=True)
    
    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_end']-this_exp_features['pitch_initial'],
                                                 rot_bout = this_exp_features['pitch_post_bout']-this_exp_features['pitch_pre_bout'],
                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                rot_l_decel=this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                rot_early_accel = pitch_mid_accel-this_exp_features['pitch_pre_bout'],
                                                rot_late_accel = this_exp_features['pitch_peak'] - pitch_mid_accel,
                                                rot_early_decel = pitch_mid_decel-this_exp_features['pitch_peak'],
                                                rot_late_decel = this_exp_features['pitch_post_bout'] - pitch_mid_decel,
                                                bout_traj = epochBouts_trajectory,
                                                bout_displ = displ,
                                                atk_ang = epochBouts_trajectory - this_exp_features['pitch_pre_bout'],
                                                # tsp_pre = this_exp_features['traj_pre_bout'] - this_exp_features['pitch_pre_bout'],
                                                tsp_peak = this_exp_features['traj_peak'] - this_exp_features['pitch_peak'],
                                                angvel_chg = this_exp_features['angvel_post_phase'] - this_exp_features['angvel_initial_phase'] 
                                                )  
    return this_exp_features

def get_bout_features(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = int(peak_idx + T_start * FRAME_RATE)
    idx_end = int(peak_idx + T_end * FRAME_RATE)
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
                # reset for each condition
                bout_features = pd.DataFrame()
                subdir_list.sort()
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # angular velocity (angVel) calculation
                    rows = []
                    night_rows = []
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch            
                    exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                    exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs())
                    # assign frame number, total_aligned frames per bout
                    exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)),
                                               expNum = expNum,
                                               exp=exp)
                    # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                    bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,'aligned_time']
                    
                    # truncate first, just incase some aligned bouts aren't complete
                    for i in bout_time.index:
                        rows.extend(list(range(i*total_aligned+int(idxRANGE[0]),i*total_aligned+int(idxRANGE[1]))))
                    
                    # assign bout numbers
                    trunc_exp_data = exp_data.loc[rows,:]
                    trunc_exp_data = trunc_exp_data.assign(
                        bout_num = trunc_exp_data.groupby(np.arange(len(trunc_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
                    )
                    this_exp_features = extract_bout_features_v4(trunc_exp_data,peak_idx,FRAME_RATE)
                    this_exp_features = this_exp_features.assign(
                        bout_time = bout_time.values,
                        expNum = expNum,
                    )
                    # day night split. also assign ztime column
                    this_ztime_exp_features = day_night_split(this_exp_features,'bout_time',ztime=which_zeitgeber)
                    
                    bout_features = pd.concat([bout_features,this_ztime_exp_features])
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
