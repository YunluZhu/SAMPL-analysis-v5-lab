import os,glob
from pickle import FRAME
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
from plot_functions.plt_tools import (day_night_split)
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter


def extract_bout_features_v4(bout_data,peak_idx, FRAME_RATE, **kwargs):
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
    T_PREP_275 = -0.275
    T_PREP_150 = -0.15
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_post_150 = 0.15
    T_END = 0.2
    T_MID_ACCEL = -0.05
    T_BODY_END_40 = -0.04
    T_BODY_END_30 = -0.03

    T_MID_DECEL = 0.05

    T_PRE_15 = -0.015
    T_POST_15 = 0.015
    T_MAX_ANGVEL = -0.05
    
    idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = round_half_up(peak_idx + T_POST_BOUT * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx + T_MID_ACCEL * FRAME_RATE)
    idx_mid_decel = round_half_up(peak_idx + T_MID_DECEL * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_END * FRAME_RATE)
    idx_body_end_40 = round_half_up(peak_idx + T_BODY_END_40 * FRAME_RATE)
    idx_body_end_30 = round_half_up(peak_idx + T_BODY_END_30 * FRAME_RATE)
    idx_max_angvel = round_half_up(peak_idx + T_MAX_ANGVEL * FRAME_RATE)
    idx_prep_200 = round_half_up(peak_idx + T_PREP_200 * FRAME_RATE)
    idx_pre_275 = round_half_up(peak_idx + T_PREP_275 * FRAME_RATE)
    idx_prep_150 = round_half_up(peak_idx + T_PREP_150 * FRAME_RATE)
    idx_post_150 = round_half_up(peak_idx + T_post_150 * FRAME_RATE)
    idx_pre_15 = round_half_up(peak_idx + T_PRE_15 * FRAME_RATE)
    idx_post_15 = round_half_up(peak_idx + T_POST_15 * FRAME_RATE)
    
    
    idx_initial_phase = np.arange(idx_pre_275,idx_initial)
    idx_prep_phase = np.arange(idx_prep_200,idx_pre_bout)
    idx_accel_phase = np.arange(idx_pre_bout,peak_idx)
    idx_decel_phase = np.arange(peak_idx,idx_post_bout)
    idx_post_phase = np.arange(idx_post_150,idx_end)
    idx_peak_phase = np.arange(idx_pre_15,idx_post_15)
    
    for key, value in kwargs.items():
        if key == 'idx_max_angvel':
            idx_max_angvel = value
    
    
    this_exp_features = pd.DataFrame(data={
        'pitch_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_pitch'].values, 
        'pitch_mid_accel':bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].values, 
        'pitch_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values, 
        'pitch_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_pitch'].values, 
        'pitch_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_pitch'].values, 
        'pitch_end': bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_pitch'].values, 
        'pitch_max_angvel': bout_data.loc[bout_data['idx']==idx_max_angvel,'propBoutAligned_pitch'].values, 
        'pitch_body_end_40':bout_data.loc[bout_data['idx']==idx_body_end_40,'propBoutAligned_pitch'].values, 
        'pitch_body_end_30': bout_data.loc[bout_data['idx']==idx_body_end_30,'propBoutAligned_pitch'].values,       
        
        
        # 'pitch_25ms_accel':bout_data.loc[bout_data['idx']==idx_pre_25,'propBoutAligned_pitch'].values, 
        
        'traj_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_instHeading'].values, 
        'traj_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_instHeading'].values, 
        'traj_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_instHeading'].values, 
        'traj_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_instHeading'].values, 
        'traj_end':bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_instHeading'].values, 
        
        'spd_peak':bout_data.loc[bout_data['idx']==peak_idx,'propBoutAligned_speed'].values, 
        
        'angvel_initial_phase': bout_data.loc[bout_data['idx'].isin(idx_initial_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 
        'angvel_prep_phase': bout_data.loc[bout_data['idx'].isin(idx_prep_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 
        'angvel_post_phase': bout_data.loc[bout_data['idx'].isin(idx_post_phase),:].groupby('bout_num')['propBoutAligned_angVel'].mean().values, 
        
        'pitch_initial_phase': bout_data.loc[bout_data['idx'].isin(idx_initial_phase),:].groupby('bout_num')['propBoutAligned_pitch'].mean().values, 
        'pitch_peak_phase': bout_data.query('idx in @idx_peak_phase').groupby('bout_num')['propBoutAligned_pitch'].mean().values, 
        'traj_peak_phase': bout_data.query('idx in @idx_peak_phase').groupby('bout_num')['propBoutAligned_instHeading'].mean().values, 
        # 'angvel_accel_phase':
        # 'angvel_decel_phase':
        # 'angvel_post_phase':
        
        # 'spd_mid_decel':bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_speed'].values, 
        # 'bout_num':bout_data.loc[bout_data['idx']==peak_idx,'bout_num'].values, 
    })
    
    # calculate attack angles
    # bout trajectory is the same as (bout_data.h5, key='prop_bout2')['epochBouts_trajectory']
    yy = (bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_y'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_y'].values)
    xx = (bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_x'].values - bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_x'].values)

    absxx = np.absolute(xx)
    epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
    displ = np.sqrt(np.square(yy) + np.square(absxx))
    
    swim_indicator = bout_data['propBoutAligned_speed'] > 4
    y_swim = bout_data.loc[swim_indicator].groupby('bout_num').head(1)['propBoutAligned_y'].values - bout_data.loc[swim_indicator].groupby('bout_num').tail(1)['propBoutAligned_y'].values
    x_swim = bout_data.loc[swim_indicator].groupby('bout_num').head(1)['propBoutAligned_x'].values - bout_data.loc[swim_indicator].groupby('bout_num').tail(1)['propBoutAligned_x'].values
    displ_swim = np.sqrt(np.square(y_swim) + np.square(x_swim))

    # pitch_mid_accel = bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].reset_index(drop=True)
    pitch_mid_decel = bout_data.loc[bout_data['idx']==idx_mid_decel,'propBoutAligned_pitch'].reset_index(drop=True)

    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_end']-this_exp_features['pitch_initial'],
                                                 rot_bout = this_exp_features['pitch_post_bout']-this_exp_features['pitch_pre_bout'],
                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],

                                                # rot_pre_50 = this_exp_features['pitch_mid_accel']-this_exp_features['pitch_initial'],
                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                rot_full_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_initial'],
                                                rot_full_decel=this_exp_features['pitch_end']-this_exp_features['pitch_peak'],

                                                rot_l_decel=this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                rot_early_accel = this_exp_features['pitch_mid_accel']-this_exp_features['pitch_pre_bout'],
                                                rot_late_accel = this_exp_features['pitch_peak'] - this_exp_features['pitch_mid_accel'],
                                                rot_early_decel = pitch_mid_decel-this_exp_features['pitch_peak'],
                                                rot_late_decel = this_exp_features['pitch_post_bout'] - pitch_mid_decel,
                                                #phased rotation for more robust calculation
                                                rot_pre_bout_phased=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial_phase'],
                                                rot_full_accel_phased=this_exp_features['pitch_peak_phase']-this_exp_features['pitch_initial_phase'],
                                                rot_early_body_change=this_exp_features['pitch_max_angvel']-this_exp_features['pitch_initial'],
                                                rot_to_max_angvel=this_exp_features['pitch_max_angvel']-this_exp_features['pitch_initial'],
                                                bout_trajectory = epochBouts_trajectory,
                                                bout_displ = displ,
                                                traj_deviation = this_exp_features['traj_peak'] -  this_exp_features['pitch_initial'],
                                                atk_ang = this_exp_features['traj_peak'] - this_exp_features['pitch_peak'],
                                                atk_ang_phased = this_exp_features['traj_peak_phase'] - this_exp_features['pitch_peak_phase'],
                                                # tsp_pre = this_exp_features['traj_pre_bout'] - this_exp_features['pitch_pre_bout'],
                                                tsp_peak = this_exp_features['traj_peak'] - this_exp_features['pitch_peak'],
                                                angvel_chg = this_exp_features['angvel_post_phase'] - this_exp_features['angvel_initial_phase'],
                                                depth_chg = yy,
                                                displ_swim = displ_swim,
                                                x_chg = xx,
                                                )  
    return this_exp_features


def get_bout_features(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]
    max_angvel_time = -60 # by default 
    max_angvel_df = pd.DataFrame()

    # for day night split
    which_zeitgeber = 'day'
    for key, value in kwargs.items():
        if key == 'ztime':
            which_zeitgeber = value
        elif key == 'max_angvel_time':
            max_angvel_df = value

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
        cond1 = all_conditions[condition_idx].split("_")[0]
        cond2 = all_conditions[condition_idx].split("_")[1]
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
                    exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)),
                                               expNum = expNum,
                                               exp=exp)
                    # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                    bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,'aligned_time']
                    
                    # truncate first, just incase some aligned bouts aren't complete
                    for i in bout_time.index:
                        rows.extend(list(range(i*total_aligned+round_half_up(idxRANGE[0]),i*total_aligned+round_half_up(idxRANGE[1]))))
                    
                    # assign bout numbers
                    trunc_exp_data = exp_data.loc[rows,:]
                    trunc_exp_data = trunc_exp_data.assign(
                        bout_num = trunc_exp_data.groupby(np.arange(len(trunc_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
                    )
                    if not max_angvel_df.empty:
                        max_angvel_time = max_angvel_df.loc[
                            (max_angvel_df.cond1 == cond1) & (max_angvel_df.cond2 == cond2), 'max_angvel_time'
                            ].item()
                    max_angvel_idx = round_half_up(peak_idx + max_angvel_time/1000*FRAME_RATE)
                        
                    this_exp_features = extract_bout_features_v4(trunc_exp_data,peak_idx,FRAME_RATE,idx_max_angvel=max_angvel_idx)
                    this_exp_features = this_exp_features.assign(
                        bout_time = bout_time.values,
                        expNum = expNum,
                    )
                    # day night split. also assign ztime column
                    this_ztime_exp_features = day_night_split(this_exp_features,'bout_time',ztime=which_zeitgeber)
                    
                    bout_features = pd.concat([bout_features,this_ztime_exp_features])
        # combine data from different conditions
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


def get_max_angvel_rot(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx - 0.1 * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx - 0.05 * FRAME_RATE)
    idx_initial = round_half_up(peak_idx - 0.25 * FRAME_RATE)

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s
    
    all_conditions = []
    folder_paths = []
    all_cond1 = []
    all_cond2 = []
    exp_data_all = pd.DataFrame()
    idxRANGE = [peak_idx-round_half_up(BEFORE_PEAK*FRAME_RATE),peak_idx+round_half_up(AFTER_PEAK*FRAME_RATE)]

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

    all_cond1 = []
    all_cond2 = []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                # reset for each condition
                this_cond_data = pd.DataFrame()
                subdir_list.sort()
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
                    
                    adj_angvel = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)

                    selected_range = selected_range.assign(
                        adj_angvel = adj_angvel,
                    )

                    exp_data = selected_range
                    exp_data = exp_data.assign(
                        time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                        expNum = expNum)
                    this_cond_data = pd.concat([this_cond_data,exp_data.loc[rows,:]])
                
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
    
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    # calculate time of max angaccel, mean of each exp, then average
    mean_angAccel = exp_data_all.groupby(['time_ms','cond1','cond2','expNum'])['adj_angvel'].median().reset_index()
    mean_angAccel = mean_angAccel.loc[mean_angAccel['time_ms']<0]
    idx_mean_max = mean_angAccel.groupby(['cond1','cond2','expNum'])['adj_angvel'].apply(
        lambda y: np.argmax(y)
    )
    time_by_bout_max = ((idx_mean_max/FRAME_RATE - BEFORE_PEAK)*1000).reset_index()
    # condition_match = exp_data_all.groupby(['expNum','cond2'])['cond2','cond1'].head(1)

    time_by_bout_max.columns = ['cond1','cond2','expNum','max_angvel_time']
    # time_by_bout_max = time_by_bout_max.assign(
    #     cond1 = condition_match['cond1'].values,
    #     cond2 = condition_match['cond2'].values,
    # )
    max_angvel_time = time_by_bout_max.groupby(['cond1','cond2'])['max_angvel_time'].mean()
    max_angvel_time = max_angvel_time.reset_index()
    
    return max_angvel_time, all_cond1, all_cond2

def get_max_angvel_rot_individualRep(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx - 0.1 * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx - 0.05 * FRAME_RATE)
    idx_initial = round_half_up(peak_idx - 0.25 * FRAME_RATE)

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s
    
    all_conditions = []
    folder_paths = []
    all_cond1 = []
    all_cond2 = []
    exp_data_all = pd.DataFrame()
    idxRANGE = [peak_idx-round_half_up(BEFORE_PEAK*FRAME_RATE),peak_idx+round_half_up(AFTER_PEAK*FRAME_RATE)]

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

    all_cond1 = []
    all_cond2 = []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                # reset for each condition
                this_cond_data = pd.DataFrame()
                subdir_list.sort()
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
                    
                    adj_angvel = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)

                    selected_range = selected_range.assign(
                        adj_angvel = adj_angvel,
                    )

                    exp_data = selected_range
                    exp_data = exp_data.assign(
                        time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                        expNum = expNum)
                    this_cond_data = pd.concat([this_cond_data,exp_data.loc[rows,:]])
                
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
    
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    # calculate time of max angaccel, mean of each exp, then average
    mean_angAccel = exp_data_all.groupby(['time_ms','cond1','cond2','expNum'])['adj_angvel'].median().reset_index()
    mean_angAccel = mean_angAccel.loc[mean_angAccel['time_ms']<0]
    idx_mean_max = mean_angAccel.groupby(['cond1','cond2','expNum'])['adj_angvel'].apply(
        lambda y: np.argmax(y)
    )
    time_by_bout_max = ((idx_mean_max/FRAME_RATE - BEFORE_PEAK)*1000).reset_index()
    # condition_match = exp_data_all.groupby(['expNum','cond2'])['cond2','cond1'].head(1)

    time_by_bout_max.columns = ['cond1','cond2','expNum','max_angvel_time']
    # time_by_bout_max = time_by_bout_max.assign(
    #     cond1 = condition_match['cond1'].values,
    #     cond2 = condition_match['cond2'].values,
    # )
    # max_angvel_time = time_by_bout_max.groupby(['cond1','cond2'])['max_angvel_time'].mean()
    # max_angvel_time = max_angvel_time.reset_index()
    
    return time_by_bout_max, all_cond1, all_cond2


def get_max_angvel_rot_byBout(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx - 0.1 * FRAME_RATE)
    idx_mid_accel = round_half_up(peak_idx - 0.05 * FRAME_RATE)
    idx_initial = round_half_up(peak_idx - 0.25 * FRAME_RATE)

    BEFORE_PEAK = 0.3 # s
    AFTER_PEAK = 0.2 #s
    
    all_conditions = []
    folder_paths = []
    all_cond1 = []
    all_cond2 = []
    exp_data_all = pd.DataFrame()
    idxRANGE = [peak_idx-round_half_up(BEFORE_PEAK*FRAME_RATE),peak_idx+round_half_up(AFTER_PEAK*FRAME_RATE)]

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

    all_cond1 = []
    all_cond2 = []
    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                # reset for each condition
                this_cond_data = pd.DataFrame()
                subdir_list.sort()
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
                    
                    adj_angvel = selected_range['propBoutAligned_angVel_sm'] * (np.repeat(adj_by_which,(idxRANGE[1]-idxRANGE[0])).values)

                    selected_range = selected_range.assign(
                        adj_angvel = adj_angvel,
                    )

                    exp_data = selected_range
                    exp_data = exp_data.assign(
                        time_ms = (exp_data['idx']-peak_idx)/FRAME_RATE*1000,
                        expNum = expNum)
                    this_cond_data = pd.concat([this_cond_data,exp_data.loc[rows,:]])
                
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
    
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    # calculate time of max angaccel, mean of each exp, then average
    # mean_angAccel = exp_data_all.groupby(['time_ms','cond1','cond2','expNum'])['adj_angvel'].median().reset_index()
    
    before_peak_data = exp_data_all.loc[(exp_data_all['time_ms']<0) & (exp_data_all['time_ms']>-200)]
    before_peak_data = before_peak_data.assign(
        bout_number = before_peak_data.groupby(np.arange(len(before_peak_data))//(idxRANGE[1]-idxRANGE[0])).ngroup().values
    )
    idx_max = before_peak_data.groupby(['cond1','cond2','expNum','bout_number'])['adj_angvel'].apply(
        lambda y: np.argmax(y)
    )
    time_by_bout_max = ((idx_max/FRAME_RATE - BEFORE_PEAK)*1000).reset_index()
    # condition_match = exp_data_all.groupby(['expNum','cond2'])['cond2','cond1'].head(1)
    time_byExp = time_by_bout_max.groupby(['cond1','cond2','expNum']).mean().reset_index()

    time_byExp.columns = ['cond1','cond2','expNum','bout_number','max_angvel_time']
    time_byExp.drop(columns='bout_number',inplace=True)
    return time_byExp, all_cond1, all_cond2