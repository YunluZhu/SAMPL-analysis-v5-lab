import os,glob
import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import day_night_split
from plot_functions.get_index import get_index
from plot_functions.plt_tools import jackknife_list
from plot_functions.get_bout_features import (get_bout_features,extract_bout_features_v4)
from scipy.stats import pearsonr 

def jackknife_kinetics(df,col):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    exp_df = df.groupby(col).size()
    jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = df.loc[df[col].isin(exp_group),:]
        this_group_kinetics = get_kinetics(this_group_data)
        this_group_kinetics = this_group_kinetics.append(pd.Series(data={
            'jackknife_group':j
        }))
        output = pd.concat([output,this_group_kinetics],axis=1)
    output = output.T.reset_index(drop=True)
    return output

def get_kinetics(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    righting_fit_early = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_early_decel'], deg=1)
    righting_fit_late = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_late_decel'], deg=1)

    steering_fit = np.polyfit(x=df['pitch_peak'], y=df['traj_peak'], deg=1)
    # if 'direction' in df.columns:
    #     righting_fit_dn = np.polyfit(x=df.loc[df['direction']=='dive','pitch_pre_bout'], 
    #                                 y=df.loc[df['direction']=='dive','rot_l_decel'], 
    #                                 deg=1)
    #     righting_fit_up = np.polyfit(x=df.loc[df['direction']=='climb','pitch_pre_bout'], 
    #                                 y=df.loc[df['direction']=='climb','rot_l_decel'], 
    #                                 deg=1)
        # corr_rot_lateAccel_decel_up = pearsonr(
        #     x=df.loc[df['direction']=='climb','rot_late_accel'],
        #     y=df.loc[df['direction']=='climb','rot_l_decel'])
    angvel_correct_fit_dn = np.polyfit(x=df.loc[df['angvel_initial_phase']<0,'angvel_initial_phase'],
                                        y=df.loc[df['angvel_initial_phase']<0,'angvel_chg'], 
                                        deg=1)
    angvel_correct_fit_up = np.polyfit(x=df.loc[df['angvel_initial_phase']>0,'angvel_initial_phase'],
                                        y=df.loc[df['angvel_initial_phase']>0,'angvel_chg'], 
                                        deg=1)   
    # posture_deviation = np.polyfit(x=df['pitch_peak'], y=df['tsp'], deg=1)
    set_point_new = np.polyfit(x=df['rot_total'], y=df['pitch_initial'], deg=1)
    set_point_ori = np.polyfit(x=df['rot_l_decel'], y=df['pitch_pre_bout'], deg=1)

    corr_rot_accel_decel = pearsonr(x=df['rot_l_accel'],
                                    y=df['rot_l_decel'])
    corr_rot_lateAccel_decel = pearsonr(x=df['rot_late_accel'],
                            y=df['rot_l_decel'])

    # corr_rot_earlyAccel_decel = pearsonr(x=df['rot_early_accel'],
    #                         y=df['rot_l_decel'])
    corr_rot_preBout_decel = pearsonr(x=df['rot_pre_bout'],
                            y=df['rot_l_decel'])
    # corr_rot_accel_decel = pearsonr(y=df['rot_l_accel'],
    #                                 x=df['rot_l_decel'])
    
    kinetics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'righting_gain_early': -1 * righting_fit_early[0],
        'righting_gain_late': -1 * righting_fit_late[0],

        'steering_gain': steering_fit[0],
        'corr_rot_accel_decel': corr_rot_accel_decel[0],
        'corr_rot_lateAccel_decel': corr_rot_lateAccel_decel[0],

        # 'corr_rot_earlyAccel_decel': corr_rot_earlyAccel_decel[0],
        'corr_rot_preBout_decel': corr_rot_preBout_decel[0],
        # 'posture_deviation_slope': posture_deviation[0],
        # 'posture_deviation_y': posture_deviation[1],
        # 'set_point_new':set_point_new[1],
        'set_point':set_point_ori[1],

        'angvel_gain_neg': -1 * angvel_correct_fit_dn[0],
        'angvel_gain_pos': -1 * angvel_correct_fit_up[0],
    })
    if 'direction' in df.columns:
        direction_kinetics = pd.Series(data={
            # 'righting_gain_dn':  -1 * righting_fit_dn[0],
            # 'righting_gain_up':  -1 * righting_fit_up[0],
            # 'corr_rot_lateAccel_decel_up': corr_rot_lateAccel_decel_up[0],
            
        })
        kinetics = pd.concat([kinetics, direction_kinetics])
    return kinetics

def get_bout_kinetics(root, FRAME_RATE,**kwargs):
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = int(peak_idx + T_start * FRAME_RATE)
    idx_end = int(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]
    spd_bins = np.arange(3,24,3)

    TSP_THRESHOLD = [-np.Inf,-50,50,np.Inf]

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
    all_kinetic_cond = pd.DataFrame()
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
                bout_kinetics = pd.DataFrame()

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
                    exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))

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
                    
                    this_ztime_exp_features = this_ztime_exp_features.assign(
                        direction = pd.cut(this_ztime_exp_features['pitch_peak'],[-90,10,90],labels=['dive','climb'])
                        )
                    
                    tsp_filter = pd.cut(this_ztime_exp_features['tsp_peak'],TSP_THRESHOLD,labels=['too_neg','select','too_pos'])
                    this_ztime_exp_features = this_ztime_exp_features.loc[tsp_filter=='select',:].reset_index(drop=True)
                    
                    this_exp_kinetics = this_ztime_exp_features.groupby('ztime').apply(
                        lambda x: get_kinetics(x)
                    ).reset_index()
                    this_exp_kinetics = this_exp_kinetics.assign(expNum = expNum)
                    
                    bout_features = pd.concat([bout_features,this_ztime_exp_features])
                    bout_kinetics = pd.concat([bout_kinetics,this_exp_kinetics], ignore_index=True)
                
            # combine data from different conditions
            cond1 = all_conditions[condition_idx].split("_")[0]
            cond2 = all_conditions[condition_idx].split("_")[1]
            all_cond1.append(cond1)
            all_cond2.append(cond2)
            all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
                dpf=cond1,
                condition=cond2
                )])
            all_kinetic_cond = pd.concat([all_kinetic_cond, bout_kinetics.assign(
                dpf=cond1,
                condition=cond2
                )])
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()
    
    all_feature_cond = all_feature_cond.assign(
        direction = pd.cut(all_feature_cond['pitch_peak'],[-80,0,80],labels=['dive','climb']),
        speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
    )
    
    # calculate jackknifed kinetics
    kinetics_jackknife = pd.DataFrame()
    for name, group in all_feature_cond.groupby(['condition','dpf','ztime']):
        this_group_kinetics = jackknife_kinetics(group,'expNum')
        this_group_kinetics = this_group_kinetics.assign(
            condition = name[0],
            dpf = name[1],
            ztime = name[2])
        kinetics_jackknife = pd.concat([kinetics_jackknife,this_group_kinetics],ignore_index=True)
    
    cat_cols = ['jackknife_group','condition','dpf','ztime']
    kinetics_jackknife.rename(columns={c:c+'_jack' for c in kinetics_jackknife.columns if c not in cat_cols},inplace=True)
    kinetics_jackknife = kinetics_jackknife.sort_values(by=['condition','jackknife_group','dpf']).reset_index(drop=True)

    # calculate jackknifed kinetics by speed bins
    kinetics_bySpd_jackknife = pd.DataFrame()
    for name, group in all_feature_cond.groupby(['condition','dpf','ztime']):
        kinetics_all_speed = pd.DataFrame()
        for speed_bin in set(group.speed_bins):
            if pd.notna(speed_bin):
                this_speed_data = group.loc[group['speed_bins']==speed_bin,:]
                this_speed_kinetics = jackknife_kinetics(this_speed_data,'expNum')
                this_speed_kinetics = this_speed_kinetics.assign(speed_bins=speed_bin)
                kinetics_all_speed = pd.concat([kinetics_all_speed,this_speed_kinetics],ignore_index=True)
        kinetics_all_speed = kinetics_all_speed.assign(
            condition = name[0],
            dpf = name[1],
            ztime = name[2]
            )   
        kinetics_bySpd_jackknife = pd.concat([kinetics_bySpd_jackknife, kinetics_all_speed],ignore_index=True)
    kinetics_bySpd_jackknife = kinetics_bySpd_jackknife.sort_values(by=['condition','jackknife_group','dpf']).reset_index(drop=True)
   
   
    return all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2

