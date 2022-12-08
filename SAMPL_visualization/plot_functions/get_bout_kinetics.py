import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
from plot_functions.plt_tools import day_night_split
from plot_functions.get_index import get_index
from plot_functions.plt_tools import jackknife_list
from plot_functions.get_bout_features import (get_bout_features,extract_bout_features_v4)
from numpy.polynomial.polynomial import Polynomial
from scipy.stats import pearsonr 
from scipy.optimize import curve_fit


def sigmoid_fit2(x_val, y_val,func,revFunc,**kwargs):
    x0=[0.1, -10, 4, -5]
    p0 = tuple(x0)
    popt, pcov = curve_fit(func,x_val, y_val, 
                           p0 = p0,
                        )
    x = revFunc(0,*popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, x, p_sigma

def sigfunc_4free(x, a, b, c, d):
    y = c + (d)/(1 + np.exp(-(a*(x + b))))
    return y

def revSigfun(x,a,b,c,d):
    y = (-1)*b - np.log(d/(x-c)-1)/a
    return y

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
        if len(this_group_data)>10:
            this_group_kinetics = get_kinetics(this_group_data)
            this_group_kinetics = pd.concat([this_group_kinetics,pd.Series(data={'jackknife_group':j})])
            
            output = pd.concat([output,this_group_kinetics],axis=1)
    output = output.T.reset_index(drop=True)
    return output

def get_kinetics(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    steering_fit = np.polyfit(x=df['pitch_peak'], y=df['traj_peak'], deg=1)
    set_point_ori = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    corr_rot_accel_decel = pearsonr(x=df['rot_l_accel'],
                                    y=df['rot_l_decel'])
    corr_rot_lateAccel_decel = pearsonr(x=df['rot_late_accel'],
                            y=df['rot_l_decel'])
    corr_rot_preBout_decel = pearsonr(x=df['rot_pre_bout'],
                            y=df['rot_l_decel'])
    
    y_posture_corr = pearsonr(x=df['pitch_peak'],
                                y=df['depth_chg'])
    x_posture_corr = pearsonr(x=df['pitch_peak'],
                            y=df['x_chg'])

    y_posture_fit = np.polyfit(x=df['pitch_peak'], y=df['depth_chg'], deg=1)
    x_posture_fit = np.polyfit(x=df['pitch_peak'], y=df['x_chg'], deg=1)

    angvel_fit = np.polyfit(x=df['angvel_initial_phase'], y=df['angvel_chg'], deg=1)
    depth_chg_fit = np.polyfit(x=df['depth_chg'], y=df['additional_depth_chg'], deg=1)

    # sigmoid fit
    # righting_coef, righting_x_intersect, sigma = sigmoid_fit2(
    #         df['pitch_pre_bout'],
    #         df['rot_l_decel'], 
    #         func=sigfunc_4free, 
    #         revFunc=revSigfun
    #     )
    # sig_righting_gain = -1 * righting_coef.iloc[0,0]*(righting_coef.iloc[0,3])/4
    
    kinetics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'steering_gain': steering_fit[0],
        'corr_rot_accel_decel': corr_rot_accel_decel[0],
        'corr_rot_lateAccel_decel': corr_rot_lateAccel_decel[0],
        'corr_rot_preBout_decel': corr_rot_preBout_decel[0],
        'set_point':-set_point_ori[1]/set_point_ori[0],
        # 'sig_righting_gain': sig_righting_gain,
        # 'sig_set_point': righting_x_intersect,
        'y_posture_corr': y_posture_corr[0],
        'x_posture_corr': x_posture_corr[0],
        'y_efficacy': y_posture_fit[0],
        'x_efficacy': x_posture_fit[0],
        'depth_gain': depth_chg_fit[0],

        'angvel_gain': angvel_fit[1],
    })
    return kinetics

def get_set_poround_half_up(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    # steering_fit = np.polyfit(x=df['pitch_peak'], y=df['traj_peak'], deg=1)
    kinetics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        # 'steering_gain': steering_fit[0],
        'set_point':-1 * righting_fit[1]/righting_fit[0],
    })
    return kinetics

def get_kinetics_sigmoid(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_l_decel'], deg=1)
    steering_fit = np.polyfit(x=df['traj_peak'], y=df['pitch_peak'], deg=1)
    set_point_ori = np.polyfit(x=df['rot_l_decel'], y=df['pitch_pre_bout'], deg=1)

    corr_rot_accel_decel = pearsonr(x=df['rot_l_accel'],
                                    y=df['rot_l_decel'])
    corr_rot_lateAccel_decel = pearsonr(x=df['rot_late_accel'],
                            y=df['rot_l_decel'])
    corr_rot_preBout_decel = pearsonr(x=df['rot_pre_bout'],
                            y=df['rot_l_decel'])
    y_posture_corr = pearsonr(x=df['pitch_peak'],
                                    y=df['depth_chg'])
    x_posture_corr = pearsonr(x=df['pitch_peak'],
                            y=df['x_chg'])

    y_posture_fit = np.polyfit(x=df['pitch_peak'], y=df['depth_chg'], deg=1)
    x_posture_fit = np.polyfit(x=df['pitch_peak'], y=df['x_chg'], deg=1)
    
    depth_chg_fit = np.polyfit(x=df['depth_chg'], y=df['additional_depth_chg'], deg=1)

    kinetics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'steering_gain': steering_fit[0],
        'corr_rot_accel_decel': corr_rot_accel_decel[0],
        'corr_rot_lateAccel_decel': corr_rot_lateAccel_decel[0],
        'corr_rot_preBout_decel': corr_rot_preBout_decel[0],
        'set_point':set_point_ori[1],
        'y_efficacy': y_posture_fit[0],
        'x_efficacy': x_posture_fit[0],
        'y_posture_corr': y_posture_corr[0],
        # 'x_posture_corr': x_posture_corr[0],
        'depth_gain': depth_chg_fit[0],
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
    if_calc_bySpeed = 1
    sample_num = 0
    peak_idx , total_aligned = get_index(FRAME_RATE)
    T_start = -0.3
    T_end = 0.25
    idx_start = round_half_up(peak_idx + T_start * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_end * FRAME_RATE)
    idxRANGE = [idx_start,idx_end]
 
    # spd_bins = np.arange(3,18,3)  # warning, temp change
    # spd_bins = [3.5,5.5,7,10,100]
    TSP_THRESHOLD = [-np.Inf,-50,50,np.Inf]

    # for day night split
    which_zeitgeber = 'day'
    for key, value in kwargs.items():
        if key == 'ztime':
            which_zeitgeber = value
        if key == 'sample':
            sample_num = value

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
                    exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))

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
                    this_exp_features = extract_bout_features_v4(trunc_exp_data,peak_idx,FRAME_RATE)
                    this_exp_features = this_exp_features.assign(
                        bout_time = bout_time.values,
                        expNum = expNum,
                    )
                    # day night split. also assign ztime column
                    this_ztime_exp_features = day_night_split(this_exp_features,'bout_time',ztime=which_zeitgeber)
                    
                    this_ztime_exp_features = this_ztime_exp_features.assign(
                        direction = pd.cut(this_ztime_exp_features['pitch_initial'],[-90,10,90],labels=['DN','UP'])
                        )
                    
                    tsp_filter = pd.cut(this_ztime_exp_features['atk_ang'],TSP_THRESHOLD,labels=['too_neg','select','too_pos'])
                    this_ztime_exp_features = this_ztime_exp_features.loc[tsp_filter=='select',:].reset_index(drop=True)
                    if this_ztime_exp_features.groupby('ztime').size().min() < 10:
                        print(f"Too few bouts for kinetic analysis, consider removing the dataset: exp")
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
    spd_upper = np.percentile(all_feature_cond['spd_peak'],98)
    spd_lower = np.percentile(all_feature_cond['spd_peak'],2)
    spd_bins = np.arange(spd_lower,spd_upper,(spd_upper-spd_lower)/5) 
    all_feature_cond = all_feature_cond.assign(
        # direction = pd.cut(all_feature_cond['pitch_initial'],[-90,10,90],labels=['DN','UP']),
        speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
    )
    
    if sample_num != 0:
        all_feature_cond = all_feature_cond.groupby(
                ['dpf','condition','expNum','ztime']
                ).sample(
                        n=sample_num,
                        replace=True
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
    if if_calc_bySpeed == 1:
        for name, group in all_feature_cond.groupby(['condition','dpf','ztime']):
            kinetics_all_speed = pd.DataFrame()
            for speed_bin in set(group.speed_bins):
                if pd.notna(speed_bin):
                    this_speed_data = group.loc[group['speed_bins']==speed_bin,:]
                    # min_group_size = this_speed_data.groupby('expNum').size().min()
                    this_speed_kinetics = jackknife_kinetics(this_speed_data,'expNum')
                    this_speed_kinetics = this_speed_kinetics.assign(
                        speed_bins=speed_bin,
                        average_speed = this_speed_data['spd_peak'].mean(),
                        )
                    kinetics_all_speed = pd.concat([kinetics_all_speed,this_speed_kinetics],ignore_index=True)
            kinetics_all_speed = kinetics_all_speed.assign(
                condition = name[0],
                dpf = name[1],
                ztime = name[2]
                )   
            kinetics_bySpd_jackknife = pd.concat([kinetics_bySpd_jackknife, kinetics_all_speed],ignore_index=True)
        kinetics_bySpd_jackknife = kinetics_bySpd_jackknife.sort_values(by=['condition','jackknife_group','dpf']).reset_index(drop=True)

   
    return all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2

