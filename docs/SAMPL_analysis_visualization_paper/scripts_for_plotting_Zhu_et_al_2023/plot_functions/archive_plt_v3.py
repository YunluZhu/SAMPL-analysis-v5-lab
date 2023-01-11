from lib2to3.pgen2.pgen import DFAState
import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import jackknife_list
from scipy.stats import pearsonr 
from plot_functions.plt_tools import round_half_up


def extract_bout_features_v3(bout_data,peak_idx, FRAME_RATE):
    """_summary_

    Args:
        bout_data (_type_): _description_
        peak_idx (_type_): _description_
        FRAME_RATE (_type_): _description_

    Returns:
        _type_: _description_
    """
    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_END = 0.25
    T_MID_ACCEL = -0.05
    T_MID_DECEL = 0.05
    
    idx_initial = int(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = int(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = int(peak_idx + T_POST_BOUT * FRAME_RATE)
    idx_mid_accel = int(peak_idx + T_MID_ACCEL * FRAME_RATE)
    idx_mid_decel = int(peak_idx + T_MID_DECEL * FRAME_RATE)
    idx_end = int(peak_idx + T_END * FRAME_RATE)
    
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
        'spd_mid_decel':bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_speed'].values, 
        
        
        'bout_num':bout_data.loc[bout_data['idx']==peak_idx,'bout_num'].values, 
    })
    pitch_mid_accel = bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].reset_index(drop=True)
    pitch_mid_decel = bout_data.loc[bout_data['idx']==idx_mid_decel,'propBoutAligned_pitch'].reset_index(drop=True)
    
    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_post_bout']-this_exp_features['pitch_initial'],
                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                rot_righting=this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                rot_early_accel = pitch_mid_accel-this_exp_features['pitch_pre_bout'],
                                                rot_late_accel = this_exp_features['pitch_peak'] - pitch_mid_accel,
                                                rot_early_decel = pitch_mid_decel-this_exp_features['pitch_peak'],
                                                rot_late_decel = this_exp_features['pitch_post_bout'] - pitch_mid_decel,
                                                tsp=this_exp_features['traj_peak']-this_exp_features['pitch_peak'],
                                                )  
    return this_exp_features


def get_kinematics(df):
    righting_fit = np.polyfit(x=df['pitch_pre_bout'], y=df['rot_righting'], deg=1)
    steering_fit = np.polyfit(x=df['pitch_peak'], y=df['traj_peak'], deg=1)

    righting_fit_dn = np.polyfit(x=df.loc[df['direction']=='dive','pitch_pre_bout'], 
                                y=df.loc[df['direction']=='dive','rot_righting'], 
                                deg=1)
    righting_fit_up = np.polyfit(x=df.loc[df['direction']=='climb','pitch_pre_bout'], 
                                y=df.loc[df['direction']=='climb','rot_righting'], 
                                deg=1)
    # posture_deviation = np.polyfit(x=df['pitch_peak'], y=df['tsp'], deg=1)
    set_point = np.polyfit(x=df['rot_total'], y=df['pitch_initial'], deg=1)

    corr_rot_accel_decel = pearsonr(x=df['rot_l_accel'],
                                    y=df['rot_righting'])
    corr_rot_lateAccel_decel = pearsonr(x=df['rot_late_accel'],
                            y=df['rot_righting'])
    corr_rot_earlyAccel_decel = pearsonr(x=df['rot_early_accel'],
                            y=df['rot_righting'])
    corr_rot_preBout_decel = pearsonr(x=df['rot_pre_bout'],
                            y=df['rot_righting'])
    # corr_rot_accel_decel = pearsonr(y=df['rot_l_accel'],
    #                                 x=df['rot_righting'])
    
    kinematics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'righting_gain_dn': -1 * righting_fit_dn[0],
        'righting_gain_up': -1 * righting_fit_up[0],
        'steering_gain': steering_fit[0],
        'corr_rot_accel_decel': corr_rot_accel_decel[0],
        'corr_rot_lateAccel_decel': corr_rot_lateAccel_decel[0],
        'corr_rot_earlyAccel_decel': corr_rot_earlyAccel_decel[0],
        'corr_rot_preBout_decel': corr_rot_preBout_decel[0],
        # 'posture_deviation_slope': posture_deviation[0],
        # 'posture_deviation_y': posture_deviation[1],
        'set_point':set_point[1],
    })
    return kinematics




