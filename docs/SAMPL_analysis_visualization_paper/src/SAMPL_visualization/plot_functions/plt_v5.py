import pandas as pd # pandas library
import numpy as np # numpy
from plot_functions.plt_tools import jackknife_list
from plot_functions.plt_tools import round_half_up



def extract_bout_features_v5(bout_data,PEAK_IDX, FRAME_RATE,**kwargs):
    """extract bout features from analyzed bout data.

    Args:
        bout_data (dataFrame): bout data read from ('bout_data.h5', key='prop_bout_aligned')
        PEAK_IDX (numeric): index of the frame at time of peak speed
        FRAME_RATE (int): frame rate

    Returns:
        this_exp_features: extracted bout features
    """

    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_END = 0.2
    T_MID_ACCEL = -0.05
    T_MID_DECEL = 0.05
    T_MAX_ANGVEL = -0.04
    
    idx_initial = round_half_up(PEAK_IDX + T_INITIAL * FRAME_RATE)
    idx_pre_bout = round_half_up(PEAK_IDX + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = round_half_up(PEAK_IDX + T_POST_BOUT * FRAME_RATE)
    idx_end = round_half_up(PEAK_IDX + T_END * FRAME_RATE)
    idx_max_angvel = round_half_up(PEAK_IDX + T_MAX_ANGVEL * FRAME_RATE)

    for key, value in kwargs.items():
        if key == 'idx_max_angvel':
            idx_max_angvel = value
            
    this_exp_features = pd.DataFrame(data={
        'pitch_initial':bout_data.loc[bout_data['idx']==idx_initial,'propBoutAligned_pitch'].values, 
        'pitch_pre_bout':bout_data.loc[bout_data['idx']==idx_pre_bout,'propBoutAligned_pitch'].values, 
        'pitch_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_pitch'].values, 
        'pitch_post_bout':bout_data.loc[bout_data['idx']==idx_post_bout,'propBoutAligned_pitch'].values, 
        'pitch_end': bout_data.loc[bout_data['idx']==idx_end,'propBoutAligned_pitch'].values, 
        # 'pitch_mid_accel': bout_data.loc[bout_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].values, 
        'pitch_max_angvel': bout_data.loc[bout_data['idx']==idx_max_angvel,'propBoutAligned_pitch'].values, 
        'traj_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_instHeading'].values, 
        'spd_peak':bout_data.loc[bout_data['idx']==PEAK_IDX,'propBoutAligned_speed'].values, 
    })
    swim_indicator = bout_data['propBoutAligned_speed'] > 5
    y_swim = bout_data.loc[swim_indicator].groupby('bout_num').head(1)['propBoutAligned_y'].values - bout_data.loc[swim_indicator].groupby('bout_num').tail(1)['propBoutAligned_y'].values
    x_swim = bout_data.loc[swim_indicator].groupby('bout_num').head(1)['propBoutAligned_x'].values - bout_data.loc[swim_indicator].groupby('bout_num').tail(1)['propBoutAligned_x'].values
    displ = np.sqrt(np.square(y_swim) + np.square(x_swim))
    
    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_post_bout']-this_exp_features['pitch_initial'],
                                                 rot_steering = this_exp_features['pitch_peak']-this_exp_features['pitch_initial'],
                                                rot_righting = this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                rot_to_max_angvel = this_exp_features['pitch_max_angvel']-this_exp_features['pitch_initial'],
                                                bout_displ = displ,
                                                atk_ang = this_exp_features['traj_peak'] - this_exp_features['pitch_peak'],
                                                traj_deviation = this_exp_features['traj_peak'] - this_exp_features['pitch_initial'],
                                                )  
    return this_exp_features

def get_kinematics(df):
    """get kinematics measurements: righting gain, steering tgain, set point

    Args:
        df (Dataframe): dataframe with bout features

    Returns:
        series: dataframe with analyzed kinematics
    """
    righting_fit = np.polyfit(x=df['pitch_initial'], y=df['rot_righting'], deg=1)
    steering_fit = np.polyfit(x=df['traj_peak'], y=df['pitch_peak'], deg=1)
    kinematics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'steering_gain': steering_fit[0],
        'set_point':-1 * righting_fit[1]/righting_fit[0],
    })
    return kinematics

def get_set_point(df):
    righting_fit = np.polyfit(x=df['pitch_initial'], y=df['rot_righting'], deg=1)
    kinematics = pd.Series(data={
        'righting_gain': -1 * righting_fit[0],
        'set_point':-1 * righting_fit[1]/righting_fit[0],
    })
    return kinematics

def jackknife_kinematics(df,col):
    """jackknifed kinematics based on experimental repeats

    Args:
        df (datFrame): dataframe with bout features
        col (string): name of the column for jackknife calculation
    Returns:
        DataFrame: jackknife'd kinematics
    """
    exp_df = df.groupby(col).size()
    jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = df.loc[df[col].isin(exp_group),:]
        if len(this_group_data)>10:
            this_group_kinematics = get_kinematics(this_group_data)
            this_group_kinematics = pd.concat([this_group_kinematics,pd.Series(data={
                'jackknife_group':j
            }, dtype='float64')])
            output = pd.concat([output,this_group_kinematics],axis=1)
    output = output.T.reset_index(drop=True)
    return output
