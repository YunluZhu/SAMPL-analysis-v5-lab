#%%
import os
from plot_functions.plt_tools import round_half_up
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import ( get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, day_night_split)


def Fig6_pitch_timeseries(root):
    set_font_type()

    FRAME_RATE = 166
    
    folder_name = f'Pitch changes'
    folder_dir6 = get_figure_dir('Fig_6')
    fig_dir6 = os.path.join(folder_dir6, folder_name)

    try:
        os.makedirs(fig_dir6)
    except:
        pass   
    peak_idx , total_aligned = get_index(FRAME_RATE)
    idxRANGE = [peak_idx-round_half_up(0.27*FRAME_RATE),peak_idx+round_half_up(0.22*FRAME_RATE)]
    spd_bins = np.arange(5,25,4)

    # %%
    # CONSTANTS
    # %%
    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.1 #s
    T_END = 0.2

    separation_posture = 10
    idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
    idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
    idx_post_bout = round_half_up(peak_idx + T_POST_BOUT * FRAME_RATE)
    idx_end = round_half_up(peak_idx + T_END * FRAME_RATE)
    idx_dur300ms = round_half_up(0.3*FRAME_RATE)
    idx_dur250ms = round_half_up(0.25*FRAME_RATE)

    # %%
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)

    all_around_peak_data = pd.DataFrame()
    all_cond1 = []
    all_cond2 = []

    # go through each condition folders under the root
    for condition_idx, folder in enumerate(folder_paths):
        # enter each condition folder (e.g. 7dd_ctrl)
        for subpath, subdir_list, subfile_list in os.walk(folder):
            # if folder is not empty
            if subdir_list:
                # reset for each condition
                around_peak_data = pd.DataFrame()
                # loop through each sub-folder (experiment) under each condition
                for expNum, exp in enumerate(subdir_list):
                    # angular velocity (angVel) calculation
                    rows = []
                    # for each sub-folder, get the path
                    exp_path = os.path.join(subpath, exp)
                    # get pitch                
                    exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                    exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs(),
                                                yvel = exp_data['propBoutAligned_y'].diff()*FRAME_RATE,
                                                xvel = exp_data['propBoutAligned_x'].diff()*FRAME_RATE,
                                                linear_accel = exp_data['propBoutAligned_speed'].diff(),
                                                ang_accel_of_SMangVel = exp_data['propBoutAligned_angVel'].diff(),
                                            )
                    # assign frame number, total_aligned frames per bout
                    exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                    
                    # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                    bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                    # for i in bout_time.index:
                    # # if only need day or night bouts:
                    for i in day_night_split(bout_time,'aligned_time').index:
                        rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    exp_data = exp_data.assign(expNum = expNum,
                                            exp_id = condition_idx*100+expNum)
                    around_peak_data = pd.concat([around_peak_data,exp_data.loc[rows,:]])
                # combine data from different conditions
                cond1 = all_conditions[condition_idx].split("_")[0]
                all_cond1.append(cond1)
                cond2 = all_conditions[condition_idx].split("_")[1]
                all_cond2.append(cond2)
                all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(condition0=cond1,
                                                                                                condition=cond2)])
    all_around_peak_data = all_around_peak_data.assign(
        time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000,
    )
    # %% tidy data
    all_cond1 = list(set(all_cond1))
    all_cond1.sort()
    all_cond2 = list(set(all_cond2))
    all_cond2.sort()

    all_around_peak_data = all_around_peak_data.reset_index(drop=True)
    peak_speed = all_around_peak_data.loc[all_around_peak_data.idx==peak_idx,'propBoutAligned_speed'],

    grp = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))
    all_around_peak_data = all_around_peak_data.assign(
        peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])),
        bout_number = grp.ngroup(),
                                    )
    all_around_peak_data = all_around_peak_data.assign(
                                        speed_bin = pd.cut(all_around_peak_data['peak_speed'],spd_bins,labels = np.arange(len(spd_bins)-1))
                                    )
    # %%
    # cal bout features
    features_all = pd.DataFrame()
    all_around_peak_res =  pd.DataFrame()
    expNum = all_around_peak_data['expNum'].max()

    idx_list = np.array(list(range(expNum+1)))
    for excluded_exp, idx_group in enumerate(idx_list):
        group = all_around_peak_data.loc[all_around_peak_data['expNum'].isin([idx_group])]
        yy = (group.loc[group['idx']==idx_post_bout,'propBoutAligned_y'].values - group.loc[group['idx']==idx_pre_bout,'propBoutAligned_y'].values)
        absxx = np.absolute((group.loc[group['idx']==idx_post_bout,'propBoutAligned_x'].values - group.loc[group['idx']==idx_pre_bout,'propBoutAligned_x'].values))
        epochBouts_trajectory = np.degrees(np.arctan(yy/absxx)) # direction of the bout, -90:90
        pitch_pre_bout = group.loc[group.idx==idx_pre_bout,'propBoutAligned_pitch'].values
        pitch_initial = group.loc[group.idx==idx_initial,'propBoutAligned_pitch'].values
        pitch_peak = group.loc[group.idx==round_half_up(peak_idx),'propBoutAligned_pitch'].values
        # pitch_mid_accel = group.loc[group.idx==round_half_up(idx_mid_accel),'propBoutAligned_pitch'].values
        traj_peak = group.loc[group['idx']==peak_idx,'propBoutAligned_instHeading'].values

        rot_all_l_accel = pitch_peak - pitch_initial

        bout_traj_peak = group.loc[group.idx==peak_idx,'propBoutAligned_instHeading'].values
        traj_deviation = bout_traj_peak-pitch_initial
        # rot_early_accel = pitch_mid_accel - pitch_pre_bout
        bout_features = pd.DataFrame(data={'pitch_pre_bout':pitch_pre_bout,
                                            'rot_all_l_accel': rot_all_l_accel,
                                        'pitch_initial':pitch_initial,
                                        'bout_traj_peak':bout_traj_peak,
                                        'traj_peak':traj_peak, 
                                        'traj_deviation':traj_deviation,
                                        'atk_ang':traj_peak-pitch_peak,
                                        'spd_peak': group.loc[group.idx==round_half_up(peak_idx),'propBoutAligned_speed'].values,
                                        })
        features_all = pd.concat([features_all,bout_features],ignore_index=True)
        


        grp = group.groupby(np.arange(len(group))//(idxRANGE[1]-idxRANGE[0]))
        null_initial_pitch = grp.apply(
            lambda group: group.loc[(group['idx']>(peak_idx-idx_dur300ms))&(group['idx']<(peak_idx-idx_dur250ms)), 
                                    'propBoutAligned_pitch'].mean()
        )
        this_res = group.assign(
                                    pitch_pre_bout = np.repeat(pitch_pre_bout,(idxRANGE[1]-idxRANGE[0])),
                                    pitch_initial = np.repeat(pitch_initial,(idxRANGE[1]-idxRANGE[0])),
                                    bout_traj_peak = np.repeat(bout_traj_peak,(idxRANGE[1]-idxRANGE[0])),
                                    traj_peak = np.repeat(traj_peak,(idxRANGE[1]-idxRANGE[0])),
                                    pitch_peak = np.repeat(pitch_peak,(idxRANGE[1]-idxRANGE[0])),
                                    traj_deviation =  np.repeat(traj_deviation,(idxRANGE[1]-idxRANGE[0])),
                                    relative_pitch_chg = group['propBoutAligned_pitch'] - np.repeat(null_initial_pitch,(idxRANGE[1]-idxRANGE[0])).values
                                    )
        this_res = this_res.assign(
                                    atk_ang = this_res['traj_peak']-this_res['pitch_peak'],
                                    )
        all_around_peak_res = pd.concat([all_around_peak_res,this_res],ignore_index=True)
    features_all = features_all.assign(
        direction = pd.cut(features_all['pitch_pre_bout'],[-90,separation_posture,90],labels = ['Nose-down', 'Nose-up'])
    )

    grp = all_around_peak_res.groupby(np.arange(len(all_around_peak_res))//(idxRANGE[1]-idxRANGE[0]))
    all_around_peak_res = all_around_peak_res.assign(
        bout_number = grp.ngroup()
    )


    # %%
    # plot absolute pitch change
    plt_features = 'pitch (deg)'
    # sample bout groups
    sample_bouts = np.random.choice(all_around_peak_res["bout_number"].unique(), 800)
    df_sampled = all_around_peak_res.query('bout_number in @sample_bouts').rename(columns={"propBoutAligned_pitch":plt_features})

    p = sns.relplot(
            data = df_sampled, x = 'time_ms', y = plt_features,
            kind = 'line',
            aspect=2, height=2, 
            # errorbar='sd',
            estimator=None,
            units = 'bout_number',
            alpha = 0.05
            )
    p.map(
        plt.axvline, x=0, linewidth=1, color=".3", zorder=0
        )
    p.set(xlim=(-250,200))
    p.set(ylim=(-30,50))

    # p.map(
    #     plt.axvline, x=time_of_peak_angSpd_mean, linewidth=2, color="green", zorder=0
    #     )
    plt.savefig(os.path.join(fig_dir6, f"bout pitch_timeSeries sampled.pdf"),format='PDF')

# %%
if __name__ == "__main__":
    # if to use Command Line Inputs
    data_dir = input("- Data directory: where is data folder 'behavior data'? \n")
    root = os.path.join(data_dir,'DD_7dpf')
    Fig6_pitch_timeseries(root)