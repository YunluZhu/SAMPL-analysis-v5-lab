'''
Plot averaged features (pitch, inst_traj...) categorized bt pitch up/down and speed bins
Results are jackknifed mean results across experiments (expNum)

Change all_features for the features to plot

Definition of time duration picked for averaging:
prep: bout preperation phase, -200 to -100 ms before peak speed
dur: during bout, -25 to 25 ms
post: +100 to 200 ms 
see idx_bins

'''

#%%
# import sys
import os,glob
# import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir
from plot_functions.plt_tools import (set_font_type, defaultPlotting, jackknife_mean)
from tqdm import tqdm

import matplotlib as mpl
# %%
which_data = 'hc'
# bin_by = 'pitch_peak'
# bin_by = ['pitch_peak','tsp','traj_end','spd_mid_decel']
bin_by = 'pitch_peak'

# bins = np.arange(-20,30,3) # tsp bins
# bins = np.arange(-35,55,5) # pitch bins
# bins = np.arange(-30,40,4) # end pitch bins
# bins = np.arange(-50,50,5) # all peak


def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    return df_day

# %%
def main(pick_data,frame_rate,binned_feat):
    print(f'***bin by: {binned_feat}***')
    if binned_feat == 'traj_end':
        bins = np.arange(-30,40,4) # end pitch bins
    elif 'pitch' in binned_feat:
        bins = np.arange(-35,50,5) # all peak
    elif binned_feat == 'tsp':
        bins = np.arange(-20,30,3) # tsp bins
    elif 'spd' in binned_feat:
        bins = np.arange(5,25,2) # tsp bins
    elif 'rot' in binned_feat:
        bins = np.arange(-5,10,1)
    else:
        bins = np.arange(-50,50,5) # all peak
        
    TSP_THRESHOLD = [-np.Inf,-40,40,np.Inf]

    # %%
    # Paste root directory here
    root = get_data_dir(pick_data)
    FRAME_RATE = frame_rate
    folder_name = f'tmp_boutBy_{binned_feat}'
    folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/'
    fig_dir = os.path.join(folder_dir, folder_name)

    try:
        os.mkdir(fig_dir)
        print(f'fig folder created: {folder_name}')
        print(f'plotting {pick_data}')
    except:
        print(f'plotting {pick_data}')
        pass
        
    # BIN_NUM = 4
    idxRANGE = [15,46]
    spd_bins = [5,10,15,20,25]

    T_INITIAL = -0.25 #s
    T_PRE_BOUT = -0.10 #s
    T_POST_BOUT = 0.10 #s
    T_END = 0.25
    T_MID_ACCEL = -0.05
    T_MID_DECEL = 0.05

    # %%
    # CONSTANTS
    total_aligned = round_half_up(5/4 * FRAME_RATE)+1
    peak_idx = round_half_up(3/4 * FRAME_RATE)
    idx_initial = peak_idx + T_INITIAL * FRAME_RATE
    idx_pre_bout = peak_idx + T_PRE_BOUT * FRAME_RATE
    idx_post_bout = peak_idx + T_POST_BOUT * FRAME_RATE
    idx_mid_accel = peak_idx + T_MID_ACCEL * FRAME_RATE
    idx_mid_decel = peak_idx + T_MID_DECEL * FRAME_RATE
    idx_end = peak_idx + T_END * FRAME_RATE
    # %%
    set_font_type()
    # defaultPlotting()

    # %%
    all_conditions = []
    folder_paths = []
    # get the name of all folders under root
    for folder in os.listdir(root):
        if folder[0] != '.':
            folder_paths.append(root+'/'+folder)
            all_conditions.append(folder)


    all_feature_cond = pd.DataFrame()

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
                    exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                    
                    # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                    bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                    # for i in bout_time.index:
                    # # if only need day or night bouts:
                    for i in day_night_split(bout_time,'aligned_time').index:
                        rows.extend(list(range(i*total_aligned+round_half_up(idxRANGE[0]*FRAME_RATE/40),i*total_aligned+round_half_up(idxRANGE[1]*FRAME_RATE/40))))
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
                                                        'pitch_post_bout':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_post_bout,'propBoutAligned_pitch'].values, 
                                                        'pitch_end': trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_end,'propBoutAligned_pitch'].values, 

                                                        'traj_initial':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_initial,'propBoutAligned_instHeading'].values, 
                                                        'traj_pre_bout':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_pre_bout,'propBoutAligned_instHeading'].values, 
                                                        'traj_peak':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'propBoutAligned_instHeading'].values, 
                                                        'traj_post_bout':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_post_bout,'propBoutAligned_instHeading'].values, 
                                                        'traj_end':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_end,'propBoutAligned_instHeading'].values, 

                                                        'spd_peak':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'propBoutAligned_speed'].values, 
                                                        'spd_mid_decel':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_mid_accel,'propBoutAligned_speed'].values, 
                                                        
                                                        
                                                        'bout_num':trunc_day_exp_data.loc[trunc_day_exp_data['idx']==peak_idx,'bout_num'].values, 
                                                        'expNum':[expNum]*num_of_bouts,
                                                        })
                    pitch_mid_accel = trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_mid_accel,'propBoutAligned_pitch'].reset_index(drop=True)
                    pitch_mid_decel = trunc_day_exp_data.loc[trunc_day_exp_data['idx']==idx_mid_decel,'propBoutAligned_pitch'].reset_index(drop=True)
                    
                    this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_post_bout']-this_exp_features['pitch_initial'],
                                                                rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                                rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                                rot_l_decel=this_exp_features['pitch_post_bout']-this_exp_features['pitch_peak'],
                                                                rot_early_accel = pitch_mid_accel-this_exp_features['pitch_pre_bout'],
                                                                rot_late_accel = this_exp_features['pitch_peak'] - pitch_mid_accel,
                                                                rot_early_decel = pitch_mid_decel-this_exp_features['pitch_peak'],
                                                                rot_late_decel = this_exp_features['pitch_post_bout'] - pitch_mid_decel,
                                                                tsp=this_exp_features['traj_peak']-this_exp_features['pitch_peak'],
                                                                )        
                    tsp_filter = pd.cut(this_exp_features['tsp'],TSP_THRESHOLD,labels=['too_neg','select','too_pos'])
                    this_exp_features = this_exp_features.loc[tsp_filter=='select',:].reset_index(drop=True)
                    
                    bout_features = pd.concat([bout_features,this_exp_features])
                # combine data from different conditions
                all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
                    # dpf=all_conditions[condition_idx][0:2],
                    condition=all_conditions[condition_idx][4:]
                    )])
    
    # %% tidy data
    all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
    # mean_data_cond = mean_data_cond.reset_index().sort_values(by='condition').reset_index(drop=True)


    rolling_windows = pd.DataFrame(data=bins).rolling(2, min_periods=1)
    rolling_mean = rolling_windows.mean()

    all_feature_cond = all_feature_cond.assign(
        # direction = pd.cut(all_feature_cond['pitch_peak'],[-80,0,80],labels=['dive','climb']),
        binned_feat = pd.cut(all_feature_cond[binned_feat],bins,labels=rolling_mean[1:].astype(str).values.flatten()),

        speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),

    )

    # %% Jackknife resampling
    cat_cols = ['condition','expNum','binned_feat']
    mean_data = all_feature_cond.groupby(cat_cols).mean().reset_index()
    mean_data_jackknife = mean_data.groupby(['condition','binned_feat']).apply(
        lambda x: jackknife_mean(x)
    ).reset_index()

    # calculate the excluded expNum for each jackknifed result
    max_exp = mean_data.expNum.max()
    mean_data_jackknife['expNum'] = ((max_exp * (max_exp+1))/2 - max_exp * mean_data_jackknife['expNum']).astype(int)
    try:
        mean_data_jackknife.drop(columns=['level_2'],inplace=True)
    except:
        pass

    mean_data_jackknife.rename(columns={c:c+'_jack' for c in mean_data_jackknife.columns if c not in cat_cols},inplace=True)
    # %% mean
    # # %% Compare Sibs & Tau individual features
    # toplt = mean_data_jackknife
    # all_features = [c for c in toplt.columns if c not in cat_cols]

    # flatui = ["#D0D0D0"] * (toplt['expNum'].max())

    # defaultPlotting()

    # print('Point plot categorized by speed and pitch direction')
    # for feature_toplt in tqdm(all_features):
    #     g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
    #                     col="binned_feat",
    #                     height=4, aspect=0.8, kind='point',
    #                     hue='condition', markers='d',sharey=False,
    #                     ci=None, zorder=10
    #                     )
    #     g.map_dataframe(sns.pointplot, 
    #                     x = "condition", y = feature_toplt,
    #                     hue='expNum', ci=None,palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    #     plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    #     plt.clf()
    # plt.close('all')

    # %% 
    toplt = mean_data_jackknife
    all_features = [c for c in toplt.columns if c not in cat_cols]

    defaultPlotting()

    for feature_toplt in ['pitch','traj','spd','rot','tsp']:
        wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
        wide_data['id'] = wide_data.index
        long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

        df_toplt = long_data.reset_index()
        g = sns.FacetGrid(df_toplt,
                        #   row = "direction", 
                        col='feature',
                        hue = 'condition', 
                        height=5, aspect=.8, 
                        sharey=False,
                        )
        g.map_dataframe(sns.lineplot, 
                        x = 'binned_feat', y = feature_toplt,
                        err_style='band', ci='sd'
                        )
        g.map_dataframe(sns.pointplot, 
                        x = 'binned_feat', y = feature_toplt, 
                        ci=None, join=False,
                        markers='d')
        g.add_legend()
        plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
        # plt.clf()

    plt.close('all')
    # %%

if __name__ == "__main__":
    FRAME_RATE = 40

    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rc('figure', max_open_warning = 0)
    
    if type(bin_by) == list:
        for bin_feature in bin_by:
            if which_data == 'all':
                all_data = ['ori','hets','lesion','s','master','4d','ld']
                for pick_data in (all_data):
                    main(pick_data,FRAME_RATE,bin_feature)
            else:
                main(which_data,FRAME_RATE,bin_feature)
    else:
        bin_feature = bin_by
        if which_data == 'all':
                all_data = ['ori','hets','lesion','s','master','4d']
                for pick_data in (all_data):
                    main(pick_data,FRAME_RATE,bin_feature)
        else:
            main(which_data,FRAME_RATE,bin_feature)


    
    