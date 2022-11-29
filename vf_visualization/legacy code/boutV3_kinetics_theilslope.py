'''
Plot averaged features (pitch, inst_traj...) categorized bt pitch up/down and speed bins
Results are jackknifed mean results across experiments (expNum)

Change all_features for the features to plot

Definition of time duration picked for averaging:
prep: bout preperation phase, -200 to -100 ms before peak speed
dur: during bout, -25 to 25 ms
post: +100 to 200 ms 
see idx_bins

Todo: bin by initial posture

NOTE
righting rotation: 0-100ms!
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
from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from scipy.stats import mstats #, mannwhitneyu, t, kendalltau
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
# Paste root directory here
pick_data = 'ori'
root = get_data_dir(pick_data)

folder_name = f'tmp_boutV3_kinetics_{pick_data}_theilslope'
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.mkdir(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
# BIN_NUM = 4
FRAME_RATE = 40
idxRANGE = [15,46]
spd_bins = [5,10,15,20,25]

T_INITIAL = -0.25 #s
T_PRE_BOUT = -0.10 #s
T_POST_BOUT = 0.1 #s
# %%
# CONSTANTS
total_aligned = round_half_up(5/4 * FRAME_RATE)+1
peak_idx = round_half_up(3/4 * FRAME_RATE)
idx_initial = peak_idx + T_INITIAL * FRAME_RATE
idx_pre_bout = peak_idx + T_PRE_BOUT * FRAME_RATE
idx_end_bout = peak_idx + T_POST_BOUT * FRAME_RATE
# %%
set_font_type()
# defaultPlotting()

def jackmean(df):
    output = pd.DataFrame()
    for i in list(df.index):
        output = pd.concat([output, df.loc[df.index != i,:].\
            mean().to_frame().T])
    return output

def jackknife_list(ori_list):
    matrix = np.tile(ori_list,(len(ori_list),1))
    output = matrix[~np.eye(matrix.shape[0],dtype=bool)].reshape(matrix.shape[0],-1)
    return output

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    df_day = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    return df_day
# %%
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()
all_feature_cond = pd.DataFrame()
all_kinetic_cond = pd.DataFrame()
mean_data_cond = pd.DataFrame()

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
                exp_data = exp_data.assign(expNum = exp)
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
                                                    'expNum':[exp]*num_of_bouts,
                                                    })
                
                this_exp_features = this_exp_features.assign(rot_total=this_exp_features['pitch_end']-this_exp_features['pitch_initial'],
                                                             rot_pre_bout=this_exp_features['pitch_pre_bout']-this_exp_features['pitch_initial'],
                                                             rot_l_accel=this_exp_features['pitch_peak']-this_exp_features['pitch_pre_bout'],
                                                             rot_l_decel=this_exp_features['pitch_end']-this_exp_features['pitch_peak'],
                                                             tsp=this_exp_features['traj_peak']-this_exp_features['pitch_peak'],
                                                             )
                
                nose_dir = pd.cut(this_exp_features['pitch_peak'],[-80,0,80],labels=[-1,1])
                righting_fit = mstats.theilslopes(x=this_exp_features['pitch_pre_bout'], y=this_exp_features['rot_l_decel'])
                # righting_dn_fit = np.polyfit(x=this_exp_features.loc[nose_dir==-1,'pitch_pre_bout'], y=this_exp_features.loc[nose_dir==-1,'rot_l_decel'], deg=1)
                # righting_up_fit = np.polyfit(x=this_exp_features.loc[nose_dir==1,'pitch_pre_bout'], y=this_exp_features.loc[nose_dir==1,'rot_l_decel'], deg=1)
                
                steering_fit = mstats.theilslopes(x=this_exp_features['pitch_peak'], y=this_exp_features['traj_peak'])
                # steering_dn_fit = np.polyfit(x=this_exp_features.loc[nose_dir==-1,'pitch_peak'], y=this_exp_features.loc[nose_dir==-1,'traj_peak'], deg=1)
                # steering_up_fit = np.polyfit(x=this_exp_features.loc[nose_dir==1,'pitch_peak'], y=this_exp_features.loc[nose_dir==1,'traj_peak'], deg=1)
                                
                this_exp_kinetics = pd.DataFrame(data={
                    'righting_gain': -1 * righting_fit[0],
                    # 'righting_dn_gain': -1 * righting_dn_fit[0],
                    # 'righting_up_gain': -1 * righting_up_fit[0],
                    'steering_gain': steering_fit[0],
                    # 'steering_dn_gain': steering_dn_fit[0],
                    # 'steering_up_gain': steering_up_fit[0],
                    'expNum':expNum,
                }, index=[expNum])

                bout_features = pd.concat([bout_features,this_exp_features])
                bout_kinetics = pd.concat([bout_kinetics,this_exp_kinetics])
                
            # combine data from different conditions
            all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
                # dpf=all_conditions[condition_idx][0:2],
                condition=all_conditions[condition_idx][4:]
                )])
            all_kinetic_cond = pd.concat([all_kinetic_cond, bout_kinetics.assign(
                # dpf=all_conditions[condition_idx][0:2],
                condition=all_conditions[condition_idx][4:]
                )])
  
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
all_kinetic_cond = all_kinetic_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)


all_feature_cond = all_feature_cond.assign(
    direction = pd.cut(all_feature_cond['pitch_peak'],[-80,0,80],labels=['dive','climb']),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

# %% data validation
# to see the correlation of decel rot - initial pitch
# sns.kdeplot(x='pitch_initial',y='rot_l_decel',data=all_feature_cond)
# for the righting gain: 
sns.scatterplot(x='pitch_pre_bout',y='rot_l_decel',data=all_feature_cond)
# for steering gain: 
# sns.kdeplot(x=all_feature_cond['traj_peak'], y=all_feature_cond['pitch_peak'], )


# %% Jackknife resampling
kinetics_jackknife = pd.DataFrame()
exp_df = all_feature_cond.groupby('expNum').size()
jackknife_exp_matrix = jackknife_list(list(exp_df.index))
for i, condition in enumerate(set(all_feature_cond.condition)):
    this_cond_data = all_feature_cond.loc[all_feature_cond['condition']==condition,:]
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = this_cond_data.loc[this_cond_data['expNum'].isin(exp_group),:]
        righting_fit = np.polyfit(x=this_group_data['pitch_pre_bout'], y=this_group_data['rot_l_decel'], deg=1)
        steering_fit = np.polyfit(x=this_group_data['pitch_peak'], y=this_group_data['traj_peak'], deg=1)

        righting_fit_dn = np.polyfit(x=this_group_data.loc[this_group_data['direction']=='dive','pitch_pre_bout'], 
                                     y=this_group_data.loc[this_group_data['direction']=='dive','rot_l_decel'], 
                                     deg=1)
        righting_fit_up = np.polyfit(x=this_group_data.loc[this_group_data['direction']=='climb','pitch_pre_bout'], 
                                     y=this_group_data.loc[this_group_data['direction']=='climb','rot_l_decel'], 
                                     deg=1)
            
        this_group_kinetics = pd.DataFrame(data={
            'righting_gain': -1 * righting_fit[0],
            'righting_gain_dn': -1 * righting_fit_dn[0],
            'righting_gain_up': -1 * righting_fit_up[0],
            'steering_gain': steering_fit[0],
            'jackknife_group':j,
            'condition':condition,
        }, index=[i*j+j])
        
        kinetics_jackknife = pd.concat([kinetics_jackknife,this_group_kinetics])

cat_cols = ['jackknife_group','condition']
kinetics_jackknife.rename(columns={c:c+'_jack' for c in kinetics_jackknife.columns if c not in cat_cols},inplace=True)
kinetics_jackknife = kinetics_jackknife.sort_values(by=['condition','jackknife_group']).reset_index(drop=True)
# %% mean
# data to plot

# %% Compare Sibs & Tau
cat_cols = ['jackknife_group','condition']

toplt = kinetics_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('condition').size().max())

defaultPlotting()

print('Point plot categorized by speed and pitch direction')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
                    height=4, aspect=0.8, kind='point',
                    hue='condition', markers='d',sharey=False,
                    ci=False, zorder=10
                    )
    g.map_dataframe(sns.pointplot, 
                    x = "condition", y = feature_toplt,
                    hue='jackknife_group', ci=None,
                    palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    # plt.clf()
# plt.close('all')

# %%
cat_cols = ['jackknife_group','condition']

toplt = all_kinetic_cond
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('condition').size().max())

defaultPlotting()

print('Point plot categorized by speed and pitch direction')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
                    height=4, aspect=0.8, kind='point',
                    hue='condition', markers='d',sharey=False,
                    ci=False, zorder=10
                    )
    g.map_dataframe(sns.pointplot, 
                    x = "condition", y = feature_toplt,
                    hue='expNum', ci=None,
                    palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')
# %%
