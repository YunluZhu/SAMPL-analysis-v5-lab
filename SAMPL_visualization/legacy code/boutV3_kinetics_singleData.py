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
from plot_functions.plt_v3 import (jackknife_kinetics, extract_bout_features_v3,get_kinetics)
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
# Paste root directory here
pick_data = 'tmp'


FRAME_RATE = 40
idxRANGE = [15,46]
TSP_THRESHOLD = [-np.Inf,-40,40,np.Inf]
spd_bins = np.arange(3,22,3)



root = get_data_dir(pick_data)
folder_name = f'tmp_boutV3_kinetics_{pick_data}_100after'
folder_dir = '/Users/yunluzhu/Documents/Lab2/Data/SAMPL_ana/Figures/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.mkdir(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
# BIN_NUM = 4

# %%
# CONSTANTS
total_aligned = round_half_up(5/4 * FRAME_RATE)+1
peak_idx = round_half_up(3/4 * FRAME_RATE)


# %%
set_font_type()
# defaultPlotting()

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
                
                this_exp_features = extract_bout_features_v3(trunc_day_exp_data,peak_idx,FRAME_RATE)
                this_exp_features = this_exp_features.assign(
                    expNum = [expNum]*num_of_bouts,
                    direction = pd.cut(this_exp_features['pitch_peak'],[-80,0,80],labels=['dive','climb'])
                    )
                
                tsp_filter = pd.cut(this_exp_features['tsp'],TSP_THRESHOLD,labels=['too_neg','select','too_pos'])
                this_exp_features = this_exp_features.loc[tsp_filter=='select',:].reset_index(drop=True)
                
                
                this_exp_kinetics = get_kinetics(this_exp_features)
                this_exp_kinetics = this_exp_kinetics.append(pd.Series(data={'expNum': expNum}))
                
                bout_features = pd.concat([bout_features,this_exp_features])
                bout_kinetics = pd.concat([bout_kinetics,this_exp_kinetics.to_frame().T], ignore_index=True)
                       
            # combine data from different conditions
            all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
                # dpf=all_conditions[condition_idx][0:2],
                cond1=all_conditions[condition_idx][4:]
                )])
            all_kinetic_cond = pd.concat([all_kinetic_cond, bout_kinetics.assign(
                # dpf=all_conditions[condition_idx][0:2],
                cond1=all_conditions[condition_idx][4:]
                )])
  
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
all_kinetic_cond = all_kinetic_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)


all_feature_cond = all_feature_cond.assign(
    direction = pd.cut(all_feature_cond['pitch_peak'],[-80,0,80],labels=['dive','climb']),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)
print(all_feature_cond.groupby(['cond1']).size())
# %% data validation
# to see the correlation of decel rot - initial pitch
# sns.kdeplot(x='pitch_initial',y='rot_l_decel',data=all_feature_cond.sample(n=10000),alpha=0.05)
# for the righting gain: 
# sns.scatterplot(x='pitch_pre_bout',y='rot_l_decel',data=all_feature_cond.sample(n=10000),alpha=0.05)
# for steering gain: 
# sns.kdeplot(x=all_feature_cond['pitch_peak'], y=all_feature_cond['traj_peak'], )
# set point
# sns.scatterplot(x='rot_total',y='pitch_initial',data=all_feature_cond.sample(n=10000),alpha=0.05)
# tsp pitch
# sns.scatterplot(x='pitch_peak',y='tsp',hue='speed_bins',data=all_feature_cond.sample(n=5000),alpha=0.05)
# sns.kdeplot(x='pitch_peak',y='tsp',hue='speed_bins',data=all_feature_cond.sample(n=5000))
# sns.kdeplot(x='rot_late_accel', y='rot_l_decel',hue='speed_bins',data=all_feature_cond.sample(n=5000))


# %% Jackknife resampling
# calculate kinetics
kinetics_jackknife = pd.DataFrame()
for i, condition in enumerate(set(all_feature_cond.condition)):
    this_cond_data = all_feature_cond.loc[all_feature_cond['cond1']==condition,:]
    this_group_kinetics = jackknife_kinetics(this_cond_data)
    this_group_kinetics = this_group_kinetics.assign(cond1 = condition)
    # output.reset_index(inplace=True)
    kinetics_jackknife = pd.concat([kinetics_jackknife,this_group_kinetics],ignore_index=True)

cat_cols = ['jackknife_group','cond1']
kinetics_jackknife.rename(columns={c:c+'_jack' for c in kinetics_jackknife.columns if c not in cat_cols},inplace=True)
kinetics_jackknife = kinetics_jackknife.sort_values(by=['cond1','jackknife_group']).reset_index(drop=True)

# %%
# calculate kinetics by speed bins
kinetics_bySpd_jackknife = pd.DataFrame()

for condition in set(all_feature_cond.condition):
    this_cond_data = all_feature_cond.loc[all_feature_cond['cond1']==condition,:]
    kinetics_all_speed = pd.DataFrame()
    for speed_bin in set(this_cond_data.speed_bins):
        if pd.notna(speed_bin):
            this_speed_data = this_cond_data.loc[this_cond_data['speed_bins']==speed_bin,:]
            
            this_speed_kinetics = jackknife_kinetics(this_speed_data)
            
            this_speed_kinetics = this_speed_kinetics.assign(speed_bins=speed_bin)
            kinetics_all_speed = pd.concat([kinetics_all_speed,this_speed_kinetics],ignore_index=True)
    kinetics_all_speed = kinetics_all_speed.assign(cond1 = condition)    
    kinetics_bySpd_jackknife = pd.concat([kinetics_bySpd_jackknife, kinetics_all_speed],ignore_index=True)
kinetics_bySpd_jackknife = kinetics_bySpd_jackknife.sort_values(by=['cond1','jackknife_group']).reset_index(drop=True)

# %% Compare Sibs & Tau
cat_cols = ['jackknife_group','cond1','expNum']

toplt = kinetics_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('cond1').size().max())
defaultPlotting()

print('plot jackknife data')

for feature_toplt in tqdm(all_features):
    fig, ax = plt.subplots(1, figsize=[3.2,4])
    sns.pointplot(
                x = 'cond1', y = feature_toplt, data = toplt,
                hue='jackknife_group', ci=None,
                palette=sns.color_palette(flatui), scale=0.5,zorder=1,
                ax=ax)
    g = sns.pointplot(data = toplt, x = 'cond1', y = feature_toplt,
                    linewidth=0,
                    hue='cond1', markers='d',
                    ci=False, zorder=100,
                    ax=ax,
                    )
    if feature_toplt == 'righting_gain_jack':
        g.set_ylim(0.13,0.16)
    g.legend_.remove()
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')


# %% raw data. no jackknife
cat_cols = ['expNum','cond1']

toplt = all_kinetic_cond
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('cond1').size().max())

defaultPlotting()

print('plot raw data')

for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x = 'cond1', y = feature_toplt,
                    height=4, aspect=0.8, kind='point',
                    hue='cond1', markers='d',sharey=False,
                    ci=False, zorder=10
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'cond1', y = feature_toplt,
                    hue='expNum', ci=None,
                    palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')
# %% by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['speed_bins', 'cond1']
all_features = [c for c in toplt.columns if c not in cat_cols]

print("Plot with long format. as a function of speed. ")

defaultPlotting()

for feature_toplt in tqdm(['righting','set','steering','corr']):
    wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

    df_toplt = long_data.reset_index()
    g = sns.FacetGrid(df_toplt,
                      row = "feature", 
                      hue = 'cond1', 
                      height=3, aspect=1.8, 
                      sharey='row',
                      )
    g.map_dataframe(sns.lineplot, 
                    x = 'speed_bins', y = feature_toplt,
                    err_style='band', 
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'speed_bins', y = feature_toplt, 
                    ci=None, join=False,
                    markers='d')
    g.add_legend()
    plt.savefig(fig_dir+f"/{pick_data}'s _spd_{feature_toplt}.pdf",format='PDF')
    # plt.clf()

plt.close('all')

# %%
