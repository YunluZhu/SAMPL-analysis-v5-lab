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
import pandas as pd # pandas library
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
from plot_functions.get_index import get_index

from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
# Paste root directory here
pick_data = 'for_paper'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data}_boutBySpd_durMean'
folder_dir = f'/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/{pick_data}/'
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
# BIN_NUM = 4
peak_idx , total_aligned = get_index(FRAME_RATE)
spd_bins = np.arange(5,25,4)

# %%
# CONSTANTS

# %%
set_font_type()
# defaultPlotting()

def jackmean(df):
    output = pd.DataFrame()
    for i in list(df.index):
        output = pd.concat([output, df.loc[df.index != i,:].\
            mean().to_frame().T])
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
mean_data_cond = pd.DataFrame()

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data = pd.DataFrame()
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
                exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time').index:
                    rows.extend(list(range(i*total_aligned+int(idxRANGE[0]*FRAME_RATE/40),i*total_aligned+int(idxRANGE[1]*FRAME_RATE/40))))
                exp_data = exp_data.assign(expNum = expNum)
                trunc_day_exp_data = exp_data.loc[rows,:]
                time_labels = ['1start',
                                '2prep',
                                '3accel',
                                '4dur',
                                '5decel',
                                '6post',
                                '7end']
                # now extract features that are 1 per bout
                idx_bins = [0,peak_idx-0.2*FRAME_RATE-1,
                            peak_idx-0.1*FRAME_RATE-1,
                            peak_idx-0.025*FRAME_RATE-1,
                            peak_idx+0.025*FRAME_RATE,
                            peak_idx+0.1*FRAME_RATE,
                            peak_idx+0.2*FRAME_RATE,
                            exp_data['idx'].max()]
                trunc_day_exp_data = trunc_day_exp_data.assign(
                    timing_cat = pd.cut(trunc_day_exp_data['idx'],idx_bins,labels = time_labels),
                    bout_num = trunc_day_exp_data.groupby(np.arange(len(trunc_day_exp_data))//(idxRANGE[1]-idxRANGE[0])).ngroup()
                    )
                
                grp_mean = trunc_day_exp_data.groupby(['bout_num','timing_cat'])\
                    [['propBoutAligned_instHeading','propBoutAligned_pitch']].\
                        mean()
                
                this_exp_features = pd.DataFrame()
                
                for duration in time_labels:
                    this_duration = grp_mean.xs(duration,level=1)
                    this_duration = this_duration.assign(traj_sub_pitch = this_duration['propBoutAligned_instHeading']-this_duration['propBoutAligned_pitch'])
                    this_duration = this_duration.rename(
                                    columns={'propBoutAligned_instHeading':f'traj_{duration}',
                                            'propBoutAligned_pitch':f'pitch_{duration}',
                                            'traj_sub_pitch':f'tsp_{duration}'}
                                    )
                    this_exp_features = pd.concat([this_exp_features,this_duration],axis=1)


                peak = trunc_day_exp_data.loc[trunc_day_exp_data.idx==peak_idx,['propBoutAligned_speed','propBoutAligned_instHeading','propBoutAligned_pitch']]
                peak = peak.rename(
                    columns={'propBoutAligned_speed':'speed_peak',
                             'propBoutAligned_instHeading':'traj_peak',
                             'propBoutAligned_pitch':'pitch_peak'}
                ).reset_index(drop=True)
                this_exp_features = pd.concat([this_exp_features,peak],axis=1).reset_index(drop=True).assign(expNum = expNum)
                
                # concatenate with previous data
                around_peak_data = pd.concat([around_peak_data,trunc_day_exp_data])
                bout_features = pd.concat([bout_features,this_exp_features])
            # combine data from different conditions
            all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(dpf=all_conditions[condition_idx][0:2],condition=all_conditions[condition_idx][4:])])
            all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(dpf=all_conditions[condition_idx][0:2],condition=all_conditions[condition_idx][4:])])
            # this_cond_mean = bout_features.groupby('expNum').mean()
            # mean_data_cond = pd.concat([mean_data_cond, this_cond_mean.assign(dpf=all_conditions[condition_idx][0:2],condition=all_conditions[condition_idx][4:])])
            
# %% tidy data
all_around_peak_data = all_around_peak_data.sort_values(by='condition').reset_index(drop=True)
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# mean_data_cond = mean_data_cond.reset_index().sort_values(by='condition').reset_index(drop=True)

if pick_data == 'for_paper':
    # all_cond2 = ['4dpf','7dpf','14dpf']
    all_around_peak_data = all_around_peak_data.sort_values('condition'
                            , key=lambda col: col.map(
                                    {'4dpf':1,
                                      '7dpf':2,
                                      '14dpf':3}))
    all_feature_cond = all_feature_cond.sort_values('condition'
                            , key=lambda col: col.map(
                                    {'4dpf':1,
                                      '7dpf':2,
                                      '14dpf':3}))
# assign 
# all_around_peak_data = all_around_peak_data.assign(
#     pitch_dir = pd.cut(all_around_peak_data['pitch_peak'],[-180,0,180],labels=['neg_pitch','pos_pitch'])
# )
all_feature_cond = all_feature_cond.assign(
    pitch_dir = pd.cut(all_feature_cond['pitch_peak'],[-180,0,180],labels=['neg_pitch','pos_pitch']),
    speed_bins = pd.cut(all_feature_cond['speed_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

# %%
average_speed = all_feature_cond.groupby('speed_bins')['speed_peak'].mean().values

# %% Jackknife resampling
cat_cols = ['condition','expNum','pitch_dir','speed_bins']
mean_data = all_feature_cond.groupby(cat_cols).mean().reset_index()
mean_data_jackknife = mean_data.groupby(['condition','pitch_dir','speed_bins']).apply(
    lambda x: jackmean(x)
 ).reset_index()

# calculate the excluded expNum for each jackknifed result
max_exp = mean_data.expNum.max()
mean_data_jackknife['expNum'] = ((max_exp * (max_exp+1))/2 - max_exp * mean_data_jackknife['expNum']).astype(int)

mean_data_jackknife.rename(columns={c:c+'_jack' for c in mean_data_jackknife.columns if c not in cat_cols},inplace=True)
all_features = [c for c in mean_data_jackknife.columns if c not in cat_cols]
# %% mean
# data to plot
toplt = mean_data_jackknife

# %% Compare Sibs & Tau
flatui = ["#D0D0D0"] * (toplt['expNum'].max())

defaultPlotting()

print('Point plot categorized by speed and pitch direction')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
                    row="pitch_dir",col='speed_bins', 
                    height=4, aspect=0.8, kind='point',
                    hue='condition', markers='d',sharey=False,
                    ci=None, zorder=10
                    )
    g.map_dataframe(sns.pointplot, 
                    x = "condition", y = feature_toplt,
                    hue='expNum', ci=None,palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')

# %% Plot as a function of speed
# print('As a function of speed')

# toplt = mean_data_jackknife

# defaultPlotting()

# for feature_toplt in tqdm(all_features):
#     g = sns.FacetGrid(toplt,
#                       row = "pitch_dir", 
#                       hue = 'condition', 
#                       height=3, aspect=1.8, 
#                       sharey=False,
#                       )
#     g.map_dataframe(sns.lineplot, 
#                     x = 'speed_bins', y = feature_toplt,
#                     err_style='band', 
#                     )
#     g.map_dataframe(sns.pointplot, 
#                     x = 'speed_bins', y = feature_toplt, 
#                     ci=None, join=False,
#                     markers='d')
#     g.add_legend()
#     plt.savefig(fig_dir+f"/{pick_data}'s _spd_{feature_toplt}.pdf",format='PDF')
#     plt.clf()

# plt.close('all')

# %% 
print("Plot with long format. as a function of speed. col = time duration")

defaultPlotting()

for feature_toplt in ['pitch','traj','tsp']:
    wide_data = mean_data_jackknife.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}_' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='duration', sep='_', suffix='\d\w+')

    toplt = long_data.reset_index()
    g = sns.FacetGrid(toplt,
                      row = "pitch_dir", col='duration',
                      hue = 'condition', 
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
