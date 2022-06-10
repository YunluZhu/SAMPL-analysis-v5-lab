'''
For single fish experiment
Gets IEIpitch mean for each bout
Gets featurs for bouts
Plot IEIpitch as a function of number of IEI acquired
'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
# from collections import defaultdict
# from datetime import datetime
# from datetime import timedelta
# import math
# from scipy.stats import ttest_rel
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.get_bout_features import extract_bout_features_v4
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)

set_font_type()
# %%
# Paste root directory here
pick_data = 'sf'
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx, total_aligned = get_index(FRAME_RATE)

folder_name = f'{pick_data}_working'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
time50ms = int(0.05 * FRAME_RATE)
time100ms = int(0.1 * FRAME_RATE)

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def distribution_binned_average(df, bin_width, condition):
    '''
    bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
    '''
    df = df.sort_values(by='propBoutIEI_pitch')
    df = df.assign(y_boutFreq = 1/df['propBoutIEI'])
    bins = pd.cut(df['propBoutIEI_pitch'], list(np.arange(-90,90,bin_width)))
    grp = df.groupby(bins)
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0:2],condition=condition[4:])
    return df_out

# %%
# Main function
all_fishID = []
folder_paths = []
metadata_files = []
# get the name of all folders under root

for folder in os.listdir(root):
    if folder[0] != '.':
        path = os.path.join(root,folder)
        for ele in os.listdir(path):
            d = os.path.join(path,ele)
            if os.path.isdir(d):  # if a path
                folder_paths.append(d)
                all_fishID.append(ele)
            elif 'metadata.csv' in d:  # if a metadata
                metadata_files.append(d)
                
# %%
# get IEI pitch and std(pitch)

all_IEI_angles = pd.DataFrame()
all_feature_data = pd.DataFrame()
all_mean_data = pd.DataFrame()
all_std_data = pd.DataFrame()
all_kinetics = pd.DataFrame()

# go through each condition folders under the root
for fish_idx, folder in enumerate(folder_paths):
    # get IEI pitch
    this_fish_id = int(os.path.basename(folder))
    
    df = pd.read_hdf(f"{folder}/IEI_data.h5", key='prop_bout_IEI2')
    df = day_night_split(df,'propBoutIEItime',)
    # get pitch
    this_body_angles = df.loc[:,['propBoutIEI_pitch']].rename(columns={'propBoutIEI_pitch':'IEIpitch'})

    this_body_angles = this_body_angles.assign(fish_id = this_fish_id,
                                            #    fish_idx = fish_idx,
                                               condition = 'tau',
                                               IEI_number = len(this_body_angles))
            
    # get other bout features
    angles = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout_aligned')
    angles = angles.assign(
        idx=int(len(angles)/total_aligned)*list(range(0,total_aligned)),
        bout_num = list(np.arange(len(angles))//total_aligned),
        )
    this_exp_features = extract_bout_features_v4(angles, peak_idx,FRAME_RATE)


    peak_angles = angles.loc[angles['idx']==peak_idx]
    peak_angles = peak_angles.assign(
        time = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
        traj = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout2')['epochBouts_trajectory'].values,
        )  # peak angle time and bout traj
    peak_angles_day = day_night_split(peak_angles, 'time')
    all_peak_idx = peak_angles_day.index
    
    
    this_exp_features = this_exp_features.assign(
        time = peak_angles['time'].values,
        traj = peak_angles['traj'].values,
        condition = 'tau',
        )
    this_exp_features_day = day_night_split(this_exp_features, 'time')

    atk_ang = this_exp_features_day['traj'] - angles.loc[all_peak_idx-time50ms,'propBoutAligned_pitch'].values # new atk angles calculated using accel ang
    this_exp_features_day = this_exp_features_day.assign(
        atk_ang = atk_ang,
        direction = pd.cut(this_exp_features_day['pitch_peak'],[-80,0,80],labels=['dive','climb']),
        fish_id = this_fish_id,
        )
    # calculate kinetics
    this_exp_kinetics = get_kinetics(this_exp_features_day)
    this_exp_kinetics = this_exp_kinetics.append(
        pd.Series(data={'condition': 'tau',
                        'fish_id':this_fish_id}))

    # correct condition
    if not this_fish_id%100: # if sibs
        this_body_angles['condition'] = 'sibs'
        this_exp_features_day['condition'] = 'sibs'
        this_exp_kinetics['condition'] = 'sibs'
        
    all_feature_data = pd.concat([all_feature_data,this_exp_features_day], ignore_index=True)
    all_IEI_angles = pd.concat([all_IEI_angles,this_body_angles], axis=0, ignore_index=True)
    all_kinetics = pd.concat([all_kinetics,this_exp_kinetics.to_frame().T], ignore_index=True)

# get metadata for this dlm file      
all_metadata = pd.DataFrame() 
for metadata in metadata_files:
    open_metadata = pd.read_csv(metadata, index_col=0)
    this_metadata = open_metadata.loc[:,['box_number','light_cycle','num_fish','aligned_bout','fish_id']]
    all_metadata = pd.concat([all_metadata,this_metadata], ignore_index=True)
    
cat_cols = ['condition','fish_id']
all_mean_data = all_feature_data.groupby(cat_cols).mean().reset_index()
all_std_data = all_feature_data.groupby(cat_cols).std().reset_index()
all_std_IEI_angles = all_IEI_angles.groupby(cat_cols)['IEIpitch'].std().reset_index()
all_std_IEI_angles['IEI_number'] = all_IEI_angles.groupby(cat_cols)['IEI_number'].first().values  
# all_std_data = pd.concat([all_std_data,all_std_IEI_angles.loc[:,'IEIpitch']], axis=1)

fish_metadata = all_metadata.groupby('fish_id').mean().reset_index()
fish_metadata['aligned_bout'] = all_metadata.groupby('fish_id')['aligned_bout'].sum().values
# tidy up
all_mean_data.set_index('fish_id')
all_std_data.set_index('fish_id')
fish_metadata.set_index('fish_id')
all_mean_data = pd.concat([all_mean_data,fish_metadata], axis=1)
all_std_data = pd.concat([all_std_data,fish_metadata], axis=1)


# %%
# 
sns.histplot(
    data=all_mean_data,
    x='aligned_bout',
    bins=10
    )
sns.scatterplot(
    data=all_std_IEI_angles,
    x='IEI_number',
    y='IEIpitch',
    hue='condition'
    )

# %%
high_bout_data = all_feature_data.loc[all_feature_data['fish_id'].isin(fish_metadata.loc[fish_metadata.aligned_bout>100,'fish_id']),:]
high_bout_kinetics = all_kinetics.loc[all_kinetics['fish_id'].isin(fish_metadata.loc[fish_metadata.aligned_bout>100,'fish_id']),:]

# NOTE 
# traj and pitch worth to look at
# also see lower attack angle

# %%
# Plot features

features = [
'pitch_initial', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
       'pitch_end', 'traj_initial', 'traj_pre_bout', 'traj_peak',
       'traj_post_bout', 'traj_end', 'spd_peak', 'spd_mid_decel',
       'rot_total', 'rot_pre_bout', 'rot_l_accel', 'rot_l_decel',
       'rot_early_accel', 'rot_late_accel', 'rot_early_decel',
       'rot_late_decel', 'tsp', 'traj', 'atk_ang',
]
all_directions = ['dive',
                  'climb']
for dir_toplt in all_directions:

    print(dir_toplt)

    data_toplt = high_bout_data.loc[high_bout_data.direction==dir_toplt,:]
    sibs_mean = data_toplt.loc[data_toplt.condition=='sibs',features]
    sibs_mean = sibs_mean.mean()
    high_bout_zScore = pd.DataFrame()
    high_bout_zScore = data_toplt
    for feature in features:
        high_bout_zScore.loc[:,feature] = high_bout_zScore.groupby('fish_id')[feature].apply(
                lambda x: (x-sibs_mean[feature])/np.std(x)
                )
    sns.relplot(
        data=high_bout_zScore,
        x='fish_id',
        y='rot_pre_bout',
        hue='condition',
        # row='direction',
        kind='line',
        err_style='bars',
        size=0,
        facet_kws={'sharey': False, 'sharex': True}
        )

# %%
# Plot kinetics

features = [
'righting_gain', 'righting_gain_dn', 'righting_gain_up',
       'steering_gain', 'corr_rot_accel_decel', 'corr_rot_lateAccel_decel',
       'corr_rot_earlyAccel_decel', 'corr_rot_preBout_decel', 'set_point'
]

data_toplt = high_bout_kinetics

sns.relplot(
    data=data_toplt,
    x='fish_id',
    y='righting_gain_up',
    kind='line',
    err_style='bars',
    size=0,
    facet_kws={'sharey': False, 'sharex': True}
    )


# %%
