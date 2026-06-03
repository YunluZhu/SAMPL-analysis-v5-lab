'''
For single fish experiment
Gets IEIpitch mean for each bout
Gets featurs for bouts
Plot IEIpitch as a function of number of IEI acquired
'''

#%%
import sys
import os,glob
from tabnanny import check
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.power import TTestIndPower
from statsmodels.stats.power import TTestPower

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
from plot_functions.get_bout_features import extract_bout_features_v5
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)

set_font_type()
# %%
# Paste root directory here
which_ztime = 'day'
pick_data = 'sf ana'
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
POSTURE_SEP = 5 #deg
MIN_DATA_SIZE = 0
HIGH_DATA_SIZE = 100
HIGH_IEI_SIZE = 100
time50ms = round_half_up(0.05 * FRAME_RATE)
time100ms = round_half_up(0.1 * FRAME_RATE)

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
    df_out = grp[['propBoutIEI_pitch','y_boutFreq']].mean().assign(dpf=condition[0:2],cond1=condition[4:])
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

folder_paths.sort()
# go through each condition folders under the root
for fish_idx, folder in enumerate(folder_paths):
    # get IEI pitch
    this_fish_id = round_half_up(os.path.basename(folder))
    clutch_id = this_fish_id//100
    df = pd.read_hdf(f"{folder}/IEI_data.h5", key='prop_bout_IEI2')
    df = day_night_split(df,'propBoutIEItime',ztime=which_ztime)
    if len(df) > MIN_DATA_SIZE:
        # get pitch
        this_body_angles = df.loc[:,['propBoutIEI_pitch','propBoutIEI']].rename(columns={'propBoutIEI_pitch':'IEIpitch'})

        this_body_angles = this_body_angles.assign(fish_id = this_fish_id,
                                                #    fish_idx = fish_idx,
                                                cond1 = 'tau',
                                                IEI_number = len(this_body_angles),
                                                clutch_id = clutch_id)
                
        # get other bout features
        angles = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout_aligned')
        angles = angles.assign(
            idx=round_half_up(len(angles)/total_aligned)*list(range(0,total_aligned)),
            bout_num = list(np.arange(len(angles))//total_aligned),
            )
        this_exp_features = extract_bout_features_v5(angles, peak_idx,FRAME_RATE)


        peak_angles = angles.loc[angles['idx']==peak_idx]
        peak_angles = peak_angles.assign(
            time = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
            traj = pd.read_hdf(f"{folder}/bout_data.h5", key='prop_bout2')['epochBouts_trajectory'].values,
            )  # peak angle time and bout traj
        peak_angles_day = day_night_split(peak_angles, 'time',ztime=which_ztime)
        all_peak_idx = peak_angles_day.index
        
        
        this_exp_features = this_exp_features.assign(
            time = peak_angles['time'].values,
            traj = peak_angles['traj'].values,
            cond1 = 'tau',
            )
        this_exp_features_day = day_night_split(this_exp_features, 'time',ztime=which_ztime)

        atk_ang = this_exp_features_day['traj'] - angles.loc[all_peak_idx-time50ms,'propBoutAligned_pitch'].values # new atk angles calculated using accel ang
        this_exp_features_day = this_exp_features_day.assign(
            atk_ang = atk_ang,
            direction = pd.cut(this_exp_features_day['pitch_initial'],[-91,POSTURE_SEP,91],labels=['DOWN','UP']),
            fish_id = this_fish_id,
            clutch_id = clutch_id,
            )
        # # calculate kinetics
        # this_exp_kinetics = get_kinetics(this_exp_features_day)
        # this_exp_kinetics = this_exp_kinetics.append(
        #     pd.Series(data={'cond1': 'tau',
        #                     'fish_id':this_fish_id}))

        # correct condition
        if not this_fish_id%100: # if sibs
            this_body_angles['cond1'] = 'sibs'
            this_exp_features_day['cond1'] = 'sibs'
            # this_exp_kinetics['cond1'] = 'sibs'
                    
        all_feature_data = pd.concat([all_feature_data,this_exp_features_day], ignore_index=True)
        all_IEI_angles = pd.concat([all_IEI_angles,this_body_angles], axis=0, ignore_index=True)
        # all_kinetics = pd.concat([all_kinetics,this_exp_kinetics.to_frame().T], ignore_index=True)

# get metadata for this dlm file      
all_metadata = pd.DataFrame() 
for metadata in metadata_files:
    open_metadata = pd.read_csv(metadata, index_col=0)
    this_metadata = open_metadata.loc[:,['box_number','light_cycle','num_fish','aligned_bout','fish_id']]
    all_metadata = pd.concat([all_metadata,this_metadata], ignore_index=True)
    
cat_cols = ['cond1','fish_id']
all_mean_data = all_feature_data.groupby(cat_cols).mean().reset_index()
all_std_data = all_feature_data.groupby(cat_cols).std().reset_index()
all_std_IEI_angles = all_IEI_angles.groupby(cat_cols)['IEIpitch'].std().reset_index()
all_std_IEI_angles['IEI_number'] = all_IEI_angles.groupby(cat_cols)['IEI_number'].first().values  
# all_std_data = pd.concat([all_std_data,all_std_IEI_angles.loc[:,'IEIpitch']], axis=1)

fish_metadata = all_metadata.groupby('fish_id').mean().reset_index()
fish_metadata['aligned_bout'] = all_metadata.groupby('fish_id')['aligned_bout'].sum().values

# %%
# tidy up
all_mean_data = all_mean_data.merge(fish_metadata, how='inner', on='fish_id')

# %%
# 
all_mean_data.loc[all_mean_data['cond1']=='tau','aligned_bout'].mean()

plt.figure()
sns.histplot(
    data=all_mean_data,
    x='aligned_bout',
    bins=10
    )
plt.figure()
sns.scatterplot(
    data=all_std_IEI_angles,
    x='IEI_number',
    y='IEIpitch',
    hue='cond1'
    )

# %%
fish_bout_number = all_feature_data.groupby('fish_id').size().reset_index()
fish_bout_number.columns = ['fish_id','bout_num']
print(fish_bout_number)
print(f'mean of all tau boxes: {fish_bout_number.iloc[1:,1].mean()}')


high_bout_fish = fish_bout_number.loc[fish_bout_number['bout_num'] > HIGH_DATA_SIZE,'fish_id'].values
print(f'selected fish: {high_bout_fish}')

high_bout_data = all_feature_data.loc[all_feature_data['fish_id'].isin(high_bout_fish),:]
# high_bout_kinetics = all_kinetics.loc[all_kinetics['fish_id'].isin(high_bout_fish),:]
# high_bout_IEI = all_IEI_angles.loc[all_IEI_angles['fish_id'].isin(fish_metadata.loc[fish_metadata.aligned_bout>HIGH_DATA_SIZE,'fish_id']),:]
high_bout_IEI = all_IEI_angles.loc[all_IEI_angles.fish_id.isin(high_bout_fish)]
# NOTE 
# traj and pitch worth to look at
# also see lower attack angle

# %%
# parameter by parameter
def power_analysis(high_bout_data, feature_tuple):
    feature, direction, chg_sign = feature_tuple
    if direction != "all":
        dir_high_bout =  high_bout_data.loc[high_bout_data['direction']==direction]
    else:
        dir_high_bout = high_bout_data
    sib_bout = dir_high_bout.loc[dir_high_bout.cond1 == 'sibs']
    max_sib_clutch = sib_bout.groupby('clutch_id').size().idxmax()
    max_sib_clutch_data = sib_bout.loc[sib_bout.clutch_id == max_sib_clutch]
    # max_sib_clutch_data = sib_bout.loc[sib_bout.clutch_id == 220508]
    tau_data = dir_high_bout.loc[(dir_high_bout.clutch_id == max_sib_clutch) & (dir_high_bout.cond1 == 'tau')]

    # parameters for power analysis
    effect_size = []
    fish_id_list = []
    z_score = []
    n2_total = 0
    for fish in set(tau_data['fish_id']):
        tau_bout = tau_data.loc[tau_data.fish_id == fish]
        u1, u2 = max_sib_clutch_data[feature].mean(), tau_bout[feature].mean()
        n1, n2 = len(max_sib_clutch_data), len(tau_bout)
        s1, s2 = max_sib_clutch_data[feature].std(), tau_bout[feature].std()
        s = (s1+s2)/2
        d = ((u2 - u1) / s)
        z = (u2 - u1) / s1
        effect_size.append(d)
        z_score.append(z)
        fish_id_list.append(fish)
        n2_total+=n2
    n2_average = n2_total/len(effect_size)
    alpha = 0.05
    power = 0.8
    # perform power analysis
    # analysis = TTestIndPower()
    obj = TTestIndPower()
    n = obj.solve_power(nobs1 = n1, effect_size=max(np.abs(effect_size)), alpha=alpha, power=power, 
                        ratio=None, alternative='two-sided')
    n = n*n1
    output = pd.DataFrame(data={
        "feature":feature+"_"+direction,
        "effectSize_mean": np.mean(effect_size),
        "effectSize_min": min(effect_size),
        "effectSize_max": max(effect_size),
        "average_bout_number": n2_average,
        "sampleSize": n,
        "change_sign": chg_sign,
        },
                          index=[0]
                          )
    for i, id in enumerate(fish_id_list):
        fish_zScore = pd.DataFrame(data={id:z_score[i]*chg_sign},index=[0])
        output = pd.concat([output,fish_zScore],axis=1)
    return output

feature_list = [
    ("angvel_chg","UP",-1),
    # ("angvel_chg","DOWN",-1),
    # ("angvel_post_phase","UP",-1),
    ("pitch_initial","UP",-1),
    ("pitch_end","UP",-1),
    ("pitch_initial","DOWN",1),
    # ("rot_l_decel","DOWN",-1),
    # ("rot_total","DOWN",-1),
]

power_ana_res = pd.concat([power_analysis(high_bout_data, feature_check) for feature_check in feature_list], ignore_index=True)
print(power_ana_res)
power_ana_res['sampleSize'].mean()
# %%
print(power_ana_res)
# fish_list = [22080501, 22080511, 22080513, 22080515, 22080516]
fish_list = list(high_bout_fish[1:])
check_fish = power_ana_res.loc[:,['feature']+fish_list]
check_fish = check_fish.set_index('feature')

# %%
# add IEI mean data
sibs_data = high_bout_IEI.loc[high_bout_IEI.cond1=='sibs','propBoutIEI']
sibs_mean = sibs_data.mean()
sibs_sd = sibs_data.std()
IEI_zscore = ((high_bout_IEI.groupby('fish_id')['propBoutIEI'].mean() - sibs_mean)/sibs_sd)
IEI_zscore_adj = (IEI_zscore*(-1))
IEI_zscore_adj = IEI_zscore_adj.to_frame().T

check_fish_combined = pd.concat([check_fish,IEI_zscore_adj],axis=0,join="inner")
check_fish_combined.mean()
# %%
# x = [1098,1049,960,2345]
# y = [0.080822,0.146382,0.350546,0.472061]

# sns.scatterplot(x=x,y=y)
# %%
