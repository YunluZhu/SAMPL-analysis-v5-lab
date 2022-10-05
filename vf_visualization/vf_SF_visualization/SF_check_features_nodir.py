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
pick_data = 'sf DD'
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx, total_aligned = get_index(FRAME_RATE)

folder_name = f'{pick_data}_noDir'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
MIN_DATA_SIZE = 60
HIGH_DATA_SIZE = 100
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
                
# get IEI pitch and std(pitch)

all_IEI_angles = pd.DataFrame()
all_feature_data = pd.DataFrame()
all_mean_data = pd.DataFrame()
all_std_data = pd.DataFrame()
all_kinetics = pd.DataFrame()
fish_metadata = pd.DataFrame()
folder_paths.sort()
metadata_files.sort()

assert len(folder_paths) == len(metadata_files)
# go through each condition folders under the root
for fish_idx, folder in enumerate(folder_paths):
    # get IEI pitch
    this_fish_id = int(os.path.basename(folder))
    clutch_id = this_fish_id//100
    df = pd.read_hdf(f"{folder}/IEI_data.h5", key='prop_bout_IEI2')
    df = day_night_split(df,'propBoutIEItime',)
    if len(df) > MIN_DATA_SIZE:
        # get pitch
        this_body_angles = df.loc[:,['propBoutIEI_pitch','propBoutIEI']].rename(columns={'propBoutIEI_pitch':'IEIpitch'})

        this_body_angles = this_body_angles.assign(fish_id = this_fish_id,
                                                #    fish_idx = fish_idx,
                                                condition = 'tau',
                                                IEI_number = len(this_body_angles),
                                                clutch_id = clutch_id)
                
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
            fish_id = this_fish_id,
            clutch_id = clutch_id,
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

        # deal with metadata
        open_metadata = pd.read_csv(metadata_files[fish_idx], index_col=0)
        this_metadata = pd.DataFrame(data={
            'num_fish':open_metadata.groupby('box_number')['num_fish'].mean().sum(),
            'aligned_bouts':open_metadata['aligned_bout'].sum(),
            'fish_id': this_fish_id,
            'clutch_id': clutch_id,
            'number_of_boxes': open_metadata.groupby('box_number').ngroups,
            
            # 'condition': this_exp_kinetics['condition']
        },
                                     index=[0])
        

        fish_metadata = pd.concat([fish_metadata,this_metadata], ignore_index=True)

# %%
cat_cols = ['condition','fish_id']
all_mean_data = all_feature_data.groupby(cat_cols).mean().reset_index()
all_std_data = all_feature_data.groupby(cat_cols).std().reset_index().drop(columns='clutch_id')
all_std_IEI_angles = all_IEI_angles.groupby(cat_cols)['IEIpitch'].std().reset_index()
all_std_IEI_angles['IEI_number'] = all_IEI_angles.groupby(cat_cols)['IEI_number'].first().values  
all_std_IEI_angles = all_std_IEI_angles.assign(
    clutch_id = all_std_IEI_angles['fish_id']//100
)
# all_std_data = pd.concat([all_std_data,all_std_IEI_angles.loc[:,'IEIpitch']], axis=1)

all_mean_data = all_mean_data.merge(fish_metadata, how='inner', on='fish_id')
all_std_data = all_std_data.merge(fish_metadata, how='inner', on='fish_id')
all_kinetics = all_kinetics.merge(fish_metadata, how='inner', on='fish_id')

# %%
high_bout_fish_id = fish_metadata.loc[fish_metadata.aligned_bouts>HIGH_DATA_SIZE,'fish_id']
high_bout_data = all_feature_data.loc[all_feature_data['fish_id'].isin(high_bout_fish_id),:]
high_bout_kinetics = all_kinetics.loc[all_kinetics['fish_id'].isin(high_bout_fish_id),:]
high_bout_IEI = all_std_IEI_angles.loc[all_std_IEI_angles['fish_id'].isin(high_bout_fish_id),:]
high_bout_std = all_std_data.loc[all_std_data['fish_id'].isin(high_bout_fish_id),:]



# %%
# 
plt.figure()
sns.histplot(
    data=all_mean_data,
    x='aligned_bouts',
    bins=10
    )
plt.figure()
sns.scatterplot(
    data=all_std_IEI_angles,
    x='IEI_number',
    y='IEIpitch',
    hue='condition'
    )



# %%
# Plot features


# # %%
# # z score by fish id
# cat_cols = ['condition', 'ztime', 'fish_id','time','clutch_id']
# features = list(set(high_bout_data.columns).difference(set(cat_cols)))


# high_bout_zScore = pd.DataFrame()
# data_toplt = high_bout_data
# sibs_mean = data_toplt.loc[data_toplt.condition=='sibs',features]
# sibs_mean = sibs_mean.mean()
# this_z = data_toplt.reset_index(drop=True)
# for feature in features:
#     this_z.loc[:,feature] = this_z.groupby('fish_id')[feature].apply(
#             lambda x: (x-sibs_mean[feature])/np.std(x)
#             )
# high_bout_zScore = pd.concat([high_bout_zScore,this_z],ignore_index=True) 

# # # %%

# corr = high_bout_zScore[features].sort_index(axis = 1).corr()

# # Generate a mask for the upper triangle
# mask = np.triu(np.ones_like(corr, dtype=bool))

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))

# # Generate a custom diverging colormap
# cmap = sns.diverging_palette(230, 20, as_cmap=True)

# # Draw the heatmap with the mask and correct aspect ratio
# sns.heatmap(corr, cmap=cmap, vmax=1,vmin=-1, center=0,
#             square=True, linewidths=.5, cbar_kws={"shrink": .5})


# %%
# z score by clutch id
cat_cols = ['condition', 'ztime', 'fish_id','time','clutch_id']
features = list(set(high_bout_data.columns).difference(set(cat_cols)))

zscore_byClutch = pd.DataFrame()
for this_clutch, group in high_bout_data.groupby(['clutch_id']):
    this_sib = group.loc[group['condition']=='sibs',features]
    this_sib_mean = this_sib.mean()
    this_sib_std = this_sib.std()
    this_z = group.reset_index(drop=True)
    this_z.loc[:,features] = (this_z.loc[:,features]-this_sib_mean[features])/this_sib_std[features]
    zscore_byClutch = pd.concat([zscore_byClutch, this_z],ignore_index=True)
    
# percent_chg = pd.DataFrame()
# for (this_clutch, this_direction), group in high_bout_data.groupby(['clutch_id']):
#     this_sib = group.loc[group['condition']=='sibs',features]
#     this_sib_mean = this_sib.mean()
#     this_sib_std = this_sib.std()
#     this_chg = group.reset_index(drop=True)
#     this_chg.loc[:,features] = (this_chg.loc[:,features]-this_sib_mean[features])/this_sib_mean[features]
#     percent_chg = pd.concat([percent_chg, this_chg],ignore_index=True)


# %% plot raw data
feature_to_plt = [
    # 'rot_late_accel',
    'pitch_peak',
    'pitch_initial',
    'rot_l_decel',
    'rot_l_accel',
    'pitch_end',
    'angvel_chg',
    'angvel_post_phase',
    'atk_ang',
    'bout_traj',
    'traj_peak',
    ]
# by fish
for feature in feature_to_plt:
    plt.figure()
    g = sns.catplot(
        data=high_bout_data,
        y=feature,
        x='condition',
        hue='fish_id',
        # units = 'fish_id',
        col='clutch_id',
        kind='point',
        # err_style='bars',
        # size=0,
        sharey = False,
        dodge=True
        )
    plt.savefig(fig_dir+f"/{feature} raw.pdf",format='PDF')


for feature in feature_to_plt:
    plt.figure()
    g = sns.catplot(
        data=high_bout_std,
        y=feature,
        x='condition',
        hue='fish_id',
        # units = 'fish_id',
        col='clutch_id',
        kind='point',
        # err_style='bars',
        # size=0,
        sharey = True,
        dodge=True,
        )
    plt.savefig(fig_dir+f"/{feature} std.pdf",format='PDF')

    
# %%
# by clutch
for feature in feature_to_plt:
    plt.figure()
    g = sns.catplot(
        data=high_bout_data,
        y=feature,
        x='condition',
        hue='clutch_id',
        # units = 'fish_id',
        kind='point',
        # err_style='bars',
        # size=0,
        sharey = False,
        dodge=True
        )
    plt.savefig(fig_dir+f"/mean {feature} raw.pdf",format='PDF')
    
# %% IEI
plt.figure()
g = sns.catplot(
    data=high_bout_IEI,
    y='IEIpitch',
    x='condition',
    hue='fish_id',
    # units = 'fish_id',
    col='clutch_id',
    kind='point',
    # err_style='bars',
    # size=0,
    sharey = False,
    dodge=True
    )
plt.savefig(fig_dir+f"/IEI duration raw.pdf",format='PDF')
# %%
# z score by clutch

for feature in feature_to_plt:
    plt.figure()
    g = sns.catplot(
        data=zscore_byClutch,
        y=feature,
        x='condition',
        hue='fish_id',
        # units = 'fish_id',
        col='clutch_id',
        kind='point',
        # err_style='bars',
        # size=0,
        sharey = False,
        dodge=True
        )
    plt.savefig(fig_dir+f"/{feature} zScore byClutch.pdf",format='PDF')

for feature in feature_to_plt:
    plt.figure()
    g = sns.catplot(
        data=zscore_byClutch,
        y=feature,
        x='condition',
        hue='clutch_id',
        # units = 'fish_id',
        kind='point',
        # err_style='bars',
        # size=0,
        sharey = False,
        dodge=True
        )
    plt.savefig(fig_dir+f"/mean {feature} zScore byClutch.pdf",format='PDF')
    
# %%
# percent change

# for feature in feature_to_plt:
#     plt.figure()
#     g = sns.catplot(
#         data=percent_chg,
#         y=feature,
#         x='condition',
#         hue='fish_id',
#         # units = 'fish_id',

#         row='direction',
#         col='clutch_id',
#         kind='point',
#         # err_style='bars',
#         # size=0,
#         sharey = False,
#         dodge=True
#         )
#     plt.savefig(fig_dir+f"/{feature} percentChg byClutch.pdf",format='PDF')

# for feature in feature_to_plt:
#     plt.figure()
#     g = sns.catplot(
#         data=percent_chg,
#         y=feature,
#         x='condition',
#         # hue='fish_id',
#         # units = 'fish_id',

#         row='direction',
#         col='clutch_id',
#         kind='point',
#         # err_style='bars',
#         # size=0,
#         sharey = False,
#         dodge=True
#         )
#     plt.savefig(fig_dir+f"/mean {feature} percentChg byClutch.pdf",format='PDF')

# %%
# Plot kinetics

features = [
'righting_gain',
       'steering_gain', 'corr_rot_accel_decel', 'corr_rot_lateAccel_decel',
       'corr_rot_earlyAccel_decel', 'corr_rot_preBout_decel', 'set_point'
]


data_toplt = high_bout_kinetics
for kinetics in features:
    sns.relplot(
        data=data_toplt,
        x=data_toplt.index,
        y=kinetics,
        kind='line',
        err_style='bars',
        size=0,
        facet_kws={'sharey': False, 'sharex': True}
        )
# %%
#  calculate zsocre from STD dataset
zscore_std = pd.DataFrame()
features = ['bout_traj','pitch_initial','pitch_end','rot_l_decel']
for this_clutch, group in high_bout_std.groupby(['clutch_id']):
    this_sib = group.loc[group['condition']=='sibs',features]
    this_std = group[features].std()
    this_z = group.reset_index(drop=True)
    for feature in features:
        this_feature_z = (this_z[feature] - this_sib.loc[0,feature]) / this_std[feature]
        this_z[feature] = this_feature_z
    zscore_std = pd.concat([zscore_std, this_z],ignore_index=True)
    
zscore_std = zscore_std.assign(
    feature_score = zscore_std[features].mean(axis=1)
)
# %%
zscore_kina = pd.DataFrame()
features = ['steering_gain','righting_gain']
for this_clutch, group in high_bout_kinetics.groupby(['clutch_id']):
    this_sib = group.loc[group['condition']=='sibs',features]
    this_std = group[features].std()
    this_z = group.reset_index(drop=True)
    for feature in features:
        this_feature_z = (this_z[feature] - this_sib.loc[0,feature]) / this_std[feature]
        this_z[feature] = this_feature_z
    zscore_kina = pd.concat([zscore_kina, this_z],ignore_index=True)

zscore_kina = zscore_std.assign(
    kinetics_score = zscore_kina[features].mean(axis=1) * -1
)
# %%
plt.figure(figsize=(4,3))
sns.scatterplot(
    x=zscore_std['feature_score'],
    y=zscore_kina['kinetics_score'],
    # hue='fish_id',
    style=zscore_std['condition'],
    c=zscore_std['fish_id'],
    s=50
)
plt.savefig(fig_dir+f"/behavior score corr.pdf",format='PDF')

# %%
