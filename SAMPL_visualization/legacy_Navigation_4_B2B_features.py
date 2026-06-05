# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_connected_bouts)
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid2
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)


set_font_type()

# %%
pick_data = 'wt_dl'
if_day_light_narrow_bin = True
folder_name = f'Navi4_B2B_features' 
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
    
root, FRAME_RATE= get_data_dir(pick_data)

# get consecutive bouts
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, day_light_narrow_bin = if_day_light_narrow_bin)

# tidy bout uid
all_features = all_features.assign(
    epoch_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str),
)

all_features = all_features.loc[all_features['ztime'].isin(['day','night'])].reset_index(drop=True)

# %% std of directions of consecutive bouts
list_of_features = [
    'traj_peak',
    'bout_trajectory_Pre2Post',
    'y_post_swim','y_pre_swim', 'ydispl_swim', 
    # 'y_end','y_initial',
    
    'post_IBI_align_time',
    'post_IBI_time',
    
    'spd_peak',
    
    'pitch_end', 'pitch_initial','rot_total',
    
    # grab x parameters
    'x_pre_swim','x_post_swim',
    # 'x_end','x_initial',
    'xdispl_swim',
    
    'atk_ang'
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts

#####################
max_lag = 4
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)
# take absolute value of x parameters
consecutive_bout_features = consecutive_bout_features.assign(
    xdispl_swim = np.abs(consecutive_bout_features['xdispl_swim']),
    x_pre_swim = np.abs(consecutive_bout_features['x_pre_swim']),
    x_post_swim = np.abs(consecutive_bout_features['x_post_swim']),
    # x_initial = np.abs(consecutive_bout_features['x_initial']),
    # x_end = np.abs(consecutive_bout_features['x_end']),
)
# %% 
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond0', 'cond1','id']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    # swim to swim, 5 mm/s
    post_S2S_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values, np.nan),
    pre_S2S_ydispl = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values),
    # bout to bout, aligned, using next initial - previous end
    # post_B2B_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['y_end'].values, np.nan),
    # ydispl_bout = sel_consecutive_bouts.iloc[:,:]['y_end'].values - sel_consecutive_bouts.iloc[:,:]['y_initial'].values,
    
    # x displacement
    post_S2S_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values, np.nan)),
    pre_S2S_xdispl = np.abs(np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values)),
    # post_B2B_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['x_end'].values, np.nan)),
    # xdispl_bout = np.abs(sel_consecutive_bouts.iloc[:,:]['x_end'].values - sel_consecutive_bouts.iloc[:,:]['x_initial'].values),
    
    # rotation
    post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),

    bouts = sel_consecutive_bouts['lag'] + 1,
)
# IMPORTANT: because did the above calculation in the messy way, we can only pick the middle bouts. Drop the first and last bout of each series
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)

#%%
num_of_IBI_bins = 5

bout_trajectory_p2p = 'bout_trajectory_Pre2Post'

middle_bout_df = middle_bout_df.query("post_IBI_align_time > 0")

IBI_bin_edges = np.percentile(middle_bout_df['post_IBI_time'].dropna(), np.linspace(0,100,num_of_IBI_bins+1))
other_bin_ediges = np.percentile(middle_bout_df[bout_trajectory_p2p].dropna(), np.linspace(0,100,num_of_IBI_bins+1))
middle_bout_df = middle_bout_df.assign(
    # bout type
    bout_type = pd.cut(middle_bout_df['ydispl_swim'], bins=[-np.inf,0,np.inf], labels=['sink','rise']),
    post_IBIdur_bins = pd.cut(middle_bout_df['post_IBI_time'], bins=IBI_bin_edges),
    # pitch end bins
    bout_trajectory_p2p_bins = pd.cut(middle_bout_df[bout_trajectory_p2p], bins=other_bin_ediges),
)

# %% Let's plot correlation

p = sns.relplot(
    kind='scatter',
    data=middle_bout_df,
    y='post_S2S_ydispl',
    x='ydispl_swim',
    col='cond1',
    alpha=0.01
)


#%% Let's plot correlation

g = sns.relplot(
    kind='scatter',
    data=middle_bout_df,
    y='post_S2S_ydispl',
    x='ydispl_swim',
    row='post_IBIdur_bins',
    # hue='bout_type',
    col='cond1',
    alpha=0.1,
    facet_kws={'xlim':[-3,3],'ylim':[-3,3]}
)

# NOTE nope I didn't see anything interesting here
# %%
df_tocorr = middle_bout_df[[
    # 'traj_peak',
    'bout_trajectory_Pre2Post',
    'ydispl_swim', 
    'xdispl_swim',
    'post_IBI_time',
    'spd_peak',
    'post_B2B_rot',
    'rot_total',
    'post_S2S_ydispl',
    'pre_S2S_ydispl',
    # 'pre_S2S_xdispl',
    # 'post_S2S_xdispl',
    'atk_ang'
]]
# do a pairwise correlation
corr = df_tocorr.corr()
sns.heatmap(
    corr, 
    annot=True,
    #change hue
    cmap='coolwarm',
    center=0,
    #scale to -1,1
    vmin=-1,
    vmax=1,
    annot_kws={"size": 2},
    )
plt.show()
# %%
# looks like IBI rotation is tightly correlated with IBI duration is highly correlated, as expected
# let's do a linear regression
g = sns.relplot(
    kind='scatter',
    data=middle_bout_df,
    y='post_B2B_rot',
    x='post_IBI_time',
    row='post_IBIdur_bins',
    col='cond1',
    alpha=0.1
)
g.set(xlim=(0,20))

# %%
g = sns.lmplot(
    data=middle_bout_df,
    y='xdispl_swim',
    x='spd_peak',
    col='cond1',
    # alpha=0.1,
    hue='bout_trajectory_p2p_bins',
    height=2,
    #remove scatter plot
    scatter=False,
    # facet_kws={'xlim':[-3,3],'ylim':[-3,3]}
)
# %%
