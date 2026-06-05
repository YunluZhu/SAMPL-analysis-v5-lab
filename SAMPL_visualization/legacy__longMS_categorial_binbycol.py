'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# %%
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# %%

cat_cols = ['cond1','expNum','cond0','bout_time','ztime','exp','direction']
feature_to_plt = [c for c in all_feature_cond.columns if c not in cat_cols]
all_feature_cond['xdispl_swim'] = all_feature_cond['xdispl_swim'].abs()
feature_for_comp = feature_to_plt

cat_cols = ['cond0','cond1','expNum','ztime']
avg_data = all_feature_cond.groupby(cat_cols)[feature_to_plt].median()
avg_data = avg_data.reset_index()

cat_cols = ['cond0','cond1','expNum','ztime']

# %%
feature_to_plt.sort()
df_to_plt = all_feature_cond.query("cond1=='ld' and ztime=='night'")

# %% Goal: find how fish at different ages climbs

# calculate 
col1 = 'rot_to_max_angvel'
col2 = 'atk_ang'
#%
df_to_plt_selected = df_to_plt.groupby('cond0').sample(2000, replace=True)

color_par = 'pitch_initial'

# remove extreme outliers for better visualization
df_to_plt_selected = df_to_plt_selected[df_to_plt_selected[color_par]< np.percentile(df_to_plt_selected[color_par],99)]
df_to_plt_selected = df_to_plt_selected[df_to_plt_selected[color_par]< np.percentile(df_to_plt_selected[color_par],99)] 

g = sns.relplot(
    data=df_to_plt_selected,
    col='cond0',
    row=pd.cut(df_to_plt_selected[color_par],[-np.inf,10,np.inf],labels=['Negative pitch','Positive pitch']),
    x=col1,
    y=col2,
    height=3,
    col_order=all_cond0,
    hue=color_par,
    alpha=0.05,
    palette='viridis',

)
g.set(xlim=np.percentile(df_to_plt_selected.dropna()[col1],[1,99]), ylim=np.percentile(df_to_plt_selected.dropna()[col2],[1,99]))


# %%

# Index(['x_initial', 'y_initial', 'x_end', 'y_end', 'pitch_initial',
#        'pitch_mid_accel', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
#        'pitch_end', 'pitch_max_angvel', 'traj_initial', 'traj_pre_bout',
#        'traj_peak', 'traj_post_bout', 'traj_end', 'spd_initial', 'spd_peak',
#        'angvel_initial_phase', 'angvel_prep_phase', 'angvel_post_phase',
#        'traj_initial_phase', 'spd_initial_phase', 'fish_length', 'boxNum',
#        'rot_total', 'rot_bout', 'rot_pre_bout', 'rot_l_accel',
#        'rot_full_accel', 'rot_full_decel', 'rot_l_decel', 'rot_early_accel',
#        'rot_late_accel', 'rot_early_decel', 'rot_late_decel',
#        'rot_to_max_angvel', 'bout_trajectory_Pre2Post', 'bout_displ',
#        'traj_deviation', 'atk_ang', 'tsp_peak', 'angvel_chg', 'depth_chg',
#        'depth_chg_fullBout', 'x_chg', 'x_chg_fullBout', 'lift_distance',
#        'lift_distance_fullBout', 'additional_depth_chg', 'displ_swim',
#        'ydispl_swim', 'xdispl_swim', 'y_post_swim', 'y_pre_swim',
#        'x_post_swim', 'x_pre_swim', 'WHM', 'half_spd_peak', 'bout_time',
#        'expNum', 'exp', 'cond0', 'cond1', 'ztime'],
#       dtype='object')

col1 = 'pitch_peak'
col2 = 'traj_peak'
#%
df_to_plt_selected = df_to_plt.groupby('cond0').sample(2000, replace=True)

color_par = 'pitch_initial'

# remove extreme outliers for better visualization
df_to_plt_selected = df_to_plt_selected[df_to_plt_selected[color_par]< np.percentile(df_to_plt_selected[color_par],99)]
df_to_plt_selected = df_to_plt_selected[df_to_plt_selected[color_par]> np.percentile(df_to_plt_selected[color_par],1)] 
df_to_plt_selected = df_to_plt_selected.assign(
    pitch_cat = pd.cut(df_to_plt_selected[color_par],[-np.inf,10,np.inf],labels=['Negative pitch','Positive pitch']),
)

g = sns.relplot(
    data=df_to_plt_selected,
    col='cond0',
    x=col1,
    y=col2,
    height=3,
    col_order=all_cond0,
    hue=color_par,
    alpha=0.05,
    # split color
    palette=sns.color_palette("viridis", as_cmap=True),
)
g.set(xlim=np.percentile(df_to_plt_selected.dropna()[col1],[1,99]), ylim=np.percentile(df_to_plt_selected.dropna()[col2],[1,99]))

# g = sns.lmplot(
#     data=df_to_plt_selected,
#     col='cond0',
#     x=col1,
#     y=col2,
#     height=3,
#     col_order=all_cond0,
#     # palette='viridis',
# )

# %%

df_to_plt = df_to_plt.assign(
    pitch_cat = pd.cut(df_to_plt[color_par],[-np.inf,10,np.inf],labels=['Negative pitch','Positive pitch']),
)
#%%
binned_avg_dfs = []
by_col = 'depth_chg_fullBout'
bin_col = 'additional_depth_chg'

min=np.percentile(df_to_plt[by_col], 1)
max=np.percentile(df_to_plt[by_col], 99)
bins = np.linspace(min, max, 20)
mid_bins = (bins[:-1] + bins[1:]) / 2   



for cond, group in df_to_plt.groupby(['pitch_cat','cond0','expNum']):
    this_df = distribution_binned_average_opt(
        df=group,
        by_col=by_col,
        bin_col=bin_col,
        bin=bins,
        method='mean',
    )
    this_df.columns = [f'binned_{by_col}',f'binned_{bin_col}']
    this_df = this_df.reset_index()
    this_df['cond0'] = cond[1]
    this_df['pitch_cat'] = cond[0]
    this_df['expNum'] = cond[2]
    this_df['mid_bins'] = mid_bins
    binned_avg_dfs.append(this_df)
binned_avg_df = pd.concat(binned_avg_dfs, ignore_index=True)

g = sns.relplot(
    kind='line',
    data=binned_avg_df,
    hue='cond0',
    x='mid_bins',
    y=f'binned_{bin_col}',
    # hue='pitch_cat',
    height=3,
    # col_order=all_cond0,
    # marker='_',
    linestyle='-',
)

#%%
g = sns.relplot(
    kind='scatter',
    data=df_to_plt,
    hue='cond0',
    hue_order=all_cond0,
    palette=my_palette,
    x='pitch_peak',
    y='depth_chg_fullBout',
    # hue='pitch_cat',
    col='cond0',
    height=5,
    # col_order=all_cond0,
    alpha=0.05,
)
# %%
