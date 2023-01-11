'''
plot correlation of features by time
1. corr of angular vel at each timpoint with preBoutPitch / atkAngle / trajectory deviation
2. corr of ins. trajectory at each timepoint with bout trajectory
trajectory deviation (trajecgtory residual) is defined as (bout_traj_peakecgtory - pitch_pre_bout)

'''

#%%
# import sys
import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_average, day_night_split)
from tqdm import tqdm
import matplotlib as mpl
from plot_functions.get_bout_features import get_bout_features

set_font_type()

# %%
# Paste root directory here
# if_plot_by_speed = True
pick_data = '7dd_all'
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'{pick_data} corr righting_pitch'
folder_dir6 = get_figure_dir('Fig_6')
folder_dir4 = get_figure_dir('Fig_4')

fig_dir6 = os.path.join(folder_dir6, folder_name)

try:
    os.makedirs(fig_dir6)
except:
    pass   
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.27*FRAME_RATE),peak_idx+round_half_up(0.22*FRAME_RATE)]

# %% get features
all_feature_cond, all_cond0, all_cond0 = get_bout_features(root, FRAME_RATE, ztime = 'day')
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# %%
by_which_col = 'pitch_initial'
binned_rotation = 'rot_l_decel'

xname = 'Initial Pitch'
yname = 'Decel. Rotation'

# by_which_col = 'rot_narrow_ang_accel'
print("Figure 6: Scatter plot of traj. deviation vs acceleration rotation")

feature_to_plt = [by_which_col]
toplt = all_feature_cond

for feature in feature_to_plt:
    upper = np.percentile(toplt[feature], 99)
    lower = np.percentile(toplt[feature], 1)

BIN_WIDTH = 2
AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
binned_df_cond = toplt.groupby('cond1').apply(
    lambda g: distribution_binned_average(g,by_col=by_which_col,bin_col=binned_rotation,bin=AVERAGE_BIN)
)
binned_df = distribution_binned_average(toplt,by_col=by_which_col,bin_col=binned_rotation,bin=AVERAGE_BIN)
binned_df.columns=[xname,yname]
binned_df_cond.columns=[xname,yname]
binned_df_cond = binned_df_cond.reset_index()
# %%
print("Figure 6: ")
 
g = sns.relplot(
    kind='scatter',
    data = toplt.sample(frac=0.2),
    x = by_which_col,
    y = binned_rotation,
    # x_bins=np.arange(round_half_up(lower),round_half_up(upper),3),
    # x_ci=95,
    alpha=0.1,
    # hue='direction',
    # marker='+',
    linewidth = 0,
    color = 'grey',
    height=2.5,
    aspect=5/4,
)
# g.set(ylim=(-25,40))

g.set(ylim=(-5,8))
g.set(xlim=(lower,upper))
g.map(sns.lineplot,data=binned_df,
      x=xname,
      y=yname)
g.set_axis_labels(x_var = xname, y_var = yname)
sns.despine()
plt.savefig(fig_dir6+f"/decel rot vs {by_which_col}.pdf",format='PDF')
r_val = stats.pearsonr(toplt[by_which_col],toplt[binned_rotation])[0]
print(f"pearson's r = {r_val}")

# #%%
# print("Figure 4: Distibution of traj. deviation and pitch change")
# feature = 'traj_deviation'
# plt.figure(figsize=(3,2))
# upper = np.percentile(features_all[feature], 99.5)
# lower = np.percentile(features_all[feature], 1)
# xlabel = feature + " (deg)"
# g = sns.histplot(
#     data = features_all,
#     x = feature,
#     bins = 20, 
#     element="poly",
#     stat="probability",
#     pthresh=0.05,
#     binrange=(lower,upper),
#     color='grey'
# )
# g.set_xlabel(xlabel)
# sns.despine()
# plt.savefig(os.path.join(fig_dir4,f"{feature} distribution.pdf"),format='PDF')

# feature = 'rot_all_l_accel'
# plt.figure(figsize=(3,2))
# upper = np.percentile(features_all[feature], 99.5)
# lower = np.percentile(features_all[feature], 1)
# xlabel = "Pitch change from initial to peak (deg)"
# g = sns.histplot(
#     data = features_all,
#     x = feature,
#     bins = 20, 
#     element="poly",
#     stat="probability",
#     pthresh=0.05,
#     binrange=(-13,20),
#     color='grey'
# )
# g.set_xlabel(xlabel)
# sns.despine()
# plt.savefig(os.path.join(fig_dir4,f"{feature} distribution.pdf"),format='PDF')


# %%
