'''
Very useful new function!
Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

refer to the 2D distribution plot sections for custimization.
'''

#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features, get_connected_bouts
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl
from scipy.stats import linregress, theilslopes
from functools import partial

def day_night_split(df,time_col_name, narrow_bin = False, **kwargs):
    which_ztime = 'day'
    for key, value in kwargs.items():
        if key == 'ztime':
            which_ztime = value
            
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    
    if narrow_bin:
        day_night_idx = pd.cut(hour,
                        bins=[-1, 7, 9, 21, 24],  # Define edges explicitly
                        labels=['night', 'transition1', 'day', 'transition2'],  
                        ordered=True
                        )
    else:
        day_night_idx = pd.cut(hour,[-1,8,22,24],labels=['night','day','night2']) # pd.cut() doesn't support duplicated labels
        
    day_night_idx.loc[day_night_idx=='night2'] = 'night'
    df = df.assign(ztime = list(day_night_idx))
    
    if which_ztime == 'all':
        df_out = df
    else:
        df_out = df.loc[df['ztime']==which_ztime, :]
    return df_out#, day_index, night_index

##### Parameters to change #####

pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF3_distribution2D_z{which_ztime}'
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
all_feature_cond_ori, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond_ori = all_feature_cond_ori.reset_index(drop=True)
# all_feature_cond_ori, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_feature_cond_ori = all_feature_cond_ori.reset_index(drop=True)

all_feature_cond_ori = all_feature_cond_ori.assign(
    xdispl_swim = np.abs(all_feature_cond_ori['xdispl_swim']),
    x_pre_swim = np.abs(all_feature_cond_ori['x_pre_swim']),
    x_post_swim = np.abs(all_feature_cond_ori['x_post_swim']),
)

all_feature_cond_strictz = day_night_split(all_feature_cond_ori, 'bout_time', narrow_bin=True, ztime='all')

pick_df = all_feature_cond_ori.copy()

# %% 2D distribution plot

####################################
###### Plotting Starts Here ######
####################################

# %% here's an example

toplt = all_feature_cond_ori
xname = 'spd_peak'
yname = 'displ_swim'
if len(toplt) > 50000:
    toplt = toplt.sample(n=50000)
g = sns.displot(data= toplt,
                y=yname,x=xname,
                row='cond1',
                col='cond0',
                height=3,
                facet_kws={"ylim":[0,6],"xlim":[5,35]}
                )
g.add_legend()

plt.savefig(fig_dir+f"/{yname} v {xname}.pdf",format='PDF')

# %%
pick_df = all_feature_cond_ori.copy()


def calculate_slope(group, parx='xdispl_swim', pary='spd_peak'):
    """Calculate slope of spd_peak vs xdispl_swim for a given group."""
    if len(group) < 2:
        return np.nan  # Not enough data to calculate slope
    y = group[parx].values
    x = group[pary].values

    slope, _, _, _, _ = linregress(x, y)
    return slope

def process_dataframe(pick_df, groupby_cols, time_col_name='bout_time', bin_hours=2):
    """
    Groups data by groupby_cols and 2-hour bins, 
    then calculates the slope of spd_peak vs xdispl_swim.
    """
    pick_df = pick_df.copy()
    # Determine first recorded time for each experimental group
    first_entry = pick_df.groupby(['cond0', 'cond1', 'expNum'])[time_col_name].min().reset_index()
    first_entry.rename(columns={time_col_name: 'start_time'}, inplace=True)
    
    # Define 10 AM time boundaries for each group
    first_entry['start_time_10AM'] = first_entry['start_time'].dt.floor('D') + pd.Timedelta(hours=10)
    first_entry['third_day_cutoff'] = first_entry['start_time_10AM'] + pd.Timedelta(days=2)
    
    # Merge back to assign correct time boundaries per group
    pick_df = pick_df.merge(first_entry[['cond0', 'cond1', 'expNum', 'start_time_10AM', 'third_day_cutoff']], on=['cond0', 'cond1', 'expNum'])
    pick_df = pick_df[(pick_df[time_col_name] >= pick_df['start_time_10AM']) & (pick_df[time_col_name] < pick_df['third_day_cutoff'])]
    
    # Compute time since 10 AM of day 1
    pick_df['time_since_start'] = (pick_df[time_col_name] - pick_df['start_time_10AM']).dt.total_seconds() / 3600
    
    # Define bins based on bin_hours
    pick_df['hour_bin'] = (pick_df['time_since_start'] // bin_hours) * bin_hours + 10  # Align to real hours
    
    # Group by conditions and time bins
    grouped = pick_df.groupby(groupby_cols + ['hour_bin'])
    
    # Compute slope for each group
    slopes = grouped.apply(calculate_slope).reset_index()
    slopes.rename(columns={0: 'slope'}, inplace=True)
    return slopes

binned_slope = process_dataframe(pick_df, groupby_cols=['cond0', 'cond1', 'expNum'])

# %%
g = sns.relplot(
    data=binned_slope,
    x='hour_bin', y='slope',
    hue='cond1',
    kind='line',
    facet_kws={'sharey': True, 'sharex': True},
    units = 'expNum',
    estimator=None,
    alpha=0.5
)

# Add shaded regions for nighttime (11 PM - 9 AM)
for ax in g.axes.flat:
    ax.axvspan(23, 33, color='lightgray', alpha=0.3)  # 11 PM - 9 AM (first night)
    ax.axvspan(47, 57, color='lightgray', alpha=0.3)  # 11 PM - 9 AM (second night)
g
# %%
pick_df = all_feature_cond_strictz.copy()

pick_df = pick_df.sort_values(by=['cond0', 'cond1', 'expNum', 'bout_time'])  # Ensure chronological order
# Compute slope using finite differences within each (cond0, cond1, expNum) group
pick_df['ratio'] = pick_df[yname] / pick_df[xname]
#%
xpar = 'ratio'
g = sns.displot(
    data=pick_df,
    x=xpar,
    stat='probability',
    common_norm=False,
    hue='cond1',
    row='expNum',
    col='ztime',
    facet_kws={'xlim':np.percentile(pick_df[xpar],[2,98])},
    bins='scott',
    element='poly',
    height=2,
    alpha=0.2
)
plt.savefig(fig_dir+f"/ratio distribution by cond0 cond1.pdf",format='PDF')
#%%

# %% focus on light and dark


# parx = 'xdispl_swim'
# pary = 'ydispl_swim'


# pick_df = all_feature_cond_ori.reset_index()
all_feature_cond_ori = all_feature_cond_ori.assign(
    pitch_bins = pd.cut(all_feature_cond_ori.pitch_peak, np.percentile(all_feature_cond_ori.pitch_peak, [0,20,40,60,80,100])),
    dir_bins = pd.cut(all_feature_cond_ori.bout_trajectory_Pre2Post, np.percentile(all_feature_cond_ori.bout_trajectory_Pre2Post, [0,20,40,60,80,100])),
    x_displ_spd_ratio = all_feature_cond_ori.xdispl_swim / (np.cos(np.radians(all_feature_cond_ori.traj_peak)) * all_feature_cond_ori.spd_peak),#all_feature_cond_ori.spd_peak,
    spd_ratio = all_feature_cond_ori.displ_swim / all_feature_cond_ori.spd_peak,
    y_adjusted_val = all_feature_cond_ori.depth_chg,
    xspd_peak = np.cos(np.radians(all_feature_cond_ori.traj_peak)) * all_feature_cond_ori.spd_peak,
    yspd_peak = np.sin(np.radians(all_feature_cond_ori.traj_peak)) * all_feature_cond_ori.spd_peak,
    ratio_xspd_peak = all_feature_cond_ori.xdispl_swim / (np.cos(np.radians(all_feature_cond_ori.traj_peak)) * all_feature_cond_ori.spd_peak),
)

pary = 'displ_swim'
parx = 'spd_peak'

pick_df = all_feature_cond_ori.reset_index()
pick_df.dropna(subset=[parx, pary], inplace=True)
#%% sanity check by box

data_pick = all_feature_cond_strictz.query("ztime == 'night'").query("cond1 == 'ld'")
#%%
sns.relplot(
    kind='scatter',
    data=data_pick.groupby(['expNum','boxNum']).sample(n=1000, replace=True),
    # hue='cond1',
    y=pary,
    x=parx,
    col='expNum',
    row='boxNum',
    height=2,
    alpha=0.1,
    facet_kws={'xlim':np.percentile(pick_df[parx],[1,99]),
               'ylim':np.percentile(pick_df[pary],[1,99])},
)


#%%
sns.relplot(
    kind='scatter',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000),
    hue='cond1',
    y='xdispl_swim',
    x='xspd_peak',
    col='cond0',
    row='cond1',
    height=3,
    alpha=0.02,
    facet_kws={'xlim':np.percentile(pick_df['xspd_peak'],[1,99]),
               'ylim':np.percentile(pick_df['xdispl_swim'],[1,99])},
)
plt.savefig(fig_dir+f"/{'xspd_peak'}|{'xdispl_swim'} scatter.pdf",format='PDF')



sns.relplot(
    kind='scatter',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000),
    hue='cond1',
    x=parx,
    y=pary,
    col='cond0',
    row='cond1',
    height=3,
    alpha=0.02,
    facet_kws={'xlim':np.percentile(pick_df[parx],[2,98]),
               'ylim':np.percentile(pick_df[pary],[2,98])},
)
plt.savefig(fig_dir+f"/{pary}|{parx} scatter.pdf",format='PDF')

#%%
g = sns.displot(
    data=pick_df,
    x=pary,
    stat='probability',
    common_norm=False,
    hue='cond1',
    col='cond0',
    facet_kws={'xlim':np.percentile(pick_df[pary],[1,99])},
    bins='scott',
    element='poly',
    height=2,
    alpha=0.2
)
plt.savefig(fig_dir+f"/{pary} dis.pdf",format='PDF')


#%%
sns.relplot(
    kind='scatter',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000).query("cond1 != 'll'").query("ztime == 'day'"),
    hue='cond1',
    x=parx,
    y=pary,
    # col='ztime',
    # row='cond1',
    height=3,
    alpha=0.02,
    facet_kws={'xlim':np.percentile(pick_df[parx],[2,98]),
               'ylim':np.percentile(pick_df[pary],[2,98])},
)
plt.savefig(fig_dir+f"/ld dd day only {pary}|{parx} scatter.pdf",format='PDF')


sns.relplot(
    kind='scatter',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000).query("cond1 != 'll'").query("ztime == 'night'"),
    hue='cond1',
    x=parx,
    y=pary,
    # col='ztime',
    col='cond1',
    height=3,
    alpha=0.02,
    facet_kws={'xlim':np.percentile(pick_df[parx],[2,98]),
               'ylim':np.percentile(pick_df[pary],[2,98])},
)
plt.savefig(fig_dir+f"/ld dd night only {pary}|{parx} scatter.pdf",format='PDF')


sns.displot(
    kind='kde',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000).query("cond1 != 'll'").query("ztime == 'day'"),
    hue='cond1',
    x=parx,
    y=pary,
    # col='ztime',
    # row='cond1',
    height=3,
    facet_kws={'xlim':np.percentile(pick_df[parx],[1,99]),
            'ylim':np.percentile(pick_df[pary],[1,99])},
)
plt.savefig(fig_dir+f"/ld dd day only {pary}|{parx} kde.pdf",format='PDF')

# %%

# %%

#%%
data_2plot = pick_df.groupby(['cond1','ztime']).sample(n=10000).query("cond1 != 'll'").query("ztime == 'day'")

# for y_to_plot in all_feature_cond_ori.columns:
#     sns.displot(
#         kind='kde',
#         data=data_2plot,
#         hue='cond1',
#         x=parx,
#         y=y_to_plot,
#         # col='ztime',
#         # row='cond1',
#         height=3,
#         facet_kws={'xlim':np.percentile(pick_df[parx],[1,99]),
#                 'ylim':np.percentile(pick_df[y_to_plot],[1,99])},
#     )
#     plt.savefig(fig_dir+f"/ld dd day only {y_to_plot}|{parx} kde.pdf",format='PDF')


# %%

#%%

pick_bin = 'dir_bins'
parx = 'spd_peak'
pary = 'displ_swim'

pick_df = all_feature_cond_ori.reset_index()
pick_df.dropna(subset=[parx, pary], inplace=True)


sns.relplot(
    kind='scatter',
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000),
    hue='cond1',
    x=parx,
    y=pary,
    col='ztime',
    row='cond1',
    height=3,
    alpha=0.02,
    facet_kws={'xlim':np.percentile(pick_df[parx],[2,98]),
               'ylim':np.percentile(pick_df[pary],[2,98])},
)
plt.savefig(fig_dir+f"/{pary}|{parx} scatter.pdf",format='PDF')

# g = sns.displot(
#     data=pick_df,
#     x=parx,
#     stat='probability',
#     common_norm=False,
#     hue='cond1',
#     col='ztime',
#     facet_kws={'xlim':np.percentile(pick_df[parx],[1,99])},
#     bins='scott',
#     element='poly',
#     height=2,
#     alpha=0.2
# )
#%%
g = sns.displot(
    data=pick_df.groupby(['cond1','ztime']).sample(n=10000),
    y=pary,
    x=parx,
    row='cond1',
    col='ztime',
    # stat='density',
    # kind='kde',
    pthresh=0.05,
    pmax=.9,
    height=3,
    facet_kws={'xlim':np.percentile(pick_df[parx],[1,99]),
               'ylim':np.percentile(pick_df[pary],[1,99])},
    )
g.add_legend()

plt.savefig(fig_dir+f"/{pary}|{parx} dist.pdf",format='PDF')


# %%



#%%

pick_bin = 'dir_bins'
parx = 'x_adjusted_val'
pary = 'y_adjusted_val'

pick_df = all_feature_cond_ori.query("ztime=='day'")
pick_df = pick_df.query("cond1 != 'll'")

slope_function2use = partial(calculate_slope, parx=parx, pary=pary)
pick_df_res = pick_df.groupby(['cond0','cond1','expNum',pick_bin]).apply(slope_function2use).reset_index()
pick_df_res.rename(columns={0: 'slope'}, inplace=True)
# plot point plot
sns.catplot(
    data=pick_df_res,
    y='slope',
    hue='cond1',
    x=pick_bin,
    # units='expNum',
    kind='point',
    linestyle='none',
    height=3,
    aspect=3
    )

sns.relplot(
    kind='scatter',
    data=pick_df.groupby('cond1').sample(n=10000),
    row=pick_bin,
    hue='cond1',
    col='cond1',
    x=parx,
    y=pary,
    height=2,
    alpha=0.01,
    facet_kws={'xlim':np.percentile(pick_df[parx],[2,98]),
               'ylim':np.percentile(pick_df[pary],[2,98])},
)
# %%

# %%
# debug, ignore


import os,glob
from pickle import FRAME
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
from plot_functions.plt_tools import day_night_split
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
import math

ztime = which_ztime
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.2*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]
idxRANGE_features = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.25*FRAME_RATE)]

# for day night split
which_zeitgeber = 'all'


# %%
