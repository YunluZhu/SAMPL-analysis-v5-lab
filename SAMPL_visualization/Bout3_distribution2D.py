'''
Very useful new function!
Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

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
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

##### Parameters to change #####

pick_data = 'xxxxxx' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime) # type: ignore
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)

# %% organize and annotate data
# Label bouts as upward or downward based on initial pitch
all_feature_cond['direction'] = pd.cut(
    all_feature_cond['pitch_initial'],
    bins=[-90, 10, 90],
    labels=['DN', 'UP']
)

# Sort datasets to keep experiments grouped together
all_feature_cond = (
    all_feature_cond
    .sort_values(['cond1', 'expNum'])
    .reset_index(drop=True)
)

all_ibi_cond = (
    all_ibi_cond
    .sort_values(['cond1', 'expNum'])
    .reset_index(drop=True)
)

# Calculate bout frequency and label IBI direction
all_ibi_cond = all_ibi_cond.assign(
    y_boutFreq=1 / all_ibi_cond['propBoutIEI'],
    direction=pd.cut(
        all_ibi_cond['propBoutIEI_pitch'],
        bins=[-90, 10, 90],
        labels=['DN', 'UP']
    )
)

# Calculate kinetics for each condition
all_kinetics = (
    all_feature_cond
    .groupby('cond0')
    .apply(get_kinetics)
    .reset_index()
)

# Alias used in plotting examples
all_feature_UD = all_feature_cond.copy()

# %% 2D distribution plot

####################################
###### Plotting Starts Here ######
####################################

# %%
# ============================================================================
# Example 1: Simple 2D distribution plot
# ============================================================================
# Change xname and yname to any feature pair you'd like to visualize.

toplt = all_feature_UD

xname = 'traj_peak'
yname = 'atk_ang'

# Downsample for speed if dataset is very large
if len(toplt) > 10000:
    toplt = toplt.sample(n=10000)

g = sns.displot(
    data=toplt,
    x=xname,
    y=yname,
    col='cond0',
    row='cond1',
    hue='cond1',
)

g.add_legend()

plt.savefig(
    os.path.join(fig_dir, f"{yname} v {xname}.pdf"),
    format='PDF'
)

# %%
# ============================================================================
# Example 2: Batch plot multiple feature pairs
# ============================================================================
# Add or remove feature pairs below.

toplt = all_feature_cond.assign(
    xdispl_swim_abs=lambda df: df['xdispl_swim'].abs()
)

features_to_plot = [
    ('bout_trajectory_Pre2Post', 'spd_peak'),
    ('pitch_peak', 'spd_peak'),

    # Additional useful examples:
    # ('traj_peak', 'pitch_peak'),
    # ('pitch_initial', 'rot_l_decel'),
    # ('rot_to_max_angvel', 'atk_ang'),
    # ('pitch_peak', 'depth_chg'),
    # ('additional_depth_chg', 'depth_chg'),
    # ('angvel_initial_phase', 'angvel_chg'),
    # ('pitch_end', 'angvel_post_phase'),
    # ('xdispl_swim_abs', 'ydispl_swim'),
]

for xname, yname in features_to_plot:

    sampled = toplt.groupby('cond1').sample(
        n=min(30000, len(toplt))
    )

    xlim = (
        np.percentile(toplt[xname], 0.2),
        np.percentile(toplt[xname], 99.8)
    )

    ylim = (
        np.percentile(toplt[yname], 0.2),
        np.percentile(toplt[yname], 99.8)
    )

    g = sns.displot(
        data=sampled,
        x=xname,
        y=yname,
        col='cond0',
        row='cond1',
        hue='cond1',
        height=3,
    )

    g.add_legend()
    g.set(xlim=xlim, ylim=ylim)

    plt.savefig(
        os.path.join(fig_dir, f"{yname} v {xname}.pdf"),
        format='PDF'
    )

# %%
# ============================================================================
# Example 3: KDE comparison between selected conditions
# ============================================================================
# Useful for highlighting distribution shape differences.

toplt = all_feature_cond.loc[
    all_feature_cond['cond1'].isin(['ld', 'dd'])
]

for xname, yname in features_to_plot:

    plt.figure()

    ax = sns.kdeplot(
        data=toplt,
        x=xname,
        y=yname,
        hue='cond1',
        cut=0,
        pthresh=0.05,
        common_norm=False,
        levels=4,
    )

    ax.set(
        xlim=(
            np.percentile(toplt[xname], 0.2),
            np.percentile(toplt[xname], 99.8)
        ),
        ylim=(
            np.percentile(toplt[yname], 0.2),
            np.percentile(toplt[yname], 99.8)
        )
    )

    plt.savefig(
        os.path.join(fig_dir, f"{yname} v {xname} kde.pdf"),
        format='PDF'
    )

# %%
# ============================================================================
# Example 4: Custom feature engineering
# ============================================================================
# Create new variables and visualize them directly.

toplt = all_feature_cond.assign(
    xdispl_swim_abs=lambda df: df['xdispl_swim'].abs(),
    xdispl_chg_abs=lambda df: df['x_chg'].abs(),
    depth_chg_abs=lambda df: df['depth_chg'].abs(),
)

toplt = toplt.loc[toplt['cond1'].isin(['ld', 'dd'])]

xname = 'xdispl_chg_abs'
yname = 'depth_chg_abs'

g = sns.displot(
    data=toplt,
    x=xname,
    y=yname,
    col='cond1',
    hue='cond1',
    height=3,
)

g.set(
    xlim=(
        np.percentile(toplt[xname], 0),
        np.percentile(toplt[xname], 99.8)
    ),
    ylim=(
        np.percentile(toplt[yname], 0.2),
        np.percentile(toplt[yname], 99.8)
    )
)

plt.savefig(
    os.path.join(fig_dir, f"{yname} v {xname} dis.pdf"),
    format='PDF'
)

# KDE version of the same plot

g = sns.displot(
    data=toplt,
    kind='kde',
    x=xname,
    y=yname,
    hue='cond1',
    height=3,
    cut=0,
    pthresh=0.05,
    common_norm=False,
    levels=4,
)

g.set(
    xlim=(
        np.percentile(toplt[xname], 0),
        np.percentile(toplt[xname], 99.8)
    ),
    ylim=(
        np.percentile(toplt[yname], 0.2),
        np.percentile(toplt[yname], 99.8)
    )
)

plt.savefig(
    os.path.join(fig_dir, f"{yname} v {xname} kde.pdf"),
    format='PDF'
)