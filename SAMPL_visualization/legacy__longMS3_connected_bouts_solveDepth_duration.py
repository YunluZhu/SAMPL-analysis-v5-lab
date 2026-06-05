"""
For multiple comparisons across conditions and day night

"""

# %%
# import sys
import os
import random
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import get_data_dir, get_figure_dir
from plot_functions.get_bout_features import get_connected_bouts
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_tools import (
    set_font_type,
    defaultPlotting,
    distribution_binned_average,
    distribution_binned_average_opt,
)
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.get_bout_consecutive_features import (
    cal_autocorrelation_feature,
    extract_consecutive_bout_features,
)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scipy.stats as st
import statsmodels.api as sm
import statsmodels.robust.norms as norms
from sklearn.metrics import r2_score
from scipy.stats import theilslopes
from tqdm import tqdm

# %%

##### Parameters to change #####
pick_data = "wt_light_long"  # name of your dataset to plot as defined in function get_data_dir()
which_ztime = "night"  # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split("/")[-1].replace(".py", "")
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f"fig folder created: {folder_name}")
except:
    print("Notes: re-writing old figures")

set_font_type()
mpl.rc("figure", max_open_warning=0)

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(
    root,
    FRAME_RATE,
    ztime=which_ztime,
    if_strict_DayNightSplit=True,
)

# %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# tidy bout uid
all_features = all_feature_cond.assign(
    epoch_uid=all_feature_cond["cond0"]
    + all_feature_cond["cond1"]
    + all_feature_cond["expNum"].astype(str)
    + all_feature_cond["epoch_uid"],
    exp_uid=all_feature_cond["cond0"]
    + all_feature_cond["cond1"]
    + all_feature_cond["expNum"].astype(str),
)

# select dataset
all_features = all_features.query('cond1 == "ld"')


# %%
df_toplt = all_features.copy()
# remove extreme values
df_toplt = df_toplt[df_toplt["pre_IBI_time"] < 50]
df_toplt = df_toplt[df_toplt["traj_peak"].abs() < 90]
# check bout pre IBI duration and swim direction
g = sns.displot(
    data=df_toplt,
    x="pre_IBI_time",
    y="traj_peak",
    col="cond0",
    kind="hist",
    height=3,
    stat="probability",
    aspect=1,
    palette=my_palette,
    common_norm=False,
    col_order=all_cond0,
    bins="scott",
    cmap=sns.color_palette("ch:s=2.9,rot=-.1,light=1", as_cmap=True),
    cbar=True,
    vmax=0.01,
    binrange=[[0, 10], [-20, 60]],
)

# Rasterize only the histogram meshes
for ax in g.axes.flatten():
    for coll in ax.collections:
        coll.set_rasterized(True)

plt.savefig(
    os.path.join(fig_dir, "preIBI_time_vs_trajPeak_hist2d.pdf"),
    format="pdf",
    dpi=300,
)
#%%

df_toplt = all_features.copy()
df_toplt['IBI_cat'] = pd.cut(
    df_toplt['pre_IBI_time'],
    bins=[0, 1, 20],
    labels=['short', 'long'],
)

g = sns.relplot(
    kind='scatter',
    hue='IBI_cat',
    data=df_toplt.groupby(
        ['cond0'], observed=True
    ).sample(n=4000, replace=False).sort_values(by='IBI_cat', ascending=False),
    x='pitch_initial',
    y='traj_peak',
    palette=sns.color_palette(['#55aa87','#AAAAAA']),
    hue_order=['short', 'long'],
    col='cond0',
    height=3,
    # aspect=0.8,
    alpha=0.05,
    linewidth=0,
    col_order=all_cond0,
)
g.set(
    xlim=(-30, 45),
    ylim=(-30, 65),
    xticks=[-30, 0, 30],
    yticks=[-30, 0, 30,60],
)
for ax in g.axes.flatten():
    ax.set_aspect("equal", adjustable="box")
    for coll in ax.collections:
        coll.set_rasterized(True)

plt.savefig(os.path.join(fig_dir, f'pitchInitial_vs_trajPeak_byPreIBICat_scatter.pdf'), format='pdf',dpi=300)

g = sns.relplot(
    kind='scatter',
    hue='IBI_cat',
    data=df_toplt.groupby(
        ['cond0'], observed=True
    ).sample(n=4000, replace=False).sort_values(by='IBI_cat', ascending=False),
    x='pre_IBI_time',
    y='traj_peak',
    palette=sns.color_palette(['#55aa87','#AAAAAA']),
    hue_order=['short', 'long'],
    col='cond0',
    height=3,
    aspect=0.9,
    alpha=0.05,
    linewidth=0,
    col_order=all_cond0,
)
g.set(
    xlim=(0, 10),
    ylim=(-30, 65),
    xticks=[0, 5, 10],
    yticks=[-30, 0, 30,60],
)
for ax in g.axes.flatten():
    for coll in ax.collections:
        coll.set_rasterized(True)
plt.savefig(os.path.join(fig_dir, f'preIBI_vs_trajPeak_byPreIBICat_scatter.pdf'), format='pdf',dpi=300)


# #%%
# g = sns.displot(
#     data=df_toplt,
#     x="pre_IBI_time",
#     y="traj_peak",
#     col="cond0",
#     kind="hist",
#     height=3,
#     stat="probability",
#     aspect=1,
#     palette=my_palette,
#     common_norm=False,
#     col_order=all_cond0,
#     bins="scott",
#     cmap=sns.color_palette("ch:s=2.9,rot=-.1,light=1", as_cmap=True),
#     cbar=True,
#     vmax=0.01,
#     binrange=[[0, 10], [-20, 60]],
# )

# # Rasterize only the histogram meshes
# for ax in g.axes.flatten():
#     for coll in ax.collections:
#         coll.set_rasterized(True)


# %%
list_of_features = [
    "WHM",
    "pre_IBI_time",
    "post_IBI_time",
    # 'pitch_initial',
    'pitch_end',
    # 'rot_total',
    "y_initial",
    "y_end",
    # 'x_initial',
    # 'x_end',
    "depth_chg_fullBout",
    # 'atk_ang',
    # 'lift_distance',
    "traj_peak",
    "pitch_initial",
    # 'x_chg_fullBout',
    "bout_time",
    "spd_peak",
]

# %% associate consecutive bouts

# %%
# consecutive bouts vs depth change in total

max_lag = 4
# max_lag = 2
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(
    all_features, list_of_features, max_lag
)

# %%
sel_consecutive_bouts = consecutive_bout_features.sort_values(
    by=["cond1", "cond0", "id", "lag", "ztime"]
).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    bouts=sel_consecutive_bouts["lag"] + 1
)

# Compare current y_initial with next bout's y_initial
sel_consecutive_bouts["bout_direction"] = sel_consecutive_bouts.apply(
    lambda row: "climb" if row["y_initial"] < row["y_end"] else "dive", axis=1
)

selected_data = (
    sel_consecutive_bouts.groupby(["cond1", "cond0", "ztime", "expNum", "id"])
    .apply(
        lambda group: group.assign(
            preIBI_y_displ=group["y_initial"]
            - group["y_end"].shift(
                1
            ),  # preIBI_y_displ = y end from last bout - y initial from current bout
            postIBI_y_displ=group["y_initial"].shift(-1)
            - group[
                "y_end"
            ],  # postIBI_y_displ = y initial from next bout - y end from current bout
        ),
        include_groups=False,
    )
    .reset_index()  # Reset index after apply()
)

# %%
selected_data = selected_data.assign(
    ypos_afterBout_cumu=sel_consecutive_bouts["y_end"]
    - sel_consecutive_bouts["y_initial_first"],
    time_lapse = (
        sel_consecutive_bouts["bout_time"] - sel_consecutive_bouts["bout_time_first"]
    ).dt.total_seconds(),
    # cumu_IBI_duration=sel_consecutive_bouts.groupby(
    #     ["cond1", "cond0", "ztime", "expNum", "id"], as_index=False
    # )["pre_IBI_time"]
    # .cumsum()
    # .values,
    IBI_time_cumu=sel_consecutive_bouts.groupby(
        ["cond1", "cond0", "ztime", "expNum", "id"], as_index=False
    )["pre_IBI_time"]
    .apply(lambda x: x.fillna(0).cumsum()-x.fillna(0).iloc[0])
    .values,
)

# separate by traj_peak direction
selected_data = selected_data.assign(
    traj_cat=pd.cut(
        sel_consecutive_bouts["traj_peak_first"],
        bins=[-90, 0, 90],
        labels=["dive", "climb"],
    ),
    traj_cat_steep=pd.cut(
        sel_consecutive_bouts["traj_peak_first"],
        bins=[-90, 0, 20, 90],
        labels=["dive", "flat", "steep"],
    ),
    speed_cat=pd.cut(
        sel_consecutive_bouts["spd_peak"],
        bins=[0, sel_consecutive_bouts["spd_peak"].median(), np.inf],
        labels=["slow", "fast"],
    ),
    IBI_cat=pd.cut(
        sel_consecutive_bouts["post_IBI_time_first"],
        bins=[0, 1, np.inf],
        labels=["short", "long"],
    ),
)
# calculate average by expNum
bout_series_avg = (
    selected_data.groupby(
        ["cond1", "cond0", "ztime", "expNum", "lag",'traj_cat'],
        as_index=True,
        observed=True,
    )
    .agg(
        avg_ypos_cumu=("ypos_afterBout_cumu", "median"),
        avg_time_lapse=("time_lapse", "median"),
        avg_IBI_time_cumu=("IBI_time_cumu", "median"),
        avg_pre_IBI_time=("pre_IBI_time", "median"),
        # avg_traj_peak = ('traj_peak_first', 'mean'),
    )
    .reset_index()
)
# #%%
# df_toplt = selected_data.query("lag > 0").copy()
# df_toplt = df_toplt.assign(
#     IBI_bins=pd.cut(
#         df_toplt["time_lapse"], bins=np.arange(0, 25, 2), labels=np.arange(0, 24, 2)
#     ),   
#     traj_cat_bins = pd.cut(
#         df_toplt["traj_peak_first"],
#         bins=np.arange(-30, 70, 10),
#         labels=np.arange(-30, 70, 10)[:-1],
#     ),
# )
# heatmap_df = (
#     df_toplt
#     .groupby(
#         ["cond0", "IBI_bins", "traj_cat_bins"],
#         observed=True
#     )
#     .agg(
#         depth_change=("ypos_afterBout_cumu", "median"),
#         n=("ypos_afterBout_cumu", "size"),
#         nexp=("expNum", "nunique"),
#     )
#     .reset_index()
# )
# MIN_COUNT = 10

# heatmap_df = heatmap_df[heatmap_df["n"] >= MIN_COUNT]
# heatmap_df = heatmap_df[heatmap_df["nexp"] >= 4]

# for cond_val, sub in heatmap_df.groupby("cond0", sort=False):

#     Z = (
#         sub
#         .pivot(
#             index="traj_cat_bins",
#             columns="IBI_bins",
#             values="depth_change",
#         )
#         .sort_index(ascending=True)
#     )

#     fig, ax = plt.subplots(figsize=(6, 4))

#     sns.heatmap(
#         Z,
#         ax=ax,
#         cmap="Blues",
#         vmax=2.5,
#         vmin=0,
#         # cbar_kws=dict(label="Median time lapse (s)"),
#         rasterized=True,
#     )

#     ax.set_title(f"cond0 = {cond_val}")
#     # ax.set_xlabel("Lag")
#     # ax.set_ylabel("Trajectory peak bin (deg)")

#     # make y increase upward in data coordinates
#     ax.invert_yaxis()

#     fig.tight_layout()

#     fig.savefig(
#         os.path.join(fig_dir, f"heatmap_timeLapse_lag_traj_{cond_val}.pdf"),
#         dpi=300,
#     )
    
#%%
time_bins = np.arange(0, 29, 4)

IBI_bin_levels = pd.Index(
    time_bins,
    name="time_bins",
)


df_toplt = selected_data.query("lag > 0").copy()
df_toplt = df_toplt.assign(
    time_bins=pd.cut(
        df_toplt["time_lapse"], bins=time_bins, labels=time_bins[:-1]
    ),   
    traj_cat_bins = pd.cut(
        df_toplt["traj_peak_first"],
        bins=np.arange(-30, 75, 15),
        labels=np.arange(-30, 75, 15)[:-1],
    ),
)

per_exp_heatmap = (
    df_toplt
    .groupby(
        ["cond0", "expNum", "time_bins", "traj_cat_bins"],
        observed=True,
    )
    .agg(
        depth_change=("ypos_afterBout_cumu", "median"),
        n=("ypos_afterBout_cumu", "size"),
    )
    .reset_index()
)
MIN_COUNT_PER_EXP = 5
per_exp_heatmap = per_exp_heatmap.query("n >= @MIN_COUNT_PER_EXP")
heatmap_df = (
    per_exp_heatmap
    .groupby(
        ["cond0", "time_bins", "traj_cat_bins"],
        observed=True,
    )
    .agg(
        mean_depth_change=("depth_change", "mean"),   
        n_exp=("expNum", "nunique"),
    )
    .reset_index()
)
MIN_EXP = 3
heatmap_df = heatmap_df.query("n_exp >= @MIN_EXP")

traj_bin_levels = pd.Index(
    heatmap_df["traj_cat_bins"]
    .cat.categories,
    name="traj_cat_bins",
)

for cond_val, sub in heatmap_df.groupby("cond0", sort=False):

    Z = (
        sub
        .pivot(
            index="traj_cat_bins",
            columns="time_bins",
            values="mean_depth_change",
        )
        .reindex(
            index=traj_bin_levels,
            columns=IBI_bin_levels,
        )
    )

    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        Z,
        ax=ax,
        cmap=sns.diverging_palette(312, 230, s=60, l=75,center='dark', as_cmap=True),
        vmin=-3.5,
        vmax=3.5,
        mask=Z.isna(),
        rasterized=True,
    )
    ax.invert_yaxis()

    ax.set_xticks(np.arange(len(IBI_bin_levels)) + 0.5)
    ax.set_xticklabels(IBI_bin_levels.astype(int))

    ax.set_yticks(np.arange(len(traj_bin_levels)) + 0.5)
    # ax.set_yticklabels(list(traj_bin_levels.astype(int))[::-1])

    # invert y once, explicitly
    # ax.set_ylim(len(traj_bin_levels), 0)

    ax.set_xlim(0, len(IBI_bin_levels))
    plt.savefig(
        os.path.join(fig_dir, f"heatmap_depthChange_timeLapse_traj_{cond_val}.pdf"),
        dpi=300,
    )


#%%
# x_name = 'cond0'
# print(f"\n--- ANOVA for {param} ---")
# # 1. One-way ANOVA
# for bouts_num in sorted(df_toplt['lag'].unique()):
#     print(f"\nBouts: {bouts_num}")
#     bout_series_sub = df_toplt.query('lag == @bouts_num')
#     model = ols(f"{param} ~ C(cond0)", data=bout_series_sub).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(anova_table)

#     # 2. Tukey’s HSD for post hoc comparison
#     tukey = pairwise_tukeyhsd(
#         endog=bout_series_sub[param],
#         groups=bout_series_sub["cond0"],
#         alpha=0.05
#     )
#     print("\nTukey HSD:")
#     print(tukey.summary())

#%%
climb_df = bout_series_avg.query('traj_cat == "climb"')

g = sns.relplot(
    kind='line',
    data=climb_df,
    x='lag',
    hue='cond0',
    y='avg_ypos_cumu',
    palette=my_palette,
    errorbar='se',
    estimator='mean',
    height=3,
    aspect=1
)
g.set(xticks=range(1,5))
plt.savefig(os.path.join(fig_dir, f'climb_boutSeries_avgYposCumu_byCond0.pdf'), bbox_inches='tight')
#%%
# anova stats
param = 'avg_ypos_cumu'
x_name = 'cond0'
print(f"\n--- ANOVA for {param} ---")
# 1. One-way ANOVA
for bouts_num in sorted(climb_df['lag'].unique()):
    print(f"\nBouts: {bouts_num}")
    bout_series_sub = climb_df.query('lag == @bouts_num')
    model = ols(f"{param} ~ C(cond0)", data=bout_series_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=bout_series_sub[param],
        groups=bout_series_sub["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())

