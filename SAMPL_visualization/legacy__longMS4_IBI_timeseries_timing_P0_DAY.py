# this works. large variation in results across experiments
# terminal velocity are different
# why don't we just calculate terminal speed

# with bin 0-3 s and speed threshold 0.5, we get same results by fitting on median vs raw data

#%%
# import sys
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import math
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from tqdm import tqdm
from plot_functions.get_IBIangles import get_timeseriesIBIangles
import gc
from scipy.optimize import curve_fit
from tqdm import tqdm
# from scipy.ndimage import uniform_filter1d
from plot_functions.plt_functions import plt_categorical_combined_3
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from plot_functions.plt_tools import (
    set_font_type,
    defaultPlotting,
    distribution_binned_average,
    distribution_binned_average_opt,
)
from scipy.ndimage import gaussian_filter1d

#%%
##### Parameters to change #####

pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' or 'night'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]

# Create a Seaborn palette
my_palette = sns.color_palette(my_colors)
# CONSTANTS %%%%%%
BIN_WIDTH = 0.05  # (s)
AVERAGE_BIN = np.arange(0, 0.5+BIN_WIDTH, BIN_WIDTH)

##### Parameters to change #####
def set_pretty_ticks(ax, x_mids, y_mids, x_step=5, y_step=2):
    # X ticks
    xticks = np.arange(0, len(x_mids), x_step)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(np.round(x_mids[xticks], 2), rotation=0)

    # Y ticks
    yticks = np.arange(0, len(y_mids), y_step)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(np.round(y_mids[yticks], 1))

def nice_floor(x, step):
    return step * math.floor(x / step)

def nice_ceil(x, step):
    return step * math.ceil(x / step)

def fd_width(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2 * iqr / (len(x) ** (1/3))

def snap_step(fd, allowed):
    return min(allowed, key=lambda x: abs(x - fd))

#%%
# Paste root directory here
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
#

#%% model posture explicitly for P0
tv_df_all = pd.read_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')

tv_df_short = tv_df_all.query('ztime == "day"').copy()

fd_time = fd_width(tv_df_short["start"])
fd_ang  = fd_width(tv_df_short["ang"])

print(f'fd_time: {fd_time}, fd_ang: {fd_ang}')


TIME_STEP = snap_step(10 * fd_time, [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
POSTURE_STEP = snap_step(120 * fd_ang, [0.5, 1, 1.5, 2, 3, 5])

print(f'TIME_STEP: {TIME_STEP}, POSTURE_STEP: {POSTURE_STEP}')

# -----------------------------
# Infer ranges from data
# -----------------------------
t_min_data = tv_df_short["start"].min()
t_max_data = tv_df_short["start"].max()

a_min_data = np.percentile(tv_df_short["ang"], .5)
a_max_data = np.percentile(tv_df_short["ang"], 99.5)

# -----------------------------
# Snap to nice edges
# -----------------------------
t_min = nice_floor(t_min_data, TIME_STEP)
t_max = nice_ceil(t_max_data, TIME_STEP)

a_min = nice_floor(a_min_data, POSTURE_STEP)
a_max = nice_ceil(a_max_data, POSTURE_STEP)

# -----------------------------
# Define edges
# -----------------------------
t_edges = np.arange(t_min, t_max + TIME_STEP, TIME_STEP)
a_edges = np.arange(a_min, a_max + POSTURE_STEP, POSTURE_STEP)

# Midpoints (for plotting only)
t_bin_mid = (t_edges[:-1] + t_edges[1:]) / 2
a_bin_mid = (a_edges[:-1] + a_edges[1:]) / 2

tv_df_short['ang0'] = tv_df_short.groupby('unique_IBI_idx')['ang'].transform('first')
tv_df_short['ang0_bin'] = pd.cut(tv_df_short["ang0"], bins=a_edges, include_lowest=True, labels=a_bin_mid)

tv_df_short['t_bin'] = pd.cut(tv_df_short['start'], t_edges)

#%
# 1. Compute risk and events per bin
hazard_df = (
    tv_df_short
    .groupby(['cond0', 'ang0_bin', 't_bin'])
    .agg(
        n_risk=('unique_IBI_idx', 'nunique'),
        n_event=('event', 'sum')
    )
    .reset_index()
)

#%
MIN_IBI_PER_BIN = 30  # removes noisy estimates / spikes
MIN_IBI_PER_POSTURE = 200 # removes postures with too few total IBIs

# Histogram of n_risk per (ang0_bin, t_bin)
plt.figure(figsize=(6, 4))
sns.histplot(hazard_df['n_risk'], bins=50, log_scale=(False, False))
plt.xlabel("Number of IBIs per bin")
plt.ylabel("Count of (ang0_bin, t_bin) bins")
plt.title("Distribution of IBIs per bin")
plt.axvline(MIN_IBI_PER_BIN, color='red', linestyle='--', label='Current MIN_IBI_PER_BIN')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hist_nIBIs_per_bin.png'))


posture_support = hazard_df.groupby(['cond0','ang0_bin'])['n_risk'].sum().reset_index(name='n_IBIs')

plt.figure(figsize=(6, 4))
sns.histplot(posture_support['n_IBIs'], bins=30, log_scale=(True, False))
plt.xlabel("Total IBIs per posture (summed over time)")
plt.ylabel("Number of postures")
plt.title("Distribution of IBIs per posture")
plt.axvline(MIN_IBI_PER_POSTURE, color='red', linestyle='--', label='Current MIN_IBI_PER_POSTURE')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'hist_nIBIs_per_posture.png'))

#
thresholds = range(50, 5000, 200)
n_postures_kept = []

for th in thresholds:
    n_postures_kept.append((posture_support['n_IBIs'] >= th).sum())

plt.figure(figsize=(4, 3))
plt.plot(thresholds, n_postures_kept, marker='o')
plt.xlabel("MIN_IBI_PER_POSTURE")
plt.ylabel("Number of postures retained")
plt.title("Effect of MIN_IBI_PER_POSTURE on posture coverage")
plt.grid(True)
plt.savefig(os.path.join(fig_dir, 'posture_coverage_vs_threshold.png'))


#%
# 2a. Filter out sparse time bins (per-bin filter)
hazard_df = hazard_df[hazard_df['n_risk'] >= MIN_IBI_PER_BIN].copy()

# 2b. Filter out poorly supported postures (per-posture filter)
posture_support = hazard_df.groupby(['ang0_bin'])['n_risk'].sum().reset_index(name='n_IBIs')
valid_postures = posture_support[posture_support['n_IBIs'] >= MIN_IBI_PER_POSTURE]
hazard_df = hazard_df.merge(
    valid_postures['ang0_bin'],
    on=['ang0_bin'],
    how='inner'
)

# 3. Compute hazard and mid-time for plotting
hazard_df['hazard'] = hazard_df['n_event'] / hazard_df['n_risk']
hazard_df['t_mid'] = hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))

# 4. Smooth along time
SIGMA_T = 2.0  # in bins
hazard_df['hazard_smooth'] = (
    hazard_df
    .groupby(['ang0_bin'])['hazard']
    .transform(lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest'))
)

valid_bins = hazard_df.groupby('ang0_bin')['hazard_smooth'].count() > 0
valid_ang_bins = valid_bins[valid_bins].index
hazard_df_filtered = hazard_df[hazard_df['ang0_bin'].isin(valid_ang_bins)]

# 6. Sort for plotting
hazard_df_filtered = hazard_df_filtered.sort_values(['cond0','ang0_bin','t_mid'])

# 7. Plot with consistent color palette
unique_ang_bins = sorted(hazard_df_filtered['ang0_bin'].unique())


norm = mcolors.TwoSlopeNorm(vmin=hazard_df_filtered['ang0_bin'].min(),
                            vcenter=10,
                            vmax=hazard_df_filtered['ang0_bin'].max())

palette = cm.get_cmap("coolwarm_r")

sns.relplot(
    data=hazard_df_filtered,
    x='t_mid',
    y='hazard_smooth',
    hue='ang0_bin',
    hue_order=unique_ang_bins,  # ensures same colors across facets
    col='cond0',
    kind='line',
    height=3,
    alpha=0.8,
    estimator=None,
    palette=[palette(norm(val)) for val in sorted(hazard_df_filtered['ang0_bin'].unique())]
)
plt.savefig(os.path.join(fig_dir, 'hazard_timeseries_by_posture.png'))

#%%
# --- Prepare pivot table for heatmap ---
# Rows = posture quantile bins, Columns = mid-time bins
heatmaps_smooth = {}
for cond, df_c in hazard_df_filtered.groupby('cond0'):
    # pivot: rows = ang0_bin, columns = t_mid
    Z_smooth_df = df_c.pivot_table(
        index='ang0_bin', 
        columns='t_mid', 
        values='hazard_smooth', 
        fill_value=0
    )
    heatmaps_smooth[cond] = Z_smooth_df

# --- Plot polished heatmaps ---
vmax = 0.3#hazard_df_filtered['hazard_smooth'].max()  # max for color scaling
for cond, Z_smooth_df in heatmaps_smooth.items():
    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        Z_smooth_df.values,
        ax=ax,
        cmap=sns.color_palette("ch:s=2.9,rot=-.1,light=1", as_cmap=True),
        vmin=0.,
        vmax=vmax,
        cbar_kws={"label": "Event probability"},
        rasterized=True
    )

    # Invert Y so smallest quantile at bottom
    ax.invert_yaxis()

    # Set axes limits to match matrix dimensions
    ax.set_xlim(0, int(10/TIME_STEP + 1))
    ax.set_ylim(0, Z_smooth_df.shape[0])
    ax.grid(False)

    # Optional: set pretty tick labels
    # Use t_mid_selected and a_mid_selected if you have them, otherwise derive from pivot
    t_mid_selected = Z_smooth_df.columns.values
    a_mid_selected = Z_smooth_df.index.values
    set_pretty_ticks(ax, x_mids=t_mid_selected, y_mids=a_mid_selected, x_step=10, y_step=2)

    # Labels
    ax.set_title(f"Smoothed hazard — cond {cond}")
    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Posture quantile")

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"heatmap_hazard_cond{cond}_smoothed.pdf"),
        format="pdf"
    )

#%% contours

from scipy.ndimage import gaussian_filter

sigma_t = 1.0   # time bins
sigma_a = 0.8   # posture bins
levels = np.linspace(0.05, vmax, 6)

for cond, Z_smooth_df in heatmaps_smooth.items():
    
    Z = Z_smooth_df.values
    Z_gauss = gaussian_filter(
        Z,
        sigma=(sigma_a, sigma_t),
        mode="nearest"
    )
    ny, nx = Z_gauss.shape
    x = np.arange(nx) + 0.5
    y = np.arange(ny) + 0.5


    fig, ax = plt.subplots(figsize=(4, 3))

    sns.heatmap(
        Z_smooth_df,
        ax=ax,
        cmap=sns.color_palette("ch:s=2.9,rot=-.1,light=1", as_cmap=True),
        vmin=0.,
        vmax=vmax,
        cbar_kws={"label": "Event probability"},
        rasterized=True
    )

    ax.contour(
        x,y,
        Z_gauss,
        levels=levels,
        colors="k",
        linewidths=0.6,
        extent=[0, Z_gauss.shape[1], Z_gauss.shape[0], 0]
    )

    # ax.invert_yaxis()
    ax.set_xlim(0, int(10/TIME_STEP + 1))
    ax.set_ylim(0, Z.shape[0])
    ax.grid(False)

    set_pretty_ticks(
        ax,
        x_mids=Z_smooth_df.columns.values,
        y_mids=Z_smooth_df.index.values,
        x_step=20,
        y_step=2
    )

    ax.set_title(f"Smoothed hazard — cond {cond}")
    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Posture quantile")

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"heatmap_hazard_cond{cond}_gauss_contour.pdf"),
        format="pdf"
    )
    
#%%
from scipy.stats import gaussian_kde


# avoids drawing trivial 0 and max contours

Z_all = []
Z_all_orig = {}
for cond, Z_smooth_df in heatmaps_smooth.items():

    # --- Build KDE ---
    t_vals = Z_smooth_df.columns.values
    a_vals = Z_smooth_df.index.values

    T, A = np.meshgrid(t_vals, a_vals)
    weights = Z_smooth_df.values

    mask = weights.ravel() > 0
    coords = np.vstack([
        T.ravel()[mask],
        A.ravel()[mask]
    ])
    w = weights.ravel()[mask]

    kde = gaussian_kde(
        coords,
        weights=w,
        bw_method=0.25
    )

    # --- Evaluate KDE on common grid ---
    t_grid = np.linspace(0, 10, 200)
    a_grid = np.linspace(-20, 35, 150)

    TT, AA = np.meshgrid(t_grid, a_grid)
    Z_kde = kde(
        np.vstack([TT.ravel(), AA.ravel()])
    ).reshape(TT.shape)

    # compute Z_kde as before
    Z_all.append(Z_kde.ravel())
    Z_all_orig[cond] = Z_kde

Z_all = np.concatenate(Z_all)

KDE_VMIN = np.quantile(Z_all, 0.5)
KDE_VMAX = np.quantile(Z_all, 1)

N_CONTOURS = 4
KDE_LEVELS = np.linspace(
    KDE_VMIN,
    KDE_VMAX * 0.98,
    N_CONTOURS
)
for cond, Z_kde in Z_all_orig.items():
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(5, 3))

    im = ax.imshow(
        Z_kde,
        origin="lower",
        aspect="auto",
        extent=[
            t_grid.min(), t_grid.max(),
            a_grid.min(), a_grid.max()
        ],
        cmap=sns.color_palette("ch:s=2.9,rot=-.1,light=1", as_cmap=True),
        vmin=KDE_VMIN,
        vmax=KDE_VMAX,
        rasterized=True
    )

    ax.contour(
        TT, AA, Z_kde,
        levels=KDE_LEVELS,
        colors="k",
        linewidths=0.6
    )

    # Axes
    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Posture quantile")
    ax.set_title(f"KDE hazard surface — cond {cond}")

    # Shared colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("KDE-weighted hazard density")

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"heatmap_hazard_cond{cond}_KDE_fixedrange.pdf"),
        format="pdf"
    )

#%%
for i, (cond, Z_kde) in enumerate(Z_all_orig.items()):

    fig, ax = plt.subplots(figsize=(5, 3))

    N_BANDS = len(KDE_LEVELS) - 1

    ax.contour(
        TT, AA, Z_kde,
        levels=KDE_LEVELS,
        colors=[my_palette[i]] * N_BANDS,
        alpha=1
    )

    # # Optional: label contours with actual values
    # ax.clabel(
    #     cs,
    #     fmt="%.4f",
    #     fontsize=7,
    #     inline=True,
    # )

    # Axes
    ax.set_xlim(0, 10)
    ax.set_ylim(-20, 35)
    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Posture quantile")
    ax.set_title(f"KDE hazard contours — cond {cond}")

    ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"contours_hazard_cond{cond}_KDE.pdf"),
        format="pdf"
    )
# %%
