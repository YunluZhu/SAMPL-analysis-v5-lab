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

#%% relatively fast
pickle_path = f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_angles_{which_ztime}.pkl'

try:
    # Read from pickle if it exists
    if os.path.exists(pickle_path):
        IBI_angles = pd.read_pickle(pickle_path)
        cond0_all = IBI_angles['cond0'].unique()
        cond1_all = IBI_angles['cond1'].unique()
        print('Loaded IBI angles from pickle')
        group_cols = ["unique_IBI_idx", "cond1", "cond0", "expNum"]
    else:
        raise FileNotFoundError

except Exception as e:
    print(f"Loading failed ({e}), regenerating data...")
    IBI_angles, cond0_all, cond1_all = get_timeseriesIBIangles(root, FRAME_RATE, ztime=which_ztime)

    # Convert only required columns to appropriate dtypes
    IBI_angles["cond1"] = IBI_angles["cond1"].astype("category")
    IBI_angles["ztime"] = IBI_angles["ztime"].astype("category")
 
    # IBI_angles["absTime"] = pd.to_datetime(IBI_angles["absTime"]) # check if this is necessary

    IBI_angles["unique_IBI_idx"] = (
        IBI_angles["expNum"].astype("int64") * 10**12 +
        IBI_angles["boxNum"].astype("int64") * 10**8 +
        IBI_angles["epochNum"].astype("int64") * 10**4 +
        IBI_angles["IEI_matchIndex"].astype("int64")
    )

    group_cols = ["unique_IBI_idx", "cond1", "cond0", "expNum"]
    group_min = IBI_angles.groupby(group_cols, observed=True)["absTime"].transform("min")
    IBI_angles["time_relative_s"] = (IBI_angles["absTime"] - group_min).dt.total_seconds()
    
    print("> checkpoint after preprocessing")

    IBI_angles.to_pickle(pickle_path)  # smaller file size
    print('> Saved IBI angles to pickle')

#%% 20-40 s
# --- Step 2: filter ---
mask = (
    (IBI_angles["cond1"].isin(["ld"])) 
)
cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "x" ,"y", 'headx', 'heady','swimSpeed']
df_time_filtered = IBI_angles.loc[mask, cols_needed]
# Compute group max (IBI duration) and filter in-place

print("> checkpoint after early filtering")

#%% release memory
del IBI_angles
gc.collect()

#%%
#%% truncate to start from speed threshold
df_time_filtered['swimSpeed_smoothed'] = df_time_filtered.groupby('unique_IBI_idx', observed=True)['swimSpeed'].transform(
    lambda x: savgol_filter(x.values, 11, 3) if len(x) >= 11 else x.values
)

#%%
SPEED_THRESHOLD = 0.5
df_temp = df_time_filtered.copy() # Work on a copy to ensure df_time_filtered remains clean

# 1. Calculate Start Times
mask = df_temp['swimSpeed_smoothed'] <= SPEED_THRESHOLD

# Find the MINIMUM 'time_relative_s' where the mask is True for each IBI.
start_times = df_temp[mask].groupby('unique_IBI_idx')['time_relative_s'].min()

# 2. Add Start Times to the temporary DataFrame
# Use a simple assignment after turning the Series into a dictionary/mapping
df_temp['start_time'] = df_temp['unique_IBI_idx'].map(start_times)

# 3. Truncate and Re-index
# Filter rows where current time is greater than or equal to the start time
df_truncated = df_temp[
    df_temp['time_relative_s'] >= df_temp['start_time']
].copy()

# Calculate the new relative time
df_truncated['time_relative_s'] = (
    df_truncated['time_relative_s'] - df_truncated['start_time']
)

# 4. Clean Up and Update (Optional)
# Drop the temporary 'start_time' column from the result
df_truncated = df_truncated.drop(columns=['start_time'])

#%%
SPEED_THRESHOLD = 2
df_temp = df_truncated.copy()

# 1. Calculate END Times (last moment speed <= threshold)
mask = df_temp['swimSpeed_smoothed'] <= SPEED_THRESHOLD

# Find the MAXIMUM 'time_relative_s' where the mask is True for each IBI.
end_times = df_temp[mask].groupby('unique_IBI_idx')['time_relative_s'].max()

# 2. Add End Times to the temporary DataFrame
df_temp['end_time'] = df_temp['unique_IBI_idx'].map(end_times)

# 3. Truncate (keep everything BEFORE end time)
df_truncated2 = df_temp[
    df_temp['time_relative_s'] <= df_temp['end_time']
].copy()

# 4. Clean up
df_truncated2 = df_truncated2.drop(columns=['end_time'])

#%% further cleaning: remove IBIs with high velocity after 1s, likely representing translocation
df_valid = df_truncated2.query("cond1 == 'ld'")

# after_1s_mask = df_valid["time_relative_s"] > 1

# df_valid["yvel_sg"] = savgol_filter(df_valid["yvel"].values, 11, 3)
# df_valid["angvel_sg"] = savgol_filter(df_valid["angVelSmoothed"].values, 11, 3)
# df_valid["speed_sg"] = savgol_filter(df_valid["swimSpeed"].values, 11, 3)

# after = df_valid.loc[after_1s_mask]

# #%
# stats = after.groupby("unique_IBI_idx", observed=True).agg(
#     p50_yvel=("yvel_sg", lambda x: np.percentile(x, 50)),
#     p75_angVelSmoothed=("angvel_sg", lambda x: np.percentile(x, 75)),
#     p75_swimSpeed=("speed_sg", lambda x: np.percentile(x, 75)),
#     mad_swimSpeed=("speed_sg", lambda x: np.median(np.abs(x - np.median(x)))),
#     mad_yvel=("yvel_sg", lambda x: np.median(np.abs(x - np.median(x)))),
#     mad_angVelSmoothed=("angvel_sg", lambda x: np.median(np.abs(x - np.median(x))))
# )

# #%
# valid_IBI_idx = stats.index[
#     (stats.p50_yvel < 0.01) &
#     (stats.p75_angVelSmoothed < 0) &
#     (stats.p75_swimSpeed < 0.2) &
#     (stats.mad_swimSpeed < 0.04) &
#     (stats.mad_angVelSmoothed < 1)
# ]  
# # valid_IBI_idx = df_valid.unique_IBI_idx.unique()  # keep all for now
# #%%
before_08s_mask = df_valid["time_relative_s"] < 0.8
before = df_valid.loc[before_08s_mask]
stats2 = before.groupby("unique_IBI_idx", observed=True).agg(
    ang_accel=("angVelSmoothed", lambda x: x.diff().median()),
)
valid_IBI_idx2 = stats2.index[stats2.ang_accel > -0.03]


#%% remove non-contiguous IBIs

# df_filtered = df_valid[df_valid.unique_IBI_idx.isin(np.intersect1d(valid_IBI_idx2, valid_IBI_idx))].copy()
df_passed_QC = df_valid[df_valid.unique_IBI_idx.isin(valid_IBI_idx2)].copy()

from collections import Counter

# Identify non-contiguous IBIs
consecutive_change = df_passed_QC['unique_IBI_idx'] != df_passed_QC['unique_IBI_idx'].shift(1)
ibi_order = df_passed_QC['unique_IBI_idx'][consecutive_change].tolist()

# Count occurrences
ibi_counts = Counter(ibi_order)
non_contiguous_ibi = [ibi for ibi, count in ibi_counts.items() if count > 1]

print(f"Removing {len(non_contiguous_ibi)} non-contiguous IBIs:", non_contiguous_ibi)

# Remove rows belonging to those IBIs
df_filtered = df_passed_QC.loc[~df_passed_QC['unique_IBI_idx'].isin(non_contiguous_ibi)].copy()

# Optional sanity check
print("Original rows:", len(df_passed_QC), "Cleaned rows:", len(df_filtered))
print("Unique IBIs remaining:", df_filtered['unique_IBI_idx'].nunique())

print("> checkpoint after further filtering; df_filtered ready")
# # %%
# # calcualte some attributes
# grouped = df_filtered.groupby('unique_IBI_idx', observed=True)
# IBI_features = grouped.agg(
#     IBI_time=('time_relative_s', 'max'),
#     IBI_ydispl=('y', lambda x: x.iloc[-1] - x.iloc[0]),
#     IBI_xdispl=('x', lambda x: np.abs(x.iloc[-1] - x.iloc[0])),
#     IBI_rot=('ang', lambda x: x.iloc[-1] - x.iloc[0]),
#     IBI_yvel_avg=('yvel', 'median'),
#     IBI_angvel_avg=('angVelSmoothed', 'median'),
#     cond0=('cond0', 'first'),
#     cond1=('cond1', 'first'),
#     expNum=('expNum', 'first'),
#     ztime=('ztime', 'first'),
#     IBI=('unique_IBI_idx', 'count'),
#     aftBout_pitch = ('ang', lambda x: x.iloc[0]),
#     bfrBout_pitch = ('ang', lambda x: x.iloc[-1]),
#     avg_pitch = ('ang', lambda x: x.median()),
# ).reset_index()

# IBI_features = IBI_features.assign(
#     frequency = lambda df: 1 / df['IBI_time']
# )
# features_toplt =  ['aftBout_pitch','IBI_rot','IBI_yvel_avg','IBI_angvel_avg']
# # median per expNum
# print(IBI_features.groupby(['cond0','expNum']).size())
# median_res = IBI_features.groupby(['cond0','expNum'])[features_toplt].median().reset_index()


# %%
df_filtered_ldNight = df_filtered.query("cond1=='ld' and ztime=='night'")
# drop unique_IBI_idx that have the first time_relative_s not equal to 0
df_filtered_ldNight = df_filtered_ldNight[df_filtered_ldNight.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

# Compute drift per unique_IBI_idx
drift = df_filtered_ldNight.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Filter rows where drift < 1
df_negative_drift = df_filtered_ldNight[drift < 1]

df_negative_drift = df_negative_drift.assign(
    delta_y = df_negative_drift.groupby('unique_IBI_idx', observed=True)['y'].transform(lambda x: x.diff().fillna(0)),
)
df_negative_drift = df_negative_drift.assign(
    y_displ = df_negative_drift.groupby('unique_IBI_idx', observed=True)['delta_y'].transform('cumsum'),
)
df_negative_drift = df_negative_drift.assign(
    y_displ_sm = df_negative_drift.groupby('unique_IBI_idx', observed=True)['y_displ'].transform(lambda x: savgol_filter(x.values, 5, 3) if len(x) >= 5 else x.values)
)
df_negative_drift['duration'] = df_negative_drift.groupby(['cond0','unique_IBI_idx'])['time_relative_s'].transform('max')

#%%
DURATION_THRESH = np.percentile(df_negative_drift['duration'], 90) # seconds; only include IBIs shorter than this

df = df_negative_drift.query("duration <= @DURATION_THRESH").copy()

# # drop rows with missing posture or time
# df = df.dropna(subset=['unique_IBI_idx', 'time_relative_s', 'ang'])

# # ensure ordering
# df = df.sort_values(['unique_IBI_idx', 'time_relative_s']).reset_index(drop=True)

rows = []
for ibi, g in df.groupby('unique_IBI_idx', sort=False):
    g = g.sort_values('time_relative_s')
    times = g['time_relative_s'].values
    y_displ = g['y_displ'].values # or smoothed version
    # group-level metadata (assume constant per IBI)
    meta = g[['expNum', 'cond0', 'cond1']].iloc[0].to_dict()

    if len(times) < 2:
        continue  # cannot make intervals

    for i in range(len(times)-1):
        start = times[i]
        stop = times[i+1]
        y_displ_val = y_displ[i]  # covariate value at interval start
        event = 1 if i == len(times)-2 else 0  # last interval marks event
        row = {
            'unique_IBI_idx': ibi,
            'start': start,
            'stop': stop,
            'event': event,
            'y_displ': y_displ_val,
            **meta
        }
        rows.append(row)

tv_df = pd.DataFrame(rows)
print("rows:", len(tv_df), "IBIs:", tv_df['unique_IBI_idx'].nunique())

# -----------------------------
# Standardize covariates safely
# -----------------------------

# Standardize y_displ
tv_df['y_displ_z'] = (tv_df['y_displ'] - tv_df['y_displ'].mean()) / tv_df['y_displ'].std(ddof=0)

# Add tiny epsilon to avoid log(0)
eps = 1e-3
tv_df['log_time'] = np.log(tv_df['start'] + eps)

# Clip log_time based on observed data to avoid extremes
clip_lo = np.floor(tv_df['log_time'].min())
clip_hi = np.ceil(tv_df['log_time'].max())
tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

# Compute interaction and standardize it
tv_df['y_displ_z_logt'] = tv_df['y_displ_z'] * tv_df['log_time_clipped']
tv_df['y_displ_z_logt'] = (tv_df['y_displ_z_logt'] - tv_df['y_displ_z_logt'].mean()) / tv_df['y_displ_z_logt'].std(ddof=0)
# -----------------------------

# ibi_duration = tv_df.groupby('unique_IBI_idx')['start'].agg(['min','max'])
# ibi_duration['duration'] = ibi_duration['max'] - ibi_duration['min']

# # Soft threshold: remove extremely long IBIs (e.g., above 99th percentile)
# duration_threshold = ibi_duration['duration'].quantile(0.99)
# long_ibi = ibi_duration[ibi_duration['duration'] > duration_threshold].index.tolist()
# print(f"Removing {len(long_ibi)} IBIs above 99th percentile ({duration_threshold:.2f}s)")

# tv_df_clean = tv_df.loc[~tv_df['unique_IBI_idx'].isin(long_ibi)].copy()

#%%
# %% quick visualization of sampled IBIs

rng = np.random.default_rng()
sample_ibi = rng.choice(
    tv_df['unique_IBI_idx'].unique(),
    size=min(50, tv_df['unique_IBI_idx'].nunique()),
    replace=False
)

fig, ax = plt.subplots(figsize=(5,4))

for ibi in sample_ibi:
    g = tv_df.loc[tv_df['unique_IBI_idx'] == ibi]
    ax.plot(g['start'], g['y_displ'], alpha=0.3)
    ax.scatter(
        g.loc[g['event'] == 1, 'start'],
        g.loc[g['event'] == 1, 'y_displ'],
        s=15
    )

ax.set(
    xlabel='time since IBI start (s)',
    ylabel='y_displ',
    title='Time-varying y_displ within IBIs'
)
plt.tight_layout()


#%%
# probability of bout event as function of posture and time into IBI
# instantaneous event probability
# all time points considered

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
# -----------------------------

SUPPORT_THRESH = 0.8 # minimum fraction of experiments required per bin
MIN_COUNT_PER_BIN = 200 # minimum number of data points required per bin

SIGMA_T = 3.0   # smoothing along time bins
SIGMA_A = 3.0   # smoothing along posture bins

# -----------------------------
# 0. Compute IBI duration per interval
# -----------------------------
# Make sure each row has IBI duration

# -----------------------------
# 1. Filter by duration threshold
# -----------------------------
tv_df_short = tv_df  # all IBIs already filtered by DURATION_THRESH

fd_time = fd_width(tv_df_short["start"])
fd_ydispl  = fd_width(tv_df_short["y_displ"])

print(f'fd_time: {fd_time}, fd_ydispl: {fd_ydispl}')


TIME_STEP = snap_step(5 * fd_time, [0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0])
y_displ_STEP = snap_step(10 * fd_ydispl, [0.01, 0.02, 0.03, 0.05, 0.1])

print(f'TIME_STEP: {TIME_STEP}, Y_DISPL_STEP: {y_displ_STEP}')

# -----------------------------
# Infer ranges from data
# -----------------------------
t_min_data = tv_df_short["start"].min()
t_max_data = tv_df_short["start"].max()

a_min_data = tv_df_short["y_displ"].min()
a_max_data = tv_df_short["y_displ"].max()

# -----------------------------
# Snap to nice edges
# -----------------------------
t_min = nice_floor(t_min_data, TIME_STEP)
t_max = nice_ceil(t_max_data, TIME_STEP)

a_min = nice_floor(a_min_data, y_displ_STEP)
a_max = nice_ceil(a_max_data, y_displ_STEP)

# -----------------------------
# Define edges
# -----------------------------
t_edges = np.arange(t_min, t_max + TIME_STEP, TIME_STEP)
a_edges = np.arange(a_min, a_max + y_displ_STEP, y_displ_STEP)

# Midpoints (for plotting only)
t_bin_mid = (t_edges[:-1] + t_edges[1:]) / 2
a_bin_mid = (a_edges[:-1] + a_edges[1:]) / 2

# -----------------------------
# 2. Bin the data
# -----------------------------
tv_df_short["t_bin"] = pd.cut(tv_df_short["start"], bins=t_edges, include_lowest=True, labels=t_bin_mid)
tv_df_short["a_bin"] = pd.cut(tv_df_short["y_displ"], bins=a_edges, include_lowest=True, labels=a_bin_mid)
# -----------------------------
# 3. Compute bin midpoints (for plotting axes)
# -----------------------------


# Optional: store as mapping
t_bin_to_mid = dict(zip(tv_df_short["t_bin"].cat.categories, t_bin_mid))
a_bin_to_mid = dict(zip(tv_df_short["a_bin"].cat.categories, a_bin_mid))

# Example: replace index/columns in heatmap with midpoints


# -----------------------------
# 3. Compute per-exp event probability
# -----------------------------

# -----------------------------
# Compute per-exp bin counts first
# -----------------------------
per_exp_counts = (
    tv_df_short
        .groupby(["cond0", "expNum", "a_bin", "t_bin"])
        .size()
        .reset_index(name="n_points")
)

# Keep only bins with sufficient data points
per_exp_counts_filt = per_exp_counts[per_exp_counts["n_points"] >= MIN_COUNT_PER_BIN]

# -----------------------------
# Compute per-exp event probability using only filtered bins
# -----------------------------
per_exp = (
    tv_df_short
        .merge(per_exp_counts_filt[["cond0","expNum","a_bin","t_bin"]], 
               on=["cond0","expNum","a_bin","t_bin"], how="inner")
        .groupby(["cond0", "expNum", "a_bin", "t_bin"])["event"]
        .mean()
        .reset_index()
)


# -----------------------------
# 4. Generate all combinations for support calculation
# -----------------------------
import itertools

cond0s = tv_df_short["cond0"].unique()
expNums = tv_df_short["expNum"].unique()
a_bins = tv_df_short["a_bin"].cat.categories
t_bins = tv_df_short["t_bin"].cat.categories

all_combinations = pd.DataFrame(
    list(itertools.product(cond0s, expNums, a_bins, t_bins)),
    columns=["cond0", "expNum", "a_bin", "t_bin"]
)

per_exp_full = all_combinations.merge(
    per_exp,
    on=["cond0", "expNum", "a_bin", "t_bin"],
    how="left"
)

# -----------------------------
# 5. Compute support fraction per bin
# -----------------------------
bin_support = (
    per_exp_full.groupby(["cond0", "a_bin", "t_bin"])["event"]
                 .apply(lambda x: x.notna().sum())
                 .reset_index(name="n_exp")
)

n_exp_per_cond = tv_df_short.groupby("cond0")["expNum"].nunique().reset_index(name="n_exp_total")
bin_support = bin_support.merge(n_exp_per_cond, on="cond0", how="left")
bin_support["support_frac"] = bin_support["n_exp"] / bin_support["n_exp_total"]

good_bins = bin_support.loc[bin_support["support_frac"] >= SUPPORT_THRESH, ["cond0", "a_bin", "t_bin"]]

# -----------------------------
# 6. Filter per-exp data to supported bins
# -----------------------------
per_exp_filt = per_exp_full.merge(
    good_bins,
    on=["cond0", "a_bin", "t_bin"],
    how="inner"
)

# -----------------------------
# 7. Aggregate across expNum
# -----------------------------
aggregated = (
    per_exp_filt
        .groupby(["cond0", "a_bin", "t_bin"])["event"]
        .mean()
        .reset_index()
)

# -----------------------------
# 8. Build heatmap matrices per cond0
# -----------------------------
heatmaps = {
    c: sub.pivot(index="a_bin", columns="t_bin", values="event")
    for c, sub in aggregated.groupby("cond0")
}

# -----------------------------
# 9. Plot heatmaps with fixed 0–0.2 range
# -----------------------------
# invert y axis

#%
# Global bin grids (same for all conditions)
full_a_bins = pd.Index(a_bin_mid, name="a_bin")
full_t_bins = pd.Index(t_bin_mid, name="t_bin")

# # Global posture support at earliest time bin
# first_t = full_t_bins[0]

# global_valid_y = (
#     pd.concat(
#         [
#             mat.reindex(index=full_a_bins, columns=full_t_bins)[first_t]
#             for mat in heatmaps.values()
#         ],
#         axis=1
#     )
#     .notna()
#     .any(axis=1)
# )

# # Shared posture bins and midpoints
# a_bins_shared = full_a_bins[global_valid_y]
# a_mid_shared = a_bin_mid[global_valid_y.values]


# for c, mat in heatmaps.items():

#     # Reindex and apply shared posture range
#     mat = (
#         mat
#         .reindex(index=full_a_bins, columns=full_t_bins)
#         .loc[a_bins_shared]
#     )

#     fig, ax = plt.subplots(figsize=(8, 6))

#     sns.heatmap(
#         mat,
#         ax=ax,
#         cmap="viridis",
#         vmin=0,
#         vmax=0.1,
#         cbar_kws={"label": "Event probability"},
#     )

#     ax.invert_yaxis()

#     # Consistent limits
#     ax.set_xlim(0, len(full_t_bins))
#     ax.set_ylim(0, len(a_bins_shared))

#     # Pretty ticks
#     set_pretty_ticks(
#         ax,
#         x_mids=t_bin_mid,
#         y_mids=a_mid_shared,
#         x_step=5,
#         y_step=2,
#     )

#     ax.set_title(f"Event probability heatmap (short IBIs ≤ {DURATION_THRESH}s) — cond {c}")
#     ax.set_xlabel("Time into IBI (s)")
#     ax.set_ylabel("Posture at interval start (deg)")

#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(fig_dir, f"heatmap_event_prob_cond{c}_shortIBI.pdf"),
#         format="pdf"
#     )

#%

from scipy.ndimage import gaussian_filter



def gaussian_smooth_nan_threshold(mat, sigma=(SIGMA_A, SIGMA_T), min_weight_fraction=0.5):
    """
    Gaussian smoothing that respects NaNs.
    NaNs are filled only if at least `min_weight_fraction` of a single valid neighbor contributed.
    """
    arr = mat.values.astype(float)
    nan_mask = np.isnan(arr)
    arr_filled = np.where(nan_mask, 0, arr)
    weight = (~nan_mask).astype(float)

    # Smooth data and weight
    arr_smooth = gaussian_filter(arr_filled, sigma=sigma)
    weight_smooth = gaussian_filter(weight, sigma=sigma)

    # Normalize
    arr_smooth_norm = arr_smooth / np.maximum(weight_smooth, 1e-12)

    # Only fill NaNs where enough weight contributed
    threshold = min_weight_fraction
    arr_smooth_norm[nan_mask & (weight_smooth < threshold)] = np.nan

    return pd.DataFrame(arr_smooth_norm, index=mat.index, columns=mat.columns)


# full grids
full_a_bins = pd.Index(a_bin_mid, name="a_bin")
full_t_bins = pd.Index(t_bin_mid, name="t_bin")

# Count valid x (time) bins per y (posture) across all conditions
valid_counts = pd.concat(
    [
        mat.reindex(index=full_a_bins, columns=full_t_bins).notna().sum(axis=1)
        for mat in heatmaps.values()
    ],
    axis=1
).sum(axis=1)  # sum across conditions

# Keep posture bins with at least 2 valid time bins across all conditions
global_valid_y = valid_counts >= 2

# Shared posture bins and corresponding midpoints
a_bins_selected = full_a_bins[global_valid_y]
a_mid_selected = a_bin_mid[global_valid_y.values]

# Determine valid time bins (x) across all conditions
valid_x_counts = pd.concat(
    [
        mat.reindex(index=full_a_bins, columns=full_t_bins).notna().sum(axis=0)
        for mat in heatmaps.values()
    ],
    axis=1
).sum(axis=1)  # sum across conditions

valid_x_mask = valid_x_counts >= 5

# Shared time bins and corresponding midpoints
t_bins_selected = full_t_bins[valid_x_mask]
t_mid_selected = t_bin_mid[valid_x_mask.values]


# 1. Smooth full heatmap
heatmaps_smooth_full = {
    c: gaussian_smooth_nan_threshold(
        mat.reindex(index=full_a_bins, columns=full_t_bins),
        sigma=(SIGMA_A, SIGMA_T),
        min_weight_fraction=0.25
    )
    for c, mat in heatmaps.items()
}

# 2. Clip to selected a bins and t bins for plotting
heatmaps_smooth = {
    c: mat_smooth.loc[a_bins_selected, t_bins_selected]
    for c, mat_smooth in heatmaps_smooth_full.items()
}

# --- Determine vmax from all smoothed matrices ---
all_vals = np.concatenate([mat.values.ravel() for mat in heatmaps_smooth.values()])
all_vals = all_vals[~np.isnan(all_vals)]
vmax = np.percentile(all_vals, 99.5)  # robust upper bound

# --- Plot heatmaps directly from smoothed matrices ---
for c, Z_smooth_df in heatmaps_smooth.items():
    fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(
        Z_smooth_df.values,
        ax=ax,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "Event probability"},
        rasterized=True
    )

    ax.invert_yaxis()
    ax.set_xlim(0, Z_smooth_df.shape[1])
    ax.set_ylim(0, Z_smooth_df.shape[0])
    ax.grid(False)
    set_pretty_ticks(
        ax,
        x_mids=t_mid_selected,
        y_mids=a_mid_selected,
        x_step=5,
        y_step=5,
    )

    ax.set_title(f"Smoothed event probability — cond {c}")
    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Y displ chg at interval start (deg)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(fig_dir, f"heatmap_event_prob_cond{c}_smoothed_shortIBI.pdf"),
        format="pdf"
    )

# #%%
# # plot a small version for figure

# SIGMA_A2 = 3.0   # smoothing along posture bins
# SIGMA_T2 = 3.0   # smoothing along time bins

# # 1. Smooth full heatmap
# heatmaps_smooth_full = {
#     c: gaussian_smooth_nan_threshold(
#         mat.reindex(index=full_a_bins, columns=full_t_bins),
#         sigma=(SIGMA_A2, SIGMA_T2),
#         min_weight_fraction=0.25
#     )
#     for c, mat in heatmaps.items()
# }

# # 2. Clip to selected a bins and t bins for plotting
# heatmaps_smooth = {
#     c: mat_smooth.loc[a_bins_selected, t_bins_selected]
#     for c, mat_smooth in heatmaps_smooth_full.items()
# }

# # --- Determine vmax from all smoothed matrices ---
# all_vals = np.concatenate([mat.values.ravel() for mat in heatmaps_smooth.values()])
# all_vals = all_vals[~np.isnan(all_vals)]
# vmax = np.percentile(all_vals, 99.5)  # robust upper bound

# # --- Plot heatmaps directly from smoothed matrices ---
# for c, Z_smooth_df in heatmaps_smooth.items():
#     fig, ax = plt.subplots(figsize=(4, 2.5))

#     sns.heatmap(
#         Z_smooth_df.values,
#         ax=ax,
#         cmap="viridis",
#         vmin=0,
#         vmax=vmax,
#         cbar_kws={"label": "Event probability"},
#         rasterized=True
#     )

#     ax.invert_yaxis()
#     ax.set_xlim(0, Z_smooth_df.shape[1])
#     ax.set_ylim(0, Z_smooth_df.shape[0])
#     ax.grid(False)
#     set_pretty_ticks(
#         ax,
#         x_mids=t_mid_selected,
#         y_mids=a_mid_selected,
#         x_step=10,
#         y_step=5,
#     )

#     ax.set_title(f"Smoothed event probability — cond {c}")
#     ax.set_xlabel("Time into IBI (s)")
#     ax.set_ylabel("Posture at interval start (deg)")
#     plt.tight_layout()
#     plt.savefig(
#         os.path.join(fig_dir, f"heatmap_event_prob_cond{c}_smoothed_shortIBI_small.pdf"),
#         format="pdf"
#     )








#%% plot average posture vs IBI 
what_y = 'y_displ'
what_x = 'start'

data_to_plot = tv_df_short.copy()

# X range and bins
x_range = [0, full_t_bins[valid_x_mask].max()]
BIN_WIDTH = TIME_STEP*3
bins = np.linspace(
    x_range[0],
    x_range[1] + BIN_WIDTH,
    math.ceil((x_range[1] - x_range[0]) / BIN_WIDTH) + 1
)

all_cond0 = sorted(data_to_plot['cond0'].unique())


# reduce posture bins: take 1 of every 5
reduced_a_bins = full_a_bins[global_valid_y][::5]


MIN_DATAPOINTS_PER_GROUP = 200

df_binned = (
    data_to_plot
    .groupby(['cond0','expNum'])
    .filter(lambda g: len(g) >= MIN_DATAPOINTS_PER_GROUP)  
    .groupby(['cond0','expNum'])
    .apply(
        lambda group: distribution_binned_average_opt(
            df=group,
            bin_col=what_y,
            by_col=what_x,
            method="median",
            bin=bins
        )
    )
)

df_binned.columns = [f'binned_{what_x}', f'binned_{what_y}']
df_binned = df_binned.reset_index()

# compute bin centers
df_binned["bin_center"] = df_binned[f"{what_x}"].apply(
    lambda interval: (interval.left + interval.right) / 2
)
#%%
# -------------------------------
# Filter bins with low support
# -------------------------------

MIN_FRACTION_EXP = 1
# count number of unique expNum per cond0 × posture_groups × bin_center
support = (
    df_binned.dropna()
    .groupby(['cond0','bin_center'])['expNum']
    .nunique()
    .reset_index(name='n_exp')
)

# total number of expNum per cond0
n_exp_total = df_binned.groupby('cond0')['expNum'].nunique().reset_index(name='n_exp_total')

# merge and compute fraction
support = support.merge(n_exp_total, on='cond0', how='left')
support['frac_exp'] = support['n_exp'] / support['n_exp_total']

# keep only bins with enough support
valid_bins = support.loc[support['frac_exp'] >= MIN_FRACTION_EXP, ['cond0','bin_center']]

# filter df_binned
df_binned_filtered = df_binned.merge(valid_bins, on=['cond0','bin_center'], how='inner')
#%%

sns.relplot(
    col='cond0',
    data=df_binned_filtered,
    x='bin_center',
    y=f'binned_{what_y}',
    kind='line',
    palette='mako',
    hue='expNum'
    # palette = sns.color_palette("sha", n_colors=len(reduced_a_bins)-1),
)
plt.savefig(
    os.path.join(fig_dir, f"line_avg_posture_vs_time_shortIBI.pdf"),
    format="pdf"
)

#%%
df_binned_filtered_avg = (
    df_binned_filtered.groupby(['cond0','bin_center'])[f'binned_{what_y}']
    .mean()
    .reset_index()
)
for c, Z_smooth_df in heatmaps_smooth.items():
    fig, ax = plt.subplots(figsize=(6, 3.5))

    # reorder rows so low posture is at bottom
    Z_plot = Z_smooth_df.loc[a_mid_selected].values  # ensure same order as a_mid_selected
    Z_plot = Z_plot[::-1, :]  # flip rows: first row now at top, last row at bottom

    im = ax.imshow(
        Z_plot,
        origin='upper',  # keep origin='upper' since we manually flipped
        aspect='auto',
        cmap=sns.cubehelix_palette(start=.08, rot=-.2, hue=0.9, dark=0.1, light=0.82, reverse=True, as_cmap=True),
        # cmap='viridis',
        vmin=0,
        vmax=vmax,
        extent=[
            0, t_mid_selected[-1],
            a_mid_selected[0], a_mid_selected[-1]   # min at bottom, max at top
        ]
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Event probability")

    # overlay lines
    df_lines = df_binned_filtered_avg[df_binned_filtered_avg['cond0'] == c]
    # for pg, group in df_lines.groupby('y_displ'):
    ax.plot(
        df_lines['bin_center'].values,
        df_lines[f'binned_{what_y}'].values,
        lw=1.2,
        alpha=0.5,
        color='white'
    )

    ax.set_xlabel("Time into IBI (s)")
    ax.set_ylabel("Y displacement at interval start (deg)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"heatmap_with_lines_cond{c}.pdf"), format='pdf')
