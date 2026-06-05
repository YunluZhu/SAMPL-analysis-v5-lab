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
# %%
# calcualte some attributes
grouped = df_filtered.groupby('unique_IBI_idx', observed=True)
IBI_features = grouped.agg(
    IBI_time=('time_relative_s', 'max'),
    IBI_ydispl=('y', lambda x: x.iloc[-1] - x.iloc[0]),
    IBI_xdispl=('x', lambda x: np.abs(x.iloc[-1] - x.iloc[0])),
    IBI_rot=('ang', lambda x: x.iloc[-1] - x.iloc[0]),
    IBI_yvel_avg=('yvel', 'median'),
    IBI_angvel_avg=('angVelSmoothed', 'median'),
    cond0=('cond0', 'first'),
    cond1=('cond1', 'first'),
    expNum=('expNum', 'first'),
    ztime=('ztime', 'first'),
    IBI=('unique_IBI_idx', 'count'),
    aftBout_pitch = ('ang', lambda x: x.iloc[0]),
    bfrBout_pitch = ('ang', lambda x: x.iloc[-1]),
    avg_pitch = ('ang', lambda x: x.median()),
).reset_index()

IBI_features = IBI_features.assign(
    frequency = lambda df: 1 / df['IBI_time']
)
features_toplt =  ['aftBout_pitch','IBI_rot','IBI_yvel_avg','IBI_angvel_avg']
# median per expNum
print(IBI_features.groupby(['cond0','expNum']).size())
median_res = IBI_features.groupby(['cond0','expNum'])[features_toplt].median().reset_index()
    

# # %%
# # let's plot something
# data_to_plot = IBI_features.query("frequency < 10")
# what_x = 'bfrBout_pitch'
# what_y = 'frequency'
# g = sns.displot(
#     col='cond0',
#     data=data_to_plot,
#     row='ztime',
#     # row='expNum',
#     x=what_x,
#     y=what_y,
#     common_norm=False,
#     col_order=cond0_all,
# )

# x_range = np.percentile(data_to_plot[what_x], [1,99])
# g.set(
#     xlim=x_range,
#     ylim=np.percentile(data_to_plot[what_y], [1,95]),
# )


# %%
df_filtered_ldNight = df_filtered.query("cond1=='ld' and ztime=='night'")
# drop unique_IBI_idx that have the first time_relative_s not equal to 0
df_filtered_ldNight = df_filtered_ldNight[df_filtered_ldNight.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

# Compute drift per unique_IBI_idx
drift = df_filtered_ldNight.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Filter rows where drift < 1
df_negative_drift = df_filtered_ldNight[drift < 1].copy()

#%% new filtering
df_negative_drift['duration'] = df_negative_drift.groupby(['cond0','unique_IBI_idx'])['time_relative_s'].transform('max')

#%
DURATION_THRESH = np.percentile(df_negative_drift['duration'], 90) # seconds; only include IBIs shorter than this

# note, this is a different set of IBIs than before 
# censoring at 6s rather than truncating 
MAX_T = 10

rows = []

for ibi, g in df_negative_drift.groupby('unique_IBI_idx', sort=False):
    g = g.sort_values('time_relative_s')

    times = g['time_relative_s'].values
    angs = g['ang'].values

    meta = g[['expNum', 'cond0', 'cond1']].iloc[0].to_dict()

    if len(times) < 2:
        continue

    # True event time (last observed time)
    T_event = times[-1]
    event_happens = T_event <= MAX_T

    for i in range(len(times) - 1):
        start = times[i]
        stop = times[i + 1]

        # Drop intervals that start after censoring time
        if start >= MAX_T:
            break

        # Right-censor the stop time
        stop_capped = min(stop, MAX_T)

        # Event only if this is the final interval AND event happens before MAX_T
        is_last_interval = (i == len(times) - 2)
        event = int(is_last_interval and event_happens)

        rows.append({
            'unique_IBI_idx': ibi,
            'start': start,
            'stop': stop_capped,
            'event': event,
            'ang': angs[i],
            **meta
        })

tv_df = pd.DataFrame(rows)

print(
    "rows:", len(tv_df),
    "IBIs:", tv_df['unique_IBI_idx'].nunique(),
    "events:", tv_df['event'].sum()
)

# -----------------------------
# Standardize covariates safely
# -----------------------------

# Standardize ang
tv_df['ang_z'] = (tv_df['ang'] - tv_df['ang'].mean()) / tv_df['ang'].std(ddof=0)

# Add tiny epsilon to avoid log(0)
eps = 1e-3
tv_df['log_time'] = np.log(tv_df['start'] + eps)

# Clip log_time based on observed data to avoid extremes
clip_lo = np.floor(tv_df['log_time'].min())
clip_hi = np.ceil(tv_df['log_time'].max())
tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

# Compute interaction and standardize it
tv_df['ang_z_logt'] = tv_df['ang_z'] * tv_df['log_time_clipped']
tv_df['ang_z_logt'] = (tv_df['ang_z_logt'] - tv_df['ang_z_logt'].mean()) / tv_df['ang_z_logt'].std(ddof=0)
tv_df['duration'] = tv_df.groupby('unique_IBI_idx')['start'].transform(lambda x: x.max() - x.min())

#%% model posture explicitly for P0

tv_df_short = tv_df.copy()

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
    ax.set_xlim(0, Z_smooth_df.shape[1])
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