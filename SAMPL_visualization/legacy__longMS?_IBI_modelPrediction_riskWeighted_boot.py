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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
    #%
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import chi2

from scipy.ndimage import gaussian_filter1d

def nice_floor(x, step):
    return step * math.floor(x / step)

def nice_ceil(x, step):
    return step * math.ceil(x / step)

def fd_width(x):
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    return 2 * iqr / (len(x) ** (1/3))

def snap_step(fd, allowed):
    return min(allowed, key=lambda x: abs(x - fd))

def set_pretty_ticks(ax, x_mids, y_mids, x_step=5, y_step=2):
    # X ticks
    xticks = np.arange(0, len(x_mids), x_step)
    ax.set_xticks(xticks + 0.5)
    ax.set_xticklabels(np.round(x_mids[xticks], 2), rotation=0)

    # Y ticks
    yticks = np.arange(0, len(y_mids), y_step)
    ax.set_yticks(yticks + 0.5)
    ax.set_yticklabels(np.round(y_mids[yticks], 1))
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
df_filtered_ld = df_filtered.copy()
# drop unique_IBI_idx that have the first time_relative_s not equal to 0
df_filtered_ld = df_filtered_ld[df_filtered_ld.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

# Compute drift per unique_IBI_idx
drift = df_filtered_ld.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Filter rows where drift < 1
df_negative_drift = df_filtered_ld[drift < 1].copy()


#%%

MAX_T = 10  # max IBI time (seconds) for censoring, like your previous MAX_T

# -----------------------------
# Prepare tv_df_all
# -----------------------------
tv_df_all_ = []

# Filter IBIs shorter than threshold per ztime (e.g., 90th percentile)
for ztime, df_z in df_negative_drift.groupby('ztime', observed=True):
    df_z['duration'] = df_z.groupby('unique_IBI_idx', observed=True)['time_relative_s'].transform('max')
    dur_thresh = np.percentile(df_z['duration'], 90)
    df_z_thresh = df_z.query("duration <= @dur_thresh").copy()

    rows = []
    for ibi, g in df_z_thresh.groupby('unique_IBI_idx', sort=False):
        g = g.sort_values('time_relative_s')

        times = g['time_relative_s'].values
        angs = g['ang'].values
        ang0 = angs[0]  # initial posture

        # group-level metadata
        meta = g[['expNum', 'cond0', 'cond1', 'ztime']].iloc[0].to_dict()

        if len(times) < 2:
            continue  # cannot define intervals

        # Right-censoring and event assignment
        T_event = times[-1]
        event_happens = T_event <= MAX_T

        for i in range(len(times) - 1):
            start = times[i]
            stop = times[i + 1]

            # Discard intervals that start after MAX_T
            if start >= MAX_T:
                break

            # Right-censor stop
            stop_capped = min(stop, MAX_T)

            # Event only for last interval AND before MAX_T
            is_last_interval = (i == len(times) - 2)
            event = int(is_last_interval and event_happens)

            rows.append({
                'unique_IBI_idx': ibi,
                'start': start,
                'stop': stop_capped,
                'event': event,
                'ang0': ang0,
                'ang': angs[i], 
                **meta
            })

    tv_df = pd.DataFrame(rows)

    if tv_df.empty:
        continue
    
    # -----------------------------
    # Standardize covariates
    # -----------------------------
    # z-score initial posture
    tv_df['ang0_z'] = (tv_df['ang0'] - tv_df['ang0'].mean()) / tv_df['ang0'].std(ddof=0)

    # log-time with clipping
    tv_df['log_time'] = np.log(tv_df['stop']) # change from start to stop to avoid log(0) and EPS bias
    clip_lo = np.floor(tv_df['log_time'].min())
    clip_hi = np.ceil(tv_df['log_time'].max())
    tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

    # Interaction (initial posture × log-time)
    tv_df['ang0_z_logt'] = tv_df['ang0_z'] * tv_df['log_time_clipped']
    tv_df['ang0_z_logt'] = (tv_df['ang0_z_logt'] - tv_df['ang0_z_logt'].mean()) / tv_df['ang0_z_logt'].std(ddof=0)

    # ----- New quadratic term -----
    tv_df['ang0_z2'] = tv_df['ang0_z']**2
    tv_df['ang0_z2'] = (tv_df['ang0_z2'] - tv_df['ang0_z2'].mean()) / tv_df['ang0_z2'].std(ddof=0)
    # Quadratic × time interaction
    tv_df['ang0_z2_logt'] = tv_df['ang0_z2'] * tv_df['log_time_clipped']
    tv_df['ang0_z2_logt'] = (tv_df['ang0_z2_logt'] - tv_df['ang0_z2_logt'].mean()) / tv_df['ang0_z2_logt'].std(ddof=0)
    
    tv_df_all_.append(tv_df)

tv_df_all = pd.concat(tv_df_all_, ignore_index=True)

#%%
#%% model posture explicitly for P0

tv_df_short = tv_df_all.copy()

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

# tv_df_short['ang0'] = tv_df_short.groupby('unique_IBI_idx')['ang'].transform('first')
tv_df_short['ang0_bin'] = pd.cut(tv_df_short["ang0"], bins=a_edges, include_lowest=True, labels=a_bin_mid)

tv_df_short['t_bin'] = pd.cut(tv_df_short['start'], t_edges)

#%% bootstrap hazard estimation and GLM fitting
# -----------------------------
# Parameters
# -----------------------------
n_boot = 100
MIN_IBI_PER_BIN = 30
MIN_IBI_PER_POSTURE = 200
EPS = 1e-3
SIGMA_T = 2.0

# # -----------------------------
# # Step 0: Precompute hazard_df_nonboot and global stats per (cond0, ztime)
# # -----------------------------
# nonboot_results = {}
# global_stats = {}

# for (cond0, ztime), df_sub in tv_df_short.groupby(['cond0', 'ztime'], observed=True):
#     # 1. Compute hazard
#     hazard_df = (
#         df_sub
#         .groupby(['ang0_bin', 't_bin'], observed=True)
#         .agg(
#             n_risk=('unique_IBI_idx', 'nunique'),
#             n_event=('event', 'sum')
#         )
#         .reset_index()
#     )

#     hazard_df = hazard_df[hazard_df['n_risk'] >= MIN_IBI_PER_BIN]

#     posture_support = hazard_df.groupby('ang0_bin', observed=True)['n_risk'].sum()
#     valid_postures = posture_support[posture_support >= MIN_IBI_PER_POSTURE].index
#     hazard_df = hazard_df[hazard_df['ang0_bin'].isin(valid_postures)]

#     hazard_df['hazard'] = hazard_df['n_event'] / hazard_df['n_risk']
#     hazard_df['t_mid'] = hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))
#     hazard_df['hazard_smooth'] = hazard_df.groupby('ang0_bin', observed=True)['hazard'].transform(
#         lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest')
#     )
#     # hazard_df['n_event_smooth'] = np.clip(
#     #     hazard_df['hazard_smooth'] * hazard_df['n_risk'], EPS, hazard_df['n_risk'] - EPS
#     # )

#     # 2. Global standardization
#     a = hazard_df['ang0_bin'].astype(float)
#     t = hazard_df['t_mid'].astype(float)
#     a_mu, a_sd = a.mean(), a.std(ddof=0)
#     t_mu, t_sd = t.mean(), t.std(ddof=0)
#     a_z = (a - a_mu) / a_sd
#     a2 = a_z**2
#     a2_mu, a2_sd = a2.mean(), a2.std(ddof=0)

#     global_stats[(cond0, ztime)] = dict(a_mu=a_mu, a_sd=a_sd, t_mu=t_mu, t_sd=t_sd, a2_mu=a2_mu, a2_sd=a2_sd)

#     # 3. Fit GLM (non-boot)
#     hazard_df['a_z'] = a_z
#     hazard_df['t_z'] = (t - t_mu) / t_sd
#     hazard_df['a2_z'] = (a2 - a2_mu) / a2_sd
#     hazard_df['a_t'] = hazard_df['a_z'] * hazard_df['t_z']
#     hazard_df['a2_t'] = hazard_df['a2_z'] * hazard_df['t_z']

#     X = sm.add_constant(hazard_df[['a_z','t_z','a_t','a2_z','a2_t']])
#     y = np.column_stack([hazard_df['n_event'], hazard_df['n_risk'] - hazard_df['n_event']])

#     nonboot_results[(cond0, ztime)] = sm.GLM(y, X, family=sm.families.Binomial()).fit()

# -----------------------------
# Step 1: Bootstrap GLM
# -----------------------------
boot_results = []

for (cond0, ztime), df_sub in tv_df_short.groupby(['cond0', 'ztime'], observed=True):

    # Precompute indices for IBI-level bootstrap
    ibi_groups = df_sub.groupby(['expNum', 'unique_IBI_idx'], observed=True).indices
    exp_to_ibis = {}
    for (exp, ibi), idx in ibi_groups.items():
        exp_to_ibis.setdefault(exp, []).append(idx)
    n_ibi = int(np.median([len(v) for v in exp_to_ibis.values()]))

    for b in tqdm(range(n_boot), desc=f'Bootstrapping {cond0}, {ztime}'):
        # Resample IBIs within each experiment
        sel = []
        for exp, ibs in exp_to_ibis.items():
            picks = np.random.randint(0, len(ibs), size=n_ibi)
            sel.extend(ibs[i] for i in picks)
        df_boot = df_sub.iloc[np.concatenate(sel)]

        # Recompute hazard identical to non-boot
        hazard_df = (
            df_boot
            .groupby(['ang0_bin', 't_bin'], observed=True)
            .agg(n_risk=('unique_IBI_idx','nunique'), n_event=('event','sum'))
            .reset_index()
        )
        hazard_df = hazard_df[hazard_df['n_risk'] >= MIN_IBI_PER_BIN]
        posture_support = hazard_df.groupby('ang0_bin', observed=True)['n_risk'].sum()
        valid_postures = posture_support[posture_support >= MIN_IBI_PER_POSTURE].index
        hazard_df = hazard_df[hazard_df['ang0_bin'].isin(valid_postures)]
        hazard_df['hazard'] = hazard_df['n_event'] / hazard_df['n_risk']
        hazard_df['t_mid'] = hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))
        hazard_df['hazard_smooth'] = hazard_df.groupby('ang0_bin', observed=True)['hazard'].transform(
            lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest')
        )
        # hazard_df['n_event_smooth'] = np.clip(
        #     hazard_df['hazard_smooth'] * hazard_df['n_risk'], EPS, hazard_df['n_risk'] - EPS
        # )
        a = hazard_df['ang0_bin'].astype(float)
        t = hazard_df['t_mid'].astype(float)
        
        a_mu, a_sd = a.mean(), a.std(ddof=0)
        t_mu, t_sd = t.mean(), t.std(ddof=0)
        
        # Guard against division by zero if only one bin remains
        a_sd = a_sd if a_sd > 0 else 1.0
        t_sd = t_sd if t_sd > 0 else 1.0

        hazard_df['a_z'] = (a - a_mu) / a_sd
        hazard_df['t_z'] = (t - t_mu) / t_sd
        
        # Standardize using global stats
        a = hazard_df['ang0_bin'].astype(float)
        t = hazard_df['t_mid'].astype(float)
        
        # Standardize a^2 locally
        a2_raw = hazard_df['a_z']**2
        a2_mu, a2_sd = a2_raw.mean(), a2_raw.std(ddof=0)
        a2_sd = a2_sd if a2_sd > 0 else 1.0
        
        hazard_df['a2_z'] = (a2_raw - a2_mu) / a2_sd
        hazard_df['a_t'] = hazard_df['a_z'] * hazard_df['t_z']
        hazard_df['a2_t'] = hazard_df['a2_z'] * hazard_df['t_z']

        # Fit GLM
        X = sm.add_constant(hazard_df[['a_z','t_z','a_t','a2_z','a2_t']])
        y = np.column_stack([hazard_df['n_event'], hazard_df['n_risk'] - hazard_df['n_event']])
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()

        # 5. Store coefficients AND the local stats (Required for prediction)
        for param, coef in model.params.items():
            boot_results.append({
                'cond0': cond0, 'ztime': ztime, 'bootstrap': b, 
                'parameter': param, 'coef': coef,
                'a_mu': a_mu, 'a_sd': a_sd, 't_mu': t_mu, 't_sd': t_sd,
                'a2_mu': a2_mu, 'a2_sd': a2_sd
            })
boot_df = pd.DataFrame(boot_results)


#%%
from scipy.special import expit

# -----------------------------
# Prediction grid (same variables as GLM)
# -----------------------------
a_grid = np.sort(tv_df_short['ang0_bin'].dropna().astype(float).unique())
t_grid = np.sort(tv_df_short.groupby('t_bin', observed=True)['start'].mean().dropna())

AA, TT = np.meshgrid(a_grid, t_grid, indexing='ij')
a = AA.ravel()
t = TT.ravel()

boot_pred_heatmaps = {}

# -----------------------------
# Step 3: Build predicted heatmaps (Standardizing PER BOOT)
# -----------------------------
for (cond0, ztime), df_group in boot_df.groupby(['cond0', 'ztime'], observed=True):
    pred_list = []
    
    # Iterate through each bootstrap sample's model and its specific stats
    for b, df_boot in df_group.groupby('bootstrap', observed=True):
        # 1. Get stats specific to THIS bootstrap iteration
        s = df_boot.iloc[0] 
        
        # 2. Standardize grid using THIS boot's stats
        a_z = (a - s['a_mu']) / s['a_sd']
        t_z = (t - s['t_mu']) / s['t_sd']
        a2_z = (a_z**2 - s['a2_mu']) / s['a2_sd']
        
        X_grid = sm.add_constant(np.column_stack([
            a_z, t_z, a_z*t_z, a2_z, a2_z*t_z
        ]), has_constant='add')
        
        # 3. Predict
        coef = df_boot.set_index('parameter')['coef']
        beta = coef.loc[['const', 'a_z', 't_z', 'a_t', 'a2_z', 'a2_t']].values
        
        p_hat = expit(X_grid @ beta)
        pred_list.append(p_hat.reshape(AA.shape))
    
    boot_pred_heatmaps[(cond0, ztime)] = np.mean(pred_list, axis=0)


#%%

# -----------------------------
# Plot example heatmaps
# -----------------------------
import matplotlib.pyplot as plt

for (cond0, ztime), heatmap in boot_pred_heatmaps.items():
    plt.figure(figsize=(6,4))
    plt.pcolormesh(TT, AA, heatmap, shading='auto')
    plt.xlabel('Time')
    plt.ylabel('Initial posture (ang0)')
    plt.title(f'Cond={cond0}, Ztime={ztime} (bootstrapped)')
    plt.colorbar(label='Predicted P(event)')
    plt.savefig(os.path.join(fig_dir, f'GLM_bootstrap_heatmap_{cond0}_{ztime}.pdf'),format='pdf')
# %%
# Filter for the terms of interest
terms_to_plot = ['a_z', 't_z', 'a_t', 'a2_z', 'a2_t']
plot_df = boot_df[boot_df['parameter'].isin(terms_to_plot)]

# Violin / distribution plot across bootstraps
sns.catplot(
    data=plot_df,
    y='coef', x='cond0',
    row='parameter',
    col='ztime',
    kind='point',        # shows full distribution
    height=3, aspect=1.2,
    errorbar='sd',  # standard deviation error bars
    sharey=False
)
plt.savefig(os.path.join(fig_dir, 'GLM_bootstrap_coefficients.pdf'),format='pdf')

# all_coefs = []

# for r in results:
#     model = r['model']
#     # Extract params, standard errors, and p-values
#     pdf = pd.DataFrame({
#         'coef': model.params,
#         'std_err': model.bse,
#         'p_value': model.pvalues,
#         'conf_lower': model.conf_int()[0],
#         'conf_upper': model.conf_int()[1]
#     }).reset_index().rename(columns={'index': 'term'})
    
#     # Tag with experimental metadata
#     pdf['cond0'] = r['cond0']
#     pdf['ztime'] = r['ztime']
#     all_coefs.append(pdf)

# coef_df = pd.concat(all_coefs, ignore_index=True)
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Filter for the terms of interest (exclude the constant/intercept for scaling)
# terms_to_plot = ['a_z', 't_z', 'a_t', 'a2_z', 'a2_t']
# plot_df = coef_df[coef_df['term'].isin(terms_to_plot)]

# sns.catplot(
#     data=plot_df, 
#     y='coef', x='cond0', col='term',
#     row='ztime',
# )
