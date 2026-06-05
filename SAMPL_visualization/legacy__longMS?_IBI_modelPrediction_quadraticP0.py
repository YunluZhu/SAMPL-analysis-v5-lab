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
# calcualte some attributes
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
# features_toplt =  ['aftBout_pitch','IBI_rot','IBI_yvel_avg','IBI_angvel_avg','IBI_ydispl','bfrBout_pitch','avg_pitch','frequency']
# # median per expNum
# print(IBI_features.groupby(['cond0','expNum']).size())
# median_res = IBI_features.groupby(['cond0','expNum','ztime'])[features_toplt].median().reset_index()
    

# # %%
# # let's plot something
# data_to_plot = median_res

# for feature in features_toplt:
#     g = plt_categorical_combined_3(
#         data=data_to_plot,
#         x='cond0',
#         y=feature,
#         col='ztime',
#         order=cond0_all,
#         palette=my_palette,
#         height=3,
#         units='expNum',
#     )
# # save pdf
#     plt.savefig(os.path.join(fig_dir, f'IBI_feature_{feature}_byCond0.pdf'), format='pdf')
    

# %%
df_filtered_ld = df_filtered.copy()
# drop unique_IBI_idx that have the first time_relative_s not equal to 0
df_filtered_ld = df_filtered_ld[df_filtered_ld.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

# Compute drift per unique_IBI_idx
drift = df_filtered_ld.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Filter rows where drift < 1
df_negative_drift = df_filtered_ld[drift < 1].copy()


#%%

MAX_T = 5.0  # max IBI time (seconds) for censoring, like your previous MAX_T
# EPS = 1   # small epsilon to avoid log(0)
MIN_ROWS_PER_EXP = 5  # skip very small repeats

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

# "event ~ ang0_z + ang0_z2 + log_time_clipped + ang0_z_logt + ang0_z2_logt"

#%%
# -----------------------------
# Fit logistic regression per cond0 × expNum × ztime with quadratic term
# -----------------------------
results_exp = []

for ztime, tv_df in tv_df_all.groupby('ztime', observed=True):
    print(f"\n=== ztime: {ztime} ===")

    for (cond, exp), df_e in tv_df.groupby(['cond0', 'expNum'], observed=True):
        if len(df_e) < MIN_ROWS_PER_EXP:
            continue

        # Full model with quadratic term
        m_full = smf.logit(
            "event ~ ang0_z + ang0_z2 + log_time_clipped + ang0_z_logt + ang0_z2_logt",
            data=df_e
        ).fit(disp=False)

        # Store coefficients and p-values
        for param, coef in m_full.params.items():
            results_exp.append({
                'ztime': ztime,
                'cond0': cond,
                'expNum': exp,
                'parameter': param,
                'coef': coef,
                'pval': m_full.pvalues[param]
            })

results_exp_df = pd.DataFrame(results_exp)

# -----------------------------
# Plot coefficients
# -----------------------------
g = sns.catplot(
    data=results_exp_df,
    x='cond0',
    y='coef',
    hue='cond0',
    row='ztime',
    col='parameter',
    kind='point',
    join=False,
    capsize=0.1,
    palette='tab10',
    height=3,
    aspect=0.8,
    errorbar='se',
    sharey=False
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "catplot_coef_per_exp_byP0_quadratic.pdf"), format='pdf')

# too many parameters, let's bootstrap
#%%

# -----------------------------
# Parameters
# -----------------------------
n_boot = 100
MAX_T = 5.0
EPS = 1e-3

# -----------------------------
# 1. Bootstrap logistic regression with quadratic term
# -----------------------------
boot_results = []

for (ztime, cond), df_cond in tv_df_all.groupby(['ztime', 'cond0'], observed=True):
    # Pre-calculate indices to avoid repeated groupby inside the bootstrap loop
    ibi_groups = df_cond.groupby(['expNum', 'unique_IBI_idx'], observed=True).indices

    # Build dict: expNum → list of unique_IBI_idx arrays
    exp_to_ibis = {}
    for (exp, ibi), indices in ibi_groups.items():
        exp_to_ibis.setdefault(exp, []).append(indices)

    exp_list = list(exp_to_ibis.keys())
    n_ibi_per_exp = int(np.median([len(ibis) for ibis in exp_to_ibis.values()]))

    for b in tqdm(range(n_boot), desc=f"Bootstrapping {ztime}, {cond} with {n_ibi_per_exp} per exp"):
        boot_idx_list = []

        for exp in exp_list:
            available_ibis = exp_to_ibis[exp]
            sel_idx = np.random.randint(0, len(available_ibis), size=n_ibi_per_exp)
            for idx in sel_idx:
                boot_idx_list.append(available_ibis[idx])

        # Flatten and select rows
        flat_idx = np.concatenate(boot_idx_list)
        df_boot = df_cond.iloc[flat_idx]

        # Fit logistic regression with quadratic term and interactions
        try:
            m_boot = smf.logit(
                "event ~ ang0_z + log_time_clipped + ang0_z_logt + ang0_z2 + ang0_z2_logt",
                data=df_boot
            ).fit(disp=False)
        except Exception:
            continue

        # Store coefficients
        for param, coef in m_boot.params.items():
            boot_results.append({
                'ztime': ztime,
                'cond0': cond,
                'bootstrap': b,
                'parameter': param,
                'coef': coef
            })

boot_df = pd.DataFrame(boot_results)

#%%
# change parameter to par
boot_df = boot_df.rename(columns={'parameter': 'par'})
# -----------------------------
# Plot bootstrap coefficients directly
g = sns.catplot(
    data=boot_df,
    kind='point',
    x='cond0',
    row='ztime',
    y='coef',
    col='par',  
    errorbar='sd',          
    height=3,
    aspect=0.7,
    sharey=False,
)
plt.savefig(os.path.join(fig_dir, "catplot_bootstrap_coef_per_cond0_byP0.pdf"), format='pdf')


#%%
ang0_grid_points = 20  # resolution for prediction
t_grid = np.linspace(0.006, MAX_T, 100)

# -----------------------------
# 2. Generate predicted probabilities across ang0 × time
# -----------------------------
pred_rows = []

for ztime, df_z in boot_df.groupby('ztime', observed=True):
    cond_levels = df_z['cond0'].unique()
    df_tv = tv_df_all[tv_df_all['ztime'] == ztime]

    # Mean and std for reconstructing z-score
    ang0_mean = df_tv['ang0'].mean()
    ang0_std = df_tv['ang0'].std(ddof=0)

    # Log-time clipping
    log_time = np.log(t_grid)
    clip_lo = np.floor(log_time.min())
    clip_hi = np.ceil(log_time.max())
    log_time_clip = np.clip(log_time, clip_lo, clip_hi)

    for cond in cond_levels:
        coefs_cond = df_z[df_z['cond0'] == cond]
        coefs_matrix = coefs_cond.pivot(index='bootstrap', columns='par', values='coef').sort_index()
        n_boot_local = coefs_matrix.shape[0]

        # ang0 grid in original scale
        ang0_grid = np.linspace(-30, 40, ang0_grid_points)
        ang0_z_grid = (ang0_grid - ang0_mean) / ang0_std
        ang0_z2_grid = (ang0_z_grid**2 - np.mean(ang0_z_grid**2)) / np.std(ang0_z_grid**2)

        # Interaction terms
        ang0_z_logt = (ang0_z_grid[:, None] * log_time_clip[None, :])
        ang0_z_logt = (ang0_z_logt - ang0_z_logt.mean()) / ang0_z_logt.std(ddof=0)

        ang0_z2_logt = (ang0_z2_grid[:, None] * log_time_clip[None, :])
        ang0_z2_logt = (ang0_z2_logt - ang0_z2_logt.mean()) / ang0_z2_logt.std(ddof=0)

        # Design matrix: flatten all combinations
        X_mat = np.column_stack([
            np.ones(ang0_z_logt.size),           # Intercept
            np.repeat(ang0_z_grid, len(t_grid)), # ang0_z
            np.tile(log_time_clip, len(ang0_z_grid)), # log_time_clipped
            ang0_z_logt.ravel(),
            ang0_z2_grid.repeat(len(t_grid)),
            ang0_z2_logt.ravel()
        ])

        # Align columns with model
        X_df = pd.DataFrame(X_mat, columns=coefs_matrix.columns)
        X_mat = X_df.values

        B_mat = coefs_matrix.values.T  # n_params x n_boot
        linpred = X_mat @ B_mat
        p_boot = 1 / (1 + np.exp(-linpred))

        # Flatten for long-form DataFrame
        for i_ang, ang_val in enumerate(ang0_grid):
            for i_time, t_val in enumerate(t_grid):
                for b_idx in range(n_boot_local):
                    pred_rows.append({
                        'ztime': ztime,
                        'cond0': cond,
                        'ang0': ang_val,
                        'time': t_val,
                        'prob': p_boot[i_ang * len(t_grid) + i_time, b_idx]
                    })

pred_df = pd.DataFrame(pred_rows)

# -----------------------------
# 3. Plot with seaborn
# -----------------------------

for ztime, df_z in pred_df.groupby('ztime', observed=True):
    g = sns.relplot(
        col='cond0',
        row='ztime',
        data=df_z,
        x='time',
        y='prob',
        hue='ang0',
        kind='line',
        palette='coolwarm',
        height=3,
        aspect=1.5,
        legend='full',
        errorbar='sd'
    )
    g.set_titles(f"ztime = {ztime}")
    g.set_axis_labels("IBI time (s)", "Predicted swim probability")
    plt.tight_layout()
# %%
# Loop through ztime and cond0
for ztime, df_ztime in pred_df.groupby('ztime', observed=True):
    print(f"Plotting heatmaps for ztime: {ztime}")
    for cond, df_cond in df_ztime.groupby('cond0', observed=True):
        
        # Average over bootstraps
        df_avg = df_cond.groupby(['ang0', 'time'], observed=True)['prob'].mean().reset_index()
        
        # Pivot to matrix form: rows=ang0, cols=time
        heatmap_df = df_avg.pivot(index='ang0', columns='time', values='prob')
        
        # Plot heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            heatmap_df,
            cmap='coolwarm',
            cbar_kws={'label': 'Predicted swim probability'},
            vmin=0, vmax=0.01
        )
        plt.title(f"Predicted swim probability — ztime {ztime}, cond0 {cond}")
        plt.xlabel("IBI time (s)")
        plt.ylabel("Initial posture (ang0)")
        plt.tight_layout()
        plt.show()
# %%
