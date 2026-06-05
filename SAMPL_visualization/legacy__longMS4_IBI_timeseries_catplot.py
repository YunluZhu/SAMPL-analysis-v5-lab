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
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

#%%
##### Parameters to change #####

pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' or 'night'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]

# Create a Seaborn palette
my_palette = sns.color_palette(my_colors)
# CONSTANTS %%%%%%
BIN_WIDTH = 0.1  # (s)
AVERAGE_BIN = np.arange(0, 1+BIN_WIDTH, BIN_WIDTH)

##### Parameters to change #####

# Define your exponential decay model
def exp_decay(t, v_terminal, tau, v0): 
    return v_terminal + (v0 - v_terminal) * np.exp(-t / tau)

def analyze_two_vars(df, y_cols=("yvel", "angVelSmoothed"), n_grid=400, second_cat='cond1'):
    """
    Efficient version that handles both dependent variables together.
    Returns:
        df_avg_all : averaged (median) traces per cond0/expNum/variable
        df_fit_all : exponential fit parameters per cond0/expNum/variable
    """

    # --- Map trials and build time grid
    trial_ids = df["unique_IBI_idx"].unique()
    trial_to_idx = {t: i for i, t in enumerate(trial_ids)}
    n_trials = len(trial_ids)

    t_min, t_max = df["time_relative_s"].min(), df["time_relative_s"].max()
    t_grid = np.linspace(t_min, t_max, n_grid)

    # --- Prepare arrays
    n_vars = len(y_cols)
    interp_trials = np.zeros((n_vars, n_trials, n_grid), dtype=np.float32)

    # --- Interpolate once per trial, fill all y_cols
    grouped = df.groupby("unique_IBI_idx")
    for trial, group in tqdm(grouped, desc="Interpolating all vars"):
        i = trial_to_idx[trial]
        t_orig = group["time_relative_s"].values
        t_clip = np.clip(t_grid, t_orig[0], t_orig[-1])
        idx = np.searchsorted(t_orig, t_clip) - 1
        idx[idx < 0] = 0
        t0, t1 = t_orig[idx], t_orig[idx + 1]

        for v, y_col in enumerate(y_cols):
            y_orig = group[y_col].values
            y0, y1 = y_orig[idx], y_orig[idx + 1]
            interp_trials[v, i, :] = y0 + (y1 - y0) * (t_clip - t0) / (t1 - t0)

    # --- Average per (cond0, expNum)
    avg_list = []
    cond_exp_groups = df.groupby(["cond0", second_cat, "expNum"],observed=True)["unique_IBI_idx"].unique()

    for (cond, cond1, expNum), trial_subset in tqdm(cond_exp_groups.items(), desc="Averaging per cond/expNum"):
        idxs = [trial_to_idx[t] for t in trial_subset if t in trial_to_idx]
        if not idxs:
            continue

        for v, y_col in enumerate(y_cols):
            arr = interp_trials[v, idxs, :]
            y_median = np.median(arr, axis=0)
            y_std = np.std(arr, axis=0)

            avg_list.append(pd.DataFrame({
                "time": t_grid,
                "median": y_median,
                "std": y_std,
                "cond0": cond,
                "cond1":cond1,
                "expNum": expNum,
                "variable": y_col
            }))

    df_avg_all = pd.concat(avg_list, ignore_index=True)

    # --- Fit exponential decay per variable
    fit_results = []
    for (cond, cond1,expNum, var), df_group in tqdm(df_avg_all.groupby(["cond0", second_cat,"expNum", "variable"],observed=True), desc="Fitting"):
        t_vals = df_group["time"].values
        y_vals = df_group["median"].values

        v0_guess = np.median(y_vals[:20])
        v_terminal_guess = np.median(y_vals[-20:])
        tau_guess = (t_vals[-1] - t_vals[0]) / 2
        p0 = [v_terminal_guess, tau_guess, v0_guess]

        try:
            popt, _ = curve_fit(exp_decay, t_vals, y_vals, p0=p0, maxfev=1000)
            fit_results.append({
                "cond0": cond,
                "cond1":cond1,
                "expNum": expNum,
                "variable": var,
                "v_terminal": popt[0],
                "tau": popt[1],
                "v0": popt[2]
            })
        except RuntimeError:
            fit_results.append({
                "cond0": cond,
                "cond1":cond1,
                "expNum": expNum,
                "variable": var,
                "v_terminal": np.nan,
                "tau": np.nan,
                "v0": np.nan
            })

    df_fit_all = pd.DataFrame(fit_results)
    
    
    # --- fit on raw ---
    fit_results2 = []
    for var in ['yvel_sg','angvel_sg']:
        # --- Fit exponential decay per variable
        lower_bounds = [-10, 1e-3, -20]  # [v_terminal, tau, v0]
        upper_bounds = [ 10, 20,  10]
        bounds = (lower_bounds, upper_bounds)
        for (cond, cond1,expNum), df_group in tqdm(df.groupby(["cond0", second_cat,"expNum"],observed=True), desc="Fitting"):
            t_vals = df_group["time_relative_s"].values
            y_vals = df_group[var].values

            v0_guess = np.median(y_vals[:50])
            v_terminal_guess = np.median(y_vals[-50:])
            tau_guess = (t_vals[-1] - t_vals[0]) / 2
            p0 = [v_terminal_guess, tau_guess, v0_guess]

            try:
                popt, _ = curve_fit(exp_decay, t_vals, y_vals, p0=p0,bounds=bounds,maxfev=2000, method='trf')
                fit_results2.append({
                    "cond0": cond,
                    "cond1":cond1,
                    "expNum": expNum,
                    "variable": var,
                    "v_terminal": popt[0],
                    "tau": popt[1],
                    "v0": popt[2]
                })
            except RuntimeError:
                fit_results2.append({
                    "cond0": cond,
                    "cond1":cond1,
                    "expNum": expNum,
                    "variable": var,
                    "v_terminal": np.nan,
                    "tau": np.nan,
                    "v0": np.nan
                })

    df_fit_all2 = pd.DataFrame(fit_results2)
    return df_avg_all, df_fit_all, df_fit_all2


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

#%% truncate to start from speed threshold
df_time_filtered['swimSpeed_smoothed'] = df_time_filtered.groupby('unique_IBI_idx', observed=True)['swimSpeed'].transform(
    lambda x: savgol_filter(x.values, 11, 3) if len(x) >= 11 else x.values
)

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
#%%
#% compute IBI duration after truncation
IBI_features = df_truncated2.groupby(['unique_IBI_idx', 'cond1', 'cond0', 'ztime','expNum'], observed=True).agg(
    IBI_dur = ('time_relative_s', 'max'),
).reset_index()

#%%
median_df = IBI_features.groupby(['cond0', 'ztime','expNum'], observed=True)['IBI_dur'].median().reset_index()
plt_categorical_combined_3(
    median_df,
    x='cond0',
    y='IBI_dur',
    col='ztime',
    units='expNum',
    errorbar='se',
)
plt.savefig(os.path.join(fig_dir,f"IBI_duration_by_cond0.pdf"), format='PDF')

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

anova_results = {}
tukey_results = {}

for ztime, df_z in median_df.groupby('ztime', observed=True):

    print(f"\n==============================")
    print(f"ztime = {ztime}")
    print(f"==============================")

    # Require at least 2 conditions
    if df_z['cond0'].nunique() < 2:
        print("Skipped (only one condition)")
        continue

    # -----------------------------
    # 1) One-way ANOVA
    # -----------------------------
    model = ols("IBI_dur ~ C(cond0)", data=df_z).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    print("\nANOVA:")
    print(anova_table)

    anova_results[ztime] = anova_table

    # -----------------------------
    # 2) Tukey HSD post hoc
    # -----------------------------
    tukey = pairwise_tukeyhsd(
        endog=df_z['IBI_dur'],
        groups=df_z['cond0'],
        alpha=0.05
    )

    print("\nTukey HSD:")
    print(tukey.summary())

    tukey_results[ztime] = tukey.summary()