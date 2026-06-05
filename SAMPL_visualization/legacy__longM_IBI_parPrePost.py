

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
from scipy.ndimage import uniform_filter1d
from plot_functions.plt_functions import plt_categorical_combined_3

#%%
##### Parameters to change #####

my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]

# Create a Seaborn palette
my_palette = sns.color_palette(my_colors)


pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' or 'night'

# CONSTANTS %%%%%%
BIN_WIDTH = 0.1  # (s)
AVERAGE_BIN = np.arange(0, 2+BIN_WIDTH, BIN_WIDTH)

##### Parameters to change #####

# Define your exponential decay model
def exp_decay(t, v_terminal, tau, v0): 
    return v_terminal + (v0 - v_terminal) * np.exp(-t / tau)

def analyze_two_vars(df, y_cols=("yvel", "angVelSmoothed"), n_grid=500):
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
    cond_exp_groups = df.groupby(["cond0", "expNum"])["unique_IBI_idx"].unique()

    for (cond, expNum), trial_subset in tqdm(cond_exp_groups.items(), desc="Averaging per cond/expNum"):
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
                "expNum": expNum,
                "variable": y_col
            }))

    df_avg_all = pd.concat(avg_list, ignore_index=True)

    # --- Fit exponential decay per variable
    fit_results = []
    for (cond, expNum, var), df_group in tqdm(df_avg_all.groupby(["cond0", "expNum", "variable"]), desc="Fitting"):
        t_vals = df_group["time"].values
        y_vals = df_group["median"].values

        v0_guess = np.median(y_vals[:5])
        v_terminal_guess = np.median(y_vals[-5:])
        tau_guess = (t_vals[-1] - t_vals[0]) / 2
        p0 = [v_terminal_guess, tau_guess, v0_guess]

        try:
            popt, _ = curve_fit(exp_decay, t_vals, y_vals, p0=p0, maxfev=1000)
            fit_results.append({
                "cond0": cond,
                "expNum": expNum,
                "variable": var,
                "v_terminal": popt[0],
                "tau": popt[1],
                "v0": popt[2]
            })
        except RuntimeError:
            fit_results.append({
                "cond0": cond,
                "expNum": expNum,
                "variable": var,
                "v_terminal": np.nan,
                "tau": np.nan,
                "v0": np.nan
            })

    df_fit_all = pd.DataFrame(fit_results)
    return df_avg_all, df_fit_all


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
mask = ((IBI_angles["cond1"].isin(["ld"])) & (IBI_angles["ztime"] == "night")) | (IBI_angles["cond1"].isin(["dd"]))
# mask = (IBI_angles["cond1"].isin(["dd"]))
# cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "velocity", "angAccel","y","headx","heady"]
df_time_filtered = IBI_angles.loc[mask]
# Compute group max (IBI duration) and filter in-place
df_time_filtered["ibi_duration"] = df_time_filtered.groupby(
    group_cols, observed=True
)["time_relative_s"].transform("max")

print("> checkpoint after early filtering")

#%% release memory
del IBI_angles
gc.collect()

#%% 
#NOTE realign IBI by speed thresholding
# --- Define constants and group columns
window = 0.2  # seconds
spd_thresh = 1.0
group_cols = ["expNum", "cond0", "cond1", "ztime", "unique_IBI_idx"]
feature_cols = ["yvel", "angVelSmoothed", "ang", "velocity", "y",'headx','heady']

# --- Sort once for efficient slicing
df_time_filtered = df_time_filtered.sort_values(group_cols + ["time_relative_s"], kind="mergesort")

# # only keep time > 0.5 s
# df_time_filtered2 = df_time_filtered[df_time_filtered['time_relative_s'] > 0.5]
# --- Group object for fast access
gb = df_time_filtered.groupby(group_cols, observed=True, sort=False)

# --- Collect feature rows
rows = []

for keys, subdf in gb:
    t = subdf["time_relative_s"].to_numpy()
    v = subdf["velocity"].to_numpy()
    dur = subdf["ibi_duration"].iloc[0]

    # --- find first frame where abs(velocity) < 1
    below_thresh = np.flatnonzero(np.abs(v) < spd_thresh)
    if len(below_thresh) == 0:
        # fallback if never slows down
        start_time = 0.0
    else:
        start_time = t[below_thresh[0]]

    # --- define start and end masks
    start_mask = (t >= start_time) & (t <= start_time + window)
    end_mask = t >= dur - window

    start_vals = subdf.loc[start_mask, feature_cols].mean().to_numpy(dtype=np.float32)
    end_vals = subdf.loc[end_mask, feature_cols].mean().to_numpy(dtype=np.float32)
    delta_vals = end_vals - start_vals

    rows.append(
        np.concatenate(
            [np.array(keys, dtype=object),
             np.array([dur], dtype=np.float32),
             start_vals, end_vals, delta_vals]
        )
    )

col_names = (
    group_cols
    + ["ibi_duration"]
    + [f"start_{c}" for c in feature_cols]
    + [f"end_{c}" for c in feature_cols]
    + [f"delta_{c}" for c in feature_cols]
)
df_features = pd.DataFrame(rows, columns=col_names)

#%% plotting: select by IBI duration
df_toplt = df_features.query("ibi_duration > 0.5 and ibi_duration < 50")
df_toplt = df_toplt.assign(
    cond1 = 'allDark'
)
#%%
par_sel = 'ang'
# Melt start/end columns into long format
df_long = df_toplt.melt(
    id_vars=['expNum', 'cond0','cond1', 'unique_IBI_idx'], 
    value_vars=[f'start_{par_sel}', f'end_{par_sel}'],
    var_name='timepoint',
    value_name=par_sel
)

# Optional: rename for nicer x-axis
df_long['timepoint'] = df_long['timepoint'].map({f'start_{par_sel}': 'start', f'end_{par_sel}': 'end'})

df_long_avg = df_long.groupby(['cond0', 'cond1','timepoint','expNum'])[par_sel].agg('median').reset_index()

#%
# Plot the points
g = sns.catplot(
    data=df_long_avg,
    x='timepoint',
    y=par_sel,
    hue='cond0',       # optional
    palette=my_palette,
    col='cond1',
    kind='point',      # draws mean + CI by default
    dodge=False,
    height=3,
    aspect=1,
    markers='o',
    linestyles='-',    # connect the points per hue/group mean
    order=['start', 'end']
)
plt.show()


#%%
feature = 'delta_ang'
sns.catplot(
    data=df_toplt.groupby(['cond0', 'cond1','expNum'])[feature].median().reset_index(),
    y=feature,
    x='cond0',
    col='cond1',
    hue="cond0",
    kind="point",
    height=3,
    aspect=1,
    palette=my_palette
)

#%%
par_sel = 'yvel'
# Melt start/end columns into long format
df_long = df_toplt.melt(
    id_vars=['expNum', 'cond0','cond1', 'unique_IBI_idx'], 
    value_vars=[f'start_{par_sel}', f'end_{par_sel}'],
    var_name='timepoint',
    value_name=par_sel
)

# Optional: rename for nicer x-axis
df_long['timepoint'] = df_long['timepoint'].map({f'start_{par_sel}': 'start', f'end_{par_sel}': 'end'})

df_long_avg = df_long.groupby(['cond0', 'cond1','timepoint','expNum'])[par_sel].agg('median').reset_index()

# Plot the points
g = sns.catplot(
    data=df_long_avg,
    x='timepoint',
    y=par_sel,
    hue='cond0',       # optional
    palette=my_palette,
    col='cond1',
    kind='point',      # draws mean + CI by default
    dodge=False,
    height=3,
    aspect=1,
    markers='o',
    linestyles='-',    # connect the points per hue/group mean
    order=['start', 'end']
)
plt.show()

#%
feature = 'delta_y'
sns.catplot(
    data=df_toplt.groupby(['cond0', 'cond1','expNum'])[feature].median().reset_index(),
    y=feature,
    x='cond0',
    col='cond1',
    hue="cond0",
    kind="point",
    height=3,
    aspect=1,
    palette=my_palette
)

#%%
for feature in col_names[len(group_cols):]:
    sns.displot(
        data=df_toplt,
        x=feature,
        row='cond1',
        col='ztime',
        hue="cond0",
        kind="kde",
        fill=True,
        common_norm=False,
        alpha=0.5,
        height=2,
        aspect=1.5,
        
    )
# %% all features
avg_res = df_features.groupby(['cond0', 'cond1', 'ztime', 'expNum'])[col_names[len(group_cols):]].median().reset_index()
#%%
for feature in col_names[len(group_cols):]:
    sns.catplot(
        data=avg_res,
        y=feature,
        x='cond0',
        row='cond1',
        col='ztime',
        hue="cond0",
        kind="point",
        height=2,
        aspect=1.3,
    )
# %%
