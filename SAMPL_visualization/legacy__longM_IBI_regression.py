

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
mask = (IBI_angles["cond1"].isin(["ld", "dd"])) #& (IBI_angles["ztime"] == "day")
# mask = (IBI_angles["cond1"].isin(["dd"]))
cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "y"]
df_time_filtered = IBI_angles.loc[mask, cols_needed]
# Compute group max (IBI duration) and filter in-place
df_time_filtered["ibi_duration"] = df_time_filtered.groupby(
    group_cols, observed=True
)["time_relative_s"].transform("max")

print("> checkpoint after early filtering")

#%% release memory
del IBI_angles
gc.collect()
#%% 10 sec
# compute IBI duration and filter
min_bin = AVERAGE_BIN[-1]  # upper edge of last bin

# Keep only rows from valid IBIs
df_valid = df_time_filtered[df_time_filtered["ibi_duration"] >= min_bin].copy()

# 1. Compute bin centers
bin_centers = (AVERAGE_BIN[:-1] + AVERAGE_BIN[1:]) / 2

# 2. Assign each point to a bin index
bin_idx = np.digitize(df_valid["time_relative_s"], AVERAGE_BIN) - 1

# 3. Keep only points that fall within your defined bins
mask = (bin_idx >= 0) & (bin_idx < len(bin_centers))

# 4. Assign the middle value of the bin
df_valid.loc[mask, "time_bin"] = bin_centers[bin_idx[mask]]

df_valid = df_valid.loc[mask].reset_index(drop=True)

print("> checkpoint after final filtering; df_valid ready")


#%% further cleaning: max abs yvel after 1s < 1 mm/s
# 1. Flag rows after 1 seconds
after_1s_mask = df_valid["time_relative_s"] > 1

# 2. Compute max absolute yvel after 1s per IBI
max_abs_yvel_after1 = (
    df_valid.loc[after_1s_mask]
    .groupby("unique_IBI_idx", observed=True)["yvel"]
    .agg(lambda x: np.max(np.abs(x)))
)

mean_angvel_after1 = (
    df_valid.loc[after_1s_mask]
    .groupby("unique_IBI_idx", observed=True)["angVelSmoothed"]
    .agg(lambda x: np.percentile(x, 75))
)

max_angvel_after1 = (
    df_valid.loc[after_1s_mask]
    .groupby("unique_IBI_idx", observed=True)["angVelSmoothed"]
    .agg('max')
)

# 3. Get IBIs that satisfy condition
valid_IBI_idx1 = max_abs_yvel_after1[max_abs_yvel_after1 < 1].index
# valid_IBI_idx2 = mean_angvel_after2[mean_angvel_after2 < 0].index
valid_IBI_idx2 = max_angvel_after1[max_angvel_after1 < 20].index
valid_IBI_idx = valid_IBI_idx1.intersection(valid_IBI_idx2)

# 4. Filter original dataframe
df_filtered = df_valid[df_valid["unique_IBI_idx"].isin(valid_IBI_idx)].copy()

print("> checkpoint after further filtering; df_filtered ready")

#%%
# --- Step 3: group and take mean per bin ---
binned_df_cond = (
    df_filtered
    .groupby(group_cols + ["time_bin", "ztime"], observed=True, as_index=False)
    [["time_relative_s", "yvel", "angVelSmoothed"]]
    .median()
)

print("> check point after binning")

binned_df_averaged = binned_df_cond.groupby(['cond0', 'cond1', 'expNum','time_bin'], observed=True)[['yvel','angVelSmoothed']].median().reset_index()

#%% plotting
#%%
ypar = 'yvel'

gridcol = None
gridrow = 'cond1'

g = sns.relplot(
    kind='line',
    data=binned_df_averaged,
    x='time_bin',
    y=ypar,
    hue='cond0',
    col=gridcol,
    # units='expNum',
    # estimator=None,
    row=gridrow,
    aspect=3,
    height=2,
    facet_kws={'sharey': True},
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.20, binned_df_averaged.time_bin.max()+0.2))
filename = os.path.join(fig_dir,f"IBI time series by exp rep {ypar}_{gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')

#%%

ypar = 'angVelSmoothed'

gridcol = None
gridrow = 'cond1'

g = sns.relplot(
    kind='line',
    data=binned_df_averaged,
    x='time_bin',
    y=ypar,
    hue='cond0',
    col=gridcol,
    # units='expNum',
    # estimator=None,
    row=gridrow,
    aspect=3,
    height=2,
    facet_kws={'sharey': True},
    
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.20, binned_df_averaged.time_bin.max()+0.2))
filename = os.path.join(fig_dir,f"IBI time series by exp rep {ypar}_{gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')


#%%
# modeling with exponential decay
# mask = (df_filtered["cond1"].isin(["ld"]))
df_tomodel_all = df_filtered.copy()#.loc[mask].copy()

#%%
print(df_tomodel_all.groupby('unique_IBI_idx').head(1).groupby(['cond0','cond1','expNum'], observed=True).size())

df_avg_all, df_fit_all = analyze_two_vars(df_tomodel_all, n_grid=500)

#%%

g = sns.FacetGrid(df_avg_all, row="variable", col="cond0", hue="expNum", sharex=True, sharey='row')
g.map_dataframe(sns.lineplot, x="time", y="median")
g.add_legend()
sns.despine()


# %%
for var in df_fit_all['variable'].unique():
    print(f"Plotting fits for variable: {var}")
    
    plt_categorical_combined_3(
        df_fit_all[df_fit_all['variable'] == var],
        x='cond0',
        y='v0',
        hue=None,
        units='expNum',
        errorbar='se',
    )
    
    plt_categorical_combined_3(
        df_fit_all[df_fit_all['variable'] == var],
        x='cond0',
        y='tau',
        hue=None,
        units='expNum',
        errorbar='se',
    )
    
    plt_categorical_combined_3(
        df_fit_all[df_fit_all['variable'] == var],
        x='cond0',
        y='v_terminal',
        hue=None,
        units='expNum',
        errorbar='se',
    )

#%%
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Parameters to test
params = ["tau", "v_terminal", "v0"]
variables = df_fit_all["variable"].unique()

anova_results = {}
tukey_results = {}

for var in variables:
    df_var = df_fit_all[df_fit_all["variable"] == var].dropna(subset=params)
    print(f"\n=== Variable: {var} ===")
    anova_results[var] = {}
    tukey_results[var] = {}
    
    for param in params:
        print(f"\n--- ANOVA for {param} ---")
        df_param = df_var.dropna(subset=[param])
        if df_param["cond0"].nunique() < 2:
            print(f"  Skipped (only one condition for {var})")
            continue
        
        # 1. One-way ANOVA
        model = ols(f"{param} ~ C(cond0)", data=df_param).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        print(anova_table)
        anova_results[var][param] = anova_table

        # 2. Tukey’s HSD for post hoc comparison
        tukey = pairwise_tukeyhsd(
            endog=df_param[param],
            groups=df_param["cond0"],
            alpha=0.05
        )
        print("\nTukey HSD:")
        print(tukey.summary())
        tukey_results[var][param] = tukey.summary()

# %%
