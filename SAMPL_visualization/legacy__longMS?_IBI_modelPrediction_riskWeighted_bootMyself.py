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
# tv_df_all.to_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')  # smaller file size

tv_df_all = pd.read_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')

#%%
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

#%%
MIN_IBI_PER_BIN = 30  # removes noisy estimates / spikes
MIN_IBI_PER_POSTURE = 200 # removes postures with too few total IBIs
SIGMA_T = 2.0  # in bins

n_boot = 100

boot_hazard = []

for (cond0, ztime), df_sub in tv_df_short.groupby(['cond0', 'ztime'], observed=True):

    # Precompute indices for IBI-level bootstrap
    unique_ibis = df_sub['unique_IBI_idx'].unique()

    for b in tqdm(range(n_boot), desc=f'Bootstrapping {cond0}, {ztime}'):
        # Resample IBIs with replacement
        sampled_ibis = np.random.choice(unique_ibis, size=len(unique_ibis), replace=True)

        # Select all rows corresponding to the sampled IBIs
        df_boot = df_sub[df_sub['unique_IBI_idx'].isin(sampled_ibis)]

        this_hazard_df = (
            df_boot
            .groupby(['ang0_bin', 't_bin'], observed=True)
            .agg(
                n_risk=('unique_IBI_idx', 'nunique'),
                n_event=('event', 'sum')
            )
            .reset_index()
        )
        grouped_hazard = this_hazard_df.groupby('ang0_bin', observed=True)
        this_hazard_df = this_hazard_df[this_hazard_df['n_risk'] >= MIN_IBI_PER_BIN].copy()
        posture_support = grouped_hazard['n_risk'].sum().reset_index(name='n_IBIs')
        valid_postures = posture_support[posture_support['n_IBIs'] >= MIN_IBI_PER_POSTURE]
        this_hazard_df = this_hazard_df.merge(
            valid_postures,
            on=['ang0_bin'],
            how='inner'
        )
        this_hazard_df['hazard'] = this_hazard_df['n_event'] / this_hazard_df['n_risk']
        
        grouped_hazard = this_hazard_df.groupby('ang0_bin', observed=True)
        
        boot_hazard.append(
            this_hazard_df.assign(boot=b, cond0=cond0, ztime=ztime)
        )

boot_hazard_df = pd.concat(boot_hazard, ignore_index=True)
boot_hazard_df['t_mid'] = boot_hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))
boot_hazard_df['hazard_smooth'] = boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True)['hazard'].transform(
    lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest')
)

#%%

#%%
# -----------------------------
# Step 1: Adjust counts with smoothed hazard
# -----------------------------
# Use hazard_smooth to compute fractional events per bin
boot_hazard_df['n_event_smooth'] = boot_hazard_df['hazard_smooth'] * boot_hazard_df['n_risk']

# Optional: add a small pseudo-count to avoid exact 0 or 1
EPS = 1e-3
boot_hazard_df['n_event_smooth'] = np.clip(
    boot_hazard_df['n_event_smooth'], EPS, boot_hazard_df['n_risk'] - EPS
)

boot_hazard_df['total_IBIs'] = boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True)['n_IBIs'].transform('sum')
boot_hazard_df['IBI_ratio'] = boot_hazard_df['n_IBIs'] / boot_hazard_df['total_IBIs']
# -----------------------------
# Step 2: Fit GLM per cond0/ztime
# -----------------------------
results = []
# average but only if enough data points
# Minimum number of data points required per group
min_count = 50

# Compute group size and conditional averages
boot_hazard_df_average_raw = (
    boot_hazard_df
    .groupby(['cond0','ztime','ang0_bin','t_bin'], observed=True)
    .agg(
        n_rows=('n_risk', 'size'),        # count rows in the group
        n_risk=('n_risk', 'mean'),
        n_event=('n_event_smooth', 'mean'),
        hazard=('hazard_smooth', 'mean'),
        IBI_ratio=('IBI_ratio', 'mean'),
    )
    .reset_index()
)

boot_hazard_df_average = boot_hazard_df_average_raw[boot_hazard_df_average_raw['n_rows'] >= min_count]

boot_hazard_df_average['t_mid'] = boot_hazard_df_average['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))


for (cond0, ztime), df in boot_hazard_df_average.groupby(['cond0','ztime'], observed=True):
    df = df.copy()

    # Standardize predictors
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)

    a_mu, a_sd = df['a'].mean(), df['a'].std(ddof=0)
    t_mu, t_sd = df['t'].mean(), df['t'].std(ddof=0)

    df['a_z'] = (df['a'] - a_mu) / a_sd
    df['t_z'] = (df['t'] - t_mu) / t_sd

    # # Interactions
    df['a_t'] = (df['a_z'] * df['t_z']) 
    df['a2_z'] = ((df['a_z']**2 - (df['a_z']**2).mean()) / (df['a_z']**2).std(ddof=0))
    df['a2_t'] = (df['a2_z'] * df['t_z']) 
    # Design matrix
    X = sm.add_constant(df[['a_z','t_z','a_t','a2_z','a2_t']])

    # Calculate failures
    n_fail = df['n_risk'] - df['n_event']

    # Combine into a 2D array: [successes, failures]
    y = np.column_stack([df['n_event'], n_fail])

    # Fit GLM with fractional events
    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
    ).fit()

    results.append({
        'cond0': cond0,
        'ztime': ztime,
        'model': model,
        'a_mu': a_mu,
        'a_sd': a_sd,
        't_mu': t_mu,
        't_sd': t_sd,
        'a2_mu': (df['a_z']**2).mean(),
        'a2_sd': (df['a_z']**2).std(ddof=0)
    })

# -----------------------------
# Step 3: Build predicted heatmaps
# -----------------------------
a_grid = np.sort(boot_hazard_df_average['ang0_bin'].astype(float).unique())
t_grid = np.sort(boot_hazard_df_average['t_mid'].unique())

AA, TT = np.meshgrid(a_grid, t_grid, indexing='ij')

pred_heatmaps_smooth_glm = {}

for r in results:
    model = r['model']

    a = AA.ravel()
    t = TT.ravel()

    a_z = (a - r['a_mu']) / r['a_sd']
    t_z = (t - r['t_mu']) / r['t_sd']
    a2_z = ((a_z**2) - r['a2_mu']) / r['a2_sd']

    Xp = sm.add_constant(
        np.column_stack([a_z, t_z, a_z*t_z, a2_z, a2_z*t_z])
    )

    p_hat = model.predict(Xp)
    Z = p_hat.reshape(AA.shape)

    this_boot_avg = boot_hazard_df_average.loc[(boot_hazard_df_average['cond0'] == r['cond0']) &
                                               (boot_hazard_df_average['ztime'] == r['ztime'])]

    mask_obs = np.zeros_like(AA, dtype=bool)

    for i, a_val in enumerate(a_grid):
        for j, t_val in enumerate(t_grid):
            # Find if this bin was observed
            mask = (
                (this_boot_avg['ang0_bin'].astype(float) == a_val) &
                (this_boot_avg['t_mid'] == t_val)
            )
            if mask.any():
                # Only keep bins with non-zero probability (observed IBI)
                if this_boot_avg.loc[mask, 'IBI_ratio'].values[0] > 0:
                    mask_obs[i, j] = True
                    
    Z_clipped = np.copy(Z)  # original model predictions
    Z_clipped[~mask_obs] = np.nan  # set extrapolated areas to NaN

    pred_heatmaps_smooth_glm[(r['cond0'], r['ztime'])] = Z_clipped
                
for condition, heatmap in pred_heatmaps_smooth_glm.items():

    plt.figure(figsize=(6,4))
    plt.pcolormesh(TT, AA, heatmap, shading='auto')
    plt.xlabel('Time (s)')
    plt.ylabel('Initial posture (ang0)')
    plt.colorbar(label='Predicted P(event)')
    plt.title(f'Cond={condition[0]}, Ztime={condition[1]}')
    # plt.savefig(os.path.join(fig_dir, f'glm_pred_heatmap_cond{condition[0]}_ztime{condition[1]}.pdf'))


# %%

# --- Prepare raw hazard heatmaps ---
raw_heatmaps = {}
for (cond0, ztime), df_sub in boot_hazard_df_average.groupby(['cond0', 'ztime'], observed=True):
    # Create empty heatmap
    Z_raw = np.full(AA.shape, np.nan)
    for i, a_val in enumerate(a_grid):
        for j, t_val in enumerate(t_grid):
            mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
            if mask.any():
                Z_raw[i, j] = df_sub.loc[mask, 'n_event'].values[0] / df_sub.loc[mask, 'n_risk'].values[0]
    raw_heatmaps[(cond0, ztime)] = Z_raw

# --- Plot comparison ---
for condition in pred_heatmaps_smooth_glm.keys():
    Z_glm = pred_heatmaps_smooth_glm[condition]
    Z_raw = raw_heatmaps[condition]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

    # Raw hazard
    im0 = axes[0].pcolormesh(TT, AA, Z_raw, shading='auto', cmap='viridis', vmin=0, vmax=min(np.nanmax(Z_glm), np.nanmax(Z_raw)))
    axes[0].set_title(f'Raw hazard (Cond={condition[0]}, Ztime={condition[1]})')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Initial posture (ang0)')
    fig.colorbar(im0, ax=axes[0], label='P(event)')

    # GLM-predicted hazard
    im1 = axes[1].pcolormesh(TT, AA, Z_glm, shading='auto', cmap='viridis', vmin=0,vmax=min(np.nanmax(Z_glm), np.nanmax(Z_raw)))
    axes[1].set_title('GLM-predicted hazard')
    axes[1].set_xlabel('Time (s)')
    fig.colorbar(im1, ax=axes[1], label='P(event)')

    plt.tight_layout()
    plt.show()
