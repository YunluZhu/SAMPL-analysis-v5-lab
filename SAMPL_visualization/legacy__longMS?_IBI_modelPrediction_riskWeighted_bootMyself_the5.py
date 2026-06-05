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

# tv_df_all.to_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')  # smaller file size

tv_df_all = pd.read_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')

#%%
# add derivatives
tv_df_all['ang_sm'] = tv_df_all.groupby('unique_IBI_idx')['ang'].transform(lambda x: savgol_filter(x.values, 5, 2) if len(x.values) >=5 else x.values)

tv_df_all['angvel'] = tv_df_all.groupby('unique_IBI_idx')['ang_sm'].transform(lambda x: np.gradient(x.values) if len(x.values) >1 else np.nan) * FRAME_RATE
# smooth velocity
tv_df_all['angvel_sm'] = tv_df_all.groupby('unique_IBI_idx')['angvel'].transform(lambda x: savgol_filter(x.values, 3, 2) if len(x.values) >=3 else x.values)
# acceleration
tv_df_all['angacc'] = tv_df_all.groupby('unique_IBI_idx')['angvel_sm'].transform(lambda x: np.gradient(x.values) if len(x.values) >=5 else np.nan)

# remove IBIs with nan derivatives
tv_df_all = tv_df_all[~tv_df_all['ang_sm'].isna()].copy()

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

n_boot = 20

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
                n_event=('event', 'sum'),
                angvel=('angvel', 'mean'),
                angacc=('angacc', 'mean'),
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
# average but only if enough data points
# Minimum number of data points required per group
# min_count = n_boot/2

# # Compute group size and conditional averages
boot_hazard_df_average_raw = (
    boot_hazard_df
    .groupby(['cond0','ztime','ang0_bin','t_bin'], observed=True)
    .agg(
        n_rows=('n_risk', 'size'),        # count rows in the group
        n_risk=('n_risk', 'mean'),
        n_event=('n_event_smooth', 'mean'),
        hazard=('hazard_smooth', 'mean'),
        IBI_ratio=('IBI_ratio', 'mean'),
        angvel=('angvel', 'mean'),
        angacc=('angacc', 'mean'),
    )
    .reset_index()
)

boot_hazard_df['t_mid'] = boot_hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))
boot_hazard_df_average_raw['t_mid'] = boot_hazard_df_average_raw['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))

#%%
model_vars = [
    'a_z', 
    't_z', # 
    # 'a2_z', 
    # 'aABS_z',
    'azABS_z', # haha, this measures deviation from standardized 0 angle 
    # 'vel_z', 
    # 'aABS_t',
    'a_t', # generates time dependence of posture effect
    'vel_t', # prevents late onset 
    # 'azABS_t',
]
    # 'a2_t',
    # 'vel2_z',
    # 'acc_t'

def zscore(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    z = (series - mu) / sd
    return z, mu, sd

results = []
for (boot, cond0, ztime), df in boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True):
    df = df.copy()

    # ---------------------------------
    # Base variables
    # ---------------------------------
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)

    df['a_z'], a_mu, a_sd = zscore(df['a'])
    df['t_z'], t_mu, t_sd = zscore(df['t'])
    df['vel_z'], vel_mu, vel_sd = zscore(df['angvel'])
    df['acc_z'], acc_mu, acc_sd = zscore(df['angacc'])

    # ---------------------------------
    # Quadratic terms (z-scored)
    # ---------------------------------
    df['a2_z'], a2_mu, a2_sd = zscore(df['a_z']**2)
    df['vel2_z'], vel2_mu, vel2_sd = zscore(df['vel_z']**2)
    df['aABS_z'], aABS_mu, aABS_sd = zscore(np.abs(df['a']))
    df['azABS_z'], azABS_mu, azABS_sd = zscore(np.abs(df['a_z']))
    # ---------------------------------
    # Interactions
    # ---------------------------------
    df['a_t']    = df['a_z']   * df['t_z']
    df['a2_t']   = df['a2_z']  * df['t_z']
    df['vel_t']  = df['vel_z'] * df['t_z']
    df['acc_t']  = df['acc_z'] * df['t_z']
    df['aABS_t'] = df['aABS_z'] * df['t_z']
    df['azABS_t'] = df['azABS_z'] * df['t_z']
    # Design matrix
    X = sm.add_constant(df[model_vars])

    # Calculate failures
    n_fail = df['n_risk'] - df['n_event_smooth']

    # Combine into a 2D array: [successes, failures]
    y = np.column_stack([df['n_event_smooth'], n_fail])

    # Fit GLM with fractional events
    model = sm.GLM(
        y,
        X,
        family=sm.families.Binomial(),
    ).fit()

    results.append({
        'boot': boot,
        'cond0': cond0,
        'ztime': ztime,
        'model': model,

        # linear terms
        'a_mu': a_mu,
        'a_sd': a_sd,
        't_mu': t_mu,
        't_sd': t_sd,
        'vel_mu': vel_mu,
        'vel_sd': vel_sd,
        'acc_mu': acc_mu,
        'acc_sd': acc_sd,
        'aABS_mu': aABS_mu,
        'aABS_sd': aABS_sd,
        # quadratic terms
        'a2_mu': a2_mu,
        'a2_sd': a2_sd,
        'vel2_mu': vel2_mu,
        'vel2_sd': vel2_sd,
        'azABS_mu': azABS_mu,
        'azABS_sd': azABS_sd,
    })

# -----------------------------
# Step 3: Build predicted heatmaps
# -----------------------------
a_grid = np.sort(boot_hazard_df_average_raw['ang0_bin'].astype(float).unique())
t_grid = np.sort(boot_hazard_df_average_raw['t_mid'].unique())

AA, TT = np.meshgrid(a_grid, t_grid, indexing='ij')

from collections import defaultdict
pred_heatmaps_smooth_glm = defaultdict(list)


for r in tqdm(results):
    this_boot_avg = boot_hazard_df.loc[
        (boot_hazard_df['cond0'] == r['cond0']) &
        (boot_hazard_df['ztime'] == r['ztime']) &
        (boot_hazard_df['boot'] == r['boot'])
    ].copy()

    # Precompute z-scored covariates at observed bins
    this_boot_avg['a'] = this_boot_avg['ang0_bin'].astype(float)
    this_boot_avg['t'] = this_boot_avg['t_mid'].astype(float)

    this_boot_avg['a_z'] = (this_boot_avg['a'] - r['a_mu']) / r['a_sd']
    this_boot_avg['t_z'] = (this_boot_avg['t'] - r['t_mu']) / r['t_sd']
    this_boot_avg['vel_z'] = (
        (this_boot_avg['angvel'] - r['vel_mu']) / r['vel_sd']
    )
    this_boot_avg['acc_z'] = (
        (this_boot_avg['angacc'] - r['acc_mu']) / r['acc_sd']
    )
    this_boot_avg['a_t'] = this_boot_avg['a_z'] * this_boot_avg['t_z']
    this_boot_avg['vel_t'] = this_boot_avg['vel_z'] * this_boot_avg['t_z']
    this_boot_avg['a2_z'] = (
        (this_boot_avg['a_z']**2 - r['a2_mu']) / r['a2_sd']
    )
    this_boot_avg['vel2_z'] = (
        (this_boot_avg['vel_z']**2 - r['vel2_mu']) / r['vel2_sd']
    )
    this_boot_avg['a2_t'] = this_boot_avg['a2_z'] * this_boot_avg['t_z']
    this_boot_avg['acc_t'] = this_boot_avg['acc_z'] * this_boot_avg['t_z']
    this_boot_avg['aABS_z'] = (
        (np.abs(this_boot_avg['a']) - r['aABS_mu']) / r['aABS_sd']
    )
    this_boot_avg['azABS_z'] = (
        (np.abs(this_boot_avg['a_z']) - r['azABS_mu']) / r['azABS_sd']
    )
    this_boot_avg['aABS_t'] = this_boot_avg['aABS_z'] * this_boot_avg['t_z']
    this_boot_avg['azABS_t'] = this_boot_avg['azABS_z'] * this_boot_avg['t_z']
    Xp = sm.add_constant(
        this_boot_avg[model_vars]
    )

    this_boot_avg['p_hat'] = r['model'].predict(Xp)
    
    mask_obs = np.zeros_like(AA, dtype=bool)

    Z = np.full(AA.shape, np.nan)    
    
    a_to_i = {a: i for i, a in enumerate(a_grid)}
    t_to_j = {t: j for j, t in enumerate(t_grid)}

    for _, row in this_boot_avg.iterrows():
        i = a_to_i[row['a']]
        j = t_to_j[row['t']]
        Z[i, j] = row['p_hat']

    # for i, a_val in enumerate(a_grid):
    #     for j, t_val in enumerate(t_grid):
    #         # Find if this bin was observed
    #         mask = (
    #             (this_boot_avg['ang0_bin'].astype(float) == a_val) &
    #             (this_boot_avg['t_mid'] == t_val)
    #         )
    #         if mask.any():
    #             # Only keep bins with non-zero probability (observed IBI)
    #             if this_boot_avg.loc[mask, 'IBI_ratio'].values[0] > 0:
    #                 mask_obs[i, j] = True
                    
    Z_clipped = Z # drop masking because Z is masked by construction

    pred_heatmaps_smooth_glm[(r['boot'], r['cond0'], r['ztime'])] = Z_clipped

pred_heatmaps_by_cond = defaultdict(list)

for (boot, cond0, ztime), Z in pred_heatmaps_smooth_glm.items():
    pred_heatmaps_by_cond[(cond0, ztime)].append(Z)

pred_heatmaps_mean = {
    (cond0, ztime): np.nanmean(np.stack(Z_list, axis=0), axis=0)
    for (cond0, ztime), Z_list in pred_heatmaps_by_cond.items()
}


# for condition, heatmap in pred_heatmaps_mean.items():

#     plt.figure(figsize=(6,4))
#     plt.pcolormesh(TT, AA, heatmap, shading='auto')
#     plt.xlabel('Time (s)')
#     plt.ylabel('Initial posture (ang0)')
#     plt.colorbar(label='Predicted P(event)')
#     plt.title(f'Cond={condition[0]}, Ztime={condition[1]}')
#     # plt.savefig(os.path.join(fig_dir, f'glm_pred_heatmap_cond{condition[0]}_ztime{condition[1]}.pdf'))


# --- Prepare raw hazard heatmaps ---
raw_heatmaps = {}
for (cond0, ztime), df_sub in boot_hazard_df_average_raw.groupby(['cond0', 'ztime'], observed=True):
    # Create empty heatmap
    Z_raw = np.full(AA.shape, np.nan)
    for i, a_val in enumerate(a_grid):
        for j, t_val in enumerate(t_grid):
            mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
            if mask.any():
                Z_raw[i, j] = df_sub.loc[mask, 'n_event'].values[0] / df_sub.loc[mask, 'n_risk'].values[0]
    raw_heatmaps[(cond0, ztime)] = Z_raw

# --- Plot comparison ---
for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
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
    fig.savefig(os.path.join(fig_dir, f'glm_vs_raw{condition[0]}_{condition[1]}_({model_vars}).pdf'))
    plt.tight_layout()
    plt.show()

# %%

# determine goodness of fit
# RMSE, aka. on average, how far is the GLM from the empirical hazard per bin?
# apply to averaged heatmaps

def rmse(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

def weighted_rmse(Z_raw, Z_glm, W):
    mask = ~np.isnan(Z_raw) & ~np.isnan(Z_glm)
    return np.sqrt(
        np.sum(W[mask] * (Z_raw[mask] - Z_glm[mask])**2) /
        np.sum(W[mask])
    )

weight_heatmaps = {}
for (cond0, ztime), df_sub in boot_hazard_df_average_raw.groupby(['cond0','ztime'], observed=True):
    W = np.full(AA.shape, np.nan)
    for i, a_val in enumerate(a_grid):
        for j, t_val in enumerate(t_grid):
            mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
            if mask.any():
                W[i, j] = df_sub.loc[mask, 'n_risk'].values[0]
    weight_heatmaps[(cond0, ztime)] = W

gof_metrics = []

for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
    Z_raw = raw_heatmaps[condition]
    W = weight_heatmaps[condition]

    gof_metrics.append({
        'cond0': condition[0],
        'ztime': condition[1],
        'rmse': rmse(Z_raw, Z_glm),
        'w_rmse': weighted_rmse(Z_raw, Z_glm, W),
    })

gof_df = pd.DataFrame(gof_metrics)

#%
def corr(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.corrcoef(a[mask], b[mask])[0, 1]

for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
    Z_raw = raw_heatmaps[condition]

    gof_df.loc[
        (gof_df['cond0'] == condition[0]) &
        (gof_df['ztime'] == condition[1]),
        'corr'
    ] = corr(Z_raw, Z_glm)

plt.figure(figsize=(5,4))
sns.pointplot(
    data=gof_df,
    x='ztime',
    y='w_rmse',      # or 'rmse'
    hue='cond0',
    dodge=True,
    errorbar=None,
    markers='o'
)
plt.ylabel('Weighted RMSE (GLM vs raw)')
plt.xlabel('ztime')
plt.title('Goodness of fit across time')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_gof_weightedRMSE_vs_time_({model_vars}).pdf'))

plt.figure(figsize=(5,4))
sns.pointplot(
    data=gof_df,
    x='ztime',
    y='rmse',     
    hue='cond0',
    dodge=True,
    errorbar=None,
    markers='o'
)
plt.ylabel('RMMSE')
plt.xlabel('ztime')
plt.title('Goodness of fit across time')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_gof_RMSE_vs_time_({model_vars}).pdf'))

# for every bin, how far the predicted results are from the empirical results
# scatter plot of predicted vs raw hazard
for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
    Z_raw = raw_heatmaps[condition]

    mask = ~np.isnan(Z_raw) & ~np.isnan(Z_glm)

    plt.figure(figsize=(4,4))
    plt.scatter(Z_raw[mask], Z_glm[mask], s=10, alpha=0.5)
    plt.plot([0, Z_raw[mask].max()], [0, Z_raw[mask].max()], 'k--')
    plt.xlabel('Raw hazard')
    plt.ylabel('GLM-predicted hazard')
    plt.title(f'Cond={condition[0]}, Ztime={condition[1]}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'glm_vs_raw_scatter_cond{condition[0]}_ztime{condition[1]}_({model_vars}).pdf'))

#%%
# repet rmse, but per bootstrapped model

weight_heatmaps_boot = {}
for (boot, cond0, ztime), df_sub in boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True):
    W = np.full(AA.shape, np.nan)
    for i, a_val in enumerate(a_grid):
        for j, t_val in enumerate(t_grid):
            mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
            if mask.any():
                W[i, j] = df_sub.loc[mask, 'n_risk'].values[0]
    weight_heatmaps_boot[(boot, cond0, ztime)] = W

gof_metrics_boot = []

for (boot, cond0, ztime) in weight_heatmaps_boot.keys():
    Z_glm = pred_heatmaps_smooth_glm[(boot, cond0, ztime)]
    Z_raw = raw_heatmaps[(cond0, ztime)]
    W = weight_heatmaps_boot[(boot, cond0, ztime)]

    gof_metrics_boot.append({
        'boot': boot,
        'cond0': cond0,
        'ztime': ztime,
        'rmse': rmse(Z_raw, Z_glm),
        'w_rmse': weighted_rmse(Z_raw, Z_glm, W),
    })

gof_df_boot = pd.DataFrame(gof_metrics_boot)

#%
plt.figure(figsize=(6,4))
sns.catplot(
    data=gof_df_boot,
    x='ztime',
    y='w_rmse',
    hue='cond0',
    kind='point',
    errorbar='sd',
)
plt.ylabel('Weighted RMSE (GLM vs raw)')
plt.xlabel('ztime')
plt.title('Bootstrap distribution of GLM goodness-of-fit')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_gof_weightedRMSE_vs_time_bootstrap_({model_vars}).pdf'))

plt.figure(figsize=(4,4))
sns.scatterplot(
    data=gof_df_boot,
    x='rmse',
    y='w_rmse',
    hue='cond0',
    style='ztime',
    alpha=0.8
)
plt.xlabel('RMSE')
plt.ylabel('Weighted RMSE')
plt.title('Effect of risk weighting')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_gof_RMSE_vs_weightedRMSE_bootstrap_({model_vars}).pdf'))


#%%
coef_rows = []

for r in results:
    params = r['model'].params
    conf = r['model'].conf_int()
    
    for name in params.index:
        coef_rows.append({
            'boot': r['boot'],
            'cond0': r['cond0'],
            'ztime': r['ztime'],
            'param': name,
            'coef': params[name],
            'ci_low': conf.loc[name, 0],
            'ci_high': conf.loc[name, 1],
        })

coef_df = pd.DataFrame(coef_rows)
coef_df = coef_df.query("param != 'const'")
for (cond0, ztime), df_sub in coef_df.groupby(['cond0','ztime'], observed=True):
    plt.figure(figsize=(6,4))
    sns.pointplot(
        data=df_sub,
        x='param',
        y='coef',
        errorbar='sd',
        join=False
    )
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'GLM coefficients (Cond={cond0}, Ztime={ztime})')
    plt.ylabel('Coefficient (log-odds)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'glm_coefficients_cond{cond0}_ztime{ztime}_({model_vars}).pdf'))

sns.catplot(
    data=coef_df,
    x='cond0',
    col='param',
    row='ztime',
    y='coef',
    errorbar='sd',
    kind='point',
    linestyle='none',
    height=3,
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_coefficients_by_param_({model_vars}).pdf'))



#%% ============================================================================
# EXTRACT COEFFICIENTS AND MODEL STATISTICS FROM RESULTS
# ==============================================================================

print("  " + "="*80)
print("EXTRACTING MODEL RESULTS")
print("="*80)

# Extract coefficients from fitted models
coefficient_list = []
aic_list = []

for r in tqdm(results, desc='Extracting coefficients'):
    
    model = r['model']
    boot = r['boot']
    cond0 = r['cond0']
    ztime = r['ztime']
    
    # Store model fit statistics
    aic_list.append({
        'boot': boot,
        'cond0': cond0,
        'ztime': ztime,
        'AIC': model.aic,
        'BIC': model.bic_llf,
        'deviance': model.deviance,
        'num_vars': len(model_vars),
        'selected_vars': model_vars.copy()
    })
    
    # Store coefficients
    for var in model.params.index:
        coefficient_list.append({
            'boot': boot,
            'cond0': cond0,
            'ztime': ztime,
            'variable': var,
            'coefficient': model.params[var],
            'se': model.bse[var],
            'pvalue': model.pvalues[var],
            'significant': model.pvalues[var] < 0.05
        })

# Convert to DataFrames
coef_df_full = pd.DataFrame(coefficient_list)
aic_df_full = pd.DataFrame(aic_list)

print(f"  ✓ Extracted coefficients: {len(coef_df_full)} records")
print(f"✓ Model statistics: {len(aic_df_full)} models")








# %%
cond0, ztime = ('14', 'day')





#%%
def plot_partial_dependence(cond0, ztime, results, boot_hazard_df, model_vars, fig_dir, 
                           save_individual=True, return_pred_data=False):
    """
    Generate partial dependence plots for a specific condition and time period.
    
    Parameters:
    -----------
    cond0 : str
        Condition identifier (e.g., '14')
    ztime : str
        Time period ('day' or 'night')
    results : list
        List of fitted model results
    boot_hazard_df : DataFrame
        Bootstrap hazard data
    model_vars : list
        List of model variable names
    fig_dir : str
        Directory to save figures
    save_individual : bool
        Whether to save individual condition plots
    return_pred_data : bool
        Whether to return prediction data for comparison plots
        
    Returns:
    --------
    dict (optional) : Prediction data if return_pred_data=True
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import statsmodels.api as sm
    import os
    raw_data = raw_heatmaps[(cond0, ztime)]
    max_probability = np.nanmax(raw_data)

    # Get a representative model (first bootstrap)
    condition_results = [r for r in results
                        if r['cond0'] == cond0 and r['ztime'] == ztime]
    
    if not condition_results:
        print(f"Warning: No results found for {cond0}|{ztime}")
        return None
    
    # Use first bootstrap as representative
    r = condition_results[0]
    model = r['model']
    
    # Get data range for this condition
    df_cond = boot_hazard_df[
        (boot_hazard_df['cond0'] == cond0) &
        (boot_hazard_df['ztime'] == ztime)
    ].copy()
    
    # Determine realistic ranges
    a_min, a_max = df_cond['ang0_bin'].astype(float).quantile([0.01, 0.99])
    t_min, t_max = df_cond['t_mid'].astype(float).quantile([0.01, 0.99])
    vel_min, vel_max = df_cond['angvel'].quantile([0.01, 0.99])
    
    print(f"  Data ranges for {cond0}|{ztime}:")
    print(f"  Posture: [{a_min:.1f}, {a_max:.1f}] degrees")
    print(f"  Time: [{t_min:.1f}, {t_max:.1f}] seconds")
    print(f"  Velocity: [{vel_min:.1f}, {vel_max:.1f}] deg/s")
    
    # Create figure only if saving individual plots
    if save_individual or not return_pred_data:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # Reference values (median of data)
    a_ref = df_cond['ang0_bin'].astype(float).median()
    t_ref = df_cond['t_mid'].astype(float).median()
    vel_ref = df_cond['angvel'].median()
    
    # Standardize using stored parameters
    a_z_ref = (a_ref - r['a_mu']) / r['a_sd']
    t_z_ref = (t_ref - r['t_mu']) / r['t_sd']
    vel_z_ref = (vel_ref - r['vel_mu']) / r['vel_sd']
    
    # Initialize storage for prediction data
    pred_data_storage = {}
    
    # -------------------------------------------------------------------------
    # Plot 1: a_z effect (vary posture, hold others at median)
    # -------------------------------------------------------------------------
    a_range = np.linspace(a_min, a_max, 200)
    a_z_range = (a_range - r['a_mu']) / r['a_sd']
    azABS_z_range = (np.abs(a_z_range) - r['azABS_mu']) / r['azABS_sd']
    
    pred_data = pd.DataFrame({
        'a_z': a_z_range,
        't_z': t_z_ref,
        'azABS_z': azABS_z_range,
        'a_t': a_z_range * t_z_ref,
        'vel_t': vel_z_ref * t_z_ref
    })
    
    X_pred = pred_data[model_vars].copy()
    X_pred = sm.add_constant(X_pred, has_constant='add')
    prob_pred_1 = model.predict(X_pred)
    
    pred_data_storage['posture'] = {
        'x': a_range,
        'y': prob_pred_1,
        'x_ref': a_ref
    }
    
    if save_individual or not return_pred_data:
        ax = axes[0, 0]
        ax.plot(a_range, prob_pred_1, linewidth=3.5, color='steelblue')
        ax.axvline(a_ref, color='red', linestyle='--', linewidth=2, alpha=0.5,
                   label=f'Median posture ({a_ref:.1f}°)')
        ax.fill_between(a_range, 0, prob_pred_1, alpha=0.2, color='steelblue')
        ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
        ax.set_title('Posture Effect  (holding time & velocity at median)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_probability+0.1)
    
    # -------------------------------------------------------------------------
    # Plot 2: azABS_z effect AS A FUNCTION OF POSTURE
    # -------------------------------------------------------------------------
    a_range_symmetric = np.linspace(a_min, a_max, 200)
    a_z_range_symmetric = (a_range_symmetric - r['a_mu']) / r['a_sd']
    azABS_z_range_symmetric = (np.abs(a_z_range_symmetric) - r['azABS_mu']) / r['azABS_sd']
    
    pred_data = pd.DataFrame({
        'a_z': 0,
        't_z': t_z_ref,
        'azABS_z': azABS_z_range_symmetric,
        'a_t': 0 * t_z_ref,
        'vel_t': vel_z_ref * t_z_ref
    })
    
    X_pred = pred_data[model_vars].copy()
    X_pred = sm.add_constant(X_pred, has_constant='add')
    prob_pred_2 = model.predict(X_pred)
    
    pred_data_storage['abs_posture'] = {
        'x': a_range_symmetric,
        'y': prob_pred_2
    }
    
    if save_individual or not return_pred_data:
        ax = axes[0, 1]
        ax.plot(a_range_symmetric, prob_pred_2, linewidth=3.5, color='darkgreen')
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
                   label='Straight posture')
        ax.fill_between(a_range_symmetric, 0, prob_pred_2, alpha=0.2, color='darkgreen')
        ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
        ax.set_title('Absolute Posture Effect  (|z-scored posture| deviation, a_z=0)',
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        ax.set_ylim(0, max_probability+0.1)
    
    # -------------------------------------------------------------------------
    # Plot 3: Time effect (t_z)
    # -------------------------------------------------------------------------
    t_range = np.linspace(t_min, t_max, 200)
    t_z_range = (t_range - r['t_mu']) / r['t_sd']
    azABS_z_ref = (np.abs(a_z_ref) - r['azABS_mu']) / r['azABS_sd']
    
    pred_data = pd.DataFrame({
        'a_z': a_z_ref,
        't_z': t_z_range,
        'azABS_z': azABS_z_ref,
        'a_t': a_z_ref * t_z_range,
        'vel_t': vel_z_ref * t_z_range
    })
    
    X_pred = pred_data[model_vars].copy()
    X_pred = sm.add_constant(X_pred, has_constant='add')
    prob_pred_3 = model.predict(X_pred)
    
    pred_data_storage['time'] = {
        'x': t_range,
        'y': prob_pred_3,
        'x_ref': t_ref
    }
    
    if save_individual or not return_pred_data:
        ax = axes[0, 2]
        ax.plot(t_range, prob_pred_3, linewidth=3.5, color='purple')
        ax.axvline(t_ref, color='red', linestyle='--', linewidth=2, alpha=0.5,
                   label=f'Median time ({t_ref:.1f}s)')
        ax.fill_between(t_range, 0, prob_pred_3, alpha=0.2, color='purple')
        ax.set_xlabel('Time into IBI (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
        ax.set_title('Time Effect  (holding posture & velocity at median)',
                     fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_probability+0.1)
    
    # -------------------------------------------------------------------------
    # Plot 4: a_t interaction - How posture effect changes with time
    # -------------------------------------------------------------------------
    if save_individual or not return_pred_data:
        ax = axes[1, 0]
        time_slices = np.linspace(t_min, t_max, 5)
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_slices)))
        
        for t_val, color in zip(time_slices, colors):
            t_z_val = (t_val - r['t_mu']) / r['t_sd']
            azABS_z_range = (np.abs(a_z_range) - r['azABS_mu']) / r['azABS_sd']
            pred_data = pd.DataFrame({
                'a_z': a_z_range,
                't_z': t_z_val,
                'azABS_z': azABS_z_range,
                'a_t': a_z_range * t_z_val,
                'vel_t': vel_z_ref * t_z_val
            })
            X_pred = pred_data[model_vars].copy()
            X_pred = sm.add_constant(X_pred, has_constant='add')
            prob_pred = model.predict(X_pred)
            ax.plot(a_range, prob_pred, linewidth=2.5, color=color,
                    label=f'{t_val:.1f}s', alpha=0.8)
        
        ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
        ax.set_title('a_t: How Posture Effect Changes with Time',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Time into IBI', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_probability+0.1)
    
    # -------------------------------------------------------------------------
    # Plot 5: vel_t interaction - How velocity effect changes with time
    # -------------------------------------------------------------------------
    if save_individual or not return_pred_data:
        ax = axes[1, 1]
        vel_range = np.linspace(vel_min, vel_max, 200)
        vel_z_range = (vel_range - r['vel_mu']) / r['vel_sd']
        
        for t_val, color in zip(time_slices, colors):
            t_z_val = (t_val - r['t_mu']) / r['t_sd']
            azABS_z_ref = (np.abs(a_z_ref) - r['azABS_mu']) / r['azABS_sd']
            pred_data = pd.DataFrame({
                'a_z': a_z_ref,
                't_z': t_z_val,
                'azABS_z': azABS_z_ref,
                'a_t': a_z_ref * t_z_val,
                'vel_t': vel_z_range * t_z_val
            })
            X_pred = pred_data[model_vars].copy()
            X_pred = sm.add_constant(X_pred, has_constant='add')
            prob_pred = model.predict(X_pred)
            ax.plot(vel_range, prob_pred, linewidth=2.5, color=color,
                    label=f'{t_val:.1f}s', alpha=0.8)
        
        ax.axvline(0, color='gray', linestyle=':', linewidth=1.5)
        ax.set_xlabel('Angular Velocity (deg/s)  Nose-down ← 0 → Nose-up',
                      fontsize=12, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
        ax.set_title('vel_t: How Velocity Effect Changes with Time  (Early corrections prioritized)',
                     fontsize=13, fontweight='bold')
        ax.legend(title='Time into IBI', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max_probability+0.1)
    
    # -------------------------------------------------------------------------
    # Plot 6: Heatmap - Time into IBI vs Initial Posture
    # -------------------------------------------------------------------------
    if save_individual or not return_pred_data:
        ax = axes[1, 2]
        
        a_heatmap = np.linspace(a_min, a_max, 100)
        t_heatmap = np.linspace(t_min, t_max, 100)
        A_mesh, T_mesh = np.meshgrid(a_heatmap, t_heatmap)
        
        A_z_mesh = (A_mesh - r['a_mu']) / r['a_sd']
        T_z_mesh = (T_mesh - r['t_mu']) / r['t_sd']
        azABS_z_mesh = (np.abs(A_z_mesh) - r['azABS_mu']) / r['azABS_sd']
        
        n_points = A_mesh.shape[0] * A_mesh.shape[1]
        pred_data_2d = pd.DataFrame({
            'a_z': A_z_mesh.flatten(),
            't_z': T_z_mesh.flatten(),
            'azABS_z': azABS_z_mesh.flatten(),
            'a_t': A_z_mesh.flatten() * T_z_mesh.flatten(),
            'vel_t': vel_z_ref * T_z_mesh.flatten()
        })
        
        X_pred_2d = pred_data_2d[model_vars].copy()
        X_pred_2d = sm.add_constant(X_pred_2d, has_constant='add')
        prob_pred_2d = model.predict(X_pred_2d).values.reshape(A_mesh.shape)
        
        im = ax.imshow(prob_pred_2d.T, aspect='auto', cmap='plasma',
                       origin='lower', extent=[t_min, t_max, a_min, a_max])
        
        contours = ax.contour(prob_pred_2d.T, levels=8, colors='white',
                             linewidths=1.5, alpha=0.6,
                             extent=[t_min, t_max, a_min, a_max])
        ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
        ax.set_ylabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Time into IBI (s)', fontsize=12, fontweight='bold')
        ax.set_title('Posture × Time Heatmap  (velocity at median)',
                     fontsize=13, fontweight='bold')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('P(swim)', fontsize=11, fontweight='bold')
    
    # Save individual plot
    if save_individual:
        plt.suptitle(f'Partial Dependence Plots: {cond0} dpf | {ztime}  ' +
                     f'Showing isolated effect of each variable',
                     fontsize=16, fontweight='bold', y=0.995)
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f'partial_dependence_{cond0}_{ztime}.pdf'),
                    format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, f'partial_dependence_{cond0}_{ztime}.png'),
                    dpi=200, bbox_inches='tight')
        plt.show()
        print(f"  ✓ Saved partial dependence plots for {cond0}|{ztime}")
    
    if return_pred_data:
        return pred_data_storage
    
    return None


# Get all unique combinations of conditions
all_conditions = [(r['cond0'], r['ztime']) for r in results]
unique_conditions = list(set(all_conditions))

print(f"Found {len(unique_conditions)} unique condition combinations:")
for cond0, ztime in sorted(unique_conditions):
    print(f"  - {cond0} dpf, {ztime}")

# Generate plots for each condition
for cond0, ztime in sorted(unique_conditions):
    plot_partial_dependence(
        cond0=cond0,
        ztime=ztime,
        results=results,
        boot_hazard_df=boot_hazard_df,
        model_vars=model_vars,
        fig_dir=fig_dir,
        save_individual=True,
        return_pred_data=False
    )

print("\  " + "="*80)
print("✓ All individual plots generated")
print("="*80)
#%%

# Collect prediction data for all conditions
pred_data_all = {}
for cond0, ztime in sorted(unique_conditions):
    key = (cond0, ztime)
    pred_data_all[key] = plot_partial_dependence(
        cond0=cond0,
        ztime=ztime,
        results=results,
        boot_hazard_df=boot_hazard_df,
        model_vars=model_vars,
        fig_dir=fig_dir,
        save_individual=False,
        return_pred_data=True
    )

# Get unique values
unique_cond0 = sorted(set(c[0] for c in unique_conditions))
unique_ztime = sorted(set(c[1] for c in unique_conditions))

print(f"\  Conditions (cond0): {unique_cond0}")
print(f"Time periods (ztime): {unique_ztime}")

# Define colors for each cond0
cond_colors = {}
color_palette = my_palette
for i, c in enumerate(unique_cond0):
    cond_colors[c] = color_palette[i]

# Create comparison figure: 3 columns (plots 1-3) x 2 rows (ztime)
fig, axes = plt.subplots(len(unique_ztime), 3, figsize=(9, 3*len(unique_ztime)))

# Handle case where only one ztime exists
if len(unique_ztime) == 1:
    axes = axes.reshape(1, -1)

plot_titles = ['Posture Effect', 'Absolute Posture Effect', 'Time Effect']
x_labels = ['Initial Posture (degrees)', 'Initial Posture (degrees)', 'Time into IBI (s)']
plot_keys = ['posture', 'abs_posture', 'time']

for row_idx, ztime in enumerate(unique_ztime):
    for col_idx, (plot_key, title, xlabel) in enumerate(zip(plot_keys, plot_titles, x_labels)):
        ax = axes[row_idx, col_idx]
        
        # Plot each condition with different color
        for cond0 in unique_cond0:
            key = (cond0, ztime)
            if key in pred_data_all and pred_data_all[key] is not None:
                data = pred_data_all[key][plot_key]
                ax.plot(data['x'], data['y'], 
                       linewidth=3, 
                       color=cond_colors[cond0],
                       label=f'{cond0} dpf',
                       alpha=0.8)
                
                # Add reference line if available
                if 'x_ref' in data:
                    ax.axvline(data['x_ref'], 
                              color=cond_colors[cond0], 
                              linestyle='--', 
                              linewidth=1.5, 
                              alpha=0.3)
        
        # Formatting
        ax.set_xlabel(xlabel, fontsize=11, fontweight='bold')
        ax.set_ylabel('Predicted P(swim)', fontsize=11, fontweight='bold')
        ax.set_title(f'{title}\  {ztime}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.6)
        
        # Add special formatting for absolute posture plot
        if plot_key == 'abs_posture':
            ax.axvline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)

plt.suptitle(f'Comparison Across Conditions: Main Effects\  ' +
             f'Model variables: {model_vars}',
             fontsize=14, fontweight='bold', y=1.00)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'partial_dependence_comparison_all_conditions_({model_vars}).png'),
            dpi=200, bbox_inches='tight')
plt.show()

print("\  ✓ Comparison plot saved successfully")
print(f"   Location: {fig_dir}")