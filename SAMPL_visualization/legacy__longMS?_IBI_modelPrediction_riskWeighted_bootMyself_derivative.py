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

tv_df_all['angvel'] = tv_df_all.groupby('unique_IBI_idx')['ang_sm'].transform(lambda x: np.gradient(x.values) if len(x.values) >1 else np.nan)
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
    't_z', # not necessary, if having vel_t and a_t
    # 'a2_z',
    # 'aABS_z',
    'azABS_z', # haha, this measures deviation from standardized 0 angle 
    'vel_z', 
    # 'aABS_t',
    'a_t',  # generates time dependence of posture effect
    'vel_t', # prevents late onset 
    # 'azABS_t',
    # 'a2_t',
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

#%%
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


# for param in model_vars:
#     for cond0 in coef_df['cond0'].unique():
#         for ztime in coef_df['ztime'].unique():
#             df_sub = coef_df.query("param == @param & cond0 == @cond0 & ztime == @ztime")
#             if df_sub.empty:
#                 continue  # skip if no data for this combination

#             plt.figure(figsize=(4,3))
#             sns.histplot(
#                 data=df_sub,
#                 x='coef',
#                 bins=30,
#                 kde=True
#             )
#             plt.axvline(0, color='k', linestyle='--')
#             plt.title(f"{param} | cond0={cond0}, ztime={ztime}")
#             plt.xlabel("Coefficient")
#             plt.ylabel("Count")
#             plt.tight_layout()
#             plt.savefig(
#                 os.path.join(
#                     fig_dir, 
#                     f'glm_coefficient_distribution_{param}_cond{cond0}_z{ztime}.pdf'
#                 )
#             )


# %%
#%% ============================================================================
# EXTRACT COEFFICIENTS AND MODEL STATISTICS FROM RESULTS
# ==============================================================================

print("\n" + "="*80)
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

print(f"\n✓ Extracted coefficients: {len(coef_df_full)} records")
print(f"✓ Model statistics: {len(aic_df_full)} models")



#%% ============================================================================
# QUESTION 2: VARIABLE IMPORTANCE (MULTIPLE CRITERIA)
# ==============================================================================

def analyze_variable_importance_multicriteria(coef_df, aic_df, fig_dir):
    """
    Multi-criteria variable importance analysis
    
    Criteria:
    1. Effect size (median absolute coefficient)
    2. Sign consistency (proportion with same sign)
    3. Statistical significance (proportion p < 0.05)
    4. Stability (inverse coefficient of variation)
    """
    
    print("\n" + "="*80)
    print("QUESTION 2: VARIABLE IMPORTANCE (MULTI-CRITERIA)")
    print("="*80)
    
    coef_analysis = coef_df[coef_df['variable'] != 'const'].copy()
    
    importance_metrics = []
    
    for ztime in ['day', 'night']:
        for var in coef_analysis['variable'].unique():
            
            var_coefs = coef_analysis[
                (coef_analysis['variable'] == var) & 
                (coef_analysis['ztime'] == ztime)
            ]['coefficient']
            
            var_pvals = coef_analysis[
                (coef_analysis['variable'] == var) & 
                (coef_analysis['ztime'] == ztime)
            ]['pvalue']
            
            if len(var_coefs) == 0:
                continue
            
            # Criterion 1: Effect size
            median_abs_coef = np.median(np.abs(var_coefs))
            mean_abs_coef = np.mean(np.abs(var_coefs))
            
            # Criterion 2: Sign consistency
            prop_positive = (var_coefs > 0).mean()
            sign_consistency = max(prop_positive, 1 - prop_positive)
            
            # Criterion 3: Statistical significance
            prop_significant = (var_pvals < 0.05).mean()
            
            # Criterion 4: Stability (inverse CV)
            mean_coef = var_coefs.mean()
            std_coef = var_coefs.std()
            cv = abs(std_coef / mean_coef) if mean_coef != 0 else np.inf
            stability_score = 1 / (1 + cv)
            
            importance_metrics.append({
                'variable': var,
                'ztime': ztime,
                'median_abs_coef': median_abs_coef,
                'mean_abs_coef': mean_abs_coef,
                'median_coef': np.median(var_coefs),
                'sign_consistency': sign_consistency,
                'prop_significant': prop_significant,
                'stability_score': stability_score,
                'cv': cv,
                'n_bootstraps': len(var_coefs)
            })
    
    importance_df = pd.DataFrame(importance_metrics)
    
    # =========================================================================
    # Normalize criteria to 0-1 scale (within each time period)
    # =========================================================================
    
    for ztime in ['day', 'night']:
        mask = importance_df['ztime'] == ztime
        
        for criterion in ['median_abs_coef', 'sign_consistency', 'prop_significant', 'stability_score']:
            values = importance_df.loc[mask, criterion]
            
            if values.max() > values.min():
                normalized = (values - values.min()) / (values.max() - values.min())
            else:
                normalized = 0.5
            
            importance_df.loc[mask, f'{criterion}_norm'] = normalized
    
    # Combined importance score (geometric mean)
    importance_df['importance_score'] = (
        importance_df['median_abs_coef_norm'] *
        importance_df['sign_consistency_norm'] *
        importance_df['prop_significant_norm'] *
        importance_df['stability_score_norm']
    ) ** 0.25
    
    # =========================================================================
    # PLOT 1: Multi-Criteria Radar/Spider Charts
    # =========================================================================
    
    from math import pi
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw=dict(projection='polar'))
    
    criteria = ['Effect Size', 'Sign\nConsistency', 'Statistical\nSignificance', 'Stability']
    criteria_cols = ['median_abs_coef_norm', 'sign_consistency_norm', 
                     'prop_significant_norm', 'stability_score_norm']
    
    for time_idx, ztime in enumerate(['day', 'night']):
        ax = axes[time_idx]
        
        time_data = importance_df[importance_df['ztime'] == ztime].copy()
        time_data = time_data.sort_values('importance_score', ascending=False)
        
        # Number of criteria
        N = len(criteria)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Plot each variable
        colors = plt.cm.Set3(np.linspace(0, 1, len(time_data)))
        
        for idx, (_, row) in enumerate(time_data.iterrows()):
            values = [row[col] for col in criteria_cols]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, 
                   label=row['variable'], color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        # Fix axis
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(criteria, fontsize=11, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_title(f'{ztime.upper()}: Variable Importance Profile\n(Outer = Better)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'importance_radar_charts.pdf'),
               format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'importance_radar_charts.png'),
               dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved: importance_radar_charts.pdf/png")
    
    # =========================================================================
    # PLOT 2: Importance Heatmaps
    # =========================================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    heatmap_configs = [
        ('median_abs_coef', 'Effect Size\n(Median |Coefficient|)', 'Purples', 'max'),
        ('sign_consistency', 'Sign Consistency\n(Max of prop +/-)', 'Blues', 'max'),
        ('prop_significant', 'Statistical Significance\n(Prop. p < 0.05)', 'Greens', 'max'),
        ('stability_score', 'Stability\n(1 / (1 + CV))', 'Oranges', 'max'),
        ('importance_score', 'COMBINED IMPORTANCE', 'Reds', 'max')
    ]
    
    for idx, (metric, title, cmap, vmax_type) in enumerate(heatmap_configs):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        pivot = importance_df.pivot_table(
            index='variable', columns='ztime', values=metric
        )
        
        # Sort by mean
        pivot['mean'] = pivot.mean(axis=1)
        pivot = pivot.sort_values('mean', ascending=False).drop('mean', axis=1)
        
        # Determine vmax
        if vmax_type == 'max':
            vmax = pivot.max().max()
        else:
            vmax = 1.0
        
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap=cmap,
                   vmin=0, vmax=vmax,
                   cbar_kws={'label': metric.replace('_', ' ').title()},
                   ax=ax, linewidths=2.5, linecolor='white',
                   annot_kws={'fontsize': 12, 'fontweight': 'bold'})
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Time Period', fontsize=11, fontweight='bold')
        ax.set_ylabel('Variable' if col == 0 else '', fontsize=11, fontweight='bold')
    
    # Last subplot: Rankings comparison
    ax = axes[1, 2]
    
    day_data = importance_df[importance_df['ztime'] == 'day'].sort_values(
        'importance_score', ascending=True
    )
    night_data = importance_df[importance_df['ztime'] == 'night'].sort_values(
        'importance_score', ascending=True
    )
    
    # Ensure same order for comparison
    all_vars_sorted = importance_df.groupby('variable')['importance_score'].mean().sort_values(ascending=True).index
    
    y_pos = np.arange(len(all_vars_sorted))
    width = 0.35
    
    for var_idx, var in enumerate(all_vars_sorted):
        day_score = day_data[day_data['variable'] == var]['importance_score'].values
        night_score = night_data[night_data['variable'] == var]['importance_score'].values
        
        if len(day_score) > 0:
            ax.barh(var_idx - width/2, day_score[0], width,
                   label='Day' if var_idx == 0 else '', 
                   color='gold', alpha=0.8, edgecolor='black', linewidth=1)
        
        if len(night_score) > 0:
            ax.barh(var_idx + width/2, night_score[0], width,
                   label='Night' if var_idx == 0 else '',
                   color='midnightblue', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_vars_sorted, fontsize=11, fontweight='bold')
    ax.set_xlabel('Combined Importance Score', fontsize=12, fontweight='bold')
    ax.set_title('Day vs Night Rankings', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'importance_multicriteria.pdf'),
               format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'importance_multicriteria.png'),
               dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved: importance_multicriteria.pdf/png")
    
    # =========================================================================
    # PRINT RANKINGS
    # =========================================================================
    
    print("\n" + "="*80)
    print("VARIABLE IMPORTANCE RANKINGS")
    print("="*80)
    
    for ztime in ['day', 'night']:
        print(f"\n{'='*80}")
        print(f"{ztime.upper()} - Multi-Criteria Rankings")
        print('='*80)
        
        time_data = importance_df[importance_df['ztime'] == ztime].sort_values(
            'importance_score', ascending=False
        )
        
        print(f"\n{'Rank':<6} {'Variable':<12} {'Effect':<10} {'Sign':<10} "
              f"{'Sig%':<8} {'Stable':<10} {'SCORE':<10}")
        print("-"*80)
        
        for rank, (_, row) in enumerate(time_data.iterrows(), 1):
            stars = '***' if row['importance_score'] > 0.75 else '**' if row['importance_score'] > 0.5 else '*' if row['importance_score'] > 0.25 else ''
            
            print(f"{rank:<6} {row['variable']:<12} "
                  f"{row['median_abs_coef']:>9.4f} "
                  f"{row['sign_consistency']:>9.3f} "
                  f"{row['prop_significant']*100:>6.1f}% "
                  f"{row['stability_score']:>9.3f} "
                  f"{row['importance_score']:>9.4f} {stars}")
    
    # Save
    importance_df.to_csv(
        os.path.join(fig_dir, 'variable_importance_all_metrics.csv'),
        index=False
    )
    print("\n✓ Saved: variable_importance_all_metrics.csv")
    
    return importance_df



#%%
#%% ============================================================================
# DEVELOPMENTAL ANALYSIS: Pairwise Comparisons (Better for 3 timepoints)
# ==============================================================================

from scipy.stats import mannwhitneyu


def analyze_pairwise_developmental_changes(coef_df, fig_dir):
    """
    With only 3 timepoints, pairwise comparisons are more interpretable than trends
    """
    
    print("\n" + "="*80)
    print("DEVELOPMENTAL ANALYSIS: Pairwise Age Comparisons")
    print("="*80)
    
    coef_analysis = coef_df[coef_df['variable'] != 'const'].copy()
    coef_analysis['age'] = coef_analysis['cond0'].astype(int)
    
    variables = sorted(coef_analysis['variable'].unique())
    ages = sorted(coef_analysis['age'].unique())
    
    # Define pairwise comparisons
    comparisons = [
        (ages[0], ages[1], f'{ages[0]}→{ages[1]} dpf'),
        (ages[1], ages[2], f'{ages[1]}→{ages[2]} dpf'),
        (ages[0], ages[2], f'{ages[0]}→{ages[2]} dpf (overall)')
    ]
    
    
    # =========================================================================
    # Statistical Tests
    # =========================================================================
    
    pairwise_results = []
    
    for var in variables:
        for ztime in ['day', 'night']:
            
            var_data = coef_analysis[
                (coef_analysis['variable'] == var) &
                (coef_analysis['ztime'] == ztime)
            ]
            
            for age1, age2, label in comparisons:
                
                coefs1 = var_data[var_data['age'] == age1]['coefficient']
                coefs2 = var_data[var_data['age'] == age2]['coefficient']
                
                if len(coefs1) > 0 and len(coefs2) > 0:
                    
                    med1 = coefs1.median()
                    med2 = coefs2.median()
                    change = med2 - med1
                    pct_change = (change / abs(med1) * 100) if med1 != 0 else np.inf
                    
                    # Mann-Whitney U test
                    stat, pval = mannwhitneyu(coefs1, coefs2, alternative='two-sided')
                    
                    # Effect size: Cohen's d
                    pooled_std = np.sqrt((coefs1.var() + coefs2.var()) / 2)
                    cohens_d = (med2 - med1) / pooled_std if pooled_std > 0 else 0
                    
                    pairwise_results.append({
                        'variable': var,
                        'ztime': ztime,
                        'comparison': label,
                        'age1': age1,
                        'age2': age2,
                        'median_age1': med1,
                        'median_age2': med2,
                        'change': change,
                        'percent_change': pct_change,
                        'pvalue': pval,
                        'cohens_d': cohens_d,
                        'significant': pval < 0.05
                    })
    
    pairwise_df = pd.DataFrame(pairwise_results)
    
    # =========================================================================
    # PLOT 1: Change Heatmaps
    # =========================================================================
    
    fig, axes = plt.subplots(len(comparisons), 2, figsize=(16, 5*len(comparisons)))
    if len(comparisons) == 1:
        axes = axes.reshape(1, -1)
    
    for comp_idx, (age1, age2, label) in enumerate(comparisons):
        
        for time_idx, ztime in enumerate(['day', 'night']):
            ax = axes[comp_idx, time_idx]
            
            # Get data for this comparison and time
            comp_data = pairwise_df[
                (pairwise_df['comparison'] == label) &
                (pairwise_df['ztime'] == ztime)
            ].copy()
            
            if len(comp_data) == 0:
                continue
            
            # Sort by change magnitude
            comp_data = comp_data.sort_values('change', ascending=True)
            
            # Create bar plot
            colors = ['crimson' if c > 0 else 'steelblue' for c in comp_data['change']]
            bars = ax.barh(range(len(comp_data)), comp_data['change'],
                          color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add significance markers
            for i, (_, row) in enumerate(comp_data.iterrows()):
                if row['significant']:
                    marker = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*'
                    x_pos = row['change'] + (0.01 if row['change'] > 0 else -0.01)
                    ha = 'left' if row['change'] > 0 else 'right'
                    ax.text(x_pos, i, marker, ha=ha, va='center',
                           fontsize=14, fontweight='bold', color='black')
            
            ax.set_yticks(range(len(comp_data)))
            ax.set_yticklabels(comp_data['variable'], fontsize=11, fontweight='bold')
            ax.set_xlabel('Coefficient Change', fontsize=12, fontweight='bold')
            ax.set_title(f'{ztime.upper()}: {label}\n(Red=Increase, Blue=Decrease)',
                        fontsize=13, fontweight='bold')
            ax.axvline(0, color='black', linewidth=2)
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (_, row) in enumerate(comp_data.iterrows()):
                label_text = f'{row["change"]:+.3f}'
                x_pos = row['change'] + (0.02 if row['change'] > 0 else -0.02)
                ha = 'left' if row['change'] > 0 else 'right'
                ax.text(x_pos, i, label_text, ha=ha, va='center',
                       fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'developmental_pairwise_changes.pdf'),
               format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(fig_dir, 'developmental_pairwise_changes.png'),
               dpi=200, bbox_inches='tight')
    plt.show()
    
    print("\n✓ Saved: developmental_pairwise_changes.pdf/png")
    
    # =========================================================================
    # Print Summary
    # =========================================================================
    
    print("\n" + "="*80)
    print("PAIRWISE DEVELOPMENTAL CHANGES")
    print("="*80)
    
    for ztime in ['day', 'night']:
        print(f"\n{'='*80}")
        print(f"{ztime.upper()}")
        print('='*80)
        
        time_data = pairwise_df[pairwise_df['ztime'] == ztime]
        
        for age1, age2, label in comparisons:
            print(f"\n{label}:")
            print("-"*80)
            
            comp_data = time_data[time_data['comparison'] == label].sort_values(
                'change', key=abs, ascending=False
            )
            
            print(
                f"{'Variable':<12} {'Age1':<10} {'Age2':<10} {'Change':<10} "
                f"{'%Change':<10} {'Cohens d':<12} {'p-value':<10} {'Sig':<6}"
            )
            print("-"*80)
            
            for _, row in comp_data.iterrows():
                sig = '***' if row['pvalue'] < 0.001 else '**' if row['pvalue'] < 0.01 else '*' if row['pvalue'] < 0.05 else ''
                
                print(f"{row['variable']:<12} "
                      f"{row['median_age1']:>9.4f} "
                      f"{row['median_age2']:>9.4f} "
                      f"{row['change']:>9.4f} "
                      f"{row['percent_change']:>9.1f}% "
                      f"{row['cohens_d']:>11.3f} "
                      f"{row['pvalue']:>9.4f} {sig:<6}")
    
    pairwise_df.to_csv(os.path.join(fig_dir, 'developmental_pairwise_stats.csv'), index=False)
    print("\n✓ Saved: developmental_pairwise_stats.csv")
    
    return pairwise_df

# Question 1: Pairwise developmental comparisons (better for 3 timepoints)
pairwise_stats = analyze_pairwise_developmental_changes(coef_df_full, fig_dir)


#%% ============================================================================
# ISSUE 2: MODEL-BASED VARIABLE IMPORTANCE (Leave-One-Out)
# ==============================================================================

def calculate_variable_importance_by_exclusion(boot_hazard_df, model_vars, 
                                               raw_heatmaps, weight_heatmaps,
                                               a_grid, t_grid, fig_dir,
                                               n_boots_sample=20):
    """
    Calculate variable importance by fitting models with each variable excluded
    Measures importance as: degradation in model fit when variable is removed
    
    This is like permutation importance but more principled for GLMs
    """
    
    print("\n" + "="*80)
    print("VARIABLE IMPORTANCE: Leave-One-Out Analysis")
    print("="*80)
    print("This will take a while - fitting models with each variable excluded...")
    
    from itertools import combinations
    
    def zscore(series):
        mu = series.mean()
        sd = series.std(ddof=0)
        z = (series - mu) / sd
        return z, mu, sd
    
    def create_features(df):
        """Create all features (same as your main code)"""
        df = df.copy()
        df['a'] = df['ang0_bin'].astype(float)
        df['t'] = df['t_mid'].astype(float)
        
        df['a_z'], a_mu, a_sd = zscore(df['a'])
        df['t_z'], t_mu, t_sd = zscore(df['t'])
        df['vel_z'], vel_mu, vel_sd = zscore(df['angvel'])
        df['acc_z'], acc_mu, acc_sd = zscore(df['angacc'])
        
        df['a2_z'], a2_mu, a2_sd = zscore(df['a_z']**2)
        df['vel2_z'], vel2_mu, vel2_sd = zscore(df['vel_z']**2)
        df['aABS_z'], aABS_mu, aABS_sd = zscore(np.abs(df['a']))
        df['azABS_z'], azABS_mu, azABS_sd = zscore(np.abs(df['a_z']))
        
        df['a_t'] = df['a_z'] * df['t_z']
        df['a2_t'] = df['a2_z'] * df['t_z']
        df['vel_t'] = df['vel_z'] * df['t_z']
        df['acc_t'] = df['acc_z'] * df['t_z']
        df['aABS_t'] = df['aABS_z'] * df['t_z']
        df['azABS_t'] = df['azABS_z'] * df['t_z']
        
        return df
    
    def calculate_wrmse(obs_grid, pred_grid, weight_grid):
        """Weighted RMSE"""
        mask = ~np.isnan(obs_grid) & ~np.isnan(pred_grid) & ~np.isnan(weight_grid)
        if not mask.any():
            return np.nan
        diff_sq = (obs_grid[mask] - pred_grid[mask]) ** 2
        weights = weight_grid[mask]
        return np.sqrt(np.average(diff_sq, weights=weights))
    
    # =========================================================================
    # Fit models with each variable excluded
    # =========================================================================
    
    importance_results = []
    
    for (cond0, ztime), df_cond in boot_hazard_df.groupby(['cond0', 'ztime'], observed=True):
        
        print(f"\n{cond0} dpf | {ztime}")
        print("-"*50)
        
        # Sample bootstraps for speed
        available_boots = df_cond['boot'].unique()
        boots_to_use = np.random.choice(available_boots, 
                                       size=min(n_boots_sample, len(available_boots)), 
                                       replace=False)
        
        # Store predictions for each variable exclusion
        predictions_by_exclusion = {var: [] for var in model_vars}
        predictions_full = []
        
        for boot in tqdm(boots_to_use, desc=f'{cond0}|{ztime}', leave=False):
            
            df_boot = df_cond[df_cond['boot'] == boot].copy()
            df_boot = create_features(df_boot)
            
            if len(df_boot) < 20:
                continue
            
            y = np.column_stack([df_boot['n_event_smooth'],
                                df_boot['n_risk'] - df_boot['n_event_smooth']])
            
            # Build prediction grid lookup
            a_to_i = {a: i for i, a in enumerate(a_grid)}
            t_to_j = {t: j for j, t in enumerate(t_grid)}
            
            # ----------------------------------------------------------------
            # 1. Full model (all variables)
            # ----------------------------------------------------------------
            X_full = sm.add_constant(df_boot[model_vars])
            model_full = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
            
            # Predict
            pred_full = model_full.predict(X_full)
            
            Z_full = np.full((len(a_grid), len(t_grid)), np.nan)
            
            # Create lookup dictionaries
            a_to_i = {a: i for i, a in enumerate(a_grid)}
            t_to_j = {t: j for j, t in enumerate(t_grid)}
            
            # Fill grid
            for idx in range(len(df_boot)):
                a_val = df_boot.iloc[idx]['ang0_bin']  # Already float
                t_val = df_boot.iloc[idx]['t_mid']
                
                if a_val in a_to_i and t_val in t_to_j:
                    i = a_to_i[a_val]
                    j = t_to_j[t_val]
                    Z_full[i, j] = pred_full.iloc[idx]
                    
            predictions_full.append(Z_full)
            
            full_deviance = model_full.deviance
            
            
            # ----------------------------------------------------------------
            # 2. Models with each variable excluded
            # ----------------------------------------------------------------
            for excluded_var in model_vars:
                
                # Variables to include (all except excluded)
                vars_subset = [v for v in model_vars if v != excluded_var]
                
                if len(vars_subset) == 0:
                    continue
            
                X_subset = sm.add_constant(df_boot[vars_subset])
                model_subset = sm.GLM(y, X_subset, family=sm.families.Binomial()).fit()
                
                # Predict
                pred_subset = model_subset.predict(X_subset)
                
                # Build grid
                
                Z_subset = np.full((len(a_grid), len(t_grid)), np.nan)
                
                # Create lookup dictionaries
                a_to_i = {a: i for i, a in enumerate(a_grid)}
                t_to_j = {t: j for j, t in enumerate(t_grid)}
                
                # Fill grid
                for idx in range(len(df_boot)):
                    a_val = df_boot.iloc[idx]['ang0_bin']  # Already float
                    t_val = df_boot.iloc[idx]['t_mid']
                    
                    if a_val in a_to_i and t_val in t_to_j:
                        i = a_to_i[a_val]
                        j = t_to_j[t_val]
                        Z_subset[i, j] = pred_subset.iloc[idx]
                        
                predictions_by_exclusion[excluded_var].append(Z_subset)
                
                subset_deviance = model_subset.deviance
                
                # Calculate importance metrics
                deviance_increase = subset_deviance - full_deviance
                aic_increase = model_subset.aic - model_full.aic
        
        # ====================================================================
        # Calculate importance scores
        # ====================================================================
        
        if len(predictions_full) > 0:
            
            # Average predictions
            Z_full_mean = np.nanmean(np.array(predictions_full), axis=0)
            Z_obs = raw_heatmaps.get((cond0, ztime))
            Z_weight = weight_heatmaps.get((cond0, ztime))
            
            if Z_obs is not None and Z_weight is not None:
                
                # Full model RMSE
                rmse_full = calculate_wrmse(Z_obs, Z_full_mean, Z_weight)
                
                for excluded_var in model_vars:
                    
                    if len(predictions_by_exclusion[excluded_var]) > 0:
                        
                        # Average predictions without this variable
                        Z_subset_mean = np.nanmean(
                            np.array(predictions_by_exclusion[excluded_var]), axis=0
                        )
                        
                        # RMSE without this variable
                        rmse_subset = calculate_wrmse(Z_obs, Z_subset_mean, Z_weight)
                        
                        # Importance = degradation when removed
                        rmse_increase = rmse_subset - rmse_full
                        rmse_increase_pct = (rmse_increase / rmse_full * 100) if rmse_full > 0 else 0
                        
                        importance_results.append({
                            'cond0': cond0,
                            'ztime': ztime,
                            'variable': excluded_var,
                            'rmse_full': rmse_full,
                            'rmse_without': rmse_subset,
                            'rmse_increase': rmse_increase,
                            'rmse_increase_pct': rmse_increase_pct,
                            'n_bootstraps': len(predictions_by_exclusion[excluded_var])
                        })
    
    importance_loo_df = pd.DataFrame(importance_results)
    
    # =========================================================================
    # PLOT: Leave-One-Out Importance
    # =========================================================================
    
    if len(importance_loo_df) > 0:
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        for time_idx, ztime in enumerate(['day', 'night']):
            ax = axes[time_idx]
            
            time_data = importance_loo_df[importance_loo_df['ztime'] == ztime].copy()
            
            # Average across ages
            avg_importance = time_data.groupby('variable')['rmse_increase_pct'].mean().sort_values(ascending=True)
            
            colors = plt.cm.YlOrRd(avg_importance.values / avg_importance.max())
            bars = ax.barh(range(len(avg_importance)), avg_importance.values,
                          color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            
            ax.set_yticks(range(len(avg_importance)))
            ax.set_yticklabels(avg_importance.index, fontsize=12, fontweight='bold')
            ax.set_xlabel('RMSE Increase When Removed (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{ztime.upper()}: Variable Importance\n(Higher = More Important)',
                        fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='x')
            
            # Add value labels
            for i, (var, val) in enumerate(avg_importance.items()):
                ax.text(val + 0.5, i, f'{val:.1f}%', va='center',
                       fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'importance_leave_one_out.pdf'),
                   format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(fig_dir, 'importance_leave_one_out.png'),
                   dpi=200, bbox_inches='tight')
        plt.show()
        
        print("\n✓ Saved: importance_leave_one_out.pdf/png")
        
        # Print rankings
        print("\n" + "="*80)
        print("LEAVE-ONE-OUT IMPORTANCE RANKINGS")
        print("="*80)
        
        for ztime in ['day', 'night']:
            print(f"\n{ztime.upper()}:")
            print("-"*80)
            
            time_data = importance_loo_df[importance_loo_df['ztime'] == ztime]
            avg_by_var = time_data.groupby('variable')['rmse_increase_pct'].mean().sort_values(ascending=False)
            
            print(f"{'Rank':<6} {'Variable':<12} {'RMSE Increase':<16} {'Interpretation':<30}")
            print("-"*80)
            
            for rank, (var, increase) in enumerate(avg_by_var.items(), 1):
                if increase > 10:
                    interp = "⭐ CRITICAL"
                elif increase > 5:
                    interp = "Important"
                elif increase > 2:
                    interp = "Moderate"
                else:
                    interp = "Minor"
                
                print(f"{rank:<6} {var:<12} {increase:>13.2f}% {interp:<30}")
        
        importance_loo_df.to_csv(os.path.join(fig_dir, 'importance_leave_one_out_data.csv'), 
                                index=False)
        print("\n✓ Saved: importance_leave_one_out_data.csv")
    
    return importance_loo_df



#%% ============================================================================
# RUN ALL ANALYSES
# ==============================================================================

# Question 2a: Coefficient-based importance
importance_coef = analyze_variable_importance_multicriteria(coef_df_full, aic_df_full, fig_dir)

# Question 2b: Model-based importance (leave-one-out)
importance_loo = calculate_variable_importance_by_exclusion(
    boot_hazard_df, 
    model_vars,
    raw_heatmaps,
    weight_heatmaps,
    a_grid,
    t_grid,
    fig_dir,
    n_boots_sample=20  # Use 20 bootstraps for speed
)

#%%
sns.catplot(
    data=importance_loo,
    x='variable',
    y='rmse_increase_pct',
    col='ztime',
    hue='cond0',
    kind='point',
    palette=my_palette,
    height=4
)


# %%
