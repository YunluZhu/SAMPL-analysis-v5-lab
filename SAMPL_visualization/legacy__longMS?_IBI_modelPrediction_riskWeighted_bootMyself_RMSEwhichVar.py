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
    
def zscore(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    z = (series - mu) / sd
    return z, mu, sd

def rmse(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    return np.sqrt(np.mean((a[mask] - b[mask])**2))

def weighted_rmse(Z_raw, Z_glm, W):
    mask = ~np.isnan(Z_raw) & ~np.isnan(Z_glm)
    return np.sqrt(
        np.sum(W[mask] * (Z_raw[mask] - Z_glm[mask])**2) /
        np.sum(W[mask])
    )
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
from itertools import combinations

a_grid = np.sort(boot_hazard_df_average_raw['ang0_bin'].astype(float).unique())
t_grid = np.sort(boot_hazard_df_average_raw['t_mid'].unique())

AA, TT = np.meshgrid(a_grid, t_grid, indexing='ij')

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
    

all_vars = ['t_z', 'a2_z', 'vel_z', 'a_t', 'vel_t', 'aABS_z', 'aABS_t']

# Generate all subsets (except empty)
candidate_models = []
for k in range(1, len(all_vars)+1):
    candidate_models.extend(combinations(all_vars, k))

gof_results = []

#check colinearity

for (cond0, ztime), df_boot in tqdm(boot_hazard_df.groupby(['cond0','ztime'], observed=True)):
    df_boot = df_boot.copy()
    
    # Z-score covariates
    df_boot['a'] = df_boot['ang0_bin'].astype(float)
    df_boot['t'] = df_boot['t_mid'].astype(float)
    df_boot['a_z'], a_mu, a_sd = zscore(df_boot['a'])
    df_boot['t_z'], t_mu, t_sd = zscore(df_boot['t'])
    df_boot['vel_z'], vel_mu, vel_sd = zscore(df_boot['angvel'])
    df_boot['acc_z'], acc_mu, acc_sd = zscore(df_boot['angacc'])
    df_boot['a2_z'], a2_mu, a2_sd = zscore(df_boot['a_z']**2)
    df_boot['vel2_z'], vel2_mu, vel2_sd = zscore(df_boot['vel_z']**2)
    df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
    df_boot['a2_t'] = df_boot['a2_z'] * df_boot['t_z']
    df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
    df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
    df_boot['aABS_z'], aABS_mu, aABS_sd = zscore(np.abs(df_boot['a_z']))
    df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
    

    plt.figure(figsize=(8,6))
    g = sns.heatmap(df_boot[all_vars].corr(), annot=True, cmap='coolwarm',
                vmin=-1, vmax=1)
    plt.title(f'Correlation matrix: {cond0}, {ztime}')
    plt.savefig(os.path.join(fig_dir, f'correlation_matrix_{cond0}_{ztime}.pdf'))

#%%
    
# Loop through candidate models
for subset in tqdm(candidate_models):
    subset = list(subset)
    
    for (boot, cond0, ztime), df_boot in boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True):
        df_boot = df_boot.copy()
        
        # Z-score covariates
        df_boot['a'] = df_boot['ang0_bin'].astype(float)
        df_boot['t'] = df_boot['t_mid'].astype(float)
        df_boot['a_z'], a_mu, a_sd = zscore(df_boot['a'])
        df_boot['t_z'], t_mu, t_sd = zscore(df_boot['t'])
        df_boot['vel_z'], vel_mu, vel_sd = zscore(df_boot['angvel'])
        df_boot['acc_z'], acc_mu, acc_sd = zscore(df_boot['angacc'])
        df_boot['a2_z'], a2_mu, a2_sd = zscore(df_boot['a_z']**2)
        df_boot['vel2_z'], vel2_mu, vel2_sd = zscore(df_boot['vel_z']**2)
        df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
        df_boot['a2_t'] = df_boot['a2_z'] * df_boot['t_z']
        df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
        df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
        df_boot['aABS_z'], aABS_mu, aABS_sd = zscore(np.abs(df_boot['a_z']))
        df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
    
        # -----------------------------
        # Fit GLM
        # -----------------------------
        X = sm.add_constant(df_boot[subset])
        y = np.column_stack([df_boot['n_event_smooth'], df_boot['n_risk'] - df_boot['n_event_smooth']])
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        
        # -----------------------------
        # Predict and map into heatmap
        # -----------------------------
        df_boot['p_hat'] = model.predict(X)
        
        Z_pred = np.full(AA.shape, np.nan)
        W = np.full(AA.shape, np.nan)
        a_to_i = {a: i for i, a in enumerate(a_grid)}
        t_to_j = {t: j for j, t in enumerate(t_grid)}
        
        for _, row in df_boot.iterrows():
            i = a_to_i[row['a']]
            j = t_to_j[row['t']]
            Z_pred[i, j] = row['p_hat']
            W[i, j] = row['n_risk']  # exactly same as p_hat mapping
        
        # -----------------------------
        # Compute weighted RMSE
        # -----------------------------
        Z_raw = raw_heatmaps[(cond0, ztime)]
        mask_obs = ~np.isnan(Z_raw)
        w_rmse_val = weighted_rmse(Z_raw[mask_obs], Z_pred[mask_obs], W[mask_obs])
        
        gof_results.append({
            'boot': boot,
            'cond0': cond0,
            'ztime': ztime,
            'model_vars': subset,
            'w_rmse': w_rmse_val
        })

gof_df_models = pd.DataFrame(gof_results)

#%%
gof_df_models = pd.DataFrame(gof_results)
gof_df_models['model_vars_str'] = gof_df_models['model_vars'].apply(lambda x: ",".join(x))

#%%
# Compute mean ± SD per model_vars / cond0 / ztime
summary = (
    gof_df_models
    .groupby(['cond0','ztime','model_vars_str'], observed=True)
    .w_rmse
    .agg(['mean', 'std'])
    .reset_index()
)

# Sort model_vars by overall mean for plotting
median_overall = summary.groupby('model_vars_str')['mean'].mean().sort_values()
ordered_models = median_overall.index.tolist()

# Catplot for better faceting
g = sns.catplot(
    data=gof_df_models,
    x='model_vars_str',
    y='w_rmse',
    hue='cond0',
    row='ztime',
    col= 'cond0',
    kind='point',
    dodge=0.4,
    linestyle='none',
    height=4,
    aspect=2,
    order=ordered_models,
    capsize=.2,
    marker='_',
    errorbar='sd',
    sharey=False,
)
# rotate
for ax in g.axes.flatten():
    for label in ax.get_xticklabels():
        label.set_rotation(90)
        
plt.savefig(os.path.join(fig_dir, 'glm_model_comparison_wRMSE.pdf'))

#%%

delta_rmse_list = []

for var in all_vars:
    for (boot, cond0, ztime), df in gof_df_models.groupby(['boot','cond0','ztime']):
        rmse_with = df.loc[df['model_vars'].apply(lambda x: var in x), 'w_rmse'].mean()  # best model including var
        rmse_without = df.loc[df['model_vars'].apply(lambda x: var not in x), 'w_rmse'].mean()  # best model excluding var
        delta_rmse_list.append({
            'boot': boot,
            'cond0': cond0,
            'ztime': ztime,
            'var': var,
            'delta_rmse': rmse_without - rmse_with
        })

delta_rmse_df = pd.DataFrame(delta_rmse_list)

sns.heatmap(
    delta_rmse_df.pivot_table(index='var', columns=['cond0','ztime'], values='delta_rmse', aggfunc='mean'),
    cmap='coolwarm', center=0, annot=True
)

# Aggregate how often a variable appears in the 'top 3' models across bootstraps
# or simply the mean coefficient weight.
importance_matrix = delta_rmse_df.pivot_table(
    index='var', 
    columns=['cond0', 'ztime'], 
    values='delta_rmse', 
    aggfunc='mean'
)
sns.heatmap(importance_matrix, annot=True, cmap="YlGnBu")


sns.catplot(
    data=delta_rmse_df,
    x='var',
    y='delta_rmse',
    hue='cond0',
    palette=my_palette,
    row='ztime',
    kind='point',
    linestyle='none',
    height=4,
    aspect=1.2,
    capsize=.2,
    marker='_',
    errorbar='sd',
    sharey=False,
)
plt.savefig(os.path.join(fig_dir, 'glm_variable_importance_wRMSE.pdf'))

sns.catplot(
    data=delta_rmse_df,
    x='cond0',
    y='delta_rmse',
    col='var',
    row='ztime',
    kind='point',
    linestyle='none',
    hue='cond0',
    palette=my_palette,
    height=3,
    aspect=.8,
    capsize=.2,
    marker='_',
    errorbar='sd',
    sharey=True,
)
plt.savefig(os.path.join(fig_dir, 'glm_variable_importance_wRMSE_byVar.pdf'))
# %%
pivot_df = delta_rmse_df.groupby(['cond0','ztime','var']).delta_rmse.mean().reset_index()
heatmap_df = pivot_df.pivot(index='var', columns=['cond0','ztime'], values='delta_rmse')

sns.heatmap(heatmap_df, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("Average Delta RMSE per Variable")
plt.savefig(os.path.join(fig_dir, 'glm_variable_importance_wRMSE_heatmap.pdf'))
# %%
# Example pseudocode
# All possible variables
remaining_vars = all_vars.copy()
selected_vars = []
cum_delta = []

selection_dfs = []

for (cond0_val, ztime_val), delta_df_sub in delta_rmse_df.groupby(['cond0', 'ztime'], observed=True):
    remaining_vars = all_vars.copy()
    selected_vars = []
    cum_delta = []

    for _ in range(len(all_vars)):
        # Compute average delta for each remaining variable
        avg_deltas = {}
        for var in remaining_vars:
            vars_to_consider = selected_vars + [var]
            avg_deltas[var] = delta_df_sub.query("var in @vars_to_consider")['delta_rmse'].mean()
        
        # Pick variable with largest average improvement
        best_var = max(avg_deltas, key=avg_deltas.get)
        selected_vars.append(best_var)

        # Cumulative improvement so far
        cum_delta.append(delta_df_sub.query("var in @selected_vars")['delta_rmse'].mean())

        remaining_vars.remove(best_var)
    
    selection_dfs.append(
        pd.DataFrame({
            'step': range(1, len(selected_vars)+1),
            'selected_var': selected_vars,
            'cum_delta_rmse': cum_delta,
            'cond0': cond0_val,
            'ztime': ztime_val
        })
    )

# Combine all conditions and ztimes
selection_df_all = pd.concat(selection_dfs, ignore_index=True)

# Plot cumulative RMSE improvement per cond0 and ztime
g = sns.FacetGrid(
    selection_df_all,
    row='ztime',
    hue='cond0',
    palette=my_palette,
    sharey=False,
    height=4,
    aspect=1.2,
    legend_out=True
)
g.map_dataframe(sns.lineplot, x='step', y='cum_delta_rmse', marker='o', legend=True)

# Annotate each selected variable above its point
for ax in g.axes.flatten():
    # Extract which ztime this row corresponds to
    ztime_val = ax.get_title().split(' = ')[1].split(',')[0].strip()
    
    # For all cond0 values in this axis (since hue is cond0)
    cond0_vals = selection_df_all['cond0'].unique()
    
    for cond0_val in cond0_vals:
        df_plot = selection_df_all.query("ztime == @ztime_val and cond0 == @cond0_val")
        for _, r in df_plot.iterrows():
            ax.text(
                r['step']+np.random.uniform(-0.1,0.1),  # small random jitter in x
                r['cum_delta_rmse']*1+ 0.001,  # small offset above point
                r['selected_var'],
                ha='center',
                va='bottom',
                fontsize=8,
                # rotation=45,
            )
g.set_axis_labels('Selection step', 'Cumulative Δ RMSE')
g.set_titles(row_template='Ztime={row_name}', col_template='Cond0={col_name}')
plt.show()

# %%
