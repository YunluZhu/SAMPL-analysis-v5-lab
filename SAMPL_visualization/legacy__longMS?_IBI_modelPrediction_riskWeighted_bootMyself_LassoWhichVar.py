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
from scipy.stats import spearmanr


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


#%
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
        n_event=('n_event', 'mean'),
        n_event_smooth=('n_event_smooth', 'mean'),  
        hazard=('hazard', 'mean'),
        IBI_ratio=('IBI_ratio', 'mean'),
        angvel=('angvel', 'mean'),
        angacc=('angacc', 'mean'),
    )
    .reset_index()
)

boot_hazard_df['t_mid'] = boot_hazard_df['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))
boot_hazard_df_average_raw['t_mid'] = boot_hazard_df_average_raw['t_bin'].apply(lambda iv: 0.5*(iv.left + iv.right))

#%%
from sklearn.model_selection import KFold
# -----------------------------
# lasso but deal with CV first


def prepare_features_corrected(df_boot, all_vars):
    """
    Prepare features with CORRECT aABS_z calculation
    """
    df_boot = df_boot.copy()
    
    # Extract raw variables
    df_boot['a'] = df_boot['ang0_bin'].astype(float)
    df_boot['t'] = df_boot['t_mid'].astype(float)
    
    # Robust z-score base features
    df_boot['a_z'], _, _ = robust_zscore(df_boot['a'])
    df_boot['t_z'], _, _ = robust_zscore(df_boot['t'])
    df_boot['vel_z'], _, _ = robust_zscore(df_boot['angvel'])
    df_boot['acc_z'], _, _ = robust_zscore(df_boot['angacc'])
    
    # ✅ CORRECT: Z-score the ABSOLUTE angle directly
    df_boot['aABS_z'], _, _ = robust_zscore(np.abs(df_boot['a']))  # <-- FIX
    
    # Create interactions
    df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
    df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
    df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
    df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
    
    return df_boot

def robust_zscore(x):
    """
    Z-score using median and MAD (robust to outliers)
    Returns: z-scored values, median, MAD
    """
    x_clean = x.dropna() if isinstance(x, pd.Series) else x[~np.isnan(x)]
    median = np.median(x_clean)
    mad = np.median(np.abs(x_clean - median))
    
    # MAD to std conversion factor (for normal distribution)
    if mad > 0:
        z = (x - median) / (1.4826 * mad)
    else:
        z = np.zeros_like(x)
    
    return z, median, mad

import warnings  # Make sure this is at the top with other imports


def fit_final_model_with_selected_alpha(df_boot, all_vars, alpha, verbose=False):
    """
    Fit final model using selected alpha and return statsmodels GLM
    
    Returns:
    --------
    final_model : statsmodels GLM results
        Fitted model with selected variables
    selected_vars : list
        Variables that survived Lasso selection
    lasso_results : statsmodels results
        Lasso fit results
    """
    
    # Prepare data (same as CV function)
    df_boot = df_boot.copy()
    df_boot['a'] = df_boot['ang0_bin'].astype(float)
    df_boot['t'] = df_boot['t_mid'].astype(float)
    
    df_boot['a_z'], _, _ = robust_zscore(df_boot['a'])
    df_boot['t_z'], _, _ = robust_zscore(df_boot['t'])
    df_boot['vel_z'], _, _ = robust_zscore(df_boot['angvel'])
    df_boot['acc_z'], _, _ = robust_zscore(df_boot['angacc'])
    
    df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
    df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
    df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
    df_boot['aABS_z'], _, _ = robust_zscore(np.abs(df_boot['a']))
    df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
    
    # Remove problematic rows
    df_boot = df_boot[
        (df_boot['n_event'] > 0) & 
        (df_boot['n_event'] < df_boot['n_risk'])
    ].copy()
    
    # Check for sufficient data
    if len(df_boot) < 10:
        if verbose:
            print(f"  ⚠️ Too few observations ({len(df_boot)})")
        return None, [], None
    
    X_full = sm.add_constant(df_boot[all_vars])
    y = np.column_stack([df_boot['n_event'], df_boot['n_risk'] - df_boot['n_event']])
    
    # Fit Lasso
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            
            lasso_model = sm.GLM(y, X_full, family=sm.families.Binomial())
            lasso_results = lasso_model.fit_regularized(
                method='elastic_net',
                alpha=alpha,
                L1_wt=1.0,
                maxiter=1000,
                cnvrg_tol=1e-6,
                zero_tol=1e-6
            )
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Lasso fitting failed: {e}")
        return None, [], None
    
    # Identify selected variables (excluding const)
    params = lasso_results.params
    active_mask = params.abs() > 1e-4
    selected_vars_raw = params.index[active_mask].tolist()
    
    # Separate const from other variables
    selected_vars = [v for v in selected_vars_raw if v != 'const']
    
    # Ensure at least one predictor
    if len(selected_vars) == 0:
        if verbose:
            print("  ⚠️ Only intercept selected, adding time")
        selected_vars = ['t_z']  # Add at least time
    
    # NOW refit without regularization
    # IMPORTANT: Create design matrix with const explicitly
    try:
        X_selected = sm.add_constant(df_boot[selected_vars])
        final_model = sm.GLM(y, X_selected, family=sm.families.Binomial()).fit()
    except Exception as e:
        if verbose:
            print(f"  ⚠️ Final GLM fitting failed: {e}")
        return None, selected_vars, lasso_results
    
    # Return selected_vars WITHOUT 'const' since we add it when needed
    return final_model, selected_vars, lasso_results



#%%

def fit_full_model_all_conditions(boot_hazard_df, all_vars, a_grid, t_grid, fig_dir,
                                MIN_IBI_PER_BIN=30, MIN_IBI_PER_POSTURE=200):
    """
    Fit full model (all variables) to all conditions
    No feature selection - just compare coefficient magnitudes
    """
        
    AIC_r_list = []
    coefficient_list = []
    predictions_list = []
    standardization_params = []  # NEW: Track standardization
    
    print("\n" + "="*80)
    print("Fitting Full Models (No Lasso Regularization)")
    print("="*80)
    
    for (boot, cond0, ztime), df_boot in boot_hazard_df.groupby(['boot', 'cond0', 'ztime'], observed=True):
        
        df_boot = df_boot.copy()
        
        # Apply filtering
        df_boot = df_boot[df_boot['n_risk'] >= MIN_IBI_PER_BIN].copy()
        
        posture_support = df_boot.groupby('ang0_bin', observed=True)['n_risk'].sum()
        valid_postures = posture_support[posture_support >= MIN_IBI_PER_POSTURE].index
        df_boot = df_boot[df_boot['ang0_bin'].isin(valid_postures)].copy()
        
        # =====================================================================
        # CRITICAL: Prepare features WITH standardization tracking
        # =====================================================================
        
        df_boot['a'] = df_boot['ang0_bin'].astype(float)
        df_boot['t'] = df_boot['t_mid'].astype(float)
        
        # Store standardization parameters for EACH feature
        std_params = {}
        
        # Base z-scored features

        df_boot['a_z'], std_params['a_mu'], std_params['a_sd'] = zscore(df_boot['a'])
    

        df_boot['t_z'], std_params['t_mu'], std_params['t_sd'] = zscore(df_boot['t'])
    

        df_boot['vel_z'], std_params['vel_mu'], std_params['vel_sd'] = zscore(df_boot['angvel'])
    

        df_boot['acc_z'], std_params['acc_mu'], std_params['acc_sd'] = zscore(df_boot['angacc'])
    
    # Special: aABS_z (absolute angle, then z-scored)

        df_boot['aABS_z'], std_params['aABS_mu'], std_params['aABS_sd'] = zscore(np.abs(df_boot['a']))
    
    # Interaction terms (created AFTER standardization)

        df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
    

        df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
    

        df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
    

        df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
        
    
        
        # =====================================================================
        # Fit model
        # =====================================================================
        

        X_full = sm.add_constant(df_boot[all_vars])
        
        # Use smoothed events if available, otherwise raw
        event_col = 'n_event'
        y = np.column_stack([df_boot[event_col],
                            df_boot['n_risk'] - df_boot[event_col]])
        
        full_model = sm.GLM(y, X_full, family=sm.families.Binomial()).fit()
        
        # Store results
        AIC_r_list.append({
            'boot': boot,
            'cond0': cond0,
            'ztime': ztime,
            'selected_vars': all_vars.copy(),
            'num_vars': len(all_vars),
            'AIC': full_model.aic,
            'BIC': full_model.bic_llf,
            'deviance': full_model.deviance
        })
        
        # Store coefficients
        for var in full_model.params.index:
            coefficient_list.append({
                'boot': boot,
                'cond0': cond0,
                'ztime': ztime,
                'variable': var,
                'coefficient': full_model.params[var],
                'se': full_model.bse[var],
                'pvalue': full_model.pvalues[var],
                'significant': full_model.pvalues[var] < 0.05
            })
        
        # Store standardization params
        standardization_params.append({
            'boot': boot,
            'cond0': cond0,
            'ztime': ztime,
            **std_params
        })
        
        # =====================================================================
        # Generate predictions using SAME standardization
        # =====================================================================
        
        # Create prediction dataframe on observed bins only
        pred_df = df_boot[['ang0_bin', 't_mid', 'angvel', 'angacc']].copy()
        pred_df['a'] = pred_df['ang0_bin'].astype(float)
        pred_df['t'] = pred_df['t_mid'].astype(float)
        
        # Apply SAME standardization as training

        pred_df['a_z'] = (pred_df['a'] - std_params['a_mu']) / std_params['a_sd']
    

        pred_df['t_z'] = (pred_df['t'] - std_params['t_mu']) / std_params['t_sd']
    

        pred_df['vel_z'] = (pred_df['angvel'] - std_params['vel_mu']) / std_params['vel_sd']
    

        pred_df['acc_z'] = (pred_df['angacc'] - std_params['acc_mu']) / std_params['acc_sd']
    

        pred_df['aABS_z'] = (np.abs(pred_df['a']) - std_params['aABS_mu']) / std_params['aABS_sd']
        
        # Interactions
        if 'a_t' in all_vars:
            pred_df['a_t'] = pred_df['a_z'] * pred_df['t_z']
        if 'vel_t' in all_vars:
            pred_df['vel_t'] = pred_df['vel_z'] * pred_df['t_z']
        if 'acc_t' in all_vars:
            pred_df['acc_t'] = pred_df['acc_z'] * pred_df['t_z']
        if 'aABS_t' in all_vars:
            pred_df['aABS_t'] = pred_df['aABS_z'] * pred_df['t_z']
        
        # Predict
        X_pred = sm.add_constant(pred_df[all_vars])
        pred_df['p_pred'] = full_model.predict(X_pred)
        
        # Build grid (OLD METHOD - more careful)
        Z = np.full((len(a_grid), len(t_grid)), np.nan)
        a_to_i = {a: i for i, a in enumerate(a_grid)}
        t_to_j = {t: j for j, t in enumerate(t_grid)}
        
        for _, row in pred_df.iterrows():
            a_val = row['ang0_bin'].astype(float)
            t_val = row['t_mid']
            
            if a_val in a_to_i and t_val in t_to_j:
                i = a_to_i[a_val]
                j = t_to_j[t_val]
                Z[i, j] = row['p_pred']
        
        predictions_list.append({
            'key': (boot, cond0, ztime),
            'grid': Z
        })

    print(f"\n✓ Completed: {len(AIC_r_list)} successful fits")
    
    return (pd.DataFrame(AIC_r_list), 
            pd.DataFrame(coefficient_list), 
            predictions_list,
            pd.DataFrame(standardization_params))


all_vars = [
    't_z', 
    # 'a_z',
    'vel_z',
    'aABS_z', 
    'a_t', 
    'vel_t', 
    'aABS_t',
    ]

aic_df_full, coef_df_full, predictions_list_full, std_params_df = fit_full_model_all_conditions(
    boot_hazard_df,
    all_vars,
    a_grid,
    t_grid,
    fig_dir,
    MIN_IBI_PER_BIN=30,
    MIN_IBI_PER_POSTURE=200
)

#%% ============================================================================
# STEP 7: Visualize Observed vs Predicted for Each Condition
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: Creating Observed vs Predicted Heatmaps")
print("="*80)

for key in list(raw_heatmaps.keys()):
    cond0_ex, ztime_ex = key
    
    # Get the Observed Grid
    Z_observed = raw_heatmaps[key]
    
    # Get the Predicted Grid (average across bootstraps)
    relevant_preds = [
        item['grid']
        for item in predictions_list_full
        if item['key'][1] == cond0_ex and item['key'][2] == ztime_ex
    ]
    
    if len(relevant_preds) == 0:
        print(f"  Skipping {cond0_ex}|{ztime_ex} - no predictions")
        continue
    
    Z_predicted = np.nanmean(np.array(relevant_preds), axis=0)
    Z_pred_std = np.nanstd(np.array(relevant_preds), axis=0)
    
    # Compute Residuals
    Z_residual = Z_observed - Z_predicted
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Observed
    im1 = axes[0, 0].imshow(Z_observed, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 0].set_title(f"Observed Hazard\n{cond0_ex} | {ztime_ex}")
    axes[0, 0].set_xlabel('Time Bin')
    axes[0, 0].set_ylabel('Posture Bin')
    plt.colorbar(im1, ax=axes[0, 0], label='Hazard')
    
    # Plot Predicted
    im2 = axes[0, 1].imshow(Z_predicted, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 1].set_title(f"Model Prediction (Mean)")
    axes[0, 1].set_xlabel('Time Bin')
    axes[0, 1].set_ylabel('Posture Bin')
    plt.colorbar(im2, ax=axes[0, 1], label='Predicted Probability')
    
    # Plot Residuals
    limit = np.nanpercentile(np.abs(Z_residual), 95)  # Use 95th percentile
    im3 = axes[1, 0].imshow(Z_residual, aspect='auto', cmap="RdBu_r", 
                             origin='lower', interpolation='nearest',
                             vmin=-limit, vmax=limit)
    axes[1, 0].set_title("Residuals (Obs - Pred)\nRed=Underestimated, Blue=Overestimated")
    axes[1, 0].set_xlabel('Time Bin')
    axes[1, 0].set_ylabel('Posture Bin')
    plt.colorbar(im3, ax=axes[1, 0], label='Residual')
    
    # Plot Prediction Uncertainty (Std across bootstraps)
    im4 = axes[1, 1].imshow(Z_pred_std, aspect='auto', cmap="Reds", 
                             origin='lower', interpolation='nearest')
    axes[1, 1].set_title("Prediction Uncertainty (SD)")
    axes[1, 1].set_xlabel('Time Bin')
    axes[1, 1].set_ylabel('Posture Bin')
    plt.colorbar(im4, ax=axes[1, 1], label='Standard Deviation')
    
    plt.suptitle(f'Model Diagnostics: {cond0_ex} | {ztime_ex}\n({len(relevant_preds)} bootstrap samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'diagnostics_{cond0_ex}_{ztime_ex}.png'), 
               dpi=150, bbox_inches='tight')

print(f"  Saved {len(raw_heatmaps)} diagnostic plots")

#%%
from collections import defaultdict
pred_heatmaps_mean = {}
for key in list(raw_heatmaps.keys()):
    cond0_ex, ztime_ex = key

    # Get the Predicted Grid (average across bootstraps)
    relevant_preds = [
        item['grid']
        for item in predictions_list_full
        if item['key'][1] == cond0_ex and item['key'][2] == ztime_ex
    ]
    
    Z_predicted = np.nanmean(np.array(relevant_preds), axis=0)
    pred_heatmaps_mean[key] = Z_predicted
    
#%%

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
plt.savefig(os.path.join(fig_dir, f'glm_gof_weightedRMSE_vs_time.pdf'))

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
plt.ylabel('Weighted RMSE (GLM vs raw)')
plt.xlabel('ztime')
plt.title('Goodness of fit across time')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_gof_RMSE_vs_time.pdf'))


for key in list(raw_heatmaps.keys()):
    Z_obaserved = raw_heatmaps[key]
    Z_predicted = pred_heatmaps_mean[key]
    cond0_ex, ztime_ex = key
    plt.figure(figsize=(4,4))
    plt.scatter(Z_observed, Z_predicted, s=10, alpha=0.5)
    plt.plot([0, Z_observed.max()], [0, Z_observed.max()], 'k--')
    plt.xlabel('Raw hazard')
    plt.ylabel('GLM-predicted hazard')
    plt.title(f'Cond={cond0_ex}, Ztime={ztime_ex}')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'glm_vs_raw_scatter_cond{cond0_ex}_ztime{ztime_ex}.pdf'))







#%%


#%%
# comparing alpha selection methods
# Summary comparison
print("\n" + "="*80)
print("APPROACH COMPARISON")
print("="*80)

print("\nOPTION 1 (Pooled Lasso):")
print(f"  Day features: {aic_df.loc[aic_df['ztime'] == 'day', 'selected_vars'].iloc[0]}")
print(f"  Night features: {aic_df.loc[aic_df['ztime'] == 'night', 'selected_vars'].iloc[0]}")
print(f"  Day models: {len(aic_df[aic_df['ztime']=='day'])} successful fits")
print(f"  Night models: {len(aic_df[aic_df['ztime']=='night'])} successful fits")

print("\nOPTION 2 (Full model):")
print(f"  All features: {all_vars}")
print(f"  Total models: {len(aic_df_full)} successful fits")

# Compare coefficient estimates
for var in all_vars:
    day_lasso = coef_df[(coef_df['ztime']=='day') & 
                                (coef_df['variable']==var)]['coefficient'].mean()
    day_full = coef_df_full[(coef_df_full['ztime']=='day') & 
                               (coef_df_full['variable']==var)]['coefficient'].mean()
    
    print(f"\n{var} (day): Lasso={day_lasso:.4f}, Full={day_full:.4f}, " +
          f"Difference={(day_full-day_lasso)/day_full*100:.1f}%")


    day_lasso = coef_df[(coef_df['ztime']=='night') & 
                                (coef_df['variable']==var)]['coefficient'].mean()
    day_full = coef_df_full[(coef_df_full['ztime']=='night') & 
                               (coef_df_full['variable']==var)]['coefficient'].mean()
    
    print(f"\n{var} (night): Lasso={day_lasso:.4f}, Full={day_full:.4f}, " +
          f"Difference={(day_full-day_lasso)/day_full*100:.1f}%")


#%%
if len(coef_df_full) == 0:
    print("⚠️ No coefficient data available")

# =========================================================================
# PLOT 1: Coefficient Trajectories Across Development
# =========================================================================

# Get unique variables
all_variables = coef_df_full['variable'].unique()
all_variables = [v for v in all_variables if v != 'const']  # Exclude intercept for now

if len(all_variables) == 0:
    print("⚠️ No variables to plot")

# Create separate plots for day and night
for time_period in ['day', 'night']:
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Get variables selected for this time period
    time_variables = coef_time['variable'].unique()
    time_variables = [v for v in time_variables if v != 'const']
    
    n_vars = len(time_variables)
    n_cols = min(3, n_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for var_idx, var in enumerate(time_variables):
        ax = axes[var_idx]
        
        var_data = coef_time[coef_time['variable'] == var].copy()
        
        # Sort by age
        var_data['age'] = var_data['cond0'].astype(int)
        var_data = var_data.sort_values('age')
        
        # Plot individual bootstrap estimates (light)
        for age in var_data['age'].unique():
            age_data = var_data[var_data['age'] == age]
            ax.scatter([age]*len(age_data), age_data['coefficient'],
                        alpha=0.15, s=20, color='gray', zorder=1)
        
        # Calculate summary statistics per age
        summary = var_data.groupby('age')['coefficient'].agg([
            'mean', 'median', 
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('q05', lambda x: x.quantile(0.05)),
            ('q95', lambda x: x.quantile(0.95)),
            'count'
        ]).reset_index()
        
        # Plot median trajectory
        ax.plot(summary['age'], summary['median'], 'o-',
                linewidth=3, markersize=12, color='crimson',
                label='Median', zorder=3, markeredgecolor='white', 
                markeredgewidth=2)
        
        # Plot IQR
        ax.fill_between(summary['age'], summary['q25'], summary['q75'],
                        alpha=0.3, color='crimson', label='IQR (25-75%)', zorder=2)
        
        # Plot 90% CI
        ax.fill_between(summary['age'], summary['q05'], summary['q95'],
                        alpha=0.15, color='crimson', label='90% CI', zorder=1)
        
        # Zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Styling
        ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{var}\n({time_period})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xticks(summary['age'])
        
        # Add sample size annotation
        for _, row in summary.iterrows():
            ax.text(row['age'], ax.get_ylim()[1]*0.95, f"n={int(row['count'])}",
                    ha='center', fontsize=8, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='gray'))
        
        # Statistical test: Test for trend across ages
        if len(summary) >= 3:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(summary['age'], summary['median'])
            trend_text = f"Trend: r={corr:.3f}, p={pval:.3f}"
            trend_color = 'green' if pval < 0.05 else 'gray'
            ax.text(0.02, 0.02, trend_text, transform=ax.transAxes,
                    fontsize=9, color=trend_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.8, edgecolor=trend_color, linewidth=2))
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Coefficient Development: {time_period.upper()}\n' +
                f'Trajectories across ages (with bootstrap uncertainty)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'coefficient_trajectories_{time_period}.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Saved: coefficient_trajectories_{time_period}.png")


# =========================================================================
# PLOT 2: Coefficient Stability Across Development
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate coefficient of variation (CV) for each variable at each age
    stability = coef_time.groupby(['cond0', 'variable'])['coefficient'].agg([
        'mean', 'std', 'count',
        ('cv', lambda x: np.abs(x.std() / x.mean()) if x.mean() != 0 else np.nan)
    ]).reset_index()
    
    # Exclude const
    stability = stability[stability['variable'] != 'const']
    
    # Pivot for heatmap
    stability_pivot = stability.pivot(index='variable', columns='cond0', values='cv')
    
    # Sort by mean CV
    stability_pivot['mean_cv'] = stability_pivot.mean(axis=1)
    stability_pivot = stability_pivot.sort_values('mean_cv').drop('mean_cv', axis=1)
    
    sns.heatmap(stability_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Coefficient of Variation'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                vmin=0, vmax=2, center=0.5)
    
    ax.set_title(f'{time_period.upper()}: Coefficient Stability\n' +
                f'(Lower CV = More Stable Across Bootstraps)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'coefficient_stability_by_age.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: coefficient_stability_by_age.png")


# =========================================================================
# PLOT 3: Effect Size Comparison (Standardized Coefficients)
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate median absolute coefficient per variable per age
    effect_size = coef_time.groupby(['cond0', 'variable'])['coefficient'].agg([
        ('median_abs', lambda x: np.median(np.abs(x))),
        'median',
        'count'
    ]).reset_index()
    
    effect_size = effect_size[effect_size['variable'] != 'const']
    
    # Pivot
    effect_pivot = effect_size.pivot(index='variable', columns='cond0', 
                                        values='median_abs')
    
    # Sort by mean effect size
    effect_pivot['mean_effect'] = effect_pivot.mean(axis=1)
    effect_pivot = effect_pivot.sort_values('mean_effect', ascending=False).drop('mean_effect', axis=1)
    
    # Create horizontal bar plot
    ages = sorted(effect_pivot.columns)
    x_pos = np.arange(len(effect_pivot.index))
    width = 0.25
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ages)))
    
    for age_idx, age in enumerate(ages):
        offset = (age_idx - len(ages)/2 + 0.5) * width
        bars = ax.barh(x_pos + offset, effect_pivot[age], width,
                        label=f'{age} dpf', color=colors[age_idx],
                        edgecolor='white', linewidth=1.5, alpha=0.8)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(effect_pivot.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Median Absolute Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Effect Sizes Across Development',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'effect_sizes_by_age.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: effect_sizes_by_age.png")


# =========================================================================
# PLOT 4: Sign Consistency Check
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate proportion of positive coefficients
    sign_consistency = coef_time.groupby(['cond0', 'variable'])['coefficient'].apply(
        lambda x: (x > 0).sum() / len(x)
    ).reset_index(name='prop_positive')
    
    sign_consistency = sign_consistency[sign_consistency['variable'] != 'const']
    
    # Pivot
    sign_pivot = sign_consistency.pivot(index='variable', columns='cond0', 
                                        values='prop_positive')
    
    # Plot
    sns.heatmap(sign_pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion Positive'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title(f'{time_period.upper()}: Sign Consistency\n' +
                f'(0.0=Always Negative, 0.5=Mixed, 1.0=Always Positive)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'coefficient_sign_consistency.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: coefficient_sign_consistency.png")


# =========================================================================
# STATISTICAL SUMMARY
# =========================================================================

print("\n" + "="*80)
print("COEFFICIENT DEVELOPMENTAL ANALYSIS SUMMARY")
print("="*80)

for time_period in ['day', 'night']:
    print(f"\n{'='*80}")
    print(f"{time_period.upper()} COEFFICIENTS")
    print('='*80)
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        print(f"  No data for {time_period}")
        continue
    
    coef_time['age'] = coef_time['cond0'].astype(int)
    
    for var in sorted(coef_time['variable'].unique()):
        if var == 'const':
            continue
        
        print(f"\n  📊 {var}")
        print(f"  {'-'*76}")
        
        var_data = coef_time[coef_time['variable'] == var]
        
        for age in sorted(var_data['age'].unique()):
            age_data = var_data[var_data['age'] == age]['coefficient']
            
            median_val = age_data.median()
            q25, q75 = age_data.quantile([0.25, 0.75])
            prop_positive = (age_data > 0).mean()
            
            print(f"    {age} dpf: median={median_val:+.4f} " +
                    f"[IQR: {q25:+.4f} to {q75:+.4f}] " +
                    f"({prop_positive*100:.0f}% positive, n={len(age_data)})")
        
        # Test for developmental trend
        if len(var_data['age'].unique()) >= 3:
            summary = var_data.groupby('age')['coefficient'].median()
            corr, pval = spearmanr(summary.index, summary.values)
            
            if pval < 0.05:
                trend = "↑ INCREASING" if corr > 0 else "↓ DECREASING"
                print(f"\n    *** {trend} with age (r={corr:.3f}, p={pval:.4f}) ***")
            else:
                print(f"\n    → No significant trend (r={corr:.3f}, p={pval:.4f})")

print("\n" + "="*80)


#%%
#%% ============================================================================
# VARIABLE IMPORTANCE ANALYSIS (Proper Methods)
# ==============================================================================


# =========================================================================
# METHOD 1: Selection Frequency (Most Robust)
# =========================================================================

print("\n" + "="*80)
print("METHOD 1: SELECTION FREQUENCY (Stability-Based Importance)")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    aic_time = aic_df_full[aic_df_full['ztime'] == time_period].copy()
    
    if len(aic_time) == 0:
        continue
    
    # Calculate selection frequency across ALL ages and bootstraps
    exploded = aic_time.explode('selected_vars')
    total_possible = len(aic_time)
    
    selection_counts = exploded['selected_vars'].value_counts()
    selection_freq = (selection_counts / total_possible * 100).sort_values(ascending=True)
    
    # Plot
    colors = plt.cm.RdYlGn(selection_freq.values / 100)
    bars = ax.barh(range(len(selection_freq)), selection_freq.values, color=colors)
    ax.set_yticks(range(len(selection_freq)))
    ax.set_yticklabels(selection_freq.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Selection Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Variable Importance\n' +
                f'(How often selected across all ages & bootstraps)',
                fontsize=13, fontweight='bold')
    ax.axvline(80, color='red', linestyle='--', linewidth=2, alpha=0.5, label='80% threshold')
    ax.axvline(50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)
    
    # Add percentage labels
    for i, (var, freq) in enumerate(selection_freq.items()):
        ax.text(freq + 2, i, f'{freq:.0f}%', va='center', fontsize=10, fontweight='bold')
    
    # Print rankings
    # print(f"\n{time_period.upper()} - Selection Frequency Rankings:")
    # print("-"*50)
    # for rank, (var, freq) in enumerate(selection_freq.items()[::-1], 1):
    #     stars = '***' if freq > 80 else '**' if freq > 50 else '*' if freq > 20 else ''
    #     print(f"  {rank}. {var:12s}: {freq:5.1f}% {stars}")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'importance_selection_frequency.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved: importance_selection_frequency.png")


# =========================================================================
# METHOD 2: Standardized Effect Sizes (Median Absolute Coefficient)
# =========================================================================

print("\n" + "="*80)
print("METHOD 2: STANDARDIZED EFFECT SIZES")
print("="*80)
print("Note: Coefficients are already on standardized predictors (z-scored)")
print("So magnitude comparison is valid WITHIN each model")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    coef_time = coef_time[coef_time['variable'] != 'const']
    
    if len(coef_time) == 0:
        continue
    
    # Calculate median absolute coefficient (across all ages & bootstraps)
    effect_sizes = coef_time.groupby('variable')['coefficient'].apply(
        lambda x: np.median(np.abs(x))
    ).sort_values(ascending=True)
    
    # Plot
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(effect_sizes)))
    ax.barh(range(len(effect_sizes)), effect_sizes.values, color=colors)
    ax.set_yticks(range(len(effect_sizes)))
    ax.set_yticklabels(effect_sizes.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Median |Coefficient|', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Effect Size\n' +
                f'(Larger = Stronger influence on hazard)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (var, effect) in enumerate(effect_sizes.items()):
        ax.text(effect + 0.01, i, f'{effect:.3f}', va='center', fontsize=9)
    
    # # Print rankings
    # print(f"\n{time_period.upper()} - Effect Size Rankings:")
    # print("-"*50)
    # for rank, (var, effect) in enumerate(effect_sizes.items()[::-1], 1):
    #     print(f"  {rank}. {var:12s}: {effect:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'importance_effect_sizes.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved: importance_effect_sizes.png")


# =========================================================================
# METHOD 4: Age-Specific Importance
# =========================================================================

print("\n" + "="*80)
print("METHOD 4: AGE-SPECIFIC IMPORTANCE")
print("="*80)

for time_period in ['day', 'night']:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    aic_time = aic_df_full[aic_df_full['ztime'] == time_period].copy()
    
    if len(aic_time) == 0:
        continue
    
    # Calculate selection frequency per age
    age_importance = []
    
    for age in sorted(aic_time['cond0'].unique()):
        age_data = aic_time[aic_time['cond0'] == age]
        exploded = age_data.explode('selected_vars')
        sel_freq = exploded['selected_vars'].value_counts() / len(age_data)
        
        for var, freq in sel_freq.items():
            age_importance.append({
                'age': age,
                'variable': var,
                'frequency': freq
            })
    
    imp_df = pd.DataFrame(age_importance)
    imp_pivot = imp_df.pivot(index='variable', columns='age', values='frequency').fillna(0)
    
    # Plot heatmap
    sns.heatmap(imp_pivot * 100, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Selection Frequency (%)'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                vmin=0, vmax=100)
    
    ax.set_title(f'{time_period.upper()}: Variable Importance Across Development\n' +
                f'(How selection patterns change with age)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'importance_by_age_{time_period}.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✓ Saved: importance_by_age_{time_period}.png")
    
    # Print age-specific rankings
    print(f"\n{time_period.upper()} - Age-Specific Rankings:")
    for age in sorted(imp_pivot.columns):
        print(f"\n  {age} dpf:")
        print("  " + "-"*40)
        age_ranks = imp_pivot[age].sort_values(ascending=False)
        for rank, (var, freq) in enumerate(age_ranks.items(), 1):
            if freq > 0:
                print(f"    {rank}. {var:<12s}: {freq*100:>5.1f}%")

print("\n" + "="*80)


# =========================================================================
# SUMMARY TABLE
# =========================================================================

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
METHOD 1 - SELECTION FREQUENCY (Primary)
    → Most robust: How often Lasso selects this variable
    → High frequency = Consistently important across conditions
    → Use this as your MAIN importance metric

METHOD 2 - EFFECT SIZE (Secondary)
    → How much the variable changes hazard (per 1 SD change)
    → Only compare WITHIN the same model
    → Interpretation: log(hazard ratio) per SD unit change

METHOD 3 - COMBINED SCORE (Best Overall)
    → Balances frequency AND magnitude
    → Upper-right quadrant = "Star variables"
    → Use for final conclusions

METHOD 4 - AGE-SPECIFIC (Developmental)
    → Which variables matter at which ages
    → Reveals developmental transitions
    → Use to tell a developmental story

RECOMMENDATION:
    1. Report METHOD 3 (Combined) as main result
    2. Support with METHOD 1 (Selection frequency)
    3. Use METHOD 4 to discuss development
    4. DON'T directly compare raw coefficients across variables
""")
print("="*80)
#%%



#%% ============================================================================
#  1: Pool Ages Within Time Period for Alpha Selection
# ==============================================================================

def cv_lasso_pooled_by_time(boot_hazard_df, all_vars, alphas=None, 
                            MIN_IBI_PER_BIN=30, MIN_IBI_PER_POSTURE=200,
                            n_boots_to_use=50, n_folds=5, verbose=True):
    """
    Pool all ages within each time period (day/night) for alpha selection
    """
    import warnings
    from sklearn.model_selection import KFold
    
    if alphas is None:
        alphas = np.logspace(-4, 1, 30)
    
    alpha_results = []
    
    # Process each time period separately
    for ztime in ['day', 'night']:
        
        print(f"\n{'='*80}")
        print(f"Alpha Selection for {ztime.upper()} (pooled across all ages)")
        print('='*80)
        
        # Pool all ages for this time period
        df_pooled = boot_hazard_df[boot_hazard_df['ztime'] == ztime].copy()
        
        print(f"  Total observations: {len(df_pooled)}")
        print(f"  Ages included: {sorted(df_pooled['cond0'].unique())}")
        print(f"  Bootstrap samples: {df_pooled['boot'].nunique()}")
        
        # Run proper K-fold CV on pooled data
        available_boots = df_pooled['boot'].unique()
        n_boots_to_use = min(n_boots_to_use, len(available_boots))
        boots_to_use = np.random.choice(available_boots, size=n_boots_to_use, replace=False)
        
        cv_scores = {alpha: [] for alpha in alphas}
        convergence_failures = {alpha: 0 for alpha in alphas}
        
        # CV loop
        for boot_id in tqdm(boots_to_use, desc=f'{ztime} CV'):
            
            df_boot = df_pooled[df_pooled['boot'] == boot_id].copy()
            df_boot = df_boot[df_boot['n_risk'] >= MIN_IBI_PER_BIN].copy()
            posture_support = df_boot.groupby('ang0_bin', observed=True)['n_risk'].sum()
            valid_postures = posture_support[posture_support >= MIN_IBI_PER_POSTURE].index
            df_boot = df_boot[df_boot['ang0_bin'].isin(valid_postures)].copy()
            
            # Check if enough data remains
            if len(df_boot) < n_folds * 10:  # Need reasonable amount per fold
                for alpha in alphas:
                    convergence_failures[alpha] += n_folds
                continue
            
            df_boot = prepare_features_corrected(df_boot, all_vars)

            # Check for any NaN/Inf in the features we'll use
            feature_check_cols = all_vars + ['n_event', 'n_risk']
            if df_boot[feature_check_cols].isnull().any().any():
                df_boot = df_boot.dropna(subset=feature_check_cols)
            
            if df_boot[all_vars].isin([np.inf, -np.inf]).any().any():
                df_boot = df_boot.replace([np.inf, -np.inf], np.nan).dropna(subset=all_vars)
            
            # Final check
            if len(df_boot) < n_folds * 10:
                for alpha in alphas:
                    convergence_failures[alpha] += n_folds
                continue
            
            # Prepare data
            try:
                X_full = sm.add_constant(df_boot[all_vars])
                y = np.column_stack([df_boot['n_event'], 
                                    df_boot['n_risk'] - df_boot['n_event']])
            except Exception as e:
                if verbose:
                    print(f"    Error preparing data for boot {boot_id}: {e}")
                for alpha in alphas:
                    convergence_failures[alpha] += n_folds
                continue
            
            # K-FOLD CV
            kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full)):
                
                X_train, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                if len(X_val) < 3:
                    for alpha in alphas:
                        convergence_failures[alpha] += 1
                    continue
                
                # Fit each alpha
                for alpha in alphas:
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore')
                            
                            lasso_model = sm.GLM(y_train, X_train, 
                                               family=sm.families.Binomial())
                            lasso_results = lasso_model.fit_regularized(
                                method='elastic_net', alpha=alpha, L1_wt=1.0,
                                maxiter=500, cnvrg_tol=1e-5, zero_tol=1e-6
                            )
                        
                        # Predict on validation (OUT-OF-SAMPLE)
                        p_val = lasso_results.predict(X_val)
                        p_val = np.clip(p_val, 1e-10, 1 - 1e-10)
                        
                        n_events_val = y_val[:, 0]
                        n_risk_val = y_val.sum(axis=1)
                        
                        deviance = -2 * np.sum(
                            n_events_val * np.log(p_val) +
                            (n_risk_val - n_events_val) * np.log(1 - p_val)
                        )
                        
                        if np.isfinite(deviance):
                            cv_scores[alpha].append(deviance)
                        else:
                            convergence_failures[alpha] += 1
                            
                    except:
                        convergence_failures[alpha] += 1
                        continue
        
        # Calculate statistics
        mean_scores = {}
        se_scores = {}
        
        for alpha in alphas:
            if len(cv_scores[alpha]) >= (n_boots_to_use * n_folds) / 2:
                mean_scores[alpha] = np.mean(cv_scores[alpha])
                se_scores[alpha] = np.std(cv_scores[alpha]) / np.sqrt(len(cv_scores[alpha]))
            else:
                mean_scores[alpha] = np.inf
                se_scores[alpha] = np.inf
        
        valid_alphas = [a for a in alphas if mean_scores[a] != np.inf]
        
        if len(valid_alphas) == 0:
            print(f"  ⚠️ No valid alphas for {ztime}!")
            continue
        
        # Convergence rate
        total_fits = n_boots_to_use * n_folds * len(alphas)
        total_failures = sum(convergence_failures.values())
        convergence_rate = 1 - (total_failures / total_fits)
        
        # Apply 1-SE rule
        best_alpha_min = min(valid_alphas, key=lambda a: mean_scores[a])
        min_deviance = mean_scores[best_alpha_min]
        min_se = se_scores[best_alpha_min]
        
        threshold = min_deviance + min_se
        candidate_alphas = [alpha for alpha in valid_alphas 
                           if mean_scores[alpha] <= threshold]
        
        if len(candidate_alphas) == 0:
            candidate_alphas = [best_alpha_min]
        
        best_alpha_1SE = max(candidate_alphas)
        
        print(f"\n  Results for {ztime.upper()}:")
        print(f"    Convergence: {convergence_rate*100:.1f}%")
        print(f"    Min deviance α: {best_alpha_min:.4f}")
        print(f"    Selected α (1-SE): {best_alpha_1SE:.4f}")
        print(f"    CV deviance: {mean_scores[best_alpha_1SE]:.1f} ± {se_scores[best_alpha_1SE]:.1f}")
        
        alpha_results.append({
            'ztime': ztime,
            'best_alpha_1SE': best_alpha_1SE,
            'best_alpha_min': best_alpha_min,
            'mean_scores': mean_scores,
            'se_scores': se_scores,
            'convergence_rate': convergence_rate,
            'n_boots_used': n_boots_to_use,
            'n_folds': n_folds
        })
    
    return alpha_results


# ============================================================================
# STEP 2: Fit Age-Specific Models with Shared Alpha & Features
# ============================================================================

def fit_age_specific_models_shared_features(boot_hazard_df, alpha_results, all_vars, fig_dir):
    """
    Use pooled alpha selection, but fit separate models per age
    - Alpha determined from pooled day/night data
    - Feature selection also from pooled data
    - But coefficients estimated separately per age
    """
    
    AIC_r_list = []
    predictions_list = []
    coefficient_list = []
    
    # First, determine which features to use for each time period
    selected_features = {}
    
    for result in alpha_results:
        ztime = result['ztime']
        selected_alpha = result['best_alpha_1SE']
        
        print(f"\n{'='*80}")
        print(f"Feature Selection for {ztime.upper()} (α={selected_alpha:.4f})")
        print('='*80)
        
        # Pool all ages for feature selection
        df_pooled = boot_hazard_df[boot_hazard_df['ztime'] == ztime].copy()
        
        # Use a few bootstraps to determine stable features
        feature_counts = {var: 0 for var in all_vars}
        n_boots_select = 20
        
        for boot_id in tqdm(df_pooled['boot'].unique()[:n_boots_select], 
                           desc='Feature selection'):
            
            df_boot = df_pooled[df_pooled['boot'] == boot_id].copy()
            
            try:
                _, selected_vars, _ = fit_final_model_with_selected_alpha(
                    df_boot, all_vars, selected_alpha, verbose=False
                )
                
                for var in selected_vars:
                    if var in all_vars:
                        feature_counts[var] += 1
            except:
                continue
        
        # Select features that appear in >50% of bootstrap samples
        threshold = n_boots_select * 0.5
        selected_vars = [var for var, count in feature_counts.items() 
                        if count > threshold]
        
        if len(selected_vars) == 0:
            selected_vars = ['t_z']  # Fallback
        
        selected_features[ztime] = selected_vars
        
        print(f"\n  Selected features for {ztime.upper()}:")
        for var in selected_vars:
            freq = feature_counts[var] / n_boots_select * 100
            print(f"    {var:12s}: {freq:5.1f}%")
    
    # Now fit age-specific models using these shared features
    print(f"\n{'='*80}")
    print("Fitting Age-Specific Models")
    print('='*80)
    
    for (cond0, ztime), df_cond in boot_hazard_df.groupby(['cond0', 'ztime'], observed=True):
        
        if ztime not in selected_features:
            continue
        
        selected_vars = selected_features[ztime]
        selected_alpha = next(r['best_alpha_1SE'] for r in alpha_results 
                             if r['ztime'] == ztime)
        
        print(f"\n  {cond0} dpf | {ztime}: fitting with {len(selected_vars)} features")
        
        boot_ids = df_cond['boot'].unique()
        
        for boot in tqdm(boot_ids, desc=f'{cond0}|{ztime}', leave=False):
            
            df_boot = df_cond[df_cond['boot'] == boot].copy()
            
            if len(df_boot) < 20:
                continue
            
            # Fit model with SHARED feature set
            try:
                df_boot = prepare_features_corrected(df_boot, all_vars)
                
                # Clean
                df_boot = df_boot[
                    (df_boot['n_event'] > 0) & 
                    (df_boot['n_event'] < df_boot['n_risk'])
                ].copy()
                
                if len(df_boot) < 10:
                    continue
                
                # Fit using ONLY the selected features
                X_selected = sm.add_constant(df_boot[selected_vars])
                y = np.column_stack([df_boot['n_event'], 
                                    df_boot['n_risk'] - df_boot['n_event']])
                
                final_model = sm.GLM(y, X_selected, family=sm.families.Binomial()).fit()
                
                # Store results
                AIC_r_list.append({
                    'boot': boot,
                    'cond0': cond0,
                    'ztime': ztime,
                    'selected_vars': selected_vars,
                    'num_vars': len(selected_vars),
                    'AIC': final_model.aic,
                    'BIC': final_model.bic_llf,
                    'deviance': final_model.deviance,
                    'alpha_used': selected_alpha
                })
                
                # Store coefficients
                for var in final_model.params.index:
                    coefficient_list.append({
                        'boot': boot,
                        'cond0': cond0,
                        'ztime': ztime,
                        'variable': var,
                        'coefficient': final_model.params[var],
                        'se': final_model.bse[var],
                        'pvalue': final_model.pvalues[var]
                    })
                
                # Generate predictions
                pred_df = df_boot.copy()
                X_pred = sm.add_constant(pred_df[selected_vars])
                pred_df['p_hat'] = final_model.predict(X_pred)
                
                pred_grid = (
                    pred_df.pivot_table(index='ang0_bin', columns='t_mid', 
                                       values='p_hat', aggfunc='mean')
                    .reindex(index=a_grid, columns=t_grid)
                )
                
                predictions_list.append({
                    'key': (boot, cond0, ztime),
                    'grid': pred_grid.values
                })
                
            except Exception as e:
                continue
    
    return pd.DataFrame(AIC_r_list), pd.DataFrame(coefficient_list), predictions_list


# ============================================================================
# RUN POOLED ALPHA SELECTION
# ============================================================================

alpha_results_pooled = cv_lasso_pooled_by_time(
    boot_hazard_df, 
    all_vars, 
    alphas=np.logspace(-4, 1, 30),
    n_boots_to_use=50,
    n_folds=5,
    verbose=True
)

# Fit age-specific models
aic_df, coef_df, predictions_list = fit_age_specific_models_shared_features(
    boot_hazard_df, 
    alpha_results_pooled, 
    all_vars, 
    fig_dir
)

#%%
#%% ============================================================================
# STEP 7: Visualize Observed vs Predicted for Each Condition
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: Creating Observed vs Predicted Heatmaps")
print("="*80)

for key in list(raw_heatmaps.keys()):
    cond0_ex, ztime_ex = key
    
    # Get the Observed Grid
    Z_observed = raw_heatmaps[key]
    
    # Get the Predicted Grid (average across bootstraps)
    relevant_preds = [
        item['grid']
        for item in predictions_list
        if item['key'][1] == cond0_ex and item['key'][2] == ztime_ex
    ]
    
    if len(relevant_preds) == 0:
        print(f"  Skipping {cond0_ex}|{ztime_ex} - no predictions")
        continue
    
    Z_predicted = np.nanmean(np.array(relevant_preds), axis=0)
    Z_pred_std = np.nanstd(np.array(relevant_preds), axis=0)
    
    # Compute Residuals
    Z_residual = Z_observed - Z_predicted
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Observed
    im1 = axes[0, 0].imshow(Z_observed.T, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 0].set_title(f"Observed Hazard\n{cond0_ex} | {ztime_ex}")
    axes[0, 0].set_xlabel('Posture Bin')
    axes[0, 0].set_ylabel('Time Bin')
    plt.colorbar(im1, ax=axes[0, 0], label='Hazard')
    
    # Plot Predicted
    im2 = axes[0, 1].imshow(Z_predicted.T, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 1].set_title(f"Model Prediction (Mean)")
    axes[0, 1].set_xlabel('Posture Bin')
    axes[0, 1].set_ylabel('Time Bin')
    plt.colorbar(im2, ax=axes[0, 1], label='Predicted Probability')
    
    # Plot Residuals
    limit = np.nanpercentile(np.abs(Z_residual), 95)  # Use 95th percentile
    im3 = axes[1, 0].imshow(Z_residual.T, aspect='auto', cmap="RdBu_r", 
                             origin='lower', interpolation='nearest',
                             vmin=-limit, vmax=limit)
    axes[1, 0].set_title("Residuals (Obs - Pred)\nRed=Underestimated, Blue=Overestimated")
    axes[1, 0].set_xlabel('Posture Bin')
    axes[1, 0].set_ylabel('Time Bin')
    plt.colorbar(im3, ax=axes[1, 0], label='Residual')
    
    # Plot Prediction Uncertainty (Std across bootstraps)
    im4 = axes[1, 1].imshow(Z_pred_std.T, aspect='auto', cmap="Reds", 
                             origin='lower', interpolation='nearest')
    axes[1, 1].set_title("Prediction Uncertainty (SD)")
    axes[1, 1].set_xlabel('Posture Bin')
    axes[1, 1].set_ylabel('Time Bin')
    plt.colorbar(im4, ax=axes[1, 1], label='Standard Deviation')
    
    plt.suptitle(f'Model Diagnostics: {cond0_ex} | {ztime_ex}\n({len(relevant_preds)} bootstrap samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'diagnostics_{cond0_ex}_{ztime_ex}.png'), 
               dpi=150, bbox_inches='tight')

print(f"  Saved {len(raw_heatmaps)} diagnostic plots")


# # #%%

# # def cv_lasso_alpha_selection_using_bootstraps(df_all_boots, all_vars, alphas=None, 
# #                                                n_boots_to_use=50, verbose=False):
# #     """
# #     Use EXISTING bootstrap samples as cross-validation folds
    
# #     Parameters:
# #     -----------
# #     df_all_boots : DataFrame
# #         Data with 'boot' column indicating bootstrap samples
# #     all_vars : list
# #         Predictor variables
# #     alphas : array-like
# #         Alpha values to search
# #     n_boots_to_use : int
# #         How many bootstrap samples to use (for speed)
# #     """
# #     import warnings
    
# #     if alphas is None:
# #         alphas = np.logspace(-4, 1, 30)
    
# #     # Get unique bootstrap IDs
# #     available_boots = df_all_boots['boot'].unique()
    
# #     if len(available_boots) < 10:
# #         if verbose:
# #             print(f"  ⚠️ Only {len(available_boots)} bootstrap samples available")
# #         return np.nan, np.nan, {}, {}, 0.0
    
# #     # Randomly sample bootstraps to use (for speed)
# #     n_boots_to_use = min(n_boots_to_use, len(available_boots))
# #     boots_to_use = np.random.choice(available_boots, size=n_boots_to_use, replace=False)
    
# #     if verbose:
# #         print(f"  Using {n_boots_to_use} bootstrap samples for CV")
    
# #     # Store CV scores
# #     cv_scores = {alpha: [] for alpha in alphas}
# #     convergence_failures = {alpha: 0 for alpha in alphas}
    
# #     # Each bootstrap is a "fold"
# #     for boot_id in tqdm(boots_to_use, desc='CV folds', leave=False, disable=not verbose):
        
# #         df_boot = df_all_boots[df_all_boots['boot'] == boot_id].copy()
        
# #         # Prepare data with robust z-scoring
# #         df_boot['a'] = df_boot['ang0_bin'].astype(float)
# #         df_boot['t'] = df_boot['t_mid'].astype(float)
        
# #         try:
# #             df_boot['a_z'], _, _ = robust_zscore(df_boot['a'])
# #             df_boot['t_z'], _, _ = robust_zscore(df_boot['t'])
# #             df_boot['vel_z'], _, _ = robust_zscore(df_boot['angvel'])
# #             df_boot['acc_z'], _, _ = robust_zscore(df_boot['angacc'])
# #         except:
# #             continue
        
# #         df_boot['a_t'] = df_boot['a_z'] * df_boot['t_z']
# #         df_boot['vel_t'] = df_boot['vel_z'] * df_boot['t_z']
# #         df_boot['acc_t'] = df_boot['acc_z'] * df_boot['t_z']
# #         df_boot['aABS_z'], _, _ = robust_zscore(np.abs(df_boot['a']))
# #         df_boot['aABS_t'] = df_boot['aABS_z'] * df_boot['t_z']
        
# #         # Clean data
# #         df_boot = df_boot[
# #             (df_boot['n_event'] > 0) & 
# #             (df_boot['n_event'] < df_boot['n_risk']) &
# #             (df_boot['n_risk'] >= 3)
# #         ].copy()
        
# #         # Remove outliers
# #         for col in ['a_z', 't_z', 'vel_z', 'acc_z', 'aABS_z']:
# #             if col in df_boot.columns:
# #                 df_boot = df_boot[df_boot[col].abs() <= 5].copy()
        
# #         if len(df_boot) < 10:
# #             for alpha in alphas:
# #                 convergence_failures[alpha] += 1
# #             continue
        
# #         # Prepare design matrix
# #         try:
# #             X_full = sm.add_constant(df_boot[all_vars])
# #             y = np.column_stack([df_boot['n_event'], 
# #                                 df_boot['n_risk'] - df_boot['n_event']])
# #         except:
# #             for alpha in alphas:
# #                 convergence_failures[alpha] += 1
# #             continue
        
# #         # Fit each alpha on this bootstrap sample
# #         for alpha in alphas:
# #             try:
# #                 with warnings.catch_warnings():
# #                     warnings.filterwarnings('ignore')
                    
# #                     lasso_model = sm.GLM(y, X_full, family=sm.families.Binomial())
# #                     lasso_results = lasso_model.fit_regularized(
# #                         method='elastic_net',
# #                         alpha=alpha,
# #                         L1_wt=1.0,
# #                         maxiter=500,  # Reduced for speed
# #                         cnvrg_tol=1e-5,  # Slightly relaxed
# #                         zero_tol=1e-6
# #                     )
                
# #                 # Calculate deviance on THIS bootstrap sample (in-sample)
# #                 p_hat = lasso_results.predict(X_full)
# #                 p_hat = np.clip(p_hat, 1e-10, 1 - 1e-10)
                
# #                 n_events = y[:, 0]
# #                 n_risk = y.sum(axis=1)
                
# #                 deviance = -2 * np.sum(
# #                     n_events * np.log(p_hat) + 
# #                     (n_risk - n_events) * np.log(1 - p_hat)
# #                 )
                
# #                 if np.isfinite(deviance):
# #                     cv_scores[alpha].append(deviance)
# #                 else:
# #                     convergence_failures[alpha] += 1
                    
# #             except Exception as e:
# #                 convergence_failures[alpha] += 1
# #                 continue
    
# #     # Calculate statistics
# #     mean_scores = {}
# #     se_scores = {}
    
# #     for alpha in alphas:
# #         if len(cv_scores[alpha]) >= n_boots_to_use / 2:
# #             mean_scores[alpha] = np.mean(cv_scores[alpha])
# #             se_scores[alpha] = np.std(cv_scores[alpha]) / np.sqrt(len(cv_scores[alpha]))
# #         else:
# #             mean_scores[alpha] = np.inf
# #             se_scores[alpha] = np.inf
    
# #     valid_alphas = [a for a in alphas if mean_scores[a] != np.inf]
    
# #     if len(valid_alphas) == 0:
# #         if verbose:
# #             print("  ⚠️ No valid alphas!")
# #         return np.nan, np.nan, mean_scores, se_scores, 0.0
    
# #     # Convergence rate
# #     total_fits = n_boots_to_use * len(alphas)
# #     total_failures = sum(convergence_failures.values())
# #     convergence_rate = 1 - (total_failures / total_fits) if total_fits > 0 else 0.0
    
# #     # Apply 1-SE rule
# #     best_alpha_min = min(valid_alphas, key=lambda a: mean_scores[a])
# #     min_deviance = mean_scores[best_alpha_min]
# #     min_se = se_scores[best_alpha_min]
    
# #     threshold = min_deviance + min_se
# #     candidate_alphas = [alpha for alpha in valid_alphas if mean_scores[alpha] <= threshold]
    
# #     if len(candidate_alphas) == 0:
# #         candidate_alphas = [best_alpha_min]
    
# #     best_alpha_1SE = max(candidate_alphas)
    
# #     if verbose:
# #         print(f"  Convergence: {convergence_rate*100:.1f}% ({len(valid_alphas)}/{len(alphas)} alphas)")
# #         print(f"  Selected: α_1SE={best_alpha_1SE:.4f}, α_min={best_alpha_min:.4f}")
    
# #     return best_alpha_1SE, best_alpha_min, mean_scores, se_scores, convergence_rate
# # #%% alpha - extreme slow

# # print("="*80)
# # print("STEP 1: Cross-Validation for Alpha Selection (Using Existing Bootstraps)")
# # print("="*80)

# # alphas = np.logspace(-4, 1, 30)

# # alpha_results = []

# # for (cond0, ztime), df_sub in boot_hazard_df.groupby(['cond0','ztime'], observed=True):
# #     print(f"\nProcessing: {cond0} | {ztime}")
# #     print("-" * 50)
    
# #     # Count available bootstraps
# #     n_boots_available = df_sub['boot'].nunique()
# #     print(f"  Available bootstrap samples: {n_boots_available}")
    
# #     # Use existing bootstraps for CV (much faster!)
# #     best_alpha_1SE, best_alpha_min, mean_scores, se_scores, conv_rate = cv_lasso_alpha_selection_using_bootstraps(
# #         df_sub, 
# #         all_vars, 
# #         alphas=alphas,
# #         n_boots_to_use=min(50, n_boots_available),  # Use 50 bootstraps max for speed
# #         verbose=True
# #     )
    
# #     alpha_results.append({
# #         'cond0': cond0,
# #         'ztime': ztime,
# #         'best_alpha_1SE': best_alpha_1SE,
# #         'best_alpha_min': best_alpha_min,
# #         'mean_scores': mean_scores,
# #         'se_scores': se_scores,
# #         'convergence_rate': conv_rate,
# #         'n_boots_used': min(50, n_boots_available)
# #     })

# # alpha_df = pd.DataFrame([{
# #     'cond0': r['cond0'],
# #     'ztime': r['ztime'],
# #     'best_alpha_1SE': r['best_alpha_1SE'],
# #     'best_alpha_min': r['best_alpha_min'],
# #     'convergence_rate': r['convergence_rate'],
# #     'n_boots_used': r['n_boots_used']
# # } for r in alpha_results])

# # print("\n" + "="*80)
# # print("ALPHA SELECTION SUMMARY")
# # print("="*80)
# # print(alpha_df)

# #%%
# #%% ============================================================================
# # CV VISUALIZATION SUITE - Investigating Alpha Selection
# # ==============================================================================
    
# # =========================================================================
# # PLOT 1: Main CV Curves with Detailed Annotations
# # =========================================================================

# n_conditions = len(alpha_results)
# n_cols = 2
# n_rows = int(np.ceil(n_conditions / n_cols))

# fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5.5*n_rows))
# if n_conditions == 1:
#     axes = np.array([axes])
# axes = axes.flatten()

# for idx, result in enumerate(alpha_results):
#     ax = axes[idx]
    
#     alphas = np.array(list(result['mean_scores'].keys()))
#     mean_dev = np.array(list(result['mean_scores'].values()))
#     se_dev = np.array([result['se_scores'][a] for a in alphas])
    
#     # Remove infinite values
#     valid_mask = np.isfinite(mean_dev) & np.isfinite(se_dev)
#     alphas_valid = alphas[valid_mask]
#     mean_dev_valid = mean_dev[valid_mask]
#     se_dev_valid = se_dev[valid_mask]
    
#     if len(alphas_valid) == 0:
#         ax.text(0.5, 0.5, 'No valid data', ha='center', va='center',
#                 transform=ax.transAxes, fontsize=14, color='red')
#         ax.set_title(f'{result["cond0"]} dpf | {result["ztime"]}')
#         continue
    
#     # Color code by day/night
#     color = 'gold' if result['ztime'] == 'day' else 'midnightblue'
    
#     # Plot CV curve
#     ax.errorbar(alphas_valid, mean_dev_valid, yerr=se_dev_valid,
#                 fmt='o-', capsize=4, alpha=0.8, 
#                 color=color, linewidth=2.5, markersize=7,
#                 elinewidth=1.5, label='Mean deviance ± SE')
    
#     # Mark minimum
#     if not np.isnan(result['best_alpha_min']):
#         min_idx = np.where(alphas_valid == result['best_alpha_min'])[0]
#         if len(min_idx) > 0:
#             ax.axvline(result['best_alpha_min'], color='dodgerblue', 
#                         linestyle='--', linewidth=2, alpha=0.7,
#                         label=f'Min α={result["best_alpha_min"]:.3f}')
#             ax.plot(result['best_alpha_min'], mean_dev_valid[min_idx[0]],
#                     'o', color='dodgerblue', markersize=12, 
#                     markeredgecolor='white', markeredgewidth=2, zorder=5)
            
#             # Draw 1-SE threshold
#             threshold = mean_dev_valid[min_idx[0]] + se_dev_valid[min_idx[0]]
#             ax.axhline(threshold, color='crimson', linestyle=':', 
#                         linewidth=2, alpha=0.6, label='1-SE threshold')
            
#             # Shade 1-SE region
#             in_1se = mean_dev_valid <= threshold
#             if in_1se.any():
#                 alpha_min_1se = alphas_valid[in_1se].min()
#                 alpha_max_1se = alphas_valid[in_1se].max()
#                 ax.axvspan(alpha_min_1se, alpha_max_1se, 
#                             alpha=0.15, color='crimson', zorder=0,
#                             label='1-SE region')
    
#     # Mark selected alpha
#     if not np.isnan(result['best_alpha_1SE']):
#         ax.axvline(result['best_alpha_1SE'], color='crimson',
#                     linestyle='-', linewidth=3, alpha=0.9,
#                     label=f'Selected α={result["best_alpha_1SE"]:.3f}')
#         se_idx = np.where(alphas_valid == result['best_alpha_1SE'])[0]
#         if len(se_idx) > 0:
#             ax.plot(result['best_alpha_1SE'], mean_dev_valid[se_idx[0]],
#                     '*', color='crimson', markersize=25,
#                     markeredgecolor='white', markeredgewidth=2, zorder=6)
    
#     ax.set_xscale('log')
#     ax.set_xlabel('Alpha (Regularization Strength)', fontsize=12, fontweight='bold')
#     ax.set_ylabel('Cross-Validation Deviance', fontsize=12, fontweight='bold')
    
#     # Title with key info
#     title = f'{result["cond0"]} dpf | {result["ztime"].upper()}\n'
#     title += f'Selected α = {result["best_alpha_1SE"]:.4f} | '
#     title += f'Convergence: {result["convergence_rate"]*100:.0f}%'
#     ax.set_title(title, fontsize=13, fontweight='bold', 
#                 color=color, pad=15)
    
#     ax.legend(fontsize=9, loc='best', framealpha=0.9)
#     ax.grid(True, alpha=0.3, which='both', linestyle=':', linewidth=0.8)
    
#     # Add annotation box
#     stats_text = f'Min deviance: {mean_dev_valid.min():.1f}\n'
#     stats_text += f'Deviance @ selected: {mean_dev_valid[se_idx[0] if len(se_idx)>0 else 0]:.1f}\n'
#     stats_text += f'Deviance increase: {((mean_dev_valid[se_idx[0]] / mean_dev_valid.min() - 1)*100 if len(se_idx)>0 else 0):.1f}%\n'
#     stats_text += f'Bootstraps used: {result.get("n_boots_used", "N/A")}'
    
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.7, edgecolor='gray', linewidth=1.5)
#     ax.text(0.97, 0.97, stats_text, transform=ax.transAxes, 
#             fontsize=9, verticalalignment='top', horizontalalignment='right',
#             bbox=props, family='monospace')

# # Hide unused subplots
# for idx in range(n_conditions, len(axes)):
#     axes[idx].axis('off')

# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'cv_curves_detailed.png'),
#             dpi=200, bbox_inches='tight', facecolor='white')
# plt.show()
# print("✓ Saved: cv_curves_detailed.png")


# # =========================================================================
# # PLOT 2: Day vs Night Side-by-Side Comparison
# # =========================================================================

# # Organize by age
# age_groups = {}
# for result in alpha_results:
#     age = result['cond0']
#     if age not in age_groups:
#         age_groups[age] = {}
#     age_groups[age][result['ztime']] = result

# n_ages = len(age_groups)
# fig, axes = plt.subplots(n_ages, 2, figsize=(16, 5*n_ages))

# if n_ages == 1:
#     axes = axes.reshape(1, -1)

# for age_idx, (age, time_dict) in enumerate(sorted(age_groups.items())):
#     for time_idx, time in enumerate(['day', 'night']):
#         ax = axes[age_idx, time_idx]
        
#         if time not in time_dict:
#             ax.text(0.5, 0.5, f'No {time} data', ha='center', va='center',
#                     transform=ax.transAxes, fontsize=14)
#             ax.set_title(f'{age} dpf | {time.upper()}')
#             continue
        
#         result = time_dict[time]
        
#         alphas = np.array(list(result['mean_scores'].keys()))
#         mean_dev = np.array(list(result['mean_scores'].values()))
#         se_dev = np.array([result['se_scores'][a] for a in alphas])
        
#         valid_mask = np.isfinite(mean_dev) & np.isfinite(se_dev)
#         alphas_valid = alphas[valid_mask]
#         mean_dev_valid = mean_dev[valid_mask]
#         se_dev_valid = se_dev[valid_mask]
        
#         if len(alphas_valid) == 0:
#             continue
        
#         # Color
#         color = 'gold' if time == 'day' else 'midnightblue'
        
#         # Plot
#         ax.fill_between(alphas_valid, 
#                         mean_dev_valid - se_dev_valid,
#                         mean_dev_valid + se_dev_valid,
#                         alpha=0.3, color=color)
#         ax.plot(alphas_valid, mean_dev_valid, 'o-',
#                 color=color, linewidth=3, markersize=8)
        
#         # Mark selection
#         if not np.isnan(result['best_alpha_1SE']):
#             ax.axvline(result['best_alpha_1SE'], color='red',
#                         linestyle='--', linewidth=3, alpha=0.8)
#             se_idx = np.where(alphas_valid == result['best_alpha_1SE'])[0]
#             if len(se_idx) > 0:
#                 ax.plot(result['best_alpha_1SE'], mean_dev_valid[se_idx[0]],
#                         '*', color='red', markersize=30,
#                         markeredgecolor='white', markeredgewidth=2)
        
#         ax.set_xscale('log')
#         ax.set_xlabel('Alpha', fontsize=12, fontweight='bold')
        
#         if time_idx == 0:
#             ax.set_ylabel(f'{age} dpf\nCV Deviance', 
#                         fontsize=12, fontweight='bold')
        
#         title = f'{time.upper()}\n'
#         title += f'α = {result["best_alpha_1SE"]:.4f}'
#         ax.set_title(title, fontsize=14, fontweight='bold', color=color, pad=10)
        
#         ax.grid(True, alpha=0.3, which='both')
#         ax.set_facecolor('#f8f8f8')
        
#         # Convergence badge
#         conv_pct = result["convergence_rate"] * 100
#         badge_color = 'green' if conv_pct > 80 else 'orange' if conv_pct > 60 else 'red'
#         ax.text(0.05, 0.95, f'{conv_pct:.0f}%\nconverged',
#                 transform=ax.transAxes, fontsize=11, fontweight='bold',
#                 verticalalignment='top',
#                 bbox=dict(boxstyle='round', facecolor=badge_color, 
#                         alpha=0.7, edgecolor='white', linewidth=2),
#                 color='white')

# plt.suptitle('Day vs Night Comparison: CV Curves', 
#             fontsize=16, fontweight='bold', y=1.00)
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'cv_day_night_comparison.png'),
#             dpi=200, bbox_inches='tight', facecolor='white')
# plt.show()
# print("✓ Saved: cv_day_night_comparison.png")


# # =========================================================================
# # PLOT 3: Alpha Selection Heatmaps
# # =========================================================================

# alpha_df = pd.DataFrame([{
#     'cond0': r['cond0'],
#     'ztime': r['ztime'],
#     'best_alpha_1SE': r['best_alpha_1SE'],
#     'best_alpha_min': r['best_alpha_min'],
#     'convergence_rate': r['convergence_rate']
# } for r in alpha_results])

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# # Heatmap 1: Selected alpha (linear scale)
# alpha_pivot = alpha_df.pivot(index='cond0', columns='ztime', 
#                                 values='best_alpha_1SE')

# sns.heatmap(alpha_pivot, annot=True, fmt='.3f', cmap='YlOrRd',
#             cbar_kws={'label': 'Selected Alpha (1-SE)'}, ax=axes[0],
#             linewidths=3, linecolor='white', annot_kws={'fontsize': 13, 'fontweight': 'bold'})
# axes[0].set_title('Selected Alpha Values\n(Higher = More Regularization)', 
#                     fontsize=13, fontweight='bold')
# axes[0].set_xlabel('Time Period', fontsize=12, fontweight='bold')
# axes[0].set_ylabel('Age (dpf)', fontsize=12, fontweight='bold')

# # Heatmap 2: Log scale
# alpha_pivot_log = np.log10(alpha_pivot.replace(0, np.nan))
# sns.heatmap(alpha_pivot_log, annot=alpha_pivot.values, fmt='.3f',
#             cmap='RdYlBu_r', center=0,
#             cbar_kws={'label': 'log₁₀(Alpha)'}, ax=axes[1],
#             linewidths=3, linecolor='white', annot_kws={'fontsize': 13, 'fontweight': 'bold'})
# axes[1].set_title('Selected Alpha (Log Scale)\n(Easier to see relative differences)', 
#                     fontsize=13, fontweight='bold')
# axes[1].set_xlabel('Time Period', fontsize=12, fontweight='bold')
# axes[1].set_ylabel('Age (dpf)', fontsize=12, fontweight='bold')

# # Heatmap 3: Convergence rate
# conv_pivot = alpha_df.pivot(index='cond0', columns='ztime',
#                                 values='convergence_rate')
# sns.heatmap(conv_pivot * 100, annot=True, fmt='.0f', cmap='RdYlGn',
#             vmin=60, vmax=100, center=80,
#             cbar_kws={'label': 'Convergence Rate (%)'}, ax=axes[2],
#             linewidths=3, linecolor='white', annot_kws={'fontsize': 13, 'fontweight': 'bold'})
# axes[2].set_title('Model Convergence Rate\n(Quality Check)', 
#                     fontsize=13, fontweight='bold')
# axes[2].set_xlabel('Time Period', fontsize=12, fontweight='bold')
# axes[2].set_ylabel('Age (dpf)', fontsize=12, fontweight='bold')

# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'alpha_selection_heatmaps.png'),
#             dpi=200, bbox_inches='tight', facecolor='white')
# plt.show()
# print("✓ Saved: alpha_selection_heatmaps.png")


# # =========================================================================
# # PLOT 4: Data Quality Investigation
# # =========================================================================

# # Check sample sizes and data characteristics
# data_quality = []

# for (cond0, ztime), df_sub in boot_hazard_df.groupby(['cond0', 'ztime'], observed=True):
#     n_boots = df_sub['boot'].nunique()
#     n_total_obs = len(df_sub)
#     n_obs_per_boot = n_total_obs / n_boots if n_boots > 0 else 0
    
#     # Event rate
#     event_rate = df_sub['n_event'].sum() / df_sub['n_risk'].sum() if df_sub['n_risk'].sum() > 0 else 0
    
#     # Velocity/acceleration variance
#     vel_var = df_sub['angvel'].var()
#     acc_var = df_sub['angacc'].var()
    
#     data_quality.append({
#         'cond0': cond0,
#         'ztime': ztime,
#         'n_boots': n_boots,
#         'n_obs_per_boot': n_obs_per_boot,
#         'event_rate': event_rate,
#         'vel_variance': vel_var,
#         'acc_variance': acc_var
#     })

# dq_df = pd.DataFrame(data_quality)

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # Plot 1: Sample sizes
# dq_pivot = dq_df.pivot(index='cond0', columns='ztime', values='n_obs_per_boot')
# sns.heatmap(dq_pivot, annot=True, fmt='.0f', cmap='Blues',
#             cbar_kws={'label': 'Observations per Bootstrap'}, ax=axes[0, 0],
#             linewidths=2, linecolor='white', annot_kws={'fontsize': 12, 'fontweight': 'bold'})
# axes[0, 0].set_title('Sample Size per Bootstrap\n(Low samples → Need more regularization)', 
#                     fontsize=12, fontweight='bold')

# # Plot 2: Event rates
# event_pivot = dq_df.pivot(index='cond0', columns='ztime', values='event_rate')
# sns.heatmap(event_pivot * 100, annot=True, fmt='.1f', cmap='RdYlGn',
#             cbar_kws={'label': 'Event Rate (%)'}, ax=axes[0, 1],
#             linewidths=2, linecolor='white', annot_kws={'fontsize': 12, 'fontweight': 'bold'})
# axes[0, 1].set_title('Event Rate\n(Imbalance affects model complexity)', 
#                     fontsize=12, fontweight='bold')

# # Plot 3: Velocity variance
# vel_pivot = dq_df.pivot(index='cond0', columns='ztime', values='vel_variance')
# sns.heatmap(vel_pivot, annot=True, fmt='.2f', cmap='Oranges',
#             cbar_kws={'label': 'Velocity Variance'}, ax=axes[1, 0],
#             linewidths=2, linecolor='white', annot_kws={'fontsize': 12, 'fontweight': 'bold'})
# axes[1, 0].set_title('Angular Velocity Variance\n(High variance → More complex patterns)', 
#                     fontsize=12, fontweight='bold')

# # Plot 4: Correlation with selected alpha
# # Merge with alpha_df
# merged = dq_df.merge(alpha_df, on=['cond0', 'ztime'])

# axes[1, 1].scatter(merged['n_obs_per_boot'], merged['best_alpha_1SE'],
#                     c=['gold' if t=='day' else 'midnightblue' for t in merged['ztime']],
#                     s=200, alpha=0.7, edgecolors='black', linewidth=2)

# for idx, row in merged.iterrows():
#     axes[1, 1].annotate(f"{row['cond0']}\n{row['ztime']}", 
#                         (row['n_obs_per_boot'], row['best_alpha_1SE']),
#                         fontsize=9, ha='center', fontweight='bold')

# axes[1, 1].set_xlabel('Observations per Bootstrap', fontsize=12, fontweight='bold')
# axes[1, 1].set_ylabel('Selected Alpha', fontsize=12, fontweight='bold')
# axes[1, 1].set_yscale('log')
# axes[1, 1].set_title('Alpha vs Sample Size\n(Expect negative correlation)', 
#                     fontsize=12, fontweight='bold')
# axes[1, 1].grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, 'data_quality_investigation.png'),
#             dpi=200, bbox_inches='tight', facecolor='white')
# plt.show()
# print("✓ Saved: data_quality_investigation.png")


# # =========================================================================
# # PRINT SUMMARY STATISTICS
# # =========================================================================

# print("\n" + "="*80)
# print("ALPHA SELECTION INVESTIGATION SUMMARY")
# print("="*80)

# print("\n📊 Selected Alpha Values:")
# print(alpha_pivot.to_string())

# print("\n📈 Day/Night Ratios (Day Alpha / Night Alpha):")
# if 'day' in alpha_pivot.columns and 'night' in alpha_pivot.columns:
#     ratio = alpha_pivot['day'] / alpha_pivot['night']
#     for age in ratio.index:
#         print(f"  {age} dpf: {ratio[age]:6.1f}× " + 
#                 f"(day needs {ratio[age]:.1f}× MORE regularization)")

# print("\n🔬 Developmental Trends:")
# ages = sorted(alpha_pivot.index)
# if len(ages) >= 2:
#     for time in alpha_pivot.columns:
#         print(f"\n  {time.upper()}:")
#         for i in range(len(ages)-1):
#             ratio = alpha_pivot.loc[ages[i+1], time] / alpha_pivot.loc[ages[i], time]
#             direction = "↑ more reg" if ratio > 1 else "↓ less reg"
#             print(f"    {ages[i]}→{ages[i+1]} dpf: {ratio:.2f}× {direction}")

# print("\n✓ Convergence Rates:")
# print(conv_pivot.applymap(lambda x: f"{x*100:.1f}%").to_string())

# print("\n📦 Sample Sizes (obs per bootstrap):")
# print(dq_pivot.to_string())

# print("\n🎯 Event Rates:")
# print(event_pivot.applymap(lambda x: f"{x*100:.1f}%").to_string())

# # Correlation analysis
# if len(merged) > 2:
#     corr_size = np.corrcoef(merged['n_obs_per_boot'], merged['best_alpha_1SE'])[0, 1]
#     corr_event = np.corrcoef(merged['event_rate'], merged['best_alpha_1SE'])[0, 1]
    
#     print("\n🔗 Correlations with Selected Alpha:")
#     print(f"  Sample size:  r = {corr_size:+.3f} (expect negative)")
#     print(f"  Event rate:   r = {corr_event:+.3f}")

# print("\n" + "="*80)



# #%% ============================================================================
# # STEP 2: Fit Models on Existing Bootstrap Samples with Selected Alphas
# # ==============================================================================

# # Master Grid for Heatmaps (Pre-calculated once)
# a_grid = np.sort(boot_hazard_df['ang0_bin'].astype(float).unique())
# t_grid = np.sort(boot_hazard_df['t_mid'].unique())
# AA, TT = np.meshgrid(a_grid, t_grid, indexing='ij')


# raw_heatmaps = {}
# weight_heatmaps = {}
# for (cond0, ztime), df_sub in boot_hazard_df_average_raw.groupby(['cond0', 'ztime'], observed=True):
#     # Create empty heatmap
#     Z_raw = np.full(AA.shape, np.nan)
#     for i, a_val in enumerate(a_grid):
#         for j, t_val in enumerate(t_grid):
#             mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
#             if mask.any():
#                 Z_raw[i, j] = df_sub.loc[mask, 'n_event_smooth'].values[0] / df_sub.loc[mask, 'n_risk'].values[0]
#     raw_heatmaps[(cond0, ztime)] = Z_raw
#     # Weight heatmap (n_risk)
#     Z_weight = np.full(AA.shape, np.nan)
#     for i, a_val in enumerate(a_grid):
#         for j, t_val in enumerate(t_grid):
#             mask = (df_sub['ang0_bin'].astype(float) == a_val) & (df_sub['t_mid'] == t_val)
#             if mask.any():
#                 Z_weight[i, j] = df_sub.loc[mask, 'n_risk'].values[0]
#     weight_heatmaps[(cond0, ztime)] = Z_weight
    
# print("\n" + "="*80)
# print("STEP 2: Fitting Models on Bootstrap Samples with Selected Alphas")
# print("="*80)

# # Create alpha lookup dictionary
# alpha_lookup = {(row['cond0'], row['ztime']): row['best_alpha_1SE'] 
#                 for _, row in alpha_df.iterrows()}

# AIC_r_list = []
# predictions_list = []
# coefficient_list = []

# # Group by condition first, then bootstrap
# for (cond0, ztime), df_cond in boot_hazard_df.groupby(['cond0', 'ztime'], observed=True):
    
#     # Get selected alpha for this condition
#     selected_alpha = alpha_lookup.get((cond0, ztime), np.nan)
    
#     if np.isnan(selected_alpha):
#         print(f"⚠️ Skipping {cond0}|{ztime} - no valid alpha found")
#         continue
    
#     print(f"\nProcessing: {cond0} | {ztime} (α={selected_alpha:.4f})")
    
#     # Get unique bootstrap iterations for this condition
#     boot_ids = df_cond['boot'].unique()
#     n_boot = len(boot_ids)
    
#     print(f"  Found {n_boot} bootstrap samples")
    
#     # Fit model for each bootstrap sample
#     for boot in tqdm(boot_ids, desc=f'{cond0}|{ztime}'):
#         # Get data for this bootstrap iteration
#         df_boot = df_cond[df_cond['boot'] == boot].copy()
        
#         # Skip if too few data points
#         if len(df_boot) < 20:
#             continue
        
#         # Fit model with selected alpha
#         try:
#             final_model, selected_vars, lasso_results = fit_final_model_with_selected_alpha(
#                 df_boot,
#                 all_vars,
#                 alpha=selected_alpha,
#                 verbose=False
#             )
            
#             if final_model is None:
#                 continue
            
#             # Store fit metrics
#             AIC_r_list.append({
#                 'boot': boot,
#                 'cond0': cond0,
#                 'ztime': ztime,
#                 'selected_vars': selected_vars,  # Already excludes 'const'
#                 'num_vars': len(selected_vars),
#                 'AIC': final_model.aic,
#                 'BIC': final_model.bic_llf,
#                 'deviance': final_model.deviance,
#                 'alpha_used': selected_alpha
#             })
            
#             # Store coefficients (including intercept from model)
#             for param_name in final_model.params.index:
#                 coefficient_list.append({
#                     'boot': boot,
#                     'cond0': cond0,
#                     'ztime': ztime,
#                     'variable': param_name,
#                     'coefficient': final_model.params[param_name],
#                     'se': final_model.bse[param_name],
#                     'pvalue': final_model.pvalues[param_name]
#                 })
            
#             # Generate predictions
#             pred_df = df_boot.copy()
#             pred_df['a'] = pred_df['ang0_bin'].astype(float)
#             pred_df['t'] = pred_df['t_mid'].astype(float)
            
#             pred_df['a_z'], _, _ = robust_zscore(pred_df['a'])
#             pred_df['t_z'], _, _ = robust_zscore(pred_df['t'])
#             pred_df['vel_z'], _, _ = robust_zscore(pred_df['angvel'])
#             pred_df['acc_z'], _, _ = robust_zscore(pred_df['angacc'])
            
#             pred_df['a_t'] = pred_df['a_z'] * pred_df['t_z']
#             pred_df['vel_t'] = pred_df['vel_z'] * pred_df['t_z']
#             pred_df['acc_t'] = pred_df['acc_z'] * pred_df['t_z']
#             pred_df['aABS_z'], _, _ = robust_zscore(np.abs(pred_df['a']))
#             pred_df['aABS_t'] = pred_df['aABS_z'] * pred_df['t_z']
            
#             # Create design matrix for prediction - EXPLICITLY ADD CONSTANT
#             X_pred = sm.add_constant(pred_df[selected_vars])  # <-- KEY FIX
#             pred_df['p_hat'] = final_model.predict(X_pred)
            
#             # Create prediction grid
#             pred_grid = (
#                 pred_df
#                 .pivot_table(index='ang0_bin', columns='t_mid', values='p_hat', aggfunc='mean')
#                 .reindex(index=a_grid, columns=t_grid)
#             )
            
#             predictions_list.append({
#                 'key': (boot, cond0, ztime),
#                 'grid': pred_grid.values
#             })
            
#         except Exception as e:
#             print(f"  ⚠️ Boot {boot} failed: {e}")
#             continue

# # Convert results to DataFrames
# aic_df = pd.DataFrame(AIC_r_list)
# coef_df = pd.DataFrame(coefficient_list)

# print("\n" + "="*80)
# print("BOOTSTRAP FITTING COMPLETE")
# print("="*80)

# if len(aic_df) > 0:
#     print(f"Total successful fits: {len(aic_df)}")
    
#     # Summary by condition
#     summary = aic_df.groupby(['cond0', 'ztime']).agg({
#         'boot': 'count',
#         'num_vars': 'mean',
#         'AIC': 'mean',
#         'deviance': 'mean'
#     }).round(2)
#     summary.columns = ['n_fits', 'avg_vars', 'avg_AIC', 'avg_deviance']
    
#     print("\nSummary by Condition:")
#     print(summary)
# else:
#     print("⚠️ No successful fits!")
    

#%% ============================================================================
# STEP 3: Variable Stability Analysis
# ==============================================================================

print("\n" + "="*80)
print("STEP 3: Variable Stability Analysis")
print("="*80)

if len(aic_df) == 0:
    print("⚠️ Cannot perform stability analysis - no successful fits")
else:
    # Explode the list of selected variables
    stability_df = aic_df.explode('selected_vars')
    
    # Calculate selection frequency per condition
    stability_counts = (
        stability_df
        .groupby(['cond0', 'ztime', 'selected_vars'], observed=True)
        .size()
        .reset_index(name='selection_count')
    )
    
    # Get total bootstraps per condition
    n_boots_per_condition = aic_df.groupby(['ztime']).size().reset_index(name='n_boots')
    stability_counts = stability_counts.merge(n_boots_per_condition, on=['ztime'])
    stability_counts['frequency'] = stability_counts['selection_count'] / stability_counts['n_boots'] * stability_counts.cond0.nunique()
    
    # Plot stability heatmap
    pivot_stability = stability_counts.pivot_table(
        index='selected_vars',
        columns=['ztime'],
        values='frequency',
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=(max(12, len(pivot_stability.columns)*0.8), 
                                     max(6, len(pivot_stability)*0.5)))
    sns.heatmap(pivot_stability, annot=True, fmt='.2f', cmap="Greens", 
                vmin=0, vmax=1, cbar_kws={'label': 'Selection Frequency'}, ax=ax)
    ax.set_title(f"Variable Selection Stability\n(Frequency across bootstrap samples)")
    ax.set_xlabel('Condition')
    ax.set_ylabel('Variable')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'lasso_stability_selection_robust.png'), dpi=150, bbox_inches='tight')
    
    print("\nVariable Selection Frequency (averaged across conditions):")
    avg_freq = pivot_stability.mean(axis=1).sort_values(ascending=False)
    for var, freq in avg_freq.items():
        print(f"  {var:12s}: {freq:.2%}")


#%% ============================================================================
# STEP 4: Coefficient Distribution Analysis
# ==============================================================================

print("\n" + "="*80)
print("STEP 4: Coefficient Distribution Analysis")
print("="*80)

if len(coef_df) == 0:
    print("⚠️ No coefficient data available")
else:
    # Plot coefficient distributions by variable
    variables_to_plot = coef_df['variable'].unique()
    n_vars = len(variables_to_plot)
    
    if n_vars > 0:
        n_cols = min(3, n_vars)
        n_rows = int(np.ceil(n_vars / n_cols))
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
        if n_vars == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, var in enumerate(variables_to_plot):
            ax = axes[idx]
            var_data = coef_df[coef_df['variable'] == var]
            
            # Violin plot by condition
            sns.violinplot(data=var_data, x='cond0', y='coefficient', 
                          hue='ztime', ax=ax, cut=0)
            ax.axhline(0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title(f'{var}')
            ax.set_xlabel('Condition')
            ax.set_ylabel('Coefficient')
            ax.tick_params(axis='x', rotation=45)
            
            # Add median values
            medians = var_data.groupby(['cond0', 'ztime'])['coefficient'].median()
            if len(medians) > 0:
                ax.text(0.02, 0.98, f'Overall median: {var_data["coefficient"].median():.3f}',
                       transform=ax.transAxes, va='top', fontsize=8,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Hide unused subplots
        for idx in range(n_vars, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, 'coefficient_distributions_robust.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Summary statistics
        print("\nCoefficient Summary Statistics:")
        coef_summary = coef_df.groupby('variable')['coefficient'].agg([
            'count', 'mean', 'std', 
            ('median', lambda x: x.median()),
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75))
        ]).round(4)
        print(coef_summary)

#%% ============================================================================
# STEP 5: Variable Importance Summary
# ==============================================================================

print("\n" + "="*80)
print("STEP 5: Variable Importance Summary")
print("="*80)

if len(aic_df) > 0:
    # Calculate overall importance (frequency across all bootstraps)
    total_boots = len(aic_df)
    
    importance_df = (
        stability_counts
        .groupby('selected_vars')
        .agg(
            total_selections=('selection_count', 'sum'),
            mean_freq=('frequency', 'mean'),
            std_freq=('frequency', 'std'),
            n_conditions=('frequency', 'count')
        )
        .reset_index()
    )
    
    # Calculate global frequency
    # Get total possible selections (sum of n_boots across all conditions)
    total_possible = stability_counts.groupby('selected_vars')['n_boots'].sum().reset_index()
    importance_df = importance_df.merge(
        total_possible.rename(columns={'n_boots': 'total_possible'}),
        on='selected_vars'
    )
    importance_df['global_freq'] = importance_df['total_selections'] / importance_df['total_possible']
    
    importance_df = importance_df.sort_values('global_freq', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(importance_df['selected_vars'], importance_df['global_freq'], 
                   xerr=importance_df['std_freq'], alpha=0.7)
    ax.set_xlabel('Selection Frequency (across all conditions & bootstraps)')
    ax.set_ylabel('Variable')
    ax.set_title('Variable Importance')
    ax.axvline(0.8, color='r', linestyle='--', linewidth=1, alpha=0.5, label='80% Threshold')
    ax.axvline(0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='50% Threshold')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (var, freq) in enumerate(zip(importance_df['selected_vars'], 
                                         importance_df['global_freq'])):
        ax.text(freq + 0.02, i, f'{freq:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'variable_importance_robust.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nVariable Importance Ranking:")
    print(importance_df[['selected_vars', 'global_freq', 'mean_freq', 'n_conditions']].to_string(index=False))

#%% ============================================================================
# STEP 6: Model Goodness-of-Fit
# ==============================================================================

print("\n" + "="*80)
print("STEP 6: Model Goodness-of-Fit Assessment")
print("="*80)

def calculate_wrmse(obs_grid, pred_grid, weight_grid):
    """
    Computes Weighted RMSE between Observed and Predicted grids.
    Ignores NaNs (bins with no data).
    """
    mask = ~np.isnan(obs_grid) & ~np.isnan(pred_grid) & ~np.isnan(weight_grid)
    if not mask.any():
        return np.nan
    
    diff_sq = (obs_grid[mask] - pred_grid[mask]) ** 2
    weights = weight_grid[mask]
    
    # Weighted Average of Squared Errors
    w_mse = np.average(diff_sq, weights=weights)
    return np.sqrt(w_mse)

# Calculate for all conditions
gof_metrics = []

for (cond0, ztime), Z_obs in raw_heatmaps.items():
    # Retrieve matching predictions (average across bootstraps)
    relevant_preds = [p['grid'] for p in predictions_list 
                      if p['key'][1]==cond0 and p['key'][2]==ztime]
    
    if not relevant_preds:
        print(f"⚠️ No predictions for {cond0}|{ztime}")
        continue
    
    Z_pred = np.nanmean(np.array(relevant_preds), axis=0)
    
    # Calculate prediction uncertainty (std across bootstraps)
    Z_pred_std = np.nanstd(np.array(relevant_preds), axis=0)
    
    # Retrieve weights
    if (cond0, ztime) in weight_heatmaps:
        Z_weight = weight_heatmaps[(cond0, ztime)]
        wrmse = calculate_wrmse(Z_obs, Z_pred, Z_weight)
        
        # Calculate additional metrics
        mask = ~np.isnan(Z_obs) & ~np.isnan(Z_pred)
        if mask.any():
            residuals = Z_obs[mask] - Z_pred[mask]
            mae = np.mean(np.abs(residuals))
            
            # R-squared weighted
            ss_res = np.sum(Z_weight[mask] * residuals**2)
            ss_tot = np.sum(Z_weight[mask] * (Z_obs[mask] - np.average(Z_obs[mask], weights=Z_weight[mask]))**2)
            r2_weighted = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
            
            gof_metrics.append({
                'cond0': cond0,
                'ztime': ztime,
                'wRMSE': wrmse,
                'MAE': mae,
                'R2_weighted': r2_weighted,
                'n_predictions': len(relevant_preds),
                'mean_pred_uncertainty': np.nanmean(Z_pred_std)
            })

gof_df = pd.DataFrame(gof_metrics)

if len(gof_df) > 0:
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # RMSE
    sns.barplot(data=gof_df, x='cond0', y='wRMSE', hue='ztime', ax=axes[0])
    axes[0].set_ylabel('Weighted RMSE')
    axes[0].set_xlabel('Condition')
    axes[0].set_title('Model Fit Quality (Lower is Better)')
    axes[0].tick_params(axis='x', rotation=45)
    
    # R-squared
    sns.barplot(data=gof_df, x='cond0', y='R2_weighted', hue='ztime', ax=axes[1])
    axes[1].set_ylabel('Weighted R²')
    axes[1].set_xlabel('Condition')
    axes[1].set_title('Explained Variance (Higher is Better)')
    axes[1].axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, 'model_goodness_of_fit_robust.png'), 
               dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nGoodness of Fit Summary:")
    print(gof_df.to_string(index=False))
    
    print("\nOverall Statistics:")
    print(f"  Mean wRMSE: {gof_df['wRMSE'].mean():.4f} ± {gof_df['wRMSE'].std():.4f}")
    print(f"  Mean R²: {gof_df['R2_weighted'].mean():.4f} ± {gof_df['R2_weighted'].std():.4f}")
    print(f"  Mean prediction uncertainty: {gof_df['mean_pred_uncertainty'].mean():.4f}")

#%% ============================================================================
# STEP 7: Visualize Observed vs Predicted for Each Condition
# ==============================================================================

print("\n" + "="*80)
print("STEP 7: Creating Observed vs Predicted Heatmaps")
print("="*80)

for key in list(raw_heatmaps.keys()):
    cond0_ex, ztime_ex = key
    
    # Get the Observed Grid
    Z_observed = raw_heatmaps[key]
    
    # Get the Predicted Grid (average across bootstraps)
    relevant_preds = [
        item['grid']
        for item in predictions_list
        if item['key'][1] == cond0_ex and item['key'][2] == ztime_ex
    ]
    
    if len(relevant_preds) == 0:
        print(f"  Skipping {cond0_ex}|{ztime_ex} - no predictions")
        continue
    
    Z_predicted = np.nanmean(np.array(relevant_preds), axis=0)
    Z_pred_std = np.nanstd(np.array(relevant_preds), axis=0)
    
    # Compute Residuals
    Z_residual = Z_observed - Z_predicted
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot Observed
    im1 = axes[0, 0].imshow(Z_observed.T, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 0].set_title(f"Observed Hazard\n{cond0_ex} | {ztime_ex}")
    axes[0, 0].set_xlabel('Posture Bin')
    axes[0, 0].set_ylabel('Time Bin')
    plt.colorbar(im1, ax=axes[0, 0], label='Hazard')
    
    # Plot Predicted
    im2 = axes[0, 1].imshow(Z_predicted.T, aspect='auto', cmap="viridis", 
                             origin='lower', interpolation='nearest')
    axes[0, 1].set_title(f"Model Prediction (Mean)")
    axes[0, 1].set_xlabel('Posture Bin')
    axes[0, 1].set_ylabel('Time Bin')
    plt.colorbar(im2, ax=axes[0, 1], label='Predicted Probability')
    
    # Plot Residuals
    limit = np.nanpercentile(np.abs(Z_residual), 95)  # Use 95th percentile
    im3 = axes[1, 0].imshow(Z_residual.T, aspect='auto', cmap="RdBu_r", 
                             origin='lower', interpolation='nearest',
                             vmin=-limit, vmax=limit)
    axes[1, 0].set_title("Residuals (Obs - Pred)\nRed=Underestimated, Blue=Overestimated")
    axes[1, 0].set_xlabel('Posture Bin')
    axes[1, 0].set_ylabel('Time Bin')
    plt.colorbar(im3, ax=axes[1, 0], label='Residual')
    
    # Plot Prediction Uncertainty (Std across bootstraps)
    im4 = axes[1, 1].imshow(Z_pred_std.T, aspect='auto', cmap="Reds", 
                             origin='lower', interpolation='nearest')
    axes[1, 1].set_title("Prediction Uncertainty (SD)")
    axes[1, 1].set_xlabel('Posture Bin')
    axes[1, 1].set_ylabel('Time Bin')
    plt.colorbar(im4, ax=axes[1, 1], label='Standard Deviation')
    
    plt.suptitle(f'Model Diagnostics: {cond0_ex} | {ztime_ex}\n({len(relevant_preds)} bootstrap samples)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'diagnostics_{cond0_ex}_{ztime_ex}.png'), 
               dpi=150, bbox_inches='tight')

print(f"  Saved {len(raw_heatmaps)} diagnostic plots")

#%% ============================================================================
# STEP 8: Save Results
# ==============================================================================

print("\n" + "="*80)
print("STEP 8: Saving Results")
print("="*80)

# Save all results
aic_df.to_csv(os.path.join(fig_dir, 'bootstrap_results_robust.csv'), index=False)
alpha_df.to_csv(os.path.join(fig_dir, 'selected_alphas_robust.csv'), index=False)
coef_df.to_csv(os.path.join(fig_dir, 'coefficients_robust.csv'), index=False)
gof_df.to_csv(os.path.join(fig_dir, 'goodness_of_fit_robust.csv'), index=False)
importance_df.to_csv(os.path.join(fig_dir, 'variable_importance_robust.csv'), index=False)

print("\nFiles saved:")
print(f"  - bootstrap_results_robust.csv ({len(aic_df)} rows)")
print(f"  - selected_alphas_robust.csv ({len(alpha_df)} rows)")
print(f"  - coefficients_robust.csv ({len(coef_df)} rows)")
print(f"  - goodness_of_fit_robust.csv ({len(gof_df)} rows)")
print(f"  - variable_importance_robust.csv ({len(importance_df)} rows)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
print(f"\nAll outputs saved to: {fig_dir}")

#%%
# compare within z time across age


if len(coef_df_full) == 0:
    print("⚠️ No coefficient data available")

# =========================================================================
# PLOT 1: Coefficient Trajectories Across Development
# =========================================================================

# Get unique variables
all_variables = coef_df_full['variable'].unique()
all_variables = [v for v in all_variables if v != 'const']  # Exclude intercept for now

if len(all_variables) == 0:
    print("⚠️ No variables to plot")

# Create separate plots for day and night
for time_period in ['day', 'night']:
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Get variables selected for this time period
    time_variables = coef_time['variable'].unique()
    time_variables = [v for v in time_variables if v != 'const']
    
    n_vars = len(time_variables)
    n_cols = min(3, n_vars)
    n_rows = int(np.ceil(n_vars / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_vars == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for var_idx, var in enumerate(time_variables):
        ax = axes[var_idx]
        
        var_data = coef_time[coef_time['variable'] == var].copy()
        
        # Sort by age
        var_data['age'] = var_data['cond0'].astype(int)
        var_data = var_data.sort_values('age')
        
        # Plot individual bootstrap estimates (light)
        for age in var_data['age'].unique():
            age_data = var_data[var_data['age'] == age]
            ax.scatter([age]*len(age_data), age_data['coefficient'],
                        alpha=0.15, s=20, color='gray', zorder=1)
        
        # Calculate summary statistics per age
        summary = var_data.groupby('age')['coefficient'].agg([
            'mean', 'median', 
            ('q25', lambda x: x.quantile(0.25)),
            ('q75', lambda x: x.quantile(0.75)),
            ('q05', lambda x: x.quantile(0.05)),
            ('q95', lambda x: x.quantile(0.95)),
            'count'
        ]).reset_index()
        
        # Plot median trajectory
        ax.plot(summary['age'], summary['median'], 'o-',
                linewidth=3, markersize=12, color='crimson',
                label='Median', zorder=3, markeredgecolor='white', 
                markeredgewidth=2)
        
        # Plot IQR
        ax.fill_between(summary['age'], summary['q25'], summary['q75'],
                        alpha=0.3, color='crimson', label='IQR (25-75%)', zorder=2)
        
        # Plot 90% CI
        ax.fill_between(summary['age'], summary['q05'], summary['q95'],
                        alpha=0.15, color='crimson', label='90% CI', zorder=1)
        
        # Zero line
        ax.axhline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
        
        # Styling
        ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax.set_title(f'{var}\n({time_period})', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_xticks(summary['age'])
        
        # Add sample size annotation
        for _, row in summary.iterrows():
            ax.text(row['age'], ax.get_ylim()[1]*0.95, f"n={int(row['count'])}",
                    ha='center', fontsize=8, color='gray',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.7, edgecolor='gray'))
        
        # Statistical test: Test for trend across ages
        if len(summary) >= 3:
            from scipy.stats import spearmanr
            corr, pval = spearmanr(summary['age'], summary['median'])
            trend_text = f"Trend: r={corr:.3f}, p={pval:.3f}"
            trend_color = 'green' if pval < 0.05 else 'gray'
            ax.text(0.02, 0.02, trend_text, transform=ax.transAxes,
                    fontsize=9, color=trend_color, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', 
                            alpha=0.8, edgecolor=trend_color, linewidth=2))
    
    # Hide unused subplots
    for idx in range(n_vars, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Coefficient Development: {time_period.upper()}\n' +
                f'Trajectories across ages (with bootstrap uncertainty)',
                fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'coefficient_trajectories_{time_period}.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"✓ Saved: coefficient_trajectories_{time_period}.png")


# =========================================================================
# PLOT 2: Coefficient Stability Across Development
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate coefficient of variation (CV) for each variable at each age
    stability = coef_time.groupby(['cond0', 'variable'])['coefficient'].agg([
        'mean', 'std', 'count',
        ('cv', lambda x: np.abs(x.std() / x.mean()) if x.mean() != 0 else np.nan)
    ]).reset_index()
    
    # Exclude const
    stability = stability[stability['variable'] != 'const']
    
    # Pivot for heatmap
    stability_pivot = stability.pivot(index='variable', columns='cond0', values='cv')
    
    # Sort by mean CV
    stability_pivot['mean_cv'] = stability_pivot.mean(axis=1)
    stability_pivot = stability_pivot.sort_values('mean_cv').drop('mean_cv', axis=1)
    
    sns.heatmap(stability_pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                cbar_kws={'label': 'Coefficient of Variation'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                vmin=0, vmax=2, center=0.5)
    
    ax.set_title(f'{time_period.upper()}: Coefficient Stability\n' +
                f'(Lower CV = More Stable Across Bootstraps)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'coefficient_stability_by_age.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: coefficient_stability_by_age.png")


# =========================================================================
# PLOT 3: Effect Size Comparison (Standardized Coefficients)
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate median absolute coefficient per variable per age
    effect_size = coef_time.groupby(['cond0', 'variable'])['coefficient'].agg([
        ('median_abs', lambda x: np.median(np.abs(x))),
        'median',
        'count'
    ]).reset_index()
    
    effect_size = effect_size[effect_size['variable'] != 'const']
    
    # Pivot
    effect_pivot = effect_size.pivot(index='variable', columns='cond0', 
                                        values='median_abs')
    
    # Sort by mean effect size
    effect_pivot['mean_effect'] = effect_pivot.mean(axis=1)
    effect_pivot = effect_pivot.sort_values('mean_effect', ascending=False).drop('mean_effect', axis=1)
    
    # Create horizontal bar plot
    ages = sorted(effect_pivot.columns)
    x_pos = np.arange(len(effect_pivot.index))
    width = 0.25
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(ages)))
    
    for age_idx, age in enumerate(ages):
        offset = (age_idx - len(ages)/2 + 0.5) * width
        bars = ax.barh(x_pos + offset, effect_pivot[age], width,
                        label=f'{age} dpf', color=colors[age_idx],
                        edgecolor='white', linewidth=1.5, alpha=0.8)
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(effect_pivot.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Median Absolute Coefficient', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Effect Sizes Across Development',
                fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='x')
    ax.axvline(0, color='black', linewidth=1)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'effect_sizes_by_age.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: effect_sizes_by_age.png")


# =========================================================================
# PLOT 4: Sign Consistency Check
# =========================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        continue
    
    # Calculate proportion of positive coefficients
    sign_consistency = coef_time.groupby(['cond0', 'variable'])['coefficient'].apply(
        lambda x: (x > 0).sum() / len(x)
    ).reset_index(name='prop_positive')
    
    sign_consistency = sign_consistency[sign_consistency['variable'] != 'const']
    
    # Pivot
    sign_pivot = sign_consistency.pivot(index='variable', columns='cond0', 
                                        values='prop_positive')
    
    # Plot
    sns.heatmap(sign_pivot, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0.5, vmin=0, vmax=1,
                cbar_kws={'label': 'Proportion Positive'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'})
    
    ax.set_title(f'{time_period.upper()}: Sign Consistency\n' +
                f'(0.0=Always Negative, 0.5=Mixed, 1.0=Always Positive)',
                fontsize=13, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'coefficient_sign_consistency.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("✓ Saved: coefficient_sign_consistency.png")


# =========================================================================
# STATISTICAL SUMMARY
# =========================================================================

print("\n" + "="*80)
print("COEFFICIENT DEVELOPMENTAL ANALYSIS SUMMARY")
print("="*80)

for time_period in ['day', 'night']:
    print(f"\n{'='*80}")
    print(f"{time_period.upper()} COEFFICIENTS")
    print('='*80)
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    
    if len(coef_time) == 0:
        print(f"  No data for {time_period}")
        continue
    
    coef_time['age'] = coef_time['cond0'].astype(int)
    
    for var in sorted(coef_time['variable'].unique()):
        if var == 'const':
            continue
        
        print(f"\n  📊 {var}")
        print(f"  {'-'*76}")
        
        var_data = coef_time[coef_time['variable'] == var]
        
        for age in sorted(var_data['age'].unique()):
            age_data = var_data[var_data['age'] == age]['coefficient']
            
            median_val = age_data.median()
            q25, q75 = age_data.quantile([0.25, 0.75])
            prop_positive = (age_data > 0).mean()
            
            print(f"    {age} dpf: median={median_val:+.4f} " +
                    f"[IQR: {q25:+.4f} to {q75:+.4f}] " +
                    f"({prop_positive*100:.0f}% positive, n={len(age_data)})")
        
        # Test for developmental trend
        if len(var_data['age'].unique()) >= 3:
            summary = var_data.groupby('age')['coefficient'].median()
            corr, pval = spearmanr(summary.index, summary.values)
            
            if pval < 0.05:
                trend = "↑ INCREASING" if corr > 0 else "↓ DECREASING"
                print(f"\n    *** {trend} with age (r={corr:.3f}, p={pval:.4f}) ***")
            else:
                print(f"\n    → No significant trend (r={corr:.3f}, p={pval:.4f})")

print("\n" + "="*80)


#%%
#%% ============================================================================
# VARIABLE IMPORTANCE ANALYSIS (Proper Methods)
# ==============================================================================


# =========================================================================
# METHOD 1: Selection Frequency (Most Robust)
# =========================================================================

print("\n" + "="*80)
print("METHOD 1: SELECTION FREQUENCY (Stability-Based Importance)")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    aic_time = aic_df_full[aic_df_full['ztime'] == time_period].copy()
    
    if len(aic_time) == 0:
        continue
    
    # Calculate selection frequency across ALL ages and bootstraps
    exploded = aic_time.explode('selected_vars')
    total_possible = len(aic_time)
    
    selection_counts = exploded['selected_vars'].value_counts()
    selection_freq = (selection_counts / total_possible * 100).sort_values(ascending=True)
    
    # Plot
    colors = plt.cm.RdYlGn(selection_freq.values / 100)
    bars = ax.barh(range(len(selection_freq)), selection_freq.values, color=colors)
    ax.set_yticks(range(len(selection_freq)))
    ax.set_yticklabels(selection_freq.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Selection Frequency (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Variable Importance\n' +
                f'(How often selected across all ages & bootstraps)',
                fontsize=13, fontweight='bold')
    ax.axvline(80, color='red', linestyle='--', linewidth=2, alpha=0.5, label='80% threshold')
    ax.axvline(50, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='50% threshold')
    ax.set_xlim(0, 100)
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=10)
    
    # Add percentage labels
    for i, (var, freq) in enumerate(selection_freq.items()):
        ax.text(freq + 2, i, f'{freq:.0f}%', va='center', fontsize=10, fontweight='bold')
    
    # Print rankings
    # print(f"\n{time_period.upper()} - Selection Frequency Rankings:")
    # print("-"*50)
    # for rank, (var, freq) in enumerate(selection_freq.items()[::-1], 1):
    #     stars = '***' if freq > 80 else '**' if freq > 50 else '*' if freq > 20 else ''
    #     print(f"  {rank}. {var:12s}: {freq:5.1f}% {stars}")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'importance_selection_frequency.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved: importance_selection_frequency.png")


# =========================================================================
# METHOD 2: Standardized Effect Sizes (Median Absolute Coefficient)
# =========================================================================

print("\n" + "="*80)
print("METHOD 2: STANDARDIZED EFFECT SIZES")
print("="*80)
print("Note: Coefficients are already on standardized predictors (z-scored)")
print("So magnitude comparison is valid WITHIN each model")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

for time_idx, time_period in enumerate(['day', 'night']):
    ax = axes[time_idx]
    
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    coef_time = coef_time[coef_time['variable'] != 'const']
    
    if len(coef_time) == 0:
        continue
    
    # Calculate median absolute coefficient (across all ages & bootstraps)
    effect_sizes = coef_time.groupby('variable')['coefficient'].apply(
        lambda x: np.median(np.abs(x))
    ).sort_values(ascending=True)
    
    # Plot
    colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(effect_sizes)))
    ax.barh(range(len(effect_sizes)), effect_sizes.values, color=colors)
    ax.set_yticks(range(len(effect_sizes)))
    ax.set_yticklabels(effect_sizes.index, fontsize=11, fontweight='bold')
    ax.set_xlabel('Median |Coefficient|', fontsize=12, fontweight='bold')
    ax.set_title(f'{time_period.upper()}: Effect Size\n' +
                f'(Larger = Stronger influence on hazard)',
                fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (var, effect) in enumerate(effect_sizes.items()):
        ax.text(effect + 0.01, i, f'{effect:.3f}', va='center', fontsize=9)
    
    # # Print rankings
    # print(f"\n{time_period.upper()} - Effect Size Rankings:")
    # print("-"*50)
    # for rank, (var, effect) in enumerate(effect_sizes.items()[::-1], 1):
    #     print(f"  {rank}. {var:12s}: {effect:.4f}")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'importance_effect_sizes.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved: importance_effect_sizes.png")


# =========================================================================
# METHOD 3: Combined Importance Score
# =========================================================================

print("\n" + "="*80)
print("METHOD 3: COMBINED IMPORTANCE SCORE")
print("="*80)
print("Combines: Selection Frequency × Effect Size")
print("Rationale: Important if BOTH frequently selected AND large effect")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for time_idx, time_period in enumerate(['day', 'night']):
    # Get selection frequency
    aic_time = aic_df_full[aic_df_full['ztime'] == time_period].copy()
    exploded = aic_time.explode('selected_vars')
    total_possible = len(aic_time)
    selection_freq = (exploded['selected_vars'].value_counts() / total_possible)
    
    # Get effect sizes
    coef_time = coef_df_full[coef_df_full['ztime'] == time_period].copy()
    coef_time = coef_time[coef_time['variable'] != 'const']
    effect_sizes = coef_time.groupby('variable')['coefficient'].apply(
        lambda x: np.median(np.abs(x))
    )
    
    # Normalize both to 0-1 scale
    selection_norm = (selection_freq - selection_freq.min()) / (selection_freq.max() - selection_freq.min())
    effect_norm = (effect_sizes - effect_sizes.min()) / (effect_sizes.max() - effect_sizes.min())
    
    # Combine (geometric mean to require both to be high)
    combined = pd.DataFrame({
        'selection': selection_norm,
        'effect': effect_norm
    }).fillna(0)
    combined['importance'] = np.sqrt(combined['selection'] * combined['effect'])
    combined = combined.sort_values('importance', ascending=False)
    
    # Scatter plot: Selection vs Effect Size
    ax_scatter = axes[time_idx, 0]
    scatter = ax_scatter.scatter(selection_freq * 100, effect_sizes, 
                                    s=200, alpha=0.7, 
                                    c=combined['importance'], 
                                    cmap='YlOrRd', edgecolors='black', linewidth=2)
    
    for var in combined.index:
        if var in selection_freq.index and var in effect_sizes.index:
            ax_scatter.annotate(var, 
                                (selection_freq[var] * 100, effect_sizes[var]),
                                fontsize=10, fontweight='bold', ha='center',
                                xytext=(0, 10), textcoords='offset points')
    
    ax_scatter.set_xlabel('Selection Frequency (%)', fontsize=12, fontweight='bold')
    ax_scatter.set_ylabel('Median |Coefficient|', fontsize=12, fontweight='bold')
    ax_scatter.set_title(f'{time_period.upper()}: Selection vs Effect Size\n' +
                        f'(Upper right = Most important)',
                        fontsize=13, fontweight='bold')
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.axhline(effect_sizes.median(), color='red', linestyle='--', alpha=0.3)
    ax_scatter.axvline(50, color='red', linestyle='--', alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax_scatter)
    cbar.set_label('Combined Importance', fontsize=11, fontweight='bold')
    
    # Bar plot: Combined importance
    ax_bar = axes[time_idx, 1]
    colors_combined = plt.cm.YlOrRd(combined['importance'].values)
    ax_bar.barh(range(len(combined)), combined['importance'].values, 
                color=colors_combined, edgecolor='black', linewidth=1.5)
    ax_bar.set_yticks(range(len(combined)))
    ax_bar.set_yticklabels(combined.index, fontsize=11, fontweight='bold')
    ax_bar.set_xlabel('Combined Importance Score', fontsize=12, fontweight='bold')
    ax_bar.set_title(f'{time_period.upper()}: Combined Importance\n' +
                    f'(√(Selection × Effect))',
                    fontsize=13, fontweight='bold')
    ax_bar.set_xlim(0, 1)
    ax_bar.grid(True, alpha=0.3, axis='x')
    
    # Add scores
    for i, (var, score) in enumerate(combined['importance'].items()):
        ax_bar.text(score + 0.02, i, f'{score:.3f}', va='center', 
                    fontsize=9, fontweight='bold')
    
    # Print rankings
    print(f"\n{time_period.upper()} - Combined Importance Rankings:")
    print("-"*60)
    print(f"{'Rank':<6} {'Variable':<12} {'Select%':<10} {'Effect':<10} {'Combined':<10}")
    print("-"*60)
    for rank, var in enumerate(combined.index, 1):
        sel = selection_freq.get(var, 0) * 100
        eff = effect_sizes.get(var, 0)
        comb = combined.loc[var, 'importance']
        print(f"{rank:<6} {var:<12} {sel:>7.1f}%  {eff:>9.4f}  {comb:>9.4f}")

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'importance_combined_score.png'),
            dpi=200, bbox_inches='tight', facecolor='white')
plt.show()
print("\n✓ Saved: importance_combined_score.png")


# =========================================================================
# METHOD 4: Age-Specific Importance
# =========================================================================

print("\n" + "="*80)
print("METHOD 4: AGE-SPECIFIC IMPORTANCE")
print("="*80)

for time_period in ['day', 'night']:
    fig, ax = plt.subplots(figsize=(12, 8))
    
    aic_time = aic_df_full[aic_df_full['ztime'] == time_period].copy()
    
    if len(aic_time) == 0:
        continue
    
    # Calculate selection frequency per age
    age_importance = []
    
    for age in sorted(aic_time['cond0'].unique()):
        age_data = aic_time[aic_time['cond0'] == age]
        exploded = age_data.explode('selected_vars')
        sel_freq = exploded['selected_vars'].value_counts() / len(age_data)
        
        for var, freq in sel_freq.items():
            age_importance.append({
                'age': age,
                'variable': var,
                'frequency': freq
            })
    
    imp_df = pd.DataFrame(age_importance)
    imp_pivot = imp_df.pivot(index='variable', columns='age', values='frequency').fillna(0)
    
    # Plot heatmap
    sns.heatmap(imp_pivot * 100, annot=True, fmt='.0f', cmap='YlOrRd',
                cbar_kws={'label': 'Selection Frequency (%)'},
                ax=ax, linewidths=2, linecolor='white',
                annot_kws={'fontsize': 10, 'fontweight': 'bold'},
                vmin=0, vmax=100)
    
    ax.set_title(f'{time_period.upper()}: Variable Importance Across Development\n' +
                f'(How selection patterns change with age)',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Age (dpf)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Variable', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'importance_by_age_{time_period}.png'),
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.show()
    print(f"\n✓ Saved: importance_by_age_{time_period}.png")
    
    # Print age-specific rankings
    print(f"\n{time_period.upper()} - Age-Specific Rankings:")
    for age in sorted(imp_pivot.columns):
        print(f"\n  {age} dpf:")
        print("  " + "-"*40)
        age_ranks = imp_pivot[age].sort_values(ascending=False)
        for rank, (var, freq) in enumerate(age_ranks.items(), 1):
            if freq > 0:
                print(f"    {rank}. {var:<12s}: {freq*100:>5.1f}%")

print("\n" + "="*80)


# =========================================================================
# SUMMARY TABLE
# =========================================================================

print("\n" + "="*80)
print("INTERPRETATION GUIDE")
print("="*80)
print("""
METHOD 1 - SELECTION FREQUENCY (Primary)
    → Most robust: How often Lasso selects this variable
    → High frequency = Consistently important across conditions
    → Use this as your MAIN importance metric

METHOD 2 - EFFECT SIZE (Secondary)
    → How much the variable changes hazard (per 1 SD change)
    → Only compare WITHIN the same model
    → Interpretation: log(hazard ratio) per SD unit change

METHOD 3 - COMBINED SCORE (Best Overall)
    → Balances frequency AND magnitude
    → Upper-right quadrant = "Star variables"
    → Use for final conclusions

METHOD 4 - AGE-SPECIFIC (Developmental)
    → Which variables matter at which ages
    → Reveals developmental transitions
    → Use to tell a developmental story

RECOMMENDATION:
    1. Report METHOD 3 (Combined) as main result
    2. Support with METHOD 1 (Selection frequency)
    3. Use METHOD 4 to discuss development
    4. DON'T directly compare raw coefficients across variables
""")
print("="*80)


















#%% old Lasso work
