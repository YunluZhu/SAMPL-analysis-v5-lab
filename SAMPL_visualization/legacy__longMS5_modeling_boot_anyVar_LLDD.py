# this works. large variation in results across experiments
# terminal velocity are different
# why don't we just calculate terminal speed

# with bin 0-3 s and speed threshold 0.5, we get same results by fitting on median vs raw data

#%%
# import sys
import itertools
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
from collections import Counter
import statsmodels.genmod.generalized_linear_model as glm
glm.SET_USE_BIC_LLF(True)

from scipy.ndimage import gaussian_filter1d


def pd_zscore(s):
    return (s - s.mean()) / s.std(ddof=0)

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

#%% 
re_analyze = False
try:
    tv_df_all = pd.read_pickle(f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_TVdf_{pick_data}.pkl')
    print('Loaded TV df from pickle')
except Exception as e:
    print(f"Loading TV df failed ({e}), \n regenerating data...")
    re_analyze = True
    
if re_analyze:
    pickle_path = f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_angles_all.pkl'

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

    #% 20-40 s
    

    cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "y",'heady','swimSpeed']
    # only select ld
    IBI_angles['cond1'] = IBI_angles['cond1'].astype('category')    
    df_time_filtered = IBI_angles[cols_needed].loc[IBI_angles['cond1'] == 'ld']

    # Compute group max (IBI duration) and filter in-place

    print("> checkpoint after early filtering")

    #% release memory
    del IBI_angles
    gc.collect()

    g = df_time_filtered.groupby('unique_IBI_idx', observed=True)

    speed = df_time_filtered['swimSpeed'].to_numpy()
    out = np.empty_like(speed)

    for idx in g.indices.values():
        vals = speed[idx]
        if len(vals) >= 11:
            out[idx] = savgol_filter(vals, 11, 3)
        else:
            out[idx] = vals

    df_time_filtered['swimSpeed_smoothed'] = out

    dt = g['time_relative_s'].shift(-1) - g['time_relative_s'].shift(1)

    df_time_filtered['heady_vel'] = (
        (g['heady'].shift(-1) - g['heady'].shift(1)) / dt
    )

    #%
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

    #%
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

    #% further cleaning: remove IBIs with high velocity after 1s, likely representing translocation
    df_valid = df_truncated2
    # #%%
    before_08s_mask = df_valid["time_relative_s"] < 0.8
    before = df_valid.loc[before_08s_mask]
    stats2 = before.groupby("unique_IBI_idx", observed=True).agg(
        ang_accel=("angVelSmoothed", lambda x: x.diff().median()),
    )
    valid_IBI_idx2 = stats2.index[stats2.ang_accel > -0.03]


    #% remove non-contiguous IBIs

    # df_filtered = df_valid[df_valid.unique_IBI_idx.isin(np.intersect1d(valid_IBI_idx2, valid_IBI_idx))].copy()
    df_passed_QC = df_valid[df_valid.unique_IBI_idx.isin(valid_IBI_idx2)].copy()


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

    # %
    df_filtered_ld = df_filtered.copy()
    # drop unique_IBI_idx that have the first time_relative_s not equal to 0
    df_filtered_ld = df_filtered_ld[df_filtered_ld.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

    # Compute drift per unique_IBI_idx
    drift = df_filtered_ld.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

    # Filter rows where drift < 1
    df_negative_drift = df_filtered_ld[drift < 1].copy()

    #%

    MAX_T = 10  # max IBI time (seconds) for censoring

    # -----------------------------
    # Prepare tv_df_all
    # -----------------------------
    tv_df_all_ = []

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
            
            yvel = g['yvel'].values
            heady_vel = g['heady_vel'].values
            
            y_cumu = g['y'].values - g['y'].values[0]
            heady_cumu = g['heady'].values - g['heady'].values[0]

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
                    'yvel': yvel[i],
                    'heady_vel': heady_vel[i],
                    'y_cumu': y_cumu[i],
                    'heady_cumu': heady_cumu[i],
                    **meta
                })

        tv_df = pd.DataFrame(rows)

        if tv_df.empty:
            continue
        
        # -----------------------------
        # Standardize covariates
        # -----------------------------
        tv_df['ang0_z'] = pd_zscore(tv_df['ang0'])

        tv_df['log_time'] = np.log(tv_df['stop'])
        clip_lo = np.floor(tv_df['log_time'].min())
        clip_hi = np.ceil(tv_df['log_time'].max())
        tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

        # --- Interactions ---
        tv_df['ang0_z_logt'] = pd_zscore(
            tv_df['ang0_z'] * tv_df['log_time_clipped']
        )

        # --- Quadratic term ---
        tv_df['ang0_z2'] = pd_zscore(tv_df['ang0_z'] ** 2)

        tv_df['ang0_z2_logt'] = pd_zscore(
            tv_df['ang0_z2'] * tv_df['log_time_clipped']
        )

        # --- Other standardized predictors ---
        for col in ['yvel', 'heady_vel', 'y_cumu', 'heady_cumu', 'ang']:
            tv_df[f'{col}_z'] = pd_zscore(tv_df[col])
        tv_df_all_.append(tv_df)

    tv_df_all = pd.concat(tv_df_all_, ignore_index=True)
    tv_df_all.to_pickle(f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_TVdf_{pick_data}.pkl')  # smaller file size

#% old tvdf:
# tv_df_all = pd.read_pickle('/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_modelPrediction_riskWeighted.pkl')

#%%
n = len(tv_df_all)
ang_sm = np.full(n, np.nan)
angvel = np.full(n, np.nan)
angvel_sm = np.full(n, np.nan)
angacc = np.full(n, np.nan)

# get underlying arrays once
ang = tv_df_all['ang'].to_numpy()
groups = tv_df_all.groupby('unique_IBI_idx', sort=False).indices

for idx in groups.values():
    x = ang[idx]

    # 1. smooth angle
    if len(x) >= 5:
        x_sm = savgol_filter(x, 5, 2)
        ang_sm[idx] = x_sm
    else:
        x_sm = x
        ang_sm[idx] = x

    # 2. velocity
    if len(x_sm) > 1:
        v = np.gradient(x_sm) * FRAME_RATE
        angvel[idx] = v
    else:
        continue  # stays nan downstream

    # 3. smooth velocity
    if len(v) >= 3:
        v_sm = savgol_filter(v, 3, 2)
        angvel_sm[idx] = v_sm
    else:
        v_sm = v
        angvel_sm[idx] = v

    # 4. acceleration
    if len(v_sm) >= 5:
        angacc[idx] = np.gradient(v_sm)

# assign once
tv_df_all['ang_sm'] = ang_sm
tv_df_all['angvel'] = angvel
tv_df_all['angvel_sm'] = angvel_sm
tv_df_all['angacc'] = angacc

# remove IBIs with nan derivatives (same as your original logic)
tv_df_all = tv_df_all[~tv_df_all['ang_sm'].isna()].copy()

#%%
tv_df_short = tv_df_all.copy()


# fd_time = fd_width(tv_df_short["start"])
# fd_ang  = fd_width(tv_df_short["ang"])

# print(f'fd_time: {fd_time}, fd_ang: {fd_ang}')


TIME_STEP_LIGHT = 0.08 # to match sample distribution in light condition
POSTURE_STEP = 5 #snap_step(120 * fd_ang, [0.5, 1, 1.5, 2, 3, 5])
TIME_STEP_DARK = 0.3 #to match sample distribution in dark condition

print(f"Using posture step: {POSTURE_STEP}, time step light: {TIME_STEP_LIGHT}, time step dark: {TIME_STEP_DARK}")

# -----------------------------
# Infer ranges from data
# -----------------------------
t_min_data = tv_df_short["start"].min()
t_max_data = tv_df_short["start"].max()

a_min_data = np.percentile(tv_df_short["ang"], .5)
a_max_data = np.percentile(tv_df_short["ang"], 99.5)
a_min = nice_floor(a_min_data, POSTURE_STEP)
a_max = nice_ceil(a_max_data, POSTURE_STEP)
a_edges = np.arange(a_min, a_max + POSTURE_STEP, POSTURE_STEP)
a_bin_mid = (a_edges[:-1] + a_edges[1:]) / 2
tv_df_short['ang0_bin'] = pd.cut(tv_df_short["ang0"], bins=a_edges, include_lowest=True, labels=a_bin_mid)

all_tv_df_sub = []

for zname, TIME_STEP in [('day', TIME_STEP_LIGHT), ('night', TIME_STEP_DARK)]:
    # -----------------------------
    # Snap to nice edges
    # -----------------------------
    t_min = nice_floor(t_min_data, TIME_STEP)
    t_max = nice_ceil(t_max_data, TIME_STEP)

    # -----------------------------
    # Define edges
    # -----------------------------
    t_edges = np.arange(t_min, t_max + TIME_STEP, TIME_STEP)

    # Midpoints (for plotting only)
    t_bin_mid = (t_edges[:-1] + t_edges[1:]) / 2

    # tv_df_short['ang0'] = tv_df_short.groupby('unique_IBI_idx')['ang'].transform('first')
    index_to_assign = tv_df_short['ztime'] == zname
    tv_df_sub = tv_df_short.loc[index_to_assign, :].copy()
    tv_df_sub['t_bin'] = pd.cut(tv_df_sub["start"], bins=t_edges, include_lowest=True, labels=t_bin_mid)
    all_tv_df_sub.append(tv_df_sub)

tv_df_short = pd.concat(all_tv_df_sub, ignore_index=True)


#%%
MIN_IBI_PER_BIN = 40  # removes noisy estimates / spikes
MIN_IBI_PER_POSTURE = 200 # removes postures with too few total IBIs
SIGMA_T = 2.0  # in bins

n_boot = 40

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
            .groupby(['ang0_bin', f't_bin'], observed=True)
            .agg(
                n_risk=('unique_IBI_idx', 'nunique'),
                n_event=('event', 'sum'),
                angvel=('angvel', 'mean'),
                yvel=('yvel', 'mean'),
                heady_vel=('heady_vel', 'mean'),
                y_cumu=('y_cumu', 'mean'),
                heady_cumu=('heady_cumu', 'mean'),
                ang=('ang', 'mean'),
                # angacc=('angacc', 'mean'),
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
        this_hazard_df['t_mid'] = this_hazard_df[f't_bin']#.apply(lambda iv: 0.5*(iv.left + iv.right))
        # grouped_hazard = this_hazard_df.groupby('ang0_bin', observed=True)
        
        boot_hazard.append(
            this_hazard_df.assign(boot=b, cond0=cond0, ztime=ztime)
        )

boot_hazard_df = pd.concat(boot_hazard, ignore_index=True)
boot_hazard_df['hazard_smooth'] = boot_hazard_df.groupby(['boot','cond0','ztime'], observed=True)['hazard'].transform(
    lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest')
)


#%%
# -----------------------------
# Step 1: Adjust counts with smoothed hazard (disabled)
# -----------------------------
# Use hazard_smooth to compute fractional events per bin
boot_hazard_df['n_event_smooth'] = boot_hazard_df['hazard'] * boot_hazard_df['n_risk']

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
        hazard_smooth=('hazard_smooth', 'mean'),
        hazard=('hazard', 'mean'),
        IBI_ratio=('IBI_ratio', 'mean'),
        angvel=('angvel', 'mean'),
        # angacc=('angacc', 'mean'),
    )
    .reset_index()
)

boot_hazard_df_average_raw['t_mid'] = boot_hazard_df_average_raw['t_bin']

#%%
# model_vars = [
#     # 'a_z', 
#     't_z', # 
#     'ang_z', # controls for angle-specific baseline hazard
#     # 'yvel_z', # controls for speed-specific baseline hazard
#     'azABS_z', # this measures deviation from standardized 0 angle 
#     # 'angvel_z', 
#     'a_t', # generates time dependence of posture effect
#     'ang_t',
#     'angvel_t', # prevents late onset 
#     'azABS_t',
#     # 'yvel_t',
#     # 'heady_vel_t',
# ]

#%%
# ---------------------------------------------------------
# Step 1: Define Hypotheses and Compare via AIC/BIC
# ---------------------------------------------------------

# Define the base hypotheses
hypotheses = {
    'PostureOnly': ['t_z','azABS_z', 'ang_z','angvel_z'],
    'PostureInteraction': ['t_z', 'azABS_t', 'ang_t','angvel_t'],
    'yvelInteraction': ['t_z','yvel_t'],
    'yvelZ': ['t_z','yvel_z'],
}

def generate_model_combinations(base_hypotheses):
    """Generates all individual and combined models from base hypotheses."""
    combined_models = {}
    names = list(base_hypotheses.keys())
    
    # Iterate through all combination lengths (1 to 3)
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            combo_name = " + ".join(combo)
            # Combine the variable lists, using set to drop duplicate 't_z'
            combo_vars = list(set(itertools.chain.from_iterable(
                [base_hypotheses[name] for name in combo]
            )))
            # Sort just to keep 't_z' or 'a_z' in a predictable order visually
            combo_vars.sort() 
            combined_models[combo_name] = combo_vars
            
    return combined_models

def compare_models_ic(df, model_dict):
    """
    Fits GLMs for a dictionary of models and returns AIC/BIC rankings.
    Assumes df already has features generated via `create_features(df)`.
    """
    results_list = []
    
    # Construct y once
    n_fail = df['n_risk'] - df['n_event_smooth']
    y = np.column_stack([df['n_event_smooth'], n_fail])
    
    for model_name, vars_list in model_dict.items():
        # Check if all vars exist in the df (prevents KeyError if 'yvel_t' is missing, etc.)
        missing_vars = [v for v in vars_list if v not in df.columns]
        if missing_vars:
            print(f"Skipping '{model_name}': missing {missing_vars}")
            continue
            
        X = sm.add_constant(df[vars_list])
        model = sm.GLM(y, X, family=sm.families.Binomial()).fit()
        
        # statsmodels Binomial GLM natively supports AIC and BIC
        results_list.append({
            'Model': model_name,
            'Variables': ", ".join(vars_list),
            'Num_Vars': len(vars_list),
            'AIC': model.aic,
            'BIC': model.bic,
            'Deviance': model.deviance
        })
        
    # Create DataFrame and sort by AIC (lowest is best)
    comparison_df = pd.DataFrame(results_list).sort_values('AIC').reset_index(drop=True)
    
    # Calculate Delta AIC to easily see how much worse other models are compared to the best
    best_aic = comparison_df['AIC'].iloc[0]
    comparison_df['Delta_AIC'] = comparison_df['AIC'] - best_aic
    
    return comparison_df

# ---------------------------------------------------------
# Step 2: Lasso Regularization to Find Surviving Parameters
# ---------------------------------------------------------

def run_lasso_selection(df, model_vars, alpha=0.01, threshold=1e-4, print_output=True):
    """
    Applies L1 regularization (Lasso) to a specific model to see which parameters survive.
    alpha: penalty weight (higher = more parameters dropped to 0).
    """
    if print_output:
        print(f"\n--- Running Lasso Selection (alpha={alpha}) ---")
    
    X = sm.add_constant(df[model_vars])
    n_fail = df['n_risk'] - df['n_event_smooth']
    y = np.column_stack([df['n_event_smooth'], n_fail])
    
    # L1_wt=1.0 strictly enforces Lasso (0.0 would be Ridge)
    lasso_model = sm.GLM(y, X, family=sm.families.Binomial())
    lasso_results = lasso_model.fit_regularized(method='elastic_net', 
                                                alpha=alpha, 
                                                L1_wt=0.8,
                                                maxiter=2000,
                                                cnvrg_tol=1e-6)
    
    # Extract surviving parameters (ignoring constant)
    # Use a small threshold to account for floating-point inaccuracies
    params = lasso_results.params
    survivors = params[params.abs() > threshold].index.tolist()
    
    # Format output
    if 'const' in survivors:
        survivors.remove('const')
        
    dropped = [v for v in model_vars if v not in survivors]
    
    if print_output:
        print(f"Original variables ({len(model_vars)}): {model_vars}")
        print(f"Surviving variables ({len(survivors)}): {survivors}")
        print(f"Dropped variables ({len(dropped)}): {dropped}")
        print("\nDetailed Coefficients:")
        print(params.round(4))
    
    return survivors

def zscore(series):
    mu = series.mean()
    sd = series.std(ddof=0)
    z = (series - mu) / sd
    return z, mu, sd

def zscore_and_track(series, stats_dict, name):
    """Z-score a series and track its mean/std."""
    mu = series.mean()
    sd = series.std()
    stats_dict[f'{name}_mu'] = mu
    stats_dict[f'{name}_sd'] = sd
    return (series - mu) / sd

def create_features(df):
    """Create all engineered features for the model."""
    df = df.copy()
    stats = {}
    
    # Base variables
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)
    
    # Z-scored linear terms
    df['a_z'] = zscore_and_track(df['a'], stats, 'a')
    df['t_z'] = zscore_and_track(df['t'], stats, 't')
    df['angvel_z'] = zscore_and_track(df['angvel'], stats, 'angvel')
    df['ang_z'] = zscore_and_track(df['ang'], stats, 'ang')
    df['yvel_z'] = zscore_and_track(df['yvel'], stats, 'yvel')
    df['heady_vel_z'] = zscore_and_track(df['heady_vel'], stats, 'heady_vel')

    # Quadratic/transformed terms
    df['a2_z'] = zscore_and_track(df['a_z']**2, stats, 'a2')
    df['aABS_z'] = zscore_and_track(np.abs(df['a']), stats, 'aABS')
    df['azABS_z'] = zscore_and_track(np.abs(df['a_z']), stats, 'azABS')
    
    # Interactions with time
    time_interactions = {
        'a_t': df['a_z'],
        'a2_t': df['a2_z'],
        'angvel_t': df['angvel_z'],
        'aABS_t': df['aABS_z'],
        'azABS_t': df['azABS_z'],
        'yvel_t': df['yvel_z'],
        'heady_vel_t': df['heady_vel_z'],
        'ang_t': df['ang_z'],
    }
    
    for name, feature in time_interactions.items():
        df[name] = feature * df['t_z']
    return df, stats

#%%
lasso_results_list = []
for condition, sample_group in tqdm(boot_hazard_df.groupby(['ztime', 'cond0','boot'], observed=True),   desc='Processing conditions'):
    if condition[2] >= 10:
        continue
    else:
        df_sample, _ = create_features(sample_group)

        # 2. Run Information Criteria comparison
        all_models = generate_model_combinations(hypotheses)
        ic_results = compare_models_ic(df_sample, all_models)
        
        # print(f"\n=== Condition: {condition} ===")

        # print("\n--- Model Comparison (Ranked by AIC) ---")
        # print(ic_results[['Model', 'AIC', 'Delta_AIC', 'BIC']])

        # 3. User selects a model to push through Lasso
        target_model_name = 'PostureOnly + PostureInteraction + yvelInteraction + yvelZ'  # <-- change this to select different models
        target_vars = all_models[target_model_name]

        # 4. Determine survivors
        # You may need to tweak alpha depending on the scale of your y data
        surviving_vars = run_lasso_selection(df_sample, target_vars, alpha=0.05, print_output=False)
        lasso_results_list.append(
            {'cond0': condition[1],
            'ztime': condition[0],
            'survivors': surviving_vars,
            'boot': condition[2],
            'ic_results': ic_results}
    )
        
kk=[]
for item in lasso_results_list:
    kk.append(item['ic_results'])
kkd = pd.concat(kk)
print(kkd.groupby('Model')['Delta_AIC'].sum().sort_values())
#%%
# 1. Collect survivors into a long-format DataFrame
# Assuming you have a list of dicts: lasso_results_list = [{'cond': 'WT', 'boot': 1, 'survivors': [...]}...]
lasso_summary = []
all_possible_vars = [item for sublist in hypotheses.values() for item in sublist]  # flatten list of variables from all hypotheses
all_possible_vars = set(all_possible_vars)  # unique variables
for res in lasso_results_list:
    for var in all_possible_vars:
        lasso_summary.append({
            'Condition': res['cond0'],
            'ZTime': res['ztime'],
            'Variable': var,
            'Selected': 1 if var in res['survivors'] else 0,
            'Boot': res['boot']
        })

summary_df = pd.DataFrame(lasso_summary)


selection_freq = summary_df.groupby(['ZTime','Condition', 'Variable'])['Selected'].mean()*100
selection_freq = selection_freq.unstack('Variable')

# 3. Plot the Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(selection_freq, annot=True, cmap="YlGnBu", vmin=0, vmax=100,fmt='.0f')
plt.title("Variable Selection Frequency (Lasso Consistency)")
plt.ylabel("Condition / Time")
plt.xlabel("Variable")  
plt.show()

# collapse all conditions and plot selection frequency per par
var_freq = selection_freq.median()
var_freq = var_freq.sort_values(ascending=False)
plt.figure(figsize=(4,3))
sns.barplot(x=var_freq.index, y=var_freq.values)
plt.xticks(rotation=45, ha='right')
plt.ylabel("Selection Frequency")
plt.title("Overall Variable Selection Frequency")
plt.ylim(0, 100)
plt.show()

# quantify how many times each variable is selected 80% of the time per condition
threshold = 80
consistent_vars = selection_freq.apply(lambda col: (col >= threshold).sum(), axis=0)
print("Variables selected in at least 80% of bootstraps per condition:")
print(consistent_vars[consistent_vars > 0])
# %%
