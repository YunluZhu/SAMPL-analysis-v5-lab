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
from collections import Counter, defaultdict
import itertools


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
    tv_df_all = pd.read_pickle(f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_TVdf_{pick_data}LLDD.pkl')
    print('Loaded TV df from pickle')
except Exception as e:
    print(f"Loading TV df failed ({e}), \n regenerating data...")
    re_analyze = True
    
if re_analyze:
    
    pickle_path = f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_angles_all.pkl'

        # Read from pickle if it exists
    if os.path.exists(pickle_path):
        IBI_angles = pd.read_pickle(pickle_path)
        cond0_all = IBI_angles['cond0'].unique()
        cond1_all = IBI_angles['cond1'].unique()
        print('Loaded IBI angles from pickle')
        group_cols = ["unique_IBI_idx", "cond1", "cond0", "expNum"]

    else:
        print(f"Loading failed, regenerating data...")
        IBI_angles, cond0_all, cond1_all = get_timeseriesIBIangles(root, FRAME_RATE, ztime=which_ztime)

        # Convert only required columns to appropriate dtypes
        IBI_angles["cond1"] = IBI_angles["cond1"].astype("category")
        IBI_angles["ztime"] = IBI_angles["ztime"].astype("category")
    
        # IBI_angles["absTime"] = pd.to_datetime(IBI_angles["absTime"]) # check if this is necessary

        # Create categorical codes for string columns
        IBI_angles["cond0_code"] = pd.Categorical(IBI_angles["cond0"]).codes
        IBI_angles["cond1_code"] = pd.Categorical(IBI_angles["cond1"]).codes

        IBI_angles["unique_IBI_idx"] = (
            IBI_angles["cond0_code"].astype("int64") * 10**14 +
            IBI_angles["cond1_code"].astype("int64") * 10**12 +
            IBI_angles["expNum"].astype("int64") * 10**8 +
            IBI_angles["boxNum"].astype("int64") * 10**4 +
            IBI_angles["epochNum"].astype("int64") * 10**2 +
            IBI_angles["IEI_matchIndex"].astype("int64")
        )
        
        print("> checkpoint after preprocessing")

        IBI_angles.to_pickle(pickle_path)  # smaller file size
        print('> Saved IBI angles to pickle')

    #% 20-40 s
    # split time again based on strict criterior using day_night_split, to ensure no time overlap between day and night

    cols_needed = ["absTime", "yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "angVelSmoothed", "ang", "y",'heady','swimSpeed']
    # only select ld
    IBI_angles['cond1'] = IBI_angles['cond1'].astype('category')    
    df_time_filtered = IBI_angles[cols_needed].loc[~(IBI_angles['cond1'] == 'ld')].copy()
    
    # Compute group max (IBI duration) and filter in-place

    print("> checkpoint after early filtering")

    #% release memory
    del IBI_angles
    gc.collect()
    
    # ===== SECTION 1: Initial calculations =====
    # Single groupby - reuse it
    g = df_time_filtered.groupby('unique_IBI_idx', observed=True, sort=False)

    # 1. Calculate time_relative_s (in-place)
    group_min = g["absTime"].transform("min")
    df_time_filtered["time_relative_s"] = (df_time_filtered["absTime"] - group_min).dt.total_seconds()

    # 2. Smooth swimSpeed - optimized with pre-allocation
    speed = df_time_filtered['swimSpeed'].to_numpy()
    out = speed.copy()

    for idx in g.indices.values():
        if len(idx) >= 11:
            out[idx] = savgol_filter(speed[idx], 11, 3)

    df_time_filtered['swimSpeed_smoothed'] = out
    del speed, out  # Free memory

    # 3. Calculate heady_vel - avoid intermediate columns
    time_rel = df_time_filtered['time_relative_s'].to_numpy()
    heady = df_time_filtered['heady'].to_numpy()

    time_next = g['time_relative_s'].shift(-1).to_numpy()
    time_prev = g['time_relative_s'].shift(1).to_numpy()
    heady_next = g['heady'].shift(-1).to_numpy()
    heady_prev = g['heady'].shift(1).to_numpy()

    df_time_filtered['heady_vel'] = (heady_next - heady_prev) / (time_next - time_prev)

    del time_next, time_prev, heady_next, heady_prev  # Free memory

    # 4. Ztime classification - direct categorical assignment
    df_time_filtered['ztime'] = df_time_filtered['cond1'].map({'ll': 'day', 'dd': 'night'}).astype('category')

    # ===== SECTION 2: Speed threshold filtering (0.5) =====
    SPEED_THRESHOLD = 0.5

    # Calculate start times directly without copy
    mask = df_time_filtered['swimSpeed_smoothed'] <= SPEED_THRESHOLD
    start_times = df_time_filtered.loc[mask].groupby('unique_IBI_idx')['time_relative_s'].min()

    # Filter and adjust in one step
    df_time_filtered['start_time'] = df_time_filtered['unique_IBI_idx'].map(start_times)
    mask_valid = df_time_filtered['time_relative_s'] >= df_time_filtered['start_time']

    df_truncated = df_time_filtered.loc[mask_valid].copy()
    df_truncated['time_relative_s'] = df_truncated['time_relative_s'] - df_truncated['start_time']
    df_truncated.drop(columns=['start_time'], inplace=True)

    del df_time_filtered  # Free memory if no longer needed

    # ===== SECTION 3: Speed threshold filtering (2.0) =====
    SPEED_THRESHOLD = 2

    mask = df_truncated['swimSpeed_smoothed'] <= SPEED_THRESHOLD
    end_times = df_truncated.loc[mask].groupby('unique_IBI_idx')['time_relative_s'].max()

    df_truncated['end_time'] = df_truncated['unique_IBI_idx'].map(end_times)
    mask_valid = df_truncated['time_relative_s'] <= df_truncated['end_time']

    df_truncated2 = df_truncated.loc[mask_valid].copy()
    df_truncated2.drop(columns=['end_time'], inplace=True)

    del df_truncated  # Free memory

    # ===== SECTION 4: Angular acceleration filter =====
    mask_before_08s = df_truncated2["time_relative_s"] < 0.8

    # Use boolean indexing instead of loc for speed
    ang_accel_median = (
        df_truncated2.loc[mask_before_08s]
        .groupby("unique_IBI_idx", observed=True)["angVelSmoothed"]
        .apply(lambda x: x.diff().median())
    )

    valid_IBI_idx2 = ang_accel_median.index[ang_accel_median > -0.03]
    df_passed_QC = df_truncated2[df_truncated2['unique_IBI_idx'].isin(valid_IBI_idx2)].copy()

    del df_truncated2  # Free memory

    # ===== SECTION 5: Remove non-contiguous IBIs =====
    # Optimized: use numpy for counting
    consecutive_change = df_passed_QC['unique_IBI_idx'].to_numpy()[1:] != df_passed_QC['unique_IBI_idx'].to_numpy()[:-1]
    ibi_order = df_passed_QC['unique_IBI_idx'].to_numpy()[np.concatenate([[True], consecutive_change])]

    ibi_counts = Counter(ibi_order)
    non_contiguous_ibi = [ibi for ibi, count in ibi_counts.items() if count > 1]

    print(f"Removing {len(non_contiguous_ibi)} non-contiguous IBIs")

    df_filtered = df_passed_QC.loc[~df_passed_QC['unique_IBI_idx'].isin(non_contiguous_ibi)].copy()

    print("Original rows:", len(df_passed_QC), "Cleaned rows:", len(df_filtered))
    print("Unique IBIs remaining:", df_filtered['unique_IBI_idx'].nunique())

    del df_passed_QC  # Free memory

    # ===== SECTION 6: Final filtering =====
    # Combine filters to avoid multiple copies
    g_filtered = df_filtered.groupby('unique_IBI_idx', observed=True, sort=False)

    # Filter 1: first time_relative_s == 0
    mask_starts_at_zero = g_filtered['time_relative_s'].transform('min') == 0

    # Filter 2: drift < 1
    ang_first = g_filtered['ang'].transform('first')
    ang_last = g_filtered['ang'].transform('last')
    mask_drift = (ang_last - ang_first) < 1

    # Apply both filters at once
    df_negative_drift = df_filtered.loc[mask_starts_at_zero & mask_drift].copy()

    del df_filtered  # Free memory

    print("> checkpoint after further filtering; df_negative_drift ready")

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

for lighting_name, TIME_STEP in [('ll', TIME_STEP_LIGHT), ('dd', TIME_STEP_DARK)]:
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
    index_to_assign = tv_df_short['cond1'] == lighting_name
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

for (cond0, cond1), df_sub in tv_df_short.groupby(['cond0', 'cond1'], observed=True):

    # Precompute indices for IBI-level bootstrap
    unique_ibis = df_sub['unique_IBI_idx'].unique()

    for b in tqdm(range(n_boot), desc=f'Bootstrapping {cond0}, {cond1}'):
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
            this_hazard_df.assign(boot=b, cond0=cond0, cond1=cond1)
        )

boot_hazard_df = pd.concat(boot_hazard, ignore_index=True)
boot_hazard_df['hazard_smooth'] = boot_hazard_df.groupby(['boot','cond0','cond1'], observed=True)['hazard'].transform(
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

boot_hazard_df['total_IBIs'] = boot_hazard_df.groupby(['boot','cond0','cond1'], observed=True)['n_IBIs'].transform('sum')
boot_hazard_df['IBI_ratio'] = boot_hazard_df['n_IBIs'] / boot_hazard_df['total_IBIs']
# -----------------------------
# Step 2: Fit GLM per cond0/cond1
# -----------------------------
# average but only if enough data points
# Minimum number of data points required per group
# min_count = n_boot/2

# # Compute group size and conditional averages

boot_hazard_df_average_raw = (
    boot_hazard_df
    .groupby(['cond0','cond1','ang0_bin','t_bin'], observed=True)
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
                                                L1_wt=1,
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
for condition, sample_group in tqdm(boot_hazard_df.groupby(['cond1', 'cond0','boot'], observed=True),   desc='Processing conditions'):
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
            'cond1': condition[0],
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
            'Cond1': res['cond1'],
            'Variable': var,
            'Selected': 1 if var in res['survivors'] else 0,
            'Boot': res['boot']
        })

summary_df = pd.DataFrame(lasso_summary)


selection_freq = summary_df.groupby(['Cond1','Condition', 'Variable'])['Selected'].mean()*100
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

#%%%%% okay let's plot with this model
#%%
model_vars = [
    'ang_z',
    # 'ang_t',
    'angvel_t',
    'angvel_z',
    'azABS_t',
    'azABS_z',
    't_z',
    # 'yvel_t',
    'yvel_z',
]
    # 'a2_t',
    # 'vel2_z',
    # 'acc_t'
    
results = []

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

def fit_binomial_model(df, model_vars):
    """Fit a binomial GLM to the data."""
    X = sm.add_constant(df[model_vars])
    n_fail = df['n_risk'] - df['n_event_smooth']
    y = np.column_stack([df['n_event_smooth'], n_fail])
    
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()

# Main loop
results = []
for (boot, cond0, cond1), group_df in boot_hazard_df.groupby(
    ['boot', 'cond0', 'cond1'], observed=True
):
    df, feature_stats = create_features(group_df)
    model = fit_binomial_model(df, model_vars)
    
    results.append({
        'boot': boot,
        'cond0': cond0,
        'cond1': cond1,
        'model': model,
        **feature_stats  # Unpack all the mu/sd stats
    })

# -----------------------------
# Step 3: Build predicted heatmaps
# -----------------------------
a_grid = {}
a_grid['ll'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['cond1'] == 'll', 'ang0_bin'].astype(float).unique())
a_grid['dd'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['cond1'] == 'dd', 'ang0_bin'].astype(float).unique())
t_grid = {}
t_grid['ll'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['cond1'] == 'll', 't_mid'].unique())
t_grid['dd'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['cond1'] == 'dd', 't_mid'].unique())

TT = {}
AA = {}
AA['ll'], TT['ll'] = np.meshgrid(a_grid['ll'], t_grid['ll'], indexing='ij')
AA['dd'], TT['dd'] = np.meshgrid(a_grid['dd'], t_grid['dd'], indexing='ij')

from collections import defaultdict
pred_heatmaps_smooth_glm = defaultdict(list)


for r in tqdm(results):
    this_boot_avg = boot_hazard_df.loc[
        (boot_hazard_df['cond0'] == r['cond0']) &
        (boot_hazard_df['cond1'] == r['cond1']) &
        (boot_hazard_df['boot'] == r['boot'])
    ].copy()

    # Precompute z-scored covariates at observed bins
    this_boot_avg['a'] = this_boot_avg['ang0_bin'].astype(float)
    this_boot_avg['t'] = this_boot_avg['t_mid'].astype(float)

    this_boot_avg['a_z'] = (this_boot_avg['a'] - r['a_mu']) / r['a_sd']
    this_boot_avg['t_z'] = (this_boot_avg['t'] - r['t_mu']) / r['t_sd']
    this_boot_avg['angvel_z'] = (
        (this_boot_avg['angvel'] - r['angvel_mu']) / r['angvel_sd']
    )
    this_boot_avg['ang_z'] = (
        (this_boot_avg['ang'] - r['ang_mu']) / r['ang_sd']
    )
    this_boot_avg['yvel_z'] = (
        (this_boot_avg['yvel'] - r['yvel_mu']) / r['yvel_sd']
    )
    this_boot_avg['heady_vel_z'] = (
        (this_boot_avg['heady_vel'] - r['heady_vel_mu']) / r['heady_vel_sd']
    )
    
    # this_boot_avg['acc_z'] = (
    #     (this_boot_avg['angacc'] - r['acc_mu']) / r['acc_sd']
    # )
    this_boot_avg['a_t'] = this_boot_avg['a_z'] * this_boot_avg['t_z']
    this_boot_avg['ang_t'] = this_boot_avg['ang_z'] * this_boot_avg['t_z']
    this_boot_avg['angvel_t'] = this_boot_avg['angvel_z'] * this_boot_avg['t_z']
    this_boot_avg['yvel_t'] = this_boot_avg['yvel_z'] * this_boot_avg['t_z']
    this_boot_avg['heady_vel_t'] = this_boot_avg['heady_vel_z'] * this_boot_avg['t_z']
    this_boot_avg['a2_z'] = (
        (this_boot_avg['a_z']**2 - r['a2_mu']) / r['a2_sd']
    )

    # this_boot_avg['acc_t'] = this_boot_avg['acc_z'] * this_boot_avg['t_z']
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
    
    mask_obs = np.zeros_like(AA[r['cond1']], dtype=bool)

    Z = np.full(AA[r['cond1']].shape, np.nan)    
    
    a_to_i = {a: i for i, a in enumerate(a_grid[r['cond1']])}
    t_to_j = {t: j for j, t in enumerate(t_grid[r['cond1']])}

    for _, row in this_boot_avg.iterrows():
        i = a_to_i[row['a']]
        j = t_to_j[row['t']]
        Z[i, j] = row['p_hat']

    # for i, a_val in enumerate(a_grid[r['ztime']]):
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

    pred_heatmaps_smooth_glm[(r['boot'], r['cond0'], r['cond1'])] = Z_clipped


# --- Collect GLM predictions in long format ---
glm_long = []

for (boot, cond0, cond1), Z in pred_heatmaps_smooth_glm.items():
    for i, a_val in enumerate(a_grid[cond1]):
        for j, t_val in enumerate(t_grid[cond1]):
            glm_long.append({
                "boot": boot,
                "cond0": cond0,
                "cond1": cond1,
                "ang0_bin": a_val,
                "t_mid": t_val,
                "hazard_pred": Z[i, j]
            })

glm_long_df = pd.DataFrame(glm_long)

glm_mean_df = (
    glm_long_df
    .groupby(["cond0", "cond1", "ang0_bin", "t_mid"], observed=True)
    ["hazard_pred"]
    .mean()
    .reset_index()
)

pred_heatmaps_mean = {}

for (cond0, cond1), df_sub in glm_mean_df.groupby(["cond0", "cond1"], observed=True):
    Z_glm_df = df_sub.pivot_table(
        index="ang0_bin",
        columns="t_mid",
        values="hazard_pred",
        fill_value=np.nan
    )
    pred_heatmaps_mean[(cond0, cond1)] = Z_glm_df


# --- Prepare raw hazard heatmaps ---
raw_heatmaps = {}
for (cond0, cond1), df_sub in boot_hazard_df_average_raw.groupby(['cond0', 'cond1'], observed=True):
    # Create empty heatmap
    Z_raw = df_sub.pivot_table(
        index='ang0_bin', 
        columns='t_mid', 
        values='hazard_smooth', 
        fill_value=np.nan,
        observed=True
    )
    raw_heatmaps[(cond0, cond1)] = Z_raw
    

# # --- Plot comparison ---
# for condition in pred_heatmaps_mean.keys():
#     Z_glm = pred_heatmaps_mean[condition]
#     Z_raw = raw_heatmaps[condition]

#     fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True, sharey=True)

#     # Raw hazard
#     im0 = axes[0].pcolormesh(TT, AA, Z_raw, shading='auto', cmap='viridis', vmin=0, 
#                             #  vmax=min(np.nanmax(Z_glm), np.nanmax(Z_raw))
#                             vmax=0.3
#                              )
#     axes[0].set_title(f'Raw hazard (Cond={condition[0]}, Ztime={condition[1]})')
#     axes[0].set_xlabel('Time (s)')
#     axes[0].set_ylabel('Initial posture (ang0)')
#     fig.colorbar(im0, ax=axes[0], label='P(event)')

#     # GLM-predicted hazard
#     im1 = axes[1].pcolormesh(TT, AA, Z_glm, shading='auto', cmap='viridis', vmin=0,
#                             #  vmax=min(np.nanmax(Z_glm), np.nanmax(Z_raw))
#                                 vmax=0.3
#                              )
#     axes[1].set_title('GLM-predicted hazard')
#     axes[1].set_xlabel('Time (s)')
#     fig.colorbar(im1, ax=axes[1], label='P(event)')
#     fig.savefig(os.path.join(fig_dir, f'glm_vs_raw{condition[0]}_{condition[1]}_({model_vars}).pdf'))
#     plt.tight_layout()
#     plt.show()


#%%
vmax = 0.3  # shared scale for fair comparison

palette_cmap = {}
palette_cmap['ll'] = sns.light_palette("#1d2e5e",as_cmap=True)
palette_cmap['dd'] = sns.light_palette("#4c0a4c", as_cmap=True)


for (cond0, cond1) in pred_heatmaps_mean.keys():

    Z_glm_df = pred_heatmaps_mean[(cond0, cond1)]
    Z_raw_df = raw_heatmaps[(cond0, cond1)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        sharex=True,
        sharey=True
    )

    # -----------------------------
    # Raw hazard
    # -----------------------------
    sns.heatmap(
        Z_raw_df,
        ax=axes[0],
        cmap=palette_cmap[cond1],
        vmin=0,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "P(event)"},
        rasterized=True
    )

    axes[0].invert_yaxis()
    axes[0].set_title(f"Raw hazard\nCond={cond0}, {cond1}")
    axes[0].set_xlabel("Time into IBI (s)")
    axes[0].set_ylabel("Posture quantile")

    # -----------------------------
    # GLM-predicted hazard
    # -----------------------------
    sns.heatmap(
        Z_glm_df,
        ax=axes[1],
        cmap=palette_cmap[cond1],
        vmin=0,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "P(event)"},
        rasterized=True
    )

    axes[1].invert_yaxis()
    axes[1].set_title("GLM-predicted hazard")
    axes[1].set_xlabel("Time into IBI (s)")
    axes[1].set_ylabel("")

    # -----------------------------
    # Pretty ticks (optional but recommended)
    # -----------------------------
    set_pretty_ticks(
        axes[0],
        x_mids=Z_raw_df.columns.values,
        y_mids=Z_raw_df.index.values,
        x_step=10,
        y_step=2
    )

    set_pretty_ticks(
        axes[1],
        x_mids=Z_glm_df.columns.values,
        y_mids=Z_glm_df.index.values,
        x_step=10,
        y_step=2
    )

    for ax in axes:
        ax.grid(False)

    plt.tight_layout()
    plt.savefig(
        os.path.join(
            fig_dir,
            f"glm_vs_raw_cond{cond0}_{cond1}.pdf"
        ),
        format="pdf"
    )

#%% RMSE per boot
from scipy.stats import pearsonr



rmse_per_boot = []

for (boot, cond0, cond1), Z_glm in pred_heatmaps_smooth_glm.items():

    df_raw = boot_hazard_df[
        (boot_hazard_df['boot'] == boot) &
        (boot_hazard_df['cond0'] == cond0) &
        (boot_hazard_df['cond1'] == cond1)
    ]

    Z_raw = np.full(Z_glm.shape, np.nan)
    W = np.zeros(Z_glm.shape)

    a_to_i = {a: i for i, a in enumerate(a_grid[cond1])}
    t_to_j = {t: j for j, t in enumerate(t_grid[cond1])}

    for _, row in df_raw.iterrows():
        a = float(row['ang0_bin'])
        t = row['t_mid']
        if a in a_to_i and t in t_to_j:
            i = a_to_i[a]
            j = t_to_j[t]
            Z_raw[i, j] = row['hazard']
            W[i, j] = row['n_risk']

    mask = ~np.isnan(Z_glm) & ~np.isnan(Z_raw) & (W > 0)




    if mask.sum() < 2:
        r = np.nan
    else:
        r, _ = pearsonr(
            Z_raw[mask].ravel(),
            Z_glm[mask].ravel()
        )



    if mask.sum() == 0:
        rmse = np.nan
        rmse_weighted = np.nan
    else:
        diff2 = (Z_glm - Z_raw) ** 2

        rmse = np.sqrt(np.mean(diff2[mask]))

        rmse_weighted = np.sqrt(
            np.sum(W[mask] * diff2[mask]) / np.sum(W[mask])
        )

    rmse_per_boot.append({
        "boot": boot,
        "cond0": cond0,
        "cond1": cond1,
        "RMSE": rmse,
        "RMSE_weighted": rmse_weighted,
        "r": r,
    })

rmse_boot_df = pd.DataFrame(rmse_per_boot)

#%%
g = sns.catplot(
    data=rmse_boot_df,
    x='cond0',
    y='RMSE',
    errorbar='sd',
    kind='point',
    units='boot',
    height=2.5,
    linestyle='none',
    aspect=1,
    marker='_',
    col='cond1',
    # palette=palette_cmap
)

plt.savefig(os.path.join(fig_dir, f'glm_RMSE_by_condition_({model_vars}).pdf'))

g = sns.catplot(
    data=rmse_boot_df,
    x='cond0',
    y='RMSE_weighted',
    errorbar='sd',
    units='boot',
    kind='point',
    linestyle='none',
    height=2.5,
    aspect=1,
    marker='_',
    col='cond1',
)
plt.savefig(os.path.join(fig_dir, f'glm_RMSE_weighted_by_condition_({model_vars}).pdf'))


g = sns.catplot(
    data=rmse_boot_df,
    x='cond0',
    y='r',
    errorbar='sd',
    units='boot',
    kind='point',
    linestyle='none',
    height=2.5,
    aspect=1,
    marker='_',
    col='cond1',
)
plt.savefig(os.path.join(fig_dir, f'glm_correlation_r_by_condition_({model_vars}).pdf'))
#%% # for every bin, how far the predicted results are from the empirical results
# scatter plot of predicted vs raw hazard

colors_dayNight = {
    'll': "#1d2e5e",
    'dd': "#4c0a4c"
}

rows = []

for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
    Z_raw = raw_heatmaps[condition]

    mask = ~np.isnan(Z_raw) & ~np.isnan(Z_glm)

    rows.append(
        pd.DataFrame({
            "raw_hazard": Z_raw[mask].values.ravel(),
            "glm_hazard": Z_glm[mask].values.ravel(),
            "cond0": condition[0],
            "cond1": condition[1],
        })
    )

df_scatter = pd.concat(rows, ignore_index=True)

g = sns.relplot(
    data=df_scatter,
    x="raw_hazard",
    y="glm_hazard",
    col="cond0",
    row="cond1",
    kind="scatter",
    alpha=0.17,
    s=30,
    linewidth=0,
    height=2.5,
    hue='cond1',
    palette=colors_dayNight,
    facet_kws=dict(sharex=True, sharey=True)
)
g.set(
    xlim=(0, 0.35),
    ylim=(0, 0.35),
    xticks=np.arange(0, 0.4, 0.1),
    yticks=np.arange(0, 0.4, 0.1),
)
for ax in g.axes.flat:
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

plt.savefig(os.path.join(fig_dir, f'glm_predicted_vs_raw_hazard_({model_vars}).pdf'))

# %%
# Deviance explained

def compute_deviance_explained(results):
    """
    Compute deviance explained per bootstrap per condition.

    Returns
    -------
    pd.DataFrame with columns:
        boot, cond0, cond1, deviance_explained
    """
    rows = []
    for r in results:
        model = r['model']
        D_model = model.deviance
        D_null = model.null_deviance

        dev_exp = 1.0 - (D_model / D_null)

        rows.append({
            'boot': r['boot'],
            'cond0': r['cond0'],
            'cond1': r['cond1'],
            'deviance_explained': dev_exp
        })

    return pd.DataFrame(rows)

devexp_df = compute_deviance_explained(results)


plt_categorical_combined_3(
    data=devexp_df,
    x='cond0',
    col='cond1',
    y='deviance_explained',
    hue='cond1',
    units='boot',
    errorbar='sd',
    overlay_func=False,
    height=2.5,
    aspect=1.2,
)

plt.savefig(
    os.path.join(
        fig_dir,
        f'glm_deviance_explained_by_condition_({model_vars}).pdf'
    )
)
#%%
def apply_zscore(series, mu, sd):
    """Apply z-score transformation using stored parameters."""
    return (series - mu) / sd

def build_design_matrix(df, r, model_vars):
    """
    Reconstruct GLM design matrix for a given bootstrap result r.
    """
    df = df.copy()
    
    # Base variables
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)
    
    # Apply stored z-score transformations
    base_features = {
        'a_z': (df['a'], 'a'),
        't_z': (df['t'], 't'),
        'angvel_z': (df['angvel'], 'angvel'),
        'ang_z': (df['ang'], 'ang'),
        'yvel_z': (df['yvel'], 'yvel'),
        'heady_vel_z': (df['heady_vel'], 'heady_vel'),
    }
    
    for feat_name, (series, stat_prefix) in base_features.items():
        df[feat_name] = apply_zscore(series, r[f'{stat_prefix}_mu'], r[f'{stat_prefix}_sd'])
    
    # Transformed features (z-scored quadratics and absolutes)
    transformed_features = {
        'aABS_z': (np.abs(df['a']), 'aABS'),
        'azABS_z': (np.abs(df['a_z']), 'azABS'),
    }
    
    for feat_name, (series, stat_prefix) in transformed_features.items():
        df[feat_name] = apply_zscore(series, r[f'{stat_prefix}_mu'], r[f'{stat_prefix}_sd'])
    
    # Time interactions
    time_interactions = {
        'a_t': 'a_z',
        'ang_t': 'ang_z',
        'angvel_t': 'angvel_z',
        'yvel_t': 'yvel_z',
        'heady_vel_t': 'heady_vel_z',
        'aABS_t': 'aABS_z',
        'azABS_t': 'azABS_z',
    }
    
    for interaction_name, base_feature in time_interactions.items():
        df[interaction_name] = df[base_feature] * df['t_z']
    
    X = sm.add_constant(df[model_vars], has_constant='add')
    return X

TERM_GROUPS = {
    'Time': ['t_z'],
    'Posture': ['a_z', 'azABS_z', 'ang_z','angvel_z'],
    'Posture-Time Interactions': ['a_t', 'ang_t', 'angvel_t', 'aABS_t', 'azABS_t'],
    'Kinematic': ['yvel_t','yvel_z']
}

def compute_delta_deviance_explained(results, boot_hazard_df, model_vars):
    """
    Compute delta deviance explained by dropping each term group.
    """
    rows = []

    for r in results:
        boot, cond0, cond1 = r['boot'], r['cond0'], r['cond1']

        df = boot_hazard_df.query(
            "boot == @boot and cond0 == @cond0 and cond1 == @cond1"
        ).copy()

        # Response
        n_fail = df['n_risk'] - df['n_event_smooth']
        y = np.column_stack([df['n_event_smooth'], n_fail])

        # Full model deviances
        D_full = r['model'].deviance
        D_null = r['model'].null_deviance

        # Full design matrix
        X_full = build_design_matrix(df, r, model_vars)

        # Test each term group
        for term, drop_vars in TERM_GROUPS.items():
            keep_vars = [v for v in model_vars if v not in drop_vars]
            X_reduced = sm.add_constant(X_full[keep_vars], has_constant='add')

            reduced_model = sm.GLM(
                y, X_reduced, family=sm.families.Binomial()
            ).fit()

            delta_dev_exp = (reduced_model.deviance - D_full) / D_null

            rows.append({
                'boot': boot,
                'cond0': cond0,
                'cond1': cond1,
                'term': term,
                'delta_deviance_explained': delta_dev_exp
            })

    return pd.DataFrame(rows)

delta_dev_df = compute_delta_deviance_explained(
    results,
    boot_hazard_df,
    model_vars
)


plt_categorical_combined_3(
    data=delta_dev_df,
    x='term',
    col='cond1',
    row='cond0',
    y='delta_deviance_explained',
    hue='cond1',
    errorbar='sd',
    units='boot',
    overlay_func=False,
    height=2.5,
    aspect=1,
    # x_order=['P', 'T', 'PT', 'AVT', 'YVT'],
)
plt.savefig(os.path.join(fig_dir, f'glm_delta_deviance_explained_by_term_({model_vars}).pdf'))


plt_categorical_combined_3(
    data=delta_dev_df,
    x='cond0',
    row='cond1',
    col='term',
    y='delta_deviance_explained',
    hue='cond1',
    errorbar='sd',
    units='boot',
    overlay_func=False,
    height=2.5,
    aspect=1,
    # col_order=['P', 'T', 'PT', 'AVT', 'YVT'],
)
plt.savefig(os.path.join(fig_dir, f'glm_delta_deviance_explained_by_condition_and_term_({model_vars}).pdf'))



#%%
coef_rows = []

for r in results:
    params = r['model'].params
    conf = r['model'].conf_int()
    
    for name in params.index:
        coef_rows.append({
            'boot': r['boot'],
            'cond0': r['cond0'],
            'cond1': r['cond1'],
            'param': name,
            'coef': params[name],
            'ci_low': conf.loc[name, 0],
            'ci_high': conf.loc[name, 1],
        })

coef_df = pd.DataFrame(coef_rows)
coef_df = coef_df.query("param != 'const'")
coef_df['abs_coef'] = coef_df['coef'].abs()
for (cond0, cond1), df_sub in coef_df.groupby(['cond0','cond1'], observed=True):
    plt.figure(figsize=(6,4))
    sns.pointplot(
        data=df_sub,
        x='param',
        y='coef',
        errorbar='sd',
        join=False
    )
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'GLM coefficients (Cond={cond0}, Cond1={cond1})')
    plt.ylabel('Coefficient (log-odds)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'glm_coefficients_cond{cond0}_cond1{cond1}_({model_vars}).pdf'))

sns.catplot(
    data=coef_df,
    x='cond0',
    col='param',
    row='cond1',
    y='coef',
    errorbar='sd',
    kind='point',
    linestyle='none',
    height=3,
    col_order=sorted(model_vars)
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'glm_coefficients_by_param_({model_vars}).pdf'))
