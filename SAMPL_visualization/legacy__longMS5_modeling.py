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

        IBI_angles["unique_IBI_idx"] = (
            IBI_angles["expNum"].astype("int64") * 10**12 +
            IBI_angles["boxNum"].astype("int64") * 10**8 +
            IBI_angles["epochNum"].astype("int64") * 10**4 +
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
    df_time_filtered = IBI_angles[cols_needed].loc[(IBI_angles['cond1'] == 'ld')].copy()
    
    # Compute group max (IBI duration) and filter in-place

    print("> checkpoint after early filtering")

    #% release memory
    del IBI_angles
    gc.collect()

    # Single groupby - reuse it
    g = df_time_filtered.groupby(['unique_IBI_idx', 'cond0', 'cond1'], observed=True)

    # 1. Calculate time_relative_s
    group_min = g["absTime"].transform("min")
    df_time_filtered["time_relative_s"] = (df_time_filtered["absTime"] - group_min).dt.total_seconds()

    # 2. Smooth swimSpeed - vectorized approach
    speed = df_time_filtered['swimSpeed'].to_numpy()
    out = speed.copy()  # Start with original values

    for idx in g.indices.values():
        if len(idx) >= 11:
            out[idx] = savgol_filter(speed[idx], 11, 3)

    df_time_filtered['swimSpeed_smoothed'] = out

    # 3. Calculate heady_vel - vectorized shifts
    df_time_filtered['time_rel_next'] = g['time_relative_s'].shift(-1)
    df_time_filtered['time_rel_prev'] = g['time_relative_s'].shift(1)
    df_time_filtered['heady_next'] = g['heady'].shift(-1)
    df_time_filtered['heady_prev'] = g['heady'].shift(1)

    dt = df_time_filtered['time_rel_next'] - df_time_filtered['time_rel_prev']
    df_time_filtered['heady_vel'] = (
        (df_time_filtered['heady_next'] - df_time_filtered['heady_prev']) / dt
    )

    # Clean up temp columns
    df_time_filtered.drop(columns=['time_rel_next', 'time_rel_prev', 'heady_next', 'heady_prev'], inplace=True)

    # 4. Get ztime classification - use index directly
    first_times = g['absTime'].first()

    strict_ztime = day_night_split(
        first_times.reset_index()[['unique_IBI_idx', 'absTime']],
        'absTime', 
        narrow_bin=True,
        ztime=which_ztime
    )

    # Merge instead of map for better performance
    df_time_filtered = df_time_filtered.merge(
        strict_ztime[['unique_IBI_idx', 'ztime']], 
        on='unique_IBI_idx', 
        how='left'
    )

    # Filter
    df_time_filtered.drop(
        df_time_filtered.query("ztime_y not in ['day', 'night']").index,
        inplace=True
    )
    df_time_filtered.rename(columns={'ztime_y': 'ztime'}, inplace=True)
    
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
# Step 1: Adjust counts with smoothed hazard
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
# new model selection


# -----------------------------------------------------------------------------
# 1. DEFINE CANDIDATE MODELS (THEORY-DRIVEN)
# -----------------------------------------------------------------------------

MODEL_CANDIDATES = {
    'baseline': ['t_z'],
    
    'posture': ['t_z', 'a_t', 'ang_t', 'azABS_t'],
    
    'kinematics': ['t_z', 'y_cumu_t', 'yvel_t'],
    
    'posture_kinematics': ['t_z', 'a_t', 'ang_t', 'azABS_t', 'angvel_t'],
    
    'full': ['t_z', 'a_t', 'ang_t', 'angvel_t', 'azABS_t', 'yvel_t', 'y_cumu_t'],
}

print("=" * 80)
print("MODEL SELECTION PIPELINE")
print("=" * 80)
print(f"\nCandidate models defined:")
for name, vars_list in MODEL_CANDIDATES.items():
    print(f"  {name:25s}: {len(vars_list):2d} variables")

# -----------------------------------------------------------------------------
# 2. FEATURE ENGINEERING FUNCTIONS
# -----------------------------------------------------------------------------

def zscore_and_track(series, stats_dict, name):
    """Z-score a series and track its mean/std."""
    mu = series.mean()
    sd = series.std(ddof=0)
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
    df['y_cumu_z'] = zscore_and_track(df['y_cumu'], stats, 'y_cumu')
    df['heady_vel_z'] = zscore_and_track(df['heady_vel'], stats, 'heady_vel')

    # Quadratic/transformed terms
    df['a2_z'] = zscore_and_track(df['a_z']**2, stats, 'a2')
    df['aABS_z'] = zscore_and_track(np.abs(df['a']), stats, 'aABS')
    df['azABS_z'] = zscore_and_track(np.abs(df['a_z']), stats, 'azABS')
    
    # Interactions with time
    time_interactions = {
        'a_t': df['a_z'],
        'y_cumu_t': df['y_cumu_z'],
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

def apply_feature_transforms(df, stats):
    """Apply pre-computed z-score transformations from training data."""
    df = df.copy()
    
    # Base variables
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)
    
    # Apply stored transformations
    for feat in ['a', 't', 'angvel', 'ang', 'yvel', 'heady_vel', 'y_cumu']:
        df[f'{feat}_z'] = (df[feat] - stats[f'{feat}_mu']) / stats[f'{feat}_sd']
    
    # Quadratics and absolutes
    df['a2_z'] = (df['a_z']**2 - stats['a2_mu']) / stats['a2_sd']
    df['aABS_z'] = (np.abs(df['a']) - stats['aABS_mu']) / stats['aABS_sd']
    df['azABS_z'] = (np.abs(df['a_z']) - stats['azABS_mu']) / stats['azABS_sd']
    
    # Interactions
    time_interactions = {
        'a_t': 'a_z',
        'y_cumu_t': 'y_cumu_z',
        'angvel_t': 'angvel_z',
        'aABS_t': 'aABS_z',
        'azABS_t': 'azABS_z',
        'yvel_t': 'yvel_z',
        'heady_vel_t': 'heady_vel_z',
        'ang_t': 'ang_z',
    }
    
    for interaction_name, base_feature in time_interactions.items():
        df[interaction_name] = df[base_feature] * df['t_z']
    
    return df

def fit_binomial_model(df, model_vars):
    """Fit a binomial GLM to the data."""
    X = sm.add_constant(df[model_vars])
    n_fail = df['n_risk'] - df['n_event_smooth']
    y = np.column_stack([df['n_event_smooth'], n_fail])
    
    return sm.GLM(y, X, family=sm.families.Binomial()).fit()

def compute_binomial_deviance(y_true, y_pred, n_risk):
    """
    Compute binomial deviance for model evaluation.
    
    Parameters
    ----------
    y_true : array, shape (n, 2)
        [n_events, n_failures]
    y_pred : array, shape (n,)
        Predicted probabilities
    n_risk : array, shape (n,)
        Number at risk per bin
    """
    n_event = y_true[:, 0]
    n_fail = y_true[:, 1]
    
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, 1e-10, 1 - 1e-10)
    
    # Binomial deviance
    dev = -2 * np.sum(
        n_event * np.log(y_pred) + n_fail * np.log(1 - y_pred)
    )
    
    return dev

# -----------------------------------------------------------------------------
# 3. CROSS-VALIDATION FOR MODEL SELECTION
# -----------------------------------------------------------------------------

def cross_validate_model(df_full, model_vars, n_folds=5):
    """
    Perform k-fold cross-validation using bootstrap samples as units.
    
    Returns per-fold metrics for statistical testing.
    """
    from sklearn.model_selection import KFold
    
    boots = df_full['boot'].unique()
    
    if len(boots) < n_folds:
        n_folds = len(boots)
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_results = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(boots)):
        train_boots = boots[train_idx]
        test_boots = boots[test_idx]
        
        # --- TRAINING ---
        df_train = df_full[df_full['boot'].isin(train_boots)].copy()
        
        try:
            df_train_feat, train_stats = create_features(df_train)
            model = fit_binomial_model(df_train_feat, model_vars)
        except:
            continue
        
        # --- TESTING ---
        df_test = df_full[df_full['boot'].isin(test_boots)].copy()
        df_test_feat = apply_feature_transforms(df_test, train_stats)
        
        X_test = sm.add_constant(df_test_feat[model_vars], has_constant='add')
        y_pred = model.predict(X_test)
        
        # True outcomes
        y_true = np.column_stack([
            df_test['n_event_smooth'],
            df_test['n_risk'] - df_test['n_event_smooth']
        ])
        
        # Deviance
        dev = compute_binomial_deviance(y_true, y_pred, df_test['n_risk'].values)
        
        # RMSE
        y_true_prob = df_test['n_event_smooth'] / df_test['n_risk']
        rmse = np.sqrt(np.mean((y_true_prob - y_pred) ** 2))
        
        # Weighted RMSE
        weights = df_test['n_risk'].values
        rmse_weighted = np.sqrt(np.sum(weights * (y_true_prob - y_pred) ** 2) / np.sum(weights))
        
        fold_results.append({
            'fold': fold_idx,
            'deviance': dev,
            'rmse': rmse,
            'rmse_weighted': rmse_weighted,
        })
    
    return pd.DataFrame(fold_results)

# -----------------------------------------------------------------------------
# 4. FIT ALL MODELS ACROSS BOOTSTRAPS (for paired testing)
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FITTING ALL CANDIDATE MODELS ACROSS BOOTSTRAPS")
print("=" * 80)

# Store per-bootstrap performance for each model
bootstrap_performance = []

for (boot, cond0, ztime), df_boot in tqdm(
    boot_hazard_df.groupby(['boot', 'cond0', 'ztime'], observed=True),
    desc="Fitting models per bootstrap"
):
    
    for model_name, var_list in MODEL_CANDIDATES.items():
        
        try:
            df_feat, stats = create_features(df_boot)
            model = fit_binomial_model(df_feat, var_list)
            
            # Compute metrics
            dev = model.deviance
            dev_null = model.null_deviance
            dev_explained = 1.0 - (dev / dev_null)
            aic = model.aic
            bic = model.bic_llf
            
            # In-sample RMSE
            y_pred = model.predict(sm.add_constant(df_feat[var_list]))
            y_true = df_feat['n_event_smooth'] / df_feat['n_risk']
            rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
            
            bootstrap_performance.append({
                'boot': boot,
                'cond0': cond0,
                'ztime': ztime,
                'model_name': model_name,
                'n_vars': len(var_list),
                'deviance': dev,
                'deviance_explained': dev_explained,
                'AIC': aic,
                'BIC': bic,
                'rmse': rmse,
            })
        except Exception as e:
            print(f"  Warning: Model {model_name} failed for boot={boot}, cond={cond0}, ztime={ztime}")
            continue

bootstrap_perf_df = pd.DataFrame(bootstrap_performance)

print(f"\n✓ Fitted {len(bootstrap_perf_df)} model-bootstrap combinations")

# -----------------------------------------------------------------------------
# 5. CROSS-VALIDATION RESULTS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("CROSS-VALIDATION FOR MODEL COMPARISON")
print("=" * 80)

cv_results_list = []

for (cond0, ztime), df_cond in tqdm(
    boot_hazard_df.groupby(['cond0', 'ztime'], observed=True),
    desc="Cross-validating models"
):
    for model_name, var_list in MODEL_CANDIDATES.items():
        
        fold_results = cross_validate_model(df_cond, var_list, n_folds=5)
        
        if len(fold_results) > 0:
            cv_results_list.append({
                'cond0': cond0,
                'ztime': ztime,
                'model_name': model_name,
                'cv_deviance_mean': fold_results['deviance'].mean(),
                'cv_deviance_sd': fold_results['deviance'].std(),
                'cv_rmse_mean': fold_results['rmse'].mean(),
                'cv_rmse_sd': fold_results['rmse'].std(),
            })

cv_results_df = pd.DataFrame(cv_results_list)

#%%
# -----------------------------------------------------------------------------
# 6. STATISTICAL TESTING: BOOTSTRAP-BASED COMPARISONS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("STATISTICAL TESTING: BOOTSTRAP-BASED MODEL COMPARISONS")
print("=" * 80)

def bootstrap_pairwise_test(df_perf, metric='deviance_explained'):
    """
    Perform pairwise bootstrap comparisons between all models.
    Uses bootstrap distribution to directly calculate p-values.
    
    For metrics where LOWER is better (AIC, BIC, deviance, rmse):
        p-value = proportion of bootstraps where model1 >= model2
    For metrics where HIGHER is better (deviance_explained):
        p-value = proportion of bootstraps where model1 <= model2
    
    Returns DataFrame of pairwise comparisons.
    """
    from itertools import combinations
    
    models = df_perf['model_name'].unique()
    comparisons = []
    
    for model1, model2 in combinations(models, 2):
        # Get paired observations across bootstraps
        df1 = df_perf[df_perf['model_name'] == model1].set_index('boot')[metric]
        df2 = df_perf[df_perf['model_name'] == model2].set_index('boot')[metric]
        
        # Only use boots where both models succeeded
        common_boots = df1.index.intersection(df2.index)
        
        if len(common_boots) < 3:
            continue
        
        vals1 = df1.loc[common_boots].values
        vals2 = df2.loc[common_boots].values
        
        n_boot = len(common_boots)
        
        # Calculate difference
        diff = vals1 - vals2
        mean_diff = np.mean(diff)
        
        # Calculate bootstrap p-value
        # For metrics where lower is better (AIC, BIC, deviance, rmse)
        if metric in ['deviance', 'AIC', 'BIC', 'rmse']:
            # H0: model1 is not better than model2 (i.e., model1 >= model2)
            # p-value = proportion where model1 >= model2
            p_value = np.mean(diff >= 0)
            better_model = model1 if mean_diff < 0 else model2
            better_mean = np.mean(vals1) if mean_diff < 0 else np.mean(vals2)
            worse_mean = np.mean(vals2) if mean_diff < 0 else np.mean(vals1)
        else:
            # For metrics where higher is better (deviance_explained)
            # H0: model1 is not better than model2 (i.e., model1 <= model2)
            # p-value = proportion where model1 <= model2
            p_value = np.mean(diff <= 0)
            better_model = model1 if mean_diff > 0 else model2
            better_mean = np.mean(vals1) if mean_diff > 0 else np.mean(vals2)
            worse_mean = np.mean(vals2) if mean_diff > 0 else np.mean(vals1)
        
        # Two-sided p-value (for when we don't have a directional hypothesis)
        p_value_twosided = 2 * min(p_value, 1 - p_value)
        
        # Calculate 95% confidence interval for the difference
        ci_lower = np.percentile(diff, 2.5)
        ci_upper = np.percentile(diff, 97.5)
        
        # Effect size: standardized difference
        effect_size = mean_diff / np.std(diff, ddof=1)
        
        # Probability that model1 is better
        if metric in ['deviance', 'AIC', 'BIC', 'rmse']:
            prob_model1_better = np.mean(vals1 < vals2)
        else:
            prob_model1_better = np.mean(vals1 > vals2)
        
        comparisons.append({
            'model1': model1,
            'model2': model2,
            'mean1': np.mean(vals1),
            'sd1': np.std(vals1, ddof=1),
            'mean2': np.mean(vals2),
            'sd2': np.std(vals2, ddof=1),
            'mean_diff': mean_diff,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'p_value_onesided': p_value,
            'p_value_twosided': p_value_twosided,
            'effect_size': effect_size,
            'prob_model1_better': prob_model1_better,
            'n_boots': n_boot,
            'better_model': better_model,
            'better_mean': better_mean,
            'worse_mean': worse_mean,
            'significant_onesided': p_value < 0.05,
            'significant_twosided': p_value_twosided < 0.05,
        })
    
    return pd.DataFrame(comparisons)

# Perform pairwise tests for each condition
statistical_comparisons = []

for (cond0, ztime), df_cond_perf in bootstrap_perf_df.groupby(['cond0', 'ztime']):
    
    print(f"\n{'='*70}")
    print(f"Condition: {cond0} dpf, {ztime}")
    print(f"{'='*70}")
    
    # Test on deviance explained (higher is better)
    comparisons_dev = bootstrap_pairwise_test(df_cond_perf, metric='deviance_explained')
    comparisons_dev['metric'] = 'deviance_explained'
    
    # Test on AIC (lower is better)
    comparisons_aic = bootstrap_pairwise_test(df_cond_perf, metric='AIC')
    comparisons_aic['metric'] = 'AIC'
    
    # Test on RMSE (lower is better)
    comparisons_rmse = bootstrap_pairwise_test(df_cond_perf, metric='rmse')
    comparisons_rmse['metric'] = 'rmse'
    
    # Combine
    all_comparisons = pd.concat([comparisons_dev, comparisons_aic, comparisons_rmse])
    all_comparisons['cond0'] = cond0
    all_comparisons['ztime'] = ztime
    
    statistical_comparisons.append(all_comparisons)
    
    # Print summary for this condition
    print("\n" + "-"*70)
    print("DEVIANCE EXPLAINED (higher is better):")
    print("-"*70)
    for _, row in comparisons_dev.sort_values('p_value_onesided').head(5).iterrows():
        sig = "***" if row['p_value_onesided'] < 0.001 else "**" if row['p_value_onesided'] < 0.01 else "*" if row['p_value_onesided'] < 0.05 else "ns"
        print(f"  {row['model1']:20s} vs {row['model2']:20s}")
        print(f"    Better model: {row['better_model']:20s} (p = {row['p_value_onesided']:.4f} {sig})")
        print(f"    Mean diff: {row['mean_diff']:+.4f} [95% CI: {row['ci_lower']:.4f}, {row['ci_upper']:.4f}]")
        print(f"    Prob({row['model1']} better): {row['prob_model1_better']:.1%}")
        print()
    
    print("\n" + "-"*70)
    print("AIC (lower is better):")
    print("-"*70)
    for _, row in comparisons_aic.sort_values('p_value_onesided').head(5).iterrows():
        sig = "***" if row['p_value_onesided'] < 0.001 else "**" if row['p_value_onesided'] < 0.01 else "*" if row['p_value_onesided'] < 0.05 else "ns"
        print(f"  {row['model1']:20s} vs {row['model2']:20s}")
        print(f"    Better model: {row['better_model']:20s} (p = {row['p_value_onesided']:.4f} {sig})")
        print(f"    Mean diff: {row['mean_diff']:+.1f} [95% CI: {row['ci_lower']:.1f}, {row['ci_upper']:.1f}]")
        print(f"    Prob({row['model1']} better): {row['prob_model1_better']:.1%}")
        print()

stat_comparison_df = pd.concat(statistical_comparisons, ignore_index=True)

# Apply Bonferroni correction per condition and metric
stat_comparison_df['p_value_bonf'] = stat_comparison_df.groupby(['cond0', 'ztime', 'metric'])['p_value_onesided'].transform(
    lambda x: np.minimum(x * len(x), 1.0)
)
stat_comparison_df['significant_bonf'] = stat_comparison_df['p_value_bonf'] < 0.05

# Save results
stat_comparison_df.to_csv(
    os.path.join(fig_dir, 'model_comparison_bootstrap_tests.csv'),
    index=False
)

print(f"\n✓ Bootstrap-based statistical comparisons saved")
print(f"  File: model_comparison_bootstrap_tests.csv")

# -----------------------------------------------------------------------------
# 7. MODEL SELECTION WITH BOOTSTRAP STATISTICAL JUSTIFICATION
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("MODEL SELECTION WITH BOOTSTRAP STATISTICAL JUSTIFICATION")
print("=" * 80)

best_models = {}

for (cond0, ztime), df_perf in bootstrap_perf_df.groupby(['cond0', 'ztime']):
    
    # Get mean performance across bootstraps
    model_summary = df_perf.groupby('model_name').agg({
        'AIC': ['mean', 'std'],
        'BIC': ['mean', 'std'],
        'deviance_explained': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'n_vars': 'first',
    }).reset_index()
    
    model_summary.columns = ['model_name', 'AIC_mean', 'AIC_std', 'BIC_mean', 'BIC_std', 
                              'dev_exp_mean', 'dev_exp_std', 'rmse_mean', 'rmse_std', 'n_vars']
    
    # Select model with best AIC
    best_aic_idx = model_summary['AIC_mean'].idxmin()
    best_aic_model = model_summary.loc[best_aic_idx, 'model_name']
    best_aic_nvars = model_summary.loc[best_aic_idx, 'n_vars']
    
    # Get statistical tests for this model vs others
    tests = stat_comparison_df[
        (stat_comparison_df['cond0'] == cond0) &
        (stat_comparison_df['ztime'] == ztime) &
        (stat_comparison_df['metric'] == 'AIC')
    ]
    
    # Check if best model is significantly better than simpler alternatives
    # Start with the best model
    selected_model = best_aic_model
    selection_rationale = "Lowest mean AIC across bootstraps"
    
    # Get all models simpler than the best AIC model
    simpler_models = model_summary[model_summary['n_vars'] < best_aic_nvars].sort_values('n_vars', ascending=False)
    
    for _, simpler in simpler_models.iterrows():
        simpler_name = simpler['model_name']
        simpler_nvars = simpler['n_vars']
        
        # Find comparison between best_aic_model and simpler_name
        comp = tests[
            ((tests['model1'] == best_aic_model) & (tests['model2'] == simpler_name)) |
            ((tests['model1'] == simpler_name) & (tests['model2'] == best_aic_model))
        ]
        
        if len(comp) > 0:
            row = comp.iloc[0]
            
            # Check if the best model is significantly better
            # If best_aic_model is model1
            if row['model1'] == best_aic_model:
                # We want to know: is best_aic_model significantly better (lower AIC)?
                # p_value_onesided is P(model1 >= model2), so if it's high, model1 is NOT significantly better
                is_significantly_better = row['p_value_onesided'] < 0.05  # Low p means model1 < model2 most of the time
                prob_better = 1 - row['p_value_onesided']
            else:
                # best_aic_model is model2
                is_significantly_better = row['p_value_onesided'] >= 0.95  # High p means model1 >= model2, so model2 is better
                prob_better = row['p_value_onesided']
            
            # If complex model is NOT significantly better, choose simpler model (parsimony)
            if not is_significantly_better:
                selected_model = simpler_name
                selection_rationale = f"Parsimony: Not significantly worse than {best_aic_model} (p={row['p_value_onesided']:.3f}), but {best_aic_nvars - simpler_nvars} fewer variables"
                print(f"\n  → Choosing {simpler_name} over {best_aic_model}")
                print(f"     Reason: p={row['p_value_onesided']:.3f} (not significant)")
                print(f"     AIC: {simpler['AIC_mean']:.1f} vs {model_summary.loc[best_aic_idx, 'AIC_mean']:.1f}")
                break  # Take the most complex of the non-significantly-worse simpler models
    
    # Store selection
    selected_row = model_summary[model_summary['model_name'] == selected_model].iloc[0]
    
    best_models[(cond0, ztime)] = {
        'model_name': selected_model,
        'vars': MODEL_CANDIDATES[selected_model],
        'n_vars': selected_row['n_vars'],
        'AIC_mean': selected_row['AIC_mean'],
        'AIC_std': selected_row['AIC_std'],
        'BIC_mean': selected_row['BIC_mean'],
        'BIC_std': selected_row['BIC_std'],
        'deviance_explained_mean': selected_row['dev_exp_mean'],
        'deviance_explained_std': selected_row['dev_exp_std'],
        'rmse_mean': selected_row['rmse_mean'],
        'rmse_std': selected_row['rmse_std'],
        'rationale': selection_rationale,
    }
    
    print(f"\n{'-'*70}")
    print(f"{cond0} dpf, {ztime}:")
    print(f"{'-'*70}")
    print(f"  Selected: {selected_model}")
    print(f"  N variables: {selected_row['n_vars']}")
    print(f"  AIC: {selected_row['AIC_mean']:.1f} ± {selected_row['AIC_std']:.1f}")
    print(f"  Deviance explained: {selected_row['dev_exp_mean']:.2%} ± {selected_row['dev_exp_std']:.2%}")
    print(f"  RMSE: {selected_row['rmse_mean']:.4f} ± {selected_row['rmse_std']:.4f}")
    print(f"  Rationale: {selection_rationale}")
    print(f"  Variables: {MODEL_CANDIDATES[selected_model]}")

# -----------------------------------------------------------------------------
# 8. VISUALIZATION: BOOTSTRAP DISTRIBUTIONS AND COMPARISONS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("GENERATING BOOTSTRAP COMPARISON VISUALIZATIONS")
print("=" * 80)

# Plot 1: Bootstrap distributions of key metrics
fig, axes = plt.subplots(len(best_models), 3, figsize=(15, 4*len(best_models)))

if len(best_models) == 1:
    axes = axes.reshape(1, -1)

for idx, ((cond0, ztime), model_info) in enumerate(best_models.items()):
    
    df_cond = bootstrap_perf_df[
        (bootstrap_perf_df['cond0'] == cond0) &
        (bootstrap_perf_df['ztime'] == ztime)
    ]
    
    selected_model = model_info['model_name']
    
    # AIC
    ax = axes[idx, 0]
    for model_name in MODEL_CANDIDATES.keys():
        df_model = df_cond[df_cond['model_name'] == model_name]
        color = 'red' if model_name == selected_model else 'gray'
        alpha = 1.0 if model_name == selected_model else 0.3
        lw = 2 if model_name == selected_model else 1
        ax.hist(df_model['AIC'], bins=20, alpha=alpha, label=model_name, 
                histtype='step', linewidth=lw, color=color)
    ax.set_xlabel('AIC')
    ax.set_ylabel('Frequency')
    ax.set_title(f'{cond0} dpf, {ztime}\nAIC Distribution')
    ax.legend(fontsize=8)
    ax.axvline(model_info['AIC_mean'], color='red', linestyle='--', lw=2, label='Selected')
    
    # Deviance Explained
    ax = axes[idx, 1]
    for model_name in MODEL_CANDIDATES.keys():
        df_model = df_cond[df_cond['model_name'] == model_name]
        color = 'red' if model_name == selected_model else 'gray'
        alpha = 1.0 if model_name == selected_model else 0.3
        lw = 2 if model_name == selected_model else 1
        ax.hist(df_model['deviance_explained'], bins=20, alpha=alpha, label=model_name,
                histtype='step', linewidth=lw, color=color)
    ax.set_xlabel('Deviance Explained')
    ax.set_ylabel('Frequency')
    ax.set_title('Deviance Explained Distribution')
    ax.axvline(model_info['deviance_explained_mean'], color='red', linestyle='--', lw=2)
    
    # RMSE
    ax = axes[idx, 2]
    for model_name in MODEL_CANDIDATES.keys():
        df_model = df_cond[df_cond['model_name'] == model_name]
        color = 'red' if model_name == selected_model else 'gray'
        alpha = 1.0 if model_name == selected_model else 0.3
        lw = 2 if model_name == selected_model else 1
        ax.hist(df_model['rmse'], bins=20, alpha=alpha, label=model_name,
                histtype='step', linewidth=lw, color=color)
    ax.set_xlabel('RMSE')
    ax.set_ylabel('Frequency')
    ax.set_title('RMSE Distribution')
    ax.axvline(model_info['rmse_mean'], color='red', linestyle='--', lw=2)

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'model_comparison_bootstrap_distributions.pdf'), dpi=300)
plt.show()

# Plot 2: Pairwise comparison matrices (showing probability that model is better)
n_conditions = len(best_models)
fig, axes = plt.subplots(n_conditions, 3, figsize=(15, 5*n_conditions))

if n_conditions == 1:
    axes = axes.reshape(1, -1)

for idx, ((cond0, ztime), model_info) in enumerate(best_models.items()):
    
    for metric_idx, metric in enumerate(['AIC', 'deviance_explained', 'rmse']):
        ax = axes[idx, metric_idx]
        
        # Get comparisons for this condition and metric
        df_comp = stat_comparison_df[
            (stat_comparison_df['cond0'] == cond0) &
            (stat_comparison_df['ztime'] == ztime) &
            (stat_comparison_df['metric'] == metric)
        ]
        
        if len(df_comp) == 0:
            continue
        
        # Create matrix of probabilities
        models = sorted(set(df_comp['model1'].tolist() + df_comp['model2'].tolist()))
        n_models = len(models)
        
        prob_matrix = np.zeros((n_models, n_models))
        sig_matrix = np.zeros((n_models, n_models), dtype=bool)
        
        for i, model_i in enumerate(models):
            for j, model_j in enumerate(models):
                if i == j:
                    prob_matrix[i, j] = 0.5  # Same model
                    continue
                
                # Find comparison
                comp = df_comp[
                    ((df_comp['model1'] == model_i) & (df_comp['model2'] == model_j))
                ]
                
                if len(comp) > 0:
                    prob_matrix[i, j] = comp.iloc[0]['prob_model1_better']
                    sig_matrix[i, j] = comp.iloc[0]['significant_onesided']
                else:
                    # Check reverse
                    comp = df_comp[
                        ((df_comp['model1'] == model_j) & (df_comp['model2'] == model_i))
                    ]
                    if len(comp) > 0:
                        prob_matrix[i, j] = 1 - comp.iloc[0]['prob_model1_better']
                        sig_matrix[i, j] = comp.iloc[0]['significant_onesided']
        
        # Plot heatmap
        im = ax.imshow(prob_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        
        # Add text annotations
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    text = '-'
                else:
                    text = f'{prob_matrix[i, j]:.2f}'
                    if sig_matrix[i, j]:
                        text += '*'
                ax.text(j, i, text, ha='center', va='center', 
                       color='black' if 0.3 < prob_matrix[i, j] < 0.7 else 'white',
                       fontsize=8)
        
        ax.set_xticks(range(n_models))
        ax.set_yticks(range(n_models))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(models, fontsize=9)
        ax.set_xlabel('Model')
        ax.set_ylabel('Model')
        ax.set_title(f'{cond0} dpf, {ztime}\nP(row better than column) - {metric}')
        
        # Highlight selected model
        selected_idx = models.index(model_info['model_name'])
        rect = plt.Rectangle((selected_idx-0.5, selected_idx-0.5), 1, 1, 
                             fill=False, edgecolor='blue', lw=3)
        ax.add_patch(rect)
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Probability')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'model_comparison_probability_matrices.pdf'), dpi=300)
plt.show()

print("✓ Bootstrap comparison visualizations saved")

# -----------------------------------------------------------------------------
# 9. SUMMARY TABLE WITH BOOTSTRAP STATISTICS
# -----------------------------------------------------------------------------

summary_rows = []
for (cond0, ztime), info in best_models.items():
    summary_rows.append({
        'Condition': f"{cond0} dpf",
        'Time': ztime,
        'Selected_Model': info['model_name'],
        'N_vars': info['n_vars'],
        'AIC': f"{info['AIC_mean']:.1f} ± {info['AIC_std']:.1f}",
        'RMSE': f"{info['rmse_mean']:.4f} ± {info['rmse_std']:.4f}",
        'Dev_Explained': f"{info['deviance_explained_mean']:.1%} ± {info['deviance_explained_std']:.1%}",
        'Selection_Rationale': info['rationale'],
        'Variables': ', '.join(info['vars']),
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "=" * 80)
print("FINAL MODEL SELECTION SUMMARY (BOOTSTRAP-BASED)")
print("=" * 80)
print("\n" + summary_df.to_string(index=False))

summary_df.to_csv(os.path.join(fig_dir, 'model_selection_summary_bootstrap_stats.csv'), index=False)

# Also create a detailed comparison table for the paper
detailed_comparison = []
for (cond0, ztime), df_perf in bootstrap_perf_df.groupby(['cond0', 'ztime']):
    model_summary = df_perf.groupby('model_name').agg({
        'AIC': ['mean', 'std'],
        'BIC': ['mean', 'std'],
        'deviance_explained': ['mean', 'std'],
        'rmse': ['mean', 'std'],
        'n_vars': 'first',
    }).reset_index()
    
    model_summary.columns = ['model_name', 'AIC_mean', 'AIC_std', 'BIC_mean', 'BIC_std',
                              'dev_exp_mean', 'dev_exp_std', 'rmse_mean', 'rmse_std', 'n_vars']
    
    model_summary['condition'] = f"{cond0} dpf"
    model_summary['time'] = ztime
    model_summary['selected'] = model_summary['model_name'] == best_models[(cond0, ztime)]['model_name']
    
    detailed_comparison.append(model_summary)

detailed_comparison_df = pd.concat(detailed_comparison, ignore_index=True)
detailed_comparison_df = detailed_comparison_df[[
    'condition', 'time', 'model_name', 'n_vars', 
    'AIC_mean', 'AIC_std', 'BIC_mean', 'BIC_std',
    'dev_exp_mean', 'dev_exp_std', 'rmse_mean', 'rmse_std', 'selected'
]]

detailed_comparison_df.to_csv(
    os.path.join(fig_dir, 'model_comparison_all_models_bootstrap_stats.csv'),
    index=False
)

print(f"\n✓ Detailed comparison table saved")
print(f"  Files generated:")
print(f"    - model_comparison_bootstrap_tests.csv")
print(f"    - model_selection_summary_bootstrap_stats.csv")
print(f"    - model_comparison_all_models_bootstrap_stats.csv")
print(f"    - model_comparison_bootstrap_distributions.pdf")
print(f"    - model_comparison_probability_matrices.pdf")

# -----------------------------------------------------------------------------
# 10. FIT FINAL MODELS WITH SELECTED VARIABLES
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("FITTING FINAL MODELS WITH SELECTED VARIABLES")
print("=" * 80)

results = []

for (boot, cond0, ztime), group_df in tqdm(
    boot_hazard_df.groupby(['boot', 'cond0', 'ztime'], observed=True),
    desc="Fitting final models"
):
    # Use selected model for this condition
    selected_vars = best_models[(cond0, ztime)]['vars']
    
    df, feature_stats = create_features(group_df)
    model = fit_binomial_model(df, selected_vars)
    
    results.append({
        'boot': boot,
        'cond0': cond0,
        'ztime': ztime,
        'model': model,
        'model_name': best_models[(cond0, ztime)]['model_name'],
        'model_vars': selected_vars,
        **feature_stats
    })

print(f"\n✓ Fitted {len(results)} final models")

print("\n" + "=" * 80)
print("MODEL SELECTION WITH BOOTSTRAP TESTING COMPLETE")
print("=" * 80)

# %%
# -----------------------------------------------------------------------------
# 11. VARIABLE IMPORTANCE WITHIN SELECTED MODELS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("TESTING NECESSITY OF INDIVIDUAL PREDICTORS")
print("=" * 80)

def test_variable_necessity(df_full, full_vars, variable_to_drop):
    """
    Test if dropping a single variable significantly worsens model performance.
    
    Returns bootstrap distribution of performance difference.
    """
    boots = df_full['boot'].unique()
    
    performance_full = []
    performance_reduced = []
    
    for boot in boots:
        df_boot = df_full[df_full['boot'] == boot].copy()
        
        try:
            # Fit full model
            df_feat, stats = create_features(df_boot)
            model_full = fit_binomial_model(df_feat, full_vars)
            
            # Fit reduced model (without this variable)
            reduced_vars = [v for v in full_vars if v != variable_to_drop]
            model_reduced = fit_binomial_model(df_feat, reduced_vars)
            
            performance_full.append({
                'boot': boot,
                'AIC': model_full.aic,
                'BIC': model_full.bic_llf,
                'deviance': model_full.deviance,
                'deviance_explained': 1 - (model_full.deviance / model_full.null_deviance),
            })
            
            performance_reduced.append({
                'boot': boot,
                'AIC': model_reduced.aic,
                'BIC': model_reduced.bic_llf,
                'deviance': model_reduced.deviance,
                'deviance_explained': 1 - (model_reduced.deviance / model_reduced.null_deviance),
            })
        except:
            continue
    
    df_full_perf = pd.DataFrame(performance_full)
    df_reduced_perf = pd.DataFrame(performance_reduced)
    
    # Merge on boot
    df_comparison = df_full_perf.merge(df_reduced_perf, on='boot', suffixes=('_full', '_reduced'))
    
    return df_comparison

def calculate_variable_importance(df_comparison, metric='deviance_explained'):
    """
    Calculate importance statistics for a single variable.
    """
    # For deviance_explained, higher is better
    # Importance = performance_full - performance_reduced
    # For AIC/deviance, lower is better
    # Importance = performance_reduced - performance_full
    
    if metric in ['AIC', 'BIC', 'deviance']:
        diff = df_comparison[f'{metric}_reduced'] - df_comparison[f'{metric}_full']
        # Positive diff means full model is better (has lower metric)
    else:
        diff = df_comparison[f'{metric}_full'] - df_comparison[f'{metric}_reduced']
        # Positive diff means full model is better (has higher metric)
    
    mean_diff = diff.mean()
    ci_lower = np.percentile(diff, 2.5)
    ci_upper = np.percentile(diff, 97.5)
    
    # P-value: proportion of bootstraps where reduced model is better or equal
    if metric in ['AIC', 'BIC', 'deviance']:
        p_value = np.mean(diff <= 0)  # Reduced is better when diff <= 0
    else:
        p_value = np.mean(diff <= 0)  # Reduced is better when diff <= 0
    
    return {
        'mean_diff': mean_diff,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'n_boots': len(diff),
    }

# Test each variable in the selected models
variable_importance_results = []

for (cond0, ztime), model_info in best_models.items():
    
    print(f"\n{'='*70}")
    print(f"Condition: {cond0} dpf, {ztime}")
    print(f"Model: {model_info['model_name']}")
    print(f"{'='*70}")
    
    full_vars = model_info['vars']
    
    df_cond = boot_hazard_df[
        (boot_hazard_df['cond0'] == cond0) &
        (boot_hazard_df['ztime'] == ztime)
    ]
    
    print(f"\nTesting necessity of each predictor:")
    print(f"{'-'*70}")
    
    for var in full_vars:
        
        if var == 't_z':
            # Don't test dropping time - it's always required
            print(f"\n  {var:20s}: REQUIRED (baseline time effect)")
            continue
        
        print(f"\n  Testing: {var}")
        
        # Get bootstrap comparison
        df_comparison = test_variable_necessity(df_cond, full_vars, var)
        
        # Calculate importance for different metrics
        importance_dev = calculate_variable_importance(df_comparison, 'deviance_explained')
        importance_aic = calculate_variable_importance(df_comparison, 'AIC')
        importance_deviance = calculate_variable_importance(df_comparison, 'deviance')
        
        # Store results
        variable_importance_results.append({
            'cond0': cond0,
            'ztime': ztime,
            'model_name': model_info['model_name'],
            'variable': var,
            'dev_exp_diff_mean': importance_dev['mean_diff'],
            'dev_exp_ci_lower': importance_dev['ci_lower'],
            'dev_exp_ci_upper': importance_dev['ci_upper'],
            'dev_exp_p_value': importance_dev['p_value'],
            'AIC_diff_mean': importance_aic['mean_diff'],
            'AIC_ci_lower': importance_aic['ci_lower'],
            'AIC_ci_upper': importance_aic['ci_upper'],
            'AIC_p_value': importance_aic['p_value'],
            'deviance_diff_mean': importance_deviance['mean_diff'],
            'deviance_p_value': importance_deviance['p_value'],
            'n_boots': importance_dev['n_boots'],
            'necessary': importance_dev['p_value'] < 0.05 or importance_aic['p_value'] < 0.05,
        })
        
        # Print results
        print(f"    Deviance explained:")
        print(f"      Full - Reduced: {importance_dev['mean_diff']:+.4f} [95% CI: {importance_dev['ci_lower']:.4f}, {importance_dev['ci_upper']:.4f}]")
        print(f"      p-value: {importance_dev['p_value']:.4f} {'***' if importance_dev['p_value'] < 0.001 else '**' if importance_dev['p_value'] < 0.01 else '*' if importance_dev['p_value'] < 0.05 else 'ns'}")
        
        print(f"    AIC:")
        print(f"      Reduced - Full: {importance_aic['mean_diff']:+.1f} [95% CI: {importance_aic['ci_lower']:.1f}, {importance_aic['ci_upper']:.1f}]")
        print(f"      p-value: {importance_aic['p_value']:.4f} {'***' if importance_aic['p_value'] < 0.001 else '**' if importance_aic['p_value'] < 0.01 else '*' if importance_aic['p_value'] < 0.05 else 'ns'}")
        
        if importance_dev['p_value'] < 0.05 or importance_aic['p_value'] < 0.05:
            print(f"    → NECESSARY (significant performance drop when removed)")
        else:
            print(f"    → NOT NECESSARY (no significant performance drop)")

variable_importance_df = pd.DataFrame(variable_importance_results)

# Save results
variable_importance_df.to_csv(
    os.path.join(fig_dir, 'variable_importance_within_models.csv'),
    index=False
)

print(f"\n✓ Variable importance analysis saved")

# -----------------------------------------------------------------------------
# 12. VISUALIZE VARIABLE IMPORTANCE
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VISUALIZING VARIABLE IMPORTANCE")
print("=" * 80)

# Plot 1: Coefficient magnitudes with confidence intervals
fig, axes = plt.subplots(len(best_models), 1, figsize=(10, 4*len(best_models)))

if len(best_models) == 1:
    axes = [axes]

for idx, ((cond0, ztime), model_info) in enumerate(best_models.items()):
    
    ax = axes[idx]
    
    # Get coefficients across all bootstraps for this condition
    coef_data = []
    
    for r in results:
        if r['cond0'] == cond0 and r['ztime'] == ztime:
            model = r['model']
            for var in model_info['vars']:
                if var in model.params.index:
                    coef_data.append({
                        'variable': var,
                        'coefficient': model.params[var],
                        'boot': r['boot'],
                    })
    
    coef_df = pd.DataFrame(coef_data)
    
    # Calculate mean and CI
    coef_summary = coef_df.groupby('variable')['coefficient'].agg([
        'mean',
        lambda x: np.percentile(x, 2.5),
        lambda x: np.percentile(x, 97.5),
    ]).reset_index()
    coef_summary.columns = ['variable', 'mean', 'ci_lower', 'ci_upper']
    
    # Sort by absolute mean
    coef_summary['abs_mean'] = np.abs(coef_summary['mean'])
    coef_summary = coef_summary.sort_values('abs_mean', ascending=True)
    
    # Plot
    y_pos = np.arange(len(coef_summary))
    ax.barh(y_pos, coef_summary['mean'], xerr=[
        coef_summary['mean'] - coef_summary['ci_lower'],
        coef_summary['ci_upper'] - coef_summary['mean']
    ], capsize=5, alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_summary['variable'])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Coefficient (log-odds)')
    ax.set_title(f'{cond0} dpf, {ztime}\nCoefficients with 95% Bootstrap CI')
    ax.grid(axis='x', alpha=0.3)
    
    # Add significance markers
    var_imp = variable_importance_df[
        (variable_importance_df['cond0'] == cond0) &
        (variable_importance_df['ztime'] == ztime)
    ]
    
    for i, var in enumerate(coef_summary['variable']):
        var_row = var_imp[var_imp['variable'] == var]
        if len(var_row) > 0 and var_row.iloc[0]['necessary']:
            ax.text(coef_summary.iloc[i]['mean'], i, ' *', 
                   fontsize=12, color='red', ha='left', va='center')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'variable_coefficients_with_importance.pdf'), dpi=300)
plt.show()

# Plot 2: Variable importance (change in deviance explained)
fig, axes = plt.subplots(len(best_models), 1, figsize=(10, 4*len(best_models)))

if len(best_models) == 1:
    axes = [axes]

for idx, ((cond0, ztime), model_info) in enumerate(best_models.items()):
    
    ax = axes[idx]
    
    var_imp = variable_importance_df[
        (variable_importance_df['cond0'] == cond0) &
        (variable_importance_df['ztime'] == ztime)
    ].copy()
    
    if len(var_imp) == 0:
        continue
    
    var_imp = var_imp.sort_values('dev_exp_diff_mean', ascending=True)
    
    y_pos = np.arange(len(var_imp))
    colors = ['red' if nec else 'gray' for nec in var_imp['necessary']]
    
    ax.barh(y_pos, var_imp['dev_exp_diff_mean'], 
            xerr=[
                var_imp['dev_exp_diff_mean'] - var_imp['dev_exp_ci_lower'],
                var_imp['dev_exp_ci_upper'] - var_imp['dev_exp_diff_mean']
            ],
            capsize=5, alpha=0.7, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_imp['variable'])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Δ Deviance Explained\n(Full - Reduced Model)')
    ax.set_title(f'{cond0} dpf, {ztime}\nVariable Importance\n(Red = Necessary, Gray = Not Necessary)')
    ax.grid(axis='x', alpha=0.3)
    
    # Add p-values
    for i, (_, row) in enumerate(var_imp.iterrows()):
        p_text = f"p={row['dev_exp_p_value']:.3f}"
        ax.text(ax.get_xlim()[1]*0.95, i, p_text, 
               fontsize=8, ha='right', va='center')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'variable_importance_deviance.pdf'), dpi=300)
plt.show()

# Plot 3: AIC difference when dropping each variable
fig, axes = plt.subplots(len(best_models), 1, figsize=(10, 4*len(best_models)))

if len(best_models) == 1:
    axes = [axes]

for idx, ((cond0, ztime), model_info) in enumerate(best_models.items()):
    
    ax = axes[idx]
    
    var_imp = variable_importance_df[
        (variable_importance_df['cond0'] == cond0) &
        (variable_importance_df['ztime'] == ztime)
    ].copy()
    
    if len(var_imp) == 0:
        continue
    
    var_imp = var_imp.sort_values('AIC_diff_mean', ascending=True)
    
    y_pos = np.arange(len(var_imp))
    colors = ['red' if nec else 'gray' for nec in var_imp['necessary']]
    
    ax.barh(y_pos, var_imp['AIC_diff_mean'], 
            xerr=[
                var_imp['AIC_diff_mean'] - var_imp['AIC_ci_lower'],
                var_imp['AIC_ci_upper'] - var_imp['AIC_diff_mean']
            ],
            capsize=5, alpha=0.7, color=colors)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(var_imp['variable'])
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Δ AIC (Reduced - Full)\n(Positive = worse without variable)')
    ax.set_title(f'{cond0} dpf, {ztime}\nAIC Penalty for Dropping Variable')
    ax.grid(axis='x', alpha=0.3)
    
    # Add reference line at AIC threshold (typically 2-4)
    ax.axvline(2, color='orange', linestyle=':', linewidth=1.5, alpha=0.5, label='Δ AIC = 2')

plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'variable_importance_aic.pdf'), dpi=300)
plt.show()

print("✓ Variable importance visualizations saved")

# -----------------------------------------------------------------------------
# 13. RECOMMEND SIMPLIFIED MODELS (if any variables are unnecessary)
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("SIMPLIFIED MODEL RECOMMENDATIONS")
print("=" * 80)

simplified_models = {}

for (cond0, ztime), model_info in best_models.items():
    
    var_imp = variable_importance_df[
        (variable_importance_df['cond0'] == cond0) &
        (variable_importance_df['ztime'] == ztime)
    ]
    
    # Get unnecessary variables
    unnecessary = var_imp[~var_imp['necessary']]['variable'].tolist()
    necessary = var_imp[var_imp['necessary']]['variable'].tolist()
    
    # Always keep t_z
    if 't_z' not in necessary:
        necessary = ['t_z'] + necessary
    
    if len(unnecessary) > 0:
        print(f"\n{cond0} dpf, {ztime}:")
        print(f"  Current model: {model_info['model_name']}")
        print(f"  Variables: {model_info['vars']}")
        print(f"  → Unnecessary variables: {unnecessary}")
        print(f"  → Simplified model would include: {necessary}")
        
        simplified_models[(cond0, ztime)] = {
            'original_model': model_info['model_name'],
            'original_vars': model_info['vars'],
            'simplified_vars': necessary,
            'dropped_vars': unnecessary,
        }
    else:
        print(f"\n{cond0} dpf, {ztime}:")
        print(f"  Model: {model_info['model_name']}")
        print(f"  → All variables are necessary!")

if len(simplified_models) > 0:
    print(f"\n{'-'*70}")
    print(f"RECOMMENDATION:")
    print(f"  {len(simplified_models)} model(s) could be simplified")
    print(f"  Consider refitting with only necessary variables")
    print(f"{'-'*70}")
    
    # Save simplified model recommendations
    simplified_df = pd.DataFrame([
        {
            'condition': f"{cond0} dpf",
            'time': ztime,
            'original_model': info['original_model'],
            'n_original_vars': len(info['original_vars']),
            'n_simplified_vars': len(info['simplified_vars']),
            'dropped_vars': ', '.join(info['dropped_vars']),
            'simplified_vars': ', '.join(info['simplified_vars']),
        }
        for (cond0, ztime), info in simplified_models.items()
    ])
    
    simplified_df.to_csv(
        os.path.join(fig_dir, 'simplified_model_recommendations.csv'),
        index=False
    )
    
    print(f"\n✓ Simplified model recommendations saved")
else:
    print(f"\n  All selected models are parsimonious!")

# -----------------------------------------------------------------------------
# 14. SUMMARY TABLE: VARIABLE NECESSITY
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("VARIABLE NECESSITY SUMMARY")
print("=" * 80)

summary_table = variable_importance_df.pivot_table(
    index='variable',
    columns=['cond0', 'ztime'],
    values='necessary',
    aggfunc='first',
    fill_value=False
)

print("\n" + summary_table.to_string())

print(f"\n✓ Variable importance analysis complete")
print(f"\n  Files generated:")
print(f"    - variable_importance_within_models.csv")
print(f"    - variable_coefficients_with_importance.pdf")
print(f"    - variable_importance_deviance.pdf")
print(f"    - variable_importance_aic.pdf")
if len(simplified_models) > 0:
    print(f"    - simplified_model_recommendations.csv")

print("\n" + "=" * 80)
print("COMPLETE VARIABLE NECESSITY TESTING FINISHED")
print("=" * 80)
# %%
# -----------------------------------------------------------------------------
# BUILD PREDICTED HEATMAPS (using selected models)
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("BUILDING PREDICTED HEATMAPS")
print("=" * 80)

a_grid = {}
a_grid['day'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'day', 'ang0_bin'].astype(float).unique())
a_grid['night'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'night', 'ang0_bin'].astype(float).unique())

t_grid = {}
t_grid['day'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'day', 't_mid'].unique())
t_grid['night'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'night', 't_mid'].unique())

TT = {}
AA = {}
AA['day'], TT['day'] = np.meshgrid(a_grid['day'], t_grid['day'], indexing='ij')
AA['night'], TT['night'] = np.meshgrid(a_grid['night'], t_grid['night'], indexing='ij')

pred_heatmaps_smooth_glm = defaultdict(list)

for r in tqdm(results, desc="Generating predictions"):
    this_boot_avg = boot_hazard_df.loc[
        (boot_hazard_df['cond0'] == r['cond0']) &
        (boot_hazard_df['ztime'] == r['ztime']) &
        (boot_hazard_df['boot'] == r['boot'])
    ].copy()

    # Apply transformations using stored statistics
    this_boot_avg_feat = apply_feature_transforms(this_boot_avg, r)
    
    # Get model variables for this condition
    model_vars = r['model_vars']
    
    Xp = sm.add_constant(this_boot_avg_feat[model_vars], has_constant='add')
    this_boot_avg['p_hat'] = r['model'].predict(Xp)
    
    Z = np.full(AA[r['ztime']].shape, np.nan)
    
    a_to_i = {a: i for i, a in enumerate(a_grid[r['ztime']])}
    t_to_j = {t: j for j, t in enumerate(t_grid[r['ztime']])}

    for _, row in this_boot_avg.iterrows():
        a_val = float(row['ang0_bin']) if hasattr(row['ang0_bin'], 'mid') else float(row['ang0_bin'])
        t_val = row['t_mid']
        
        if a_val in a_to_i and t_val in t_to_j:
            i = a_to_i[a_val]
            j = t_to_j[t_val]
            Z[i, j] = row['p_hat']
                    
    pred_heatmaps_smooth_glm[(r['boot'], r['cond0'], r['ztime'])] = Z

print("✓ Predicted heatmaps generated")

# -----------------------------------------------------------------------------
# AGGREGATE PREDICTIONS ACROSS BOOTSTRAPS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("AGGREGATING PREDICTIONS ACROSS BOOTSTRAPS")
print("=" * 80)

# Collect GLM predictions in long format
glm_long = []

for (boot, cond0, ztime), Z in pred_heatmaps_smooth_glm.items():
    for i, a_val in enumerate(a_grid[ztime]):
        for j, t_val in enumerate(t_grid[ztime]):
            glm_long.append({
                "boot": boot,
                "cond0": cond0,
                "ztime": ztime,
                "ang0_bin": a_val,
                "t_mid": t_val,
                "hazard_pred": Z[i, j]
            })

glm_long_df = pd.DataFrame(glm_long)

# Compute mean predictions across bootstraps
glm_mean_df = (
    glm_long_df
    .groupby(["cond0", "ztime", "ang0_bin", "t_mid"], observed=True)
    ["hazard_pred"]
    .mean()
    .reset_index()
)

pred_heatmaps_mean = {}

for (cond0, ztime), df_sub in glm_mean_df.groupby(["cond0", "ztime"], observed=True):
    Z_glm_df = df_sub.pivot_table(
        index="ang0_bin",
        columns="t_mid",
        values="hazard_pred",
        fill_value=np.nan
    )
    pred_heatmaps_mean[(cond0, ztime)] = Z_glm_df

print("✓ Bootstrap predictions aggregated")

# -----------------------------------------------------------------------------
# PREPARE RAW HAZARD HEATMAPS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PREPARING RAW HAZARD HEATMAPS")
print("=" * 80)

raw_heatmaps = {}
for (cond0, ztime), df_sub in boot_hazard_df_average_raw.groupby(['cond0', 'ztime'], observed=True):
    Z_raw = df_sub.pivot_table(
        index='ang0_bin', 
        columns='t_mid', 
        values='hazard_smooth', 
        fill_value=np.nan,
        observed=True
    )
    raw_heatmaps[(cond0, ztime)] = Z_raw

print("✓ Raw hazard heatmaps prepared")

# -----------------------------------------------------------------------------
# PLOT HEATMAP COMPARISONS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PLOTTING HEATMAP COMPARISONS")
print("=" * 80)

vmax = 0.3  # shared scale for fair comparison

palette_cmap = {}
palette_cmap['day'] = sns.light_palette("#1d2e5e", as_cmap=True)
palette_cmap['night'] = sns.light_palette("#4c0a4c", as_cmap=True)

for (cond0, ztime) in pred_heatmaps_mean.keys():

    Z_glm_df = pred_heatmaps_mean[(cond0, ztime)]
    Z_raw_df = raw_heatmaps[(cond0, ztime)]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10, 4),
        sharex=True,
        sharey=True
    )

    # -----------------------------
    # Raw hazard (smoothed observed)
    # -----------------------------
    sns.heatmap(
        Z_raw_df,
        ax=axes[0],
        cmap=palette_cmap[ztime],
        vmin=0,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "P(event)"},
        rasterized=True
    )

    axes[0].invert_yaxis()
    axes[0].set_title(f"Raw hazard (smoothed)\nCond={cond0}, Ztime={ztime}")
    axes[0].set_xlabel("Time into IBI (s)")
    axes[0].set_ylabel("Initial Posture (deg)")

    # -----------------------------
    # GLM-predicted hazard
    # -----------------------------
    sns.heatmap(
        Z_glm_df,
        ax=axes[1],
        cmap=palette_cmap[ztime],
        vmin=0,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "P(event)"},
        rasterized=True
    )

    axes[1].invert_yaxis()
    axes[1].set_title(f"GLM-predicted hazard\nModel: {best_models[(cond0, ztime)]['model_name']}")
    axes[1].set_xlabel("Time into IBI (s)")
    axes[1].set_ylabel("")

    # -----------------------------
    # Pretty ticks (optional)
    # -----------------------------
    if 'set_pretty_ticks' in dir():
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
            f"glm_vs_raw_cond{cond0}_ztime{ztime}.pdf"
        ),
        format="pdf"
    )
    plt.show()
    
    print(f"  ✓ Saved: glm_vs_raw_cond{cond0}_ztime{ztime}.pdf")

print("\n✓ All heatmap comparisons saved")

# -----------------------------------------------------------------------------
# COMPUTE RMSE PER BOOTSTRAP
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("COMPUTING MODEL FIT METRICS PER BOOTSTRAP")
print("=" * 80)

from scipy.stats import pearsonr

rmse_per_boot = []

for (boot, cond0, ztime), Z_glm in pred_heatmaps_smooth_glm.items():

    df_raw = boot_hazard_df[
        (boot_hazard_df['boot'] == boot) &
        (boot_hazard_df['cond0'] == cond0) &
        (boot_hazard_df['ztime'] == ztime)
    ]

    Z_raw = np.full(Z_glm.shape, np.nan)
    W = np.zeros(Z_glm.shape)

    a_to_i = {a: i for i, a in enumerate(a_grid[ztime])}
    t_to_j = {t: j for j, t in enumerate(t_grid[ztime])}

    for _, row in df_raw.iterrows():
        a = float(row['ang0_bin']) if hasattr(row['ang0_bin'], 'mid') else float(row['ang0_bin'])
        t = row['t_mid']
        if a in a_to_i and t in t_to_j:
            i = a_to_i[a]
            j = t_to_j[t]
            Z_raw[i, j] = row['hazard']
            W[i, j] = row['n_risk']

    mask = ~np.isnan(Z_glm) & ~np.isnan(Z_raw) & (W > 0)

    # Pearson correlation
    if mask.sum() < 2:
        r = np.nan
    else:
        r, _ = pearsonr(
            Z_raw[mask].ravel(),
            Z_glm[mask].ravel()
        )

    # RMSE
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
        "ztime": ztime,
        "RMSE": rmse,
        "RMSE_weighted": rmse_weighted,
        "r": r,
    })

rmse_boot_df = pd.DataFrame(rmse_per_boot)

print(f"✓ Computed RMSE and correlation for {len(rmse_boot_df)} bootstrap samples")

# -----------------------------------------------------------------------------
# PLOT MODEL FIT METRICS
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PLOTTING MODEL FIT METRICS")
print("=" * 80)

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
    col='ztime',
)
plt.savefig(os.path.join(fig_dir, f'glm_RMSE_by_condition.pdf'))
plt.show()

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
    col='ztime',
)
plt.savefig(os.path.join(fig_dir, f'glm_RMSE_weighted_by_condition.pdf'))
plt.show()

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
    col='ztime',
)
plt.savefig(os.path.join(fig_dir, f'glm_correlation_r_by_condition.pdf'))
plt.show()

print("✓ Model fit metrics plots saved")

# -----------------------------------------------------------------------------
# SCATTER PLOT: PREDICTED VS OBSERVED
# -----------------------------------------------------------------------------

print("\n" + "=" * 80)
print("PLOTTING PREDICTED VS OBSERVED SCATTER")
print("=" * 80)

colors_dayNight = {
    'day': "#1d2e5e",
    'night': "#4c0a4c"
}

rows = []

for condition in pred_heatmaps_mean.keys():
    Z_glm = pred_heatmaps_mean[condition]
    Z_raw = raw_heatmaps[condition]

    mask = ~np.isnan(Z_raw.values) & ~np.isnan(Z_glm.values)

    rows.append(
        pd.DataFrame({
            "raw_hazard": Z_raw.values[mask].ravel(),
            "glm_hazard": Z_glm.values[mask].ravel(),
            "cond0": condition[0],
            "ztime": condition[1],
        })
    )

df_scatter = pd.concat(rows, ignore_index=True)

g = sns.relplot(
    data=df_scatter,
    x="raw_hazard",
    y="glm_hazard",
    col="cond0",
    row="ztime",
    kind="scatter",
    alpha=0.17,
    s=30,
    linewidth=0,
    height=2.5,
    hue='ztime',
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

plt.savefig(os.path.join(fig_dir, f'glm_predicted_vs_raw_hazard.pdf'))
plt.show()

print("✓ Scatter plot saved")

print("\n" + "=" * 80)
print("HEATMAP ANALYSIS COMPLETE")
print("=" * 80)