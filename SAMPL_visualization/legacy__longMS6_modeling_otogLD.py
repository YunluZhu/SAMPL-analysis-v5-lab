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
from collections import Counter

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
### Parameters to change #####

pick_data = 'otog_ld' # name of your dataset to plot as defined in function get_data_dir()
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
# no need to rerun if having the tvdf

#%% 
re_analyze = False
try:
    tv_df_all = pd.read_pickle(f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_TVdf_{pick_data}.pkl')
    print('Loaded TV df from pickle')
except Exception as e:
    print(f"Loading TV df failed ({e}), regenerating data...")
    re_analyze = True
    
if re_analyze:
    pickle_path = f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_angles_{pick_data}.pkl'

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

    cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "x" ,"y", 'headx', 'heady','swimSpeed']
    df_time_filtered = IBI_angles.loc[:, cols_needed]
    # Compute group max (IBI duration) and filter in-place

    print("> checkpoint after early filtering")

    #% release memory
    del IBI_angles
    gc.collect()

    #%
    #% truncate to start from speed threshold
    df_time_filtered['swimSpeed_smoothed'] = df_time_filtered.groupby('unique_IBI_idx', observed=True)['swimSpeed'].transform(
        lambda x: savgol_filter(x.values, 11, 3) if len(x) >= 11 else x.values
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
                    **meta
                })

        tv_df = pd.DataFrame(rows)

        if tv_df.empty:
            continue
        
        # -----------------------------
        # Standardize covariates
        # -----------------------------
        # z-score initial posture
        tv_df['ang0_z'] = (tv_df['ang0'] - tv_df['ang0'].mean()) / tv_df['ang0'].std(ddof=0)

        # log-time with clipping
        tv_df['log_time'] = np.log(tv_df['stop']) # change from start to stop to avoid log(0) and EPS bias
        clip_lo = np.floor(tv_df['log_time'].min())
        clip_hi = np.ceil(tv_df['log_time'].max())
        tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

        # Interaction (initial posture × log-time)
        tv_df['ang0_z_logt'] = tv_df['ang0_z'] * tv_df['log_time_clipped']
        tv_df['ang0_z_logt'] = (tv_df['ang0_z_logt'] - tv_df['ang0_z_logt'].mean()) / tv_df['ang0_z_logt'].std(ddof=0)

        # ----- New quadratic term -----
        tv_df['ang0_z2'] = tv_df['ang0_z']**2
        tv_df['ang0_z2'] = (tv_df['ang0_z2'] - tv_df['ang0_z2'].mean()) / tv_df['ang0_z2'].std(ddof=0)
        # Quadratic × time interaction
        tv_df['ang0_z2_logt'] = tv_df['ang0_z2'] * tv_df['log_time_clipped']
        tv_df['ang0_z2_logt'] = (tv_df['ang0_z2_logt'] - tv_df['ang0_z2_logt'].mean()) / tv_df['ang0_z2_logt'].std(ddof=0)
        
        tv_df_all_.append(tv_df)

    tv_df_all = pd.concat(tv_df_all_, ignore_index=True)
    tv_df_all.to_pickle(f'/Users/yunluzhu/Documents/Lab2/Python_VF/script/longitudinal_lighting/IBI_TVdf_{pick_data}.pkl')  # smaller file size
#%

#%%
# add derivatives
tv_df_all['ang_sm'] = tv_df_all.groupby('unique_IBI_idx')['ang'].transform(lambda x: savgol_filter(x.values, 5, 2) if len(x.values) >=5 else x.values)

tv_df_all['angvel'] = tv_df_all.groupby('unique_IBI_idx')['ang_sm'].transform(lambda x: np.gradient(x.values) if len(x.values) >1 else np.nan) * FRAME_RATE
# # smooth velocity
tv_df_all['angvel_sm'] = tv_df_all.groupby('unique_IBI_idx')['angvel'].transform(lambda x: savgol_filter(x.values, 3, 2) if len(x.values) >=3 else x.values)
# # acceleration
tv_df_all['angacc'] = tv_df_all.groupby('unique_IBI_idx')['angvel_sm'].transform(lambda x: np.gradient(x.values) if len(x.values) >=5 else np.nan)

# remove IBIs with nan derivatives
tv_df_all = tv_df_all[~tv_df_all['ang_sm'].isna()].copy()

#%%
# filter out LD-night rows
tv_df_short = tv_df_all.loc[
    ~((tv_df_all['cond0'] == 'ld') & (tv_df_all['ztime'] == 'night'))
].copy()

# day/night -> light/dark
tv_df_short['ztime'] = tv_df_short['ztime'].replace({
    'day': 'light',
    'night': 'dark'
})

# DD + light -> dark
mask = (tv_df_short['cond0'] == 'DD') & (tv_df_short['ztime'] == 'light')
tv_df_short.loc[mask, 'ztime'] = 'dark'

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

for zname, TIME_STEP in [('light', TIME_STEP_LIGHT), ('dark', TIME_STEP_DARK)]:
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

n_boot = 20

boot_hazard = []

for (cond1, ztime), df_sub in tv_df_short.groupby(['cond1', 'ztime'], observed=True):

    # Precompute indices for IBI-level bootstrap
    unique_ibis = df_sub['unique_IBI_idx'].unique()

    for b in tqdm(range(n_boot), desc=f'Bootstrapping {cond1}, {ztime}'):
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
            this_hazard_df.assign(boot=b, cond1=cond1, ztime=ztime)
        )

boot_hazard_df = pd.concat(boot_hazard, ignore_index=True)
boot_hazard_df['hazard_smooth'] = boot_hazard_df.groupby(['boot','cond1','ztime'], observed=True)['hazard'].transform(
    lambda x: gaussian_filter1d(x, sigma=SIGMA_T, mode='nearest')
)


#%%
# -----------------------------
# Step 1: Adjust counts with smoothed hazard
# -----------------------------
# Use hazard_smooth to compute fractional events per bin
boot_hazard_df['n_event_smooth'] = boot_hazard_df['hazard'] * boot_hazard_df['n_risk'] # not using smoothed actually

# Optional: add a small pseudo-count to avoid exact 0 or 1
EPS = 1e-3
boot_hazard_df['n_event_smooth'] = np.clip(
    boot_hazard_df['n_event_smooth'], EPS, boot_hazard_df['n_risk'] - EPS
)

boot_hazard_df['total_IBIs'] = boot_hazard_df.groupby(['boot','cond1','ztime'], observed=True)['n_IBIs'].transform('sum')
boot_hazard_df['IBI_ratio'] = boot_hazard_df['n_IBIs'] / boot_hazard_df['total_IBIs']
# -----------------------------
# Step 2: Fit GLM per cond1/ztime
# -----------------------------
# average but only if enough data points
# Minimum number of data points required per group
# min_count = n_boot/2

# # Compute group size and conditional averages

boot_hazard_df_average_raw = (
    boot_hazard_df
    .groupby(['cond1','ztime','ang0_bin','t_bin'], observed=True)
    .agg(
        n_rows=('n_risk', 'size'),        # count rows in the group
        n_risk=('n_risk', 'mean'),
        n_event=('n_event_smooth', 'mean'),
        hazard=('hazard', 'mean'),
        hazard_smooth=('hazard_smooth', 'mean'),
        IBI_ratio=('IBI_ratio', 'mean'),
        angvel=('angvel', 'mean'),
        # angacc=('angacc', 'mean'),
    )
    .reset_index()
)

boot_hazard_df_average_raw['t_mid'] = boot_hazard_df_average_raw['t_bin']

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
for (boot, cond1, ztime), df in boot_hazard_df.groupby(['boot','cond1','ztime'], observed=True):
    df = df.copy()

    # ---------------------------------
    # Base variables
    # ---------------------------------
    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid']

    df['a_z'], a_mu, a_sd = zscore(df['a'])
    df['t_z'], t_mu, t_sd = zscore(df['t'])
    df['vel_z'], vel_mu, vel_sd = zscore(df['angvel'])
    # df['acc_z'], acc_mu, acc_sd = zscore(df['angacc'])

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
    # df['acc_t']  = df['acc_z'] * df['t_z']
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
        'cond1': cond1,
        'ztime': ztime,
        'model': model,

        # linear terms
        'a_mu': a_mu,
        'a_sd': a_sd,
        't_mu': t_mu,
        't_sd': t_sd,
        'vel_mu': vel_mu,
        'vel_sd': vel_sd,
        # 'acc_mu': acc_mu,
        # 'acc_sd': acc_sd,
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
a_grid = {}
a_grid['light'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'light', 'ang0_bin'].astype(float).unique())
a_grid['dark'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'dark', 'ang0_bin'].astype(float).unique())
t_grid = {}
t_grid['light'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'light', 't_mid'].unique())
t_grid['dark'] = np.sort(boot_hazard_df_average_raw.loc[boot_hazard_df_average_raw['ztime'] == 'dark', 't_mid'].unique())

TT = {}
AA = {}
AA['light'], TT['light'] = np.meshgrid(a_grid['light'], t_grid['light'], indexing='ij')
AA['dark'], TT['dark'] = np.meshgrid(a_grid['dark'], t_grid['dark'], indexing='ij')

from collections import defaultdict
pred_heatmaps_smooth_glm = defaultdict(list)


for r in tqdm(results):
    this_boot_avg = boot_hazard_df.loc[
        (boot_hazard_df['cond1'] == r['cond1']) &
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
    # this_boot_avg['acc_z'] = (
    #     (this_boot_avg['angacc'] - r['acc_mu']) / r['acc_sd']
    # )
    this_boot_avg['a_t'] = this_boot_avg['a_z'] * this_boot_avg['t_z']
    this_boot_avg['vel_t'] = this_boot_avg['vel_z'] * this_boot_avg['t_z']
    this_boot_avg['a2_z'] = (
        (this_boot_avg['a_z']**2 - r['a2_mu']) / r['a2_sd']
    )
    this_boot_avg['vel2_z'] = (
        (this_boot_avg['vel_z']**2 - r['vel2_mu']) / r['vel2_sd']
    )
    this_boot_avg['a2_t'] = this_boot_avg['a2_z'] * this_boot_avg['t_z']
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
    
    mask_obs = np.zeros_like(AA[r['ztime']], dtype=bool)

    Z = np.full(AA[r['ztime']].shape, np.nan)    
    
    a_to_i = {a: i for i, a in enumerate(a_grid[r['ztime']])}
    t_to_j = {t: j for j, t in enumerate(t_grid[r['ztime']])}

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

    pred_heatmaps_smooth_glm[(r['boot'], r['cond1'], r['ztime'])] = Z_clipped


# --- Collect GLM predictions in long format ---
glm_long = []

for (boot, cond1, ztime), Z in pred_heatmaps_smooth_glm.items():
    for i, a_val in enumerate(a_grid[ztime]):
        for j, t_val in enumerate(t_grid[ztime]):
            glm_long.append({
                "boot": boot,
                "cond1": cond1,
                "ztime": ztime,
                "ang0_bin": a_val,
                "t_mid": t_val,
                "hazard_pred": Z[i, j]
            })

glm_long_df = pd.DataFrame(glm_long)

glm_mean_df = (
    glm_long_df
    .groupby(["cond1", "ztime", "ang0_bin", "t_mid"], observed=True)
    ["hazard_pred"]
    .mean()
    .reset_index()
)

pred_heatmaps_mean = {}

for (cond1, ztime), df_sub in glm_mean_df.groupby(["cond1", "ztime"], observed=True):
    Z_glm_df = df_sub.pivot_table(
        index="ang0_bin",
        columns="t_mid",
        values="hazard_pred",
        fill_value=np.nan
    )
    pred_heatmaps_mean[(cond1, ztime)] = Z_glm_df


# --- Prepare raw hazard heatmaps ---
raw_heatmaps = {}
for (cond1, ztime), df_sub in boot_hazard_df_average_raw.groupby(['cond1', 'ztime'], observed=True):
    # Create empty heatmap
    Z_raw = df_sub.pivot_table(
        index='ang0_bin', 
        columns='t_mid', 
        values='hazard_smooth', 
        fill_value=np.nan,
        observed=True
    )
    raw_heatmaps[(cond1, ztime)] = Z_raw
    

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
palette_cmap['light'] = sns.light_palette("#1d2e5e",as_cmap=True)
palette_cmap['dark'] = sns.light_palette("#4c0a4c", as_cmap=True)


for (cond1, ztime) in pred_heatmaps_mean.keys():

    Z_glm_df = pred_heatmaps_mean[(cond1, ztime)]
    Z_raw_df = raw_heatmaps[(cond1, ztime)]

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
        cmap=palette_cmap[ztime],
        vmin=0,
        vmax=vmax,
        cbar=True,
        cbar_kws={"label": "P(event)"},
        rasterized=True
    )

    axes[0].invert_yaxis()
    axes[0].set_title(f"Raw hazard\nCond={cond1}, Ztime={ztime}")
    axes[0].set_xlabel("Time into IBI (s)")
    axes[0].set_ylabel("Posture quantile")

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
    axes[1].set_title(f"GLM-predicted hazard\nCond={cond1}, Ztime={ztime}")
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
            f"glm_vs_raw_cond{cond1}_ztime{ztime}.pdf"
        ),
        format="pdf"
    )

#%% RMSE per boot
from scipy.stats import pearsonr



rmse_per_boot = []

for (boot, cond1, ztime), Z_glm in pred_heatmaps_smooth_glm.items():

    df_raw = boot_hazard_df[
        (boot_hazard_df['boot'] == boot) &
        (boot_hazard_df['cond1'] == cond1) &
        (boot_hazard_df['ztime'] == ztime)
    ]

    Z_raw = np.full(Z_glm.shape, np.nan)
    W = np.zeros(Z_glm.shape)

    a_to_i = {a: i for i, a in enumerate(a_grid[ztime])}
    t_to_j = {t: j for j, t in enumerate(t_grid[ztime])}

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
        "cond1": cond1,
        "ztime": ztime,
        "RMSE": rmse,
        "RMSE_weighted": rmse_weighted,
        "r": r,
    })

rmse_boot_df = pd.DataFrame(rmse_per_boot)

#%%
g = sns.catplot(
    data=rmse_boot_df,
    x='cond1',
    y='RMSE',
    errorbar='sd',
    kind='point',
    units='boot',
    height=2.5,
    linestyle='none',
    aspect=1,
    marker='_',
    col='ztime',
    # palette=palette_cmap
)

plt.savefig(os.path.join(fig_dir, f'glm_RMSE_by_condition_({model_vars}).pdf'))

g = sns.catplot(
    data=rmse_boot_df,
    x='cond1',
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
plt.savefig(os.path.join(fig_dir, f'glm_RMSE_weighted_by_condition_({model_vars}).pdf'))


g = sns.catplot(
    data=rmse_boot_df,
    x='cond1',
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
plt.savefig(os.path.join(fig_dir, f'glm_correlation_r_by_condition_({model_vars}).pdf'))
#%% # for every bin, how far the predicted results are from the empirical results
# scatter plot of predicted vs raw hazard

colors_dayNight = {
    'light': "#1d2e5e",
    'dark': "#4c0a4c"
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

plt.savefig(os.path.join(fig_dir, f'glm_predicted_vs_raw_hazard_({model_vars}).pdf'))

# %%
# Deviance explained

def compute_deviance_explained(results):
    """
    Compute deviance explained per bootstrap per condition.

    Returns
    -------
    pd.DataFrame with columns:
        boot, cond1, ztime, deviance_explained
    """
    rows = []
    for r in results:
        model = r['model']
        D_model = model.deviance
        D_null = model.null_deviance

        dev_exp = 1.0 - (D_model / D_null)

        rows.append({
            'boot': r['boot'],
            'cond1': r['cond1'],
            'ztime': r['ztime'],
            'deviance_explained': dev_exp
        })

    return pd.DataFrame(rows)

devexp_df = compute_deviance_explained(results)


plt_categorical_combined_3(
    data=devexp_df,
    x='cond1',
    col='ztime',
    y='deviance_explained',
    hue='ztime',
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
# deviance explained by term
def build_design_matrix(df, r, model_vars):
    """
    Reconstruct GLM design matrix for a given bootstrap result r.
    """
    df = df.copy()

    df['a'] = df['ang0_bin'].astype(float)
    df['t'] = df['t_mid'].astype(float)

    df['a_z'] = (df['a'] - r['a_mu']) / r['a_sd']
    df['t_z'] = (df['t'] - r['t_mu']) / r['t_sd']
    df['vel_z'] = (df['angvel'] - r['vel_mu']) / r['vel_sd']

    df['a2_z'] = (df['a_z']**2 - r['a2_mu']) / r['a2_sd']
    df['vel2_z'] = (df['vel_z']**2 - r['vel2_mu']) / r['vel2_sd']

    df['aABS_z'] = (np.abs(df['a']) - r['aABS_mu']) / r['aABS_sd']
    df['azABS_z'] = (np.abs(df['a_z']) - r['azABS_mu']) / r['azABS_sd']

    df['a_t'] = df['a_z'] * df['t_z']
    df['vel_t'] = df['vel_z'] * df['t_z']
    df['a2_t'] = df['a2_z'] * df['t_z']
    df['aABS_t'] = df['aABS_z'] * df['t_z']
    df['azABS_t'] = df['azABS_z'] * df['t_z']

    X = sm.add_constant(df[model_vars], has_constant='add')
    return X

TERM_GROUPS = {
    'posture': ['a_z', 'azABS_z'],
    'time': ['t_z'],
    'posture_time': ['a_t'],
    'vel_time': ['vel_t'],
}
def compute_delta_deviance_explained(results, boot_hazard_df, model_vars):
    rows = []

    for r in results:
        boot, cond1, ztime = r['boot'], r['cond1'], r['ztime']

        df = boot_hazard_df.query(
            "boot == @boot and cond1 == @cond1 and ztime == @ztime"
        ).copy()

        # Response
        n_fail = df['n_risk'] - df['n_event_smooth']
        y = np.column_stack([df['n_event_smooth'], n_fail])

        # Full model
        D_full = r['model'].deviance
        D_null = r['model'].null_deviance

        # Full design matrix
        X_full = build_design_matrix(df, r, model_vars)

        for term, drop_vars in TERM_GROUPS.items():
            keep_vars = [v for v in model_vars if v not in drop_vars]

            X_reduced = sm.add_constant(
                X_full[keep_vars], has_constant='add'
            )

            reduced_model = sm.GLM(
                y,
                X_reduced,
                family=sm.families.Binomial()
            ).fit()

            delta_dev_exp = (reduced_model.deviance - D_full) / D_null

            rows.append({
                'boot': boot,
                'cond1': cond1,
                'ztime': ztime,
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
    col='ztime',
    row='cond1',
    y='delta_deviance_explained',
    hue='ztime',
    errorbar='sd',
    units='boot',
    overlay_func=False,
    height=2.5,
    aspect=1,
    x_order=['posture', 'time', 'posture_time', 'vel_time'],
)
plt.savefig(os.path.join(fig_dir, f'glm_delta_deviance_explained_by_term_({model_vars}).pdf'))


plt_categorical_combined_3(
    data=delta_dev_df,
    x='cond1',
    row='ztime',
    col='term',
    y='delta_deviance_explained',
    hue='ztime',
    errorbar='sd',
    units='boot',
    overlay_func=False,
    height=2.5,
    aspect=1,
    col_order=['posture', 'time', 'posture_time', 'vel_time'],
)
plt.savefig(os.path.join(fig_dir, f'glm_delta_deviance_explained_by_condition_and_term_({model_vars}).pdf'))


# %%

#%%
coef_rows = []

for r in results:
    params = r['model'].params
    conf = r['model'].conf_int()
    
    for name in params.index:
        coef_rows.append({
            'boot': r['boot'],
            'cond1': r['cond1'],
            'ztime': r['ztime'],
            'param': name,
            'coef': params[name],
            'ci_low': conf.loc[name, 0],
            'ci_high': conf.loc[name, 1],
        })

coef_df = pd.DataFrame(coef_rows)
coef_df = coef_df.query("param != 'const'")
for (cond1, ztime), df_sub in coef_df.groupby(['cond1','ztime'], observed=True):
    plt.figure(figsize=(6,4))
    sns.pointplot(
        data=df_sub,
        x='param',
        y='coef',
        errorbar='sd',
        join=False
    )
    plt.axhline(0, color='k', linestyle='--', alpha=0.5)
    plt.title(f'GLM coefficients (Cond={cond1}, Ztime={ztime})')
    plt.ylabel('Coefficient (log-odds)')
    plt.xlabel('')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f'glm_coefficients_cond{cond1}_ztime{ztime}_({model_vars}).pdf'))

sns.catplot(
    data=coef_df,
    x='cond1',
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
    cond1 = r['cond1']
    ztime = r['ztime']
    
    # Store model fit statistics
    aic_list.append({
        'boot': boot,
        'cond1': cond1,
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
            'cond1': cond1,
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

# #%%
# # on manifold plot partial dependence plots for each variable
# # aka “What would the model predict if the fish experienced posture X at the median time with median velocity?”
# # aka reference-slice partial dependence

# def plot_partial_dependence_clean(cond0, ztime, results, boot_hazard_df, model_vars, fig_dir,
#                                   save_individual=True, return_pred_data=False):
#     """
#     Generate partial dependence plots (on-manifold) for:
#         - Posture
#         - Time
#         - Angular velocity x Time

#     Parameters
#     ----------
#     cond0 : str
#         Condition identifier
#     ztime : str
#         'day' or 'night'
#     results : list
#         Fitted model results (dict per bootstrap)
#     boot_hazard_df : pd.DataFrame
#         Bootstrap hazard data
#     model_vars : list
#         GLM predictors
#     fig_dir : str
#         Save directory
#     save_individual : bool
#         Save plots individually
#     return_pred_data : bool
#         Return dict of prediction arrays

#     Returns
#     -------
#     dict (optional) : prediction arrays
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm
#     import os

#     # Pick first bootstrap as representative
#     condition_results = [r for r in results if r['cond0'] == cond0 and r['ztime'] == ztime]
#     if not condition_results:
#         print(f"No results for {cond0}|{ztime}")
#         return None
#     r = condition_results[0]
#     model = r['model']

#     df_cond = boot_hazard_df[(boot_hazard_df['cond0'] == cond0) &
#                              (boot_hazard_df['ztime'] == ztime)].copy()

#     # realistic ranges
#     a_min, a_max = df_cond['ang0_bin'].astype(float).quantile([0.01, 0.99])
#     t_min, t_max = df_cond['t_mid'].astype(float).quantile([0.01, 0.99])
#     vel_min, vel_max = df_cond['angvel'].quantile([0.01, 0.99])

#     # Reference medians
#     a_ref = df_cond['ang0_bin'].astype(float).median()
#     t_ref = df_cond['t_mid'].median()
#     vel_ref = df_cond['angvel'].median()

#     a_z_ref = (a_ref - r['a_mu']) / r['a_sd']
#     t_z_ref = (t_ref - r['t_mu']) / r['t_sd']
#     vel_z_ref = (vel_ref - r['vel_mu']) / r['vel_sd']
#     azABS_z_ref = (np.abs(a_z_ref) - r['azABS_mu']) / r['azABS_sd']

#     pred_data_storage = {}
#     fig, axes = plt.subplots(1, 3, figsize=(12, 4))

#     # ------------------------
#     # 1. Posture effect (hold t, vel at median)
#     # ------------------------
#     a_range = np.linspace(a_min, a_max, 200)
#     a_z_range = (a_range - r['a_mu']) / r['a_sd']
#     azABS_z_range = (np.abs(a_z_range) - r['azABS_mu']) / r['azABS_sd']

#     pred_data = pd.DataFrame({
#         'a_z': a_z_range,
#         'azABS_z': azABS_z_range,
#         't_z': t_z_ref,
#         'a_t': a_z_range * t_z_ref,
#         'vel_t': vel_z_ref * t_z_ref
#     })
#     X_pred = sm.add_constant(pred_data[model_vars], has_constant='add')
#     prob_posture = model.predict(X_pred)

#     pred_data_storage['posture'] = {'x': a_range, 'y': prob_posture}
#     ax = axes[0]
#     ax.plot(a_range, prob_posture, color='steelblue', lw=2)
#     ax.axvline(a_ref, color='red', linestyle='--', lw=1.5, alpha=0.6)
#     ax.set_xlabel('Initial Posture (deg)')
#     ax.set_ylabel('Predicted P(swim)')
#     ax.set_title('Posture Effect')

#     # ------------------------
#     # 2. Time effect (hold posture, velocity at median) # Hold posture asymmetry fixed at reference value in z-space

#     # ------------------------
#     t_range = np.linspace(t_min, t_max, 200)
#     t_z_range = (t_range - r['t_mu']) / r['t_sd']

#     pred_data = pd.DataFrame({
#         'a_z': a_z_ref,
#         'azABS_z': azABS_z_ref,
#         't_z': t_z_range,
#         'a_t': a_z_ref * t_z_range,
#         'vel_t': vel_z_ref * t_z_range
#     })
#     X_pred = sm.add_constant(pred_data[model_vars], has_constant='add')
#     prob_time = model.predict(X_pred)

#     pred_data_storage['time'] = {'x': t_range, 'y': prob_time}
#     ax = axes[1]
#     ax.plot(t_range, prob_time, color='purple', lw=2)
#     ax.axvline(t_ref, color='red', linestyle='--', lw=1.5, alpha=0.6)
#     ax.set_xlabel('Time into IBI (s)')
#     ax.set_title('Time Effect')

#     # ------------------------
#     # 3. Angular velocity × time interaction # Hold posture asymmetry fixed at reference value in z-space

#     # ------------------------
#     vel_range = np.linspace(vel_min, vel_max, 200)
#     colors = plt.cm.viridis(np.linspace(0.1, 0.9, 5))
#     ax = axes[2]

#     for t_val, c in zip(np.quantile(df_cond['t_mid'], [0.1, 0.3, 0.5, 0.7, 0.9]), colors):
#         t_z_val = (t_val - r['t_mu']) / r['t_sd']
#         pred_data = pd.DataFrame({
#             'a_z': a_z_ref,
#             'azABS_z': azABS_z_ref,
#             't_z': t_z_val,
#             'a_t': a_z_ref * t_z_val,
#             'vel_t': ((vel_range - r['vel_mu']) / r['vel_sd']) * t_z_val
#         })
#         X_pred = sm.add_constant(pred_data[model_vars], has_constant='add')
#         prob_vel_t = model.predict(X_pred)
#         ax.plot(vel_range, prob_vel_t, color=c, lw=2, alpha=0.8, label=f't={t_val:.1f}s')

#     ax.axvline(0, color='gray', linestyle=':', lw=1.5)
#     ax.set_xlabel('Angular Velocity (deg/s)')
#     ax.set_title('Velocity × Time Effect')
#     ax.set_ylabel('Predicted P(swim)')
#     ax.legend(title='Time into IBI', fontsize=9)
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()

#     if save_individual:
#         plt.savefig(os.path.join(fig_dir, f'partial_dependence_clean_{cond0}_{ztime}.pdf'),
#                     bbox_inches='tight', dpi=300)
#         plt.show()

#     if return_pred_data:
#         return pred_data_storage
#     return None



# all_conditions = [(r['cond0'], r['ztime']) for r in results]
# unique_conditions = list(set(all_conditions))

# print(f"Found {len(unique_conditions)} unique condition combinations:")
# for cond0, ztime in sorted(unique_conditions):
#     print(f"  - {cond0} dpf, {ztime}")

# # Generate plots for each condition
# for cond0, ztime in sorted(unique_conditions):
#     plot_partial_dependence_clean(
#         cond0=cond0,
#         ztime=ztime,
#         results=results,
#         boot_hazard_df=boot_hazard_df,
#         model_vars=model_vars,
#         fig_dir=fig_dir,
#         save_individual=True,
#         return_pred_data=False
#     )

# print("\  " + "="*80)
# print("✓ All individual plots generated")
# print("="*80)


#%% take use of all bootstraps to generate uncertainty estimates on partial dependence plots
def build_design_matrix2(pred_df, r, model_vars, reference_vals=None):
    """
    Construct GLM design matrix for a bootstrap result r.
    pred_df must include the variable(s) you sweep; others are filled with reference_vals.
    """
    df = pred_df.copy()

    # Fill missing variables with reference values
    if reference_vals is not None:
        for k, v in reference_vals.items():
            if k not in df.columns:
                df[k] = v

    # z-score all required terms
    df['a_z'] = (df['a'] - r['a_mu']) / r['a_sd']
    df['t_z'] = (df['t'] - r['t_mu']) / r['t_sd']
    df['vel_z'] = (df['vel'] - r['vel_mu']) / r['vel_sd']

    df['a2_z'] = (df['a_z']**2 - r['a2_mu']) / r['a2_sd']
    df['vel2_z'] = (df['vel_z']**2 - r['vel2_mu']) / r['vel2_sd']

    df['aABS_z'] = (np.abs(df['a']) - r['aABS_mu']) / r['aABS_sd']
    df['azABS_z'] = (np.abs(df['a_z']) - r['azABS_mu']) / r['azABS_sd']

    # interactions
    df['a_t'] = df['a_z'] * df['t_z']
    df['vel_t'] = df['vel_z'] * df['t_z']
    df['a2_t'] = df['a2_z'] * df['t_z']
    df['aABS_t'] = df['aABS_z'] * df['t_z']
    df['azABS_t'] = df['azABS_z'] * df['t_z']

    X = sm.add_constant(df[model_vars], has_constant='add')
    return X

def bootstrap_predict(results, pred_df, model_vars, reference_vals=None, agg='median', ci=(16, 84)):
    """
    Aggregate predictions across bootstrap models.
    Returns: p_center, p_lo, p_hi
    """
    prob_stack = []

    for r in results:
        X_pred = build_design_matrix2(pred_df, r, model_vars, reference_vals=reference_vals)
        p = r['model'].predict(X_pred)
        prob_stack.append(p)

    prob_stack = np.asarray(prob_stack)

    if agg == 'mean':
        p_center = prob_stack.mean(axis=0)
    else:
        p_center = np.median(prob_stack, axis=0)

    p_lo, p_hi = np.percentile(prob_stack, ci, axis=0)

    return p_center, p_lo, p_hi

def plot_partial_dependence_bootstrap2(cond1, ztime, results, boot_hazard_df, model_vars, fig_dir, save_individual=True):
    """
    Generate bootstrap-aggregated partial dependence plots for:
      - Posture (a)
      - Time (t)
      - Velocity × Time (vel × t)
    """
    

    # select relevant bootstrap results
    sub = [r for r in results if r['cond1']==cond1 and r['ztime']==ztime]
    if not sub:
        print(f"No results for {cond1} | {ztime}")
        return

    df_cond = boot_hazard_df[(boot_hazard_df['cond1']==cond1) & (boot_hazard_df['ztime']==ztime)]

    # realistic ranges
    a_range = np.linspace(df_cond['ang0_bin'].astype(float).quantile(0.01), df_cond['ang0_bin'].astype(float).quantile(0.99), 200)
    t_range = np.linspace(df_cond['t_mid'].quantile(0.01), df_cond['t_mid'].quantile(0.99), 200)
    # vel_range = np.linspace(df_cond['angvel'].quantile(0.01), df_cond['angvel'].quantile(0.99), 200)
    
    t_min, t_max = df_cond[df_cond['ztime'] == ztime]['t_mid'].astype(float).quantile([0.01, 0.99])
    t_bins = np.linspace(t_min, t_max, 5)  # 5 overlay curves

    vel_min, vel_max = df_cond['angvel'].astype(float).quantile([0.01, 0.99])
    vel_range = np.linspace(vel_min, vel_max, 200)
    
    vel_bins = boot_hazard_df['angvel'].astype(float).quantile([0.1, 0.3, 0.5, 0.7, 0.9]).values

    
    a_min, a_max = df_cond['ang0_bin'].astype(float).quantile([0.1, 0.9])
    a_bins = np.linspace(a_min, a_max, 5)


    t_val_sel = {
        'light': 0.5,
        'dark': 4
    }
    # reference medians for covariates
    ref_vals = {
        'a': df_cond['ang0_bin'].astype(float).median(),
        't': t_val_sel[ztime],
        'vel': df_cond['angvel'].median(),
    }

    fig, axes = plt.subplots(1, 6, figsize=(13,3))

    # ---------------- Posture effect ----------------
    pred_df = pd.DataFrame({'a': a_range})
    p_med, p_lo, p_hi = bootstrap_predict(sub, pred_df, model_vars, reference_vals=ref_vals)
    ax = axes[0]
    ax.plot(a_range, p_med, color='steelblue', lw=2)
    ax.fill_between(a_range, p_lo, p_hi, color='steelblue', alpha=0.2)
    ax.axvline(ref_vals['a'], color='red', linestyle='--', lw=1.5, alpha=0.6)
    ax.set_xlabel('Initial Posture (deg)')
    ax.set_ylabel('Predicted P(swim)')
    ax.set_title('Posture Effect')
    ax.set_ylim(0,0.35)

    # ---------------- Time effect ----------------
    pred_df = pd.DataFrame({'t': t_range})
    p_med, p_lo, p_hi = bootstrap_predict(sub, pred_df, model_vars, reference_vals=ref_vals)
    ax = axes[1]
    ax.plot(t_range, p_med, color='purple', lw=2)
    ax.fill_between(t_range, p_lo, p_hi, color='purple', alpha=0.2)
    ax.axvline(ref_vals['t'], color='red', linestyle='--', lw=1.5, alpha=0.6)
    ax.set_xlabel('Time into IBI (s)')
    ax.set_title('Time Effect')
    ax.set_ylim(0,0.35)

    # ---------------- Velocity × Time effect ----------------
    ax = axes[2]
    colors = plt.cm.viridis(np.linspace(0.1,0.9,5))
    for t_val, c in zip(t_bins, colors):
        # pred_df contains raw velocity and fixed time
        pred_df = pd.DataFrame({
            'vel': vel_range,
            't': t_val  # raw value, not z-scored
        })

        # pass reference medians for posture and any other covariates
        p_med, p_lo, p_hi = bootstrap_predict(
            sub, pred_df, model_vars, reference_vals=ref_vals
        )

        ax.plot(vel_range, p_med, color=c, lw=2, alpha=0.8, label=f't={t_val:.1f}s')
        ax.fill_between(vel_range, p_lo, p_hi, color=c, alpha=0.15)

    ax.axvline(0, color='gray', linestyle=':', lw=1.5)
    ax.set_xlabel('Angular Velocity (deg/s)')
    ax.set_ylabel('Predicted P(swim)')
    ax.set_title('Velocity × Time Effect')
    ax.legend(title='Time into IBI', fontsize=9)
    ax.set_ylim(0,0.35)

    # ---------------- velocity × Time effect (time on x-axis) ----------------
    ax = axes[3]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(vel_bins)))

    for vel_val, c in zip(vel_bins, colors):

        pred_df = pd.DataFrame({
            'vel': vel_val,   # fixed angular velocity
            't': t_range      # sweep time
        })

        p_med, p_lo, p_hi = bootstrap_predict(
            sub, pred_df, model_vars, reference_vals=ref_vals
        )

        ax.plot(
            t_range, p_med,
            color=c, lw=2, alpha=0.85,
            label=f'v={vel_val:.1f}°/s'
        )
        ax.fill_between(
            t_range, p_lo, p_hi,
            color=c, alpha=0.15
        )

    ax.axvline(ref_vals['t'], color='red', linestyle='--', lw=1.5, alpha=0.6)
    ax.set_xlabel('Time into IBI (s)')
    ax.set_ylabel('Predicted P(swim)')
    ax.set_title('Velocity × Time Effect')
    ax.set_ylim(0, 0.35)
    ax.legend(title='Angular velocity', fontsize=9)


    # ---------------- Posture × Time effect ----------------
    ax = axes[4]
    colors = plt.cm.viridis(np.linspace(0.1,0.9,5))
    for t_val, c in zip(t_bins, colors):

        pred_df = pd.DataFrame({
            'a': a_range,
            't': t_val
        })

        p_med, p_lo, p_hi = bootstrap_predict(
            sub, pred_df, model_vars, reference_vals=ref_vals
        )

        ax.plot(a_range, p_med, color=c, lw=2, alpha=0.8, label=f't={t_val:.1f}s')
        ax.fill_between(a_range, p_lo, p_hi, color=c, alpha=0.15)

    ax.axvline(0, color='gray', linestyle=':', lw=1.5)
    ax.set_xlabel('Initial Posture (deg)')
    ax.set_ylabel('Predicted P(swim)')
    ax.set_title('Posture × Time Effect')
    ax.legend(title='Time into IBI', fontsize=9)
    ax.set_ylim(0,0.35)
    
    
    # ---------------- Posture × Time effect (time on x-axis) ----------------
    ax = axes[5]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(a_bins)))

    for a_val, c in zip(a_bins, colors):

        pred_df = pd.DataFrame({
            'a': a_val,      # fixed posture
            't': t_range     # sweep time
        })

        p_med, p_lo, p_hi = bootstrap_predict(
            sub, pred_df, model_vars, reference_vals=ref_vals
        )

        ax.plot(t_range, p_med, color=c, lw=2, alpha=0.8,
                label=f'a={a_val:.1f}°')
        ax.fill_between(t_range, p_lo, p_hi, color=c, alpha=0.15)

    ax.axvline(ref_vals['t'], color='red', linestyle='--', lw=1.5, alpha=0.6)
    ax.set_xlabel('Time into IBI (s)')
    ax.set_ylabel('Predicted P(swim)')
    ax.set_title('Posture × Time Effect')
    ax.legend(title='Initial posture', fontsize=9)
    ax.set_ylim(0, 0.35)
    
    plt.tight_layout()

    if save_individual:
        plt.savefig(os.path.join(fig_dir, f'partial_dependence_bootstrap_{cond1}_{ztime}.pdf'), dpi=300, bbox_inches='tight')
        plt.show()


all_conditions = [(r['cond1'], r['ztime']) for r in results]
unique_conditions = list(set(all_conditions))

for cond1, ztime in sorted(unique_conditions):
    plot_partial_dependence_bootstrap2(
        cond1=cond1,
        ztime=ztime,
        results=results,
        boot_hazard_df=boot_hazard_df,
        model_vars=model_vars,
        fig_dir=fig_dir,
        save_individual=True
    )
#%%
# def plot_partial_dependence(cond0, ztime, results, boot_hazard_df, model_vars, fig_dir, 
#                            save_individual=True, return_pred_data=False):
#     """
#     Generate partial dependence plots for a specific condition and time period.
    
#     Parameters:
#     -----------
#     cond0 : str
#         Condition identifier (e.g., '14')
#     ztime : str
#         Time period ('day' or 'night')
#     results : list
#         List of fitted model results
#     boot_hazard_df : DataFrame
#         Bootstrap hazard data
#     model_vars : list
#         List of model variable names
#     fig_dir : str
#         Directory to save figures
#     save_individual : bool
#         Whether to save individual condition plots
#     return_pred_data : bool
#         Whether to return prediction data for comparison plots
        
#     Returns:
#     --------
#     dict (optional) : Prediction data if return_pred_data=True
#     """
#     import numpy as np
#     import pandas as pd
#     import matplotlib.pyplot as plt
#     import statsmodels.api as sm
#     import os
#     raw_data = raw_heatmaps[(cond0, ztime)]
#     max_probability = np.nanmax(raw_data)

#     # Get a representative model (first bootstrap)
#     condition_results = [r for r in results
#                         if r['cond0'] == cond0 and r['ztime'] == ztime]
    
#     if not condition_results:
#         print(f"Warning: No results found for {cond0}|{ztime}")
#         return None
    
#     # Use first bootstrap as representative
#     r = condition_results[0]
#     model = r['model']
    
#     # Get data range for this condition
#     df_cond = boot_hazard_df[
#         (boot_hazard_df['cond0'] == cond0) &
#         (boot_hazard_df['ztime'] == ztime)
#     ].copy()
    
#     # Determine realistic ranges
#     a_min, a_max = df_cond['ang0_bin'].astype(float).quantile([0.01, 0.99])
#     t_min, t_max = df_cond['t_mid'].astype(float).quantile([0.01, 0.99])
#     vel_min, vel_max = df_cond['angvel'].quantile([0.01, 0.99])
    
#     print(f"  Data ranges for {cond0}|{ztime}:")
#     print(f"  Posture: [{a_min:.1f}, {a_max:.1f}] degrees")
#     print(f"  Time: [{t_min:.1f}, {t_max:.1f}] seconds")
#     print(f"  Velocity: [{vel_min:.1f}, {vel_max:.1f}] deg/s")
    
#     # Create figure only if saving individual plots
#     if save_individual or not return_pred_data:
#         fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    
#     # Reference values (median of data)
#     a_ref = df_cond['ang0_bin'].astype(float).median()
#     t_ref = df_cond['t_mid'].astype(float).median()
#     vel_ref = df_cond['angvel'].median()
    
#     # Standardize using stored parameters
#     a_z_ref = (a_ref - r['a_mu']) / r['a_sd']
#     t_z_ref = (t_ref - r['t_mu']) / r['t_sd']
#     vel_z_ref = (vel_ref - r['vel_mu']) / r['vel_sd']
    
#     # Initialize storage for prediction data
#     pred_data_storage = {}
    
#     # -------------------------------------------------------------------------
#     # Plot 1: a_z effect (vary posture, hold others at median)
#     # -------------------------------------------------------------------------
#     a_range = np.linspace(a_min, a_max, 200)
#     a_z_range = (a_range - r['a_mu']) / r['a_sd']
#     azABS_z_range = (np.abs(a_z_range) - r['azABS_mu']) / r['azABS_sd']
    
#     pred_data = pd.DataFrame({
#         'a_z': a_z_range,
#         't_z': t_z_ref,
#         'azABS_z': azABS_z_range,
#         'a_t': a_z_range * t_z_ref,
#         'vel_t': vel_z_ref * t_z_ref
#     })
    
#     X_pred = pred_data[model_vars].copy()
#     X_pred = sm.add_constant(X_pred, has_constant='add')
#     prob_pred_1 = model.predict(X_pred)
    
#     pred_data_storage['posture'] = {
#         'x': a_range,
#         'y': prob_pred_1,
#         'x_ref': a_ref
#     }
    
#     if save_individual or not return_pred_data:
#         ax = axes[0, 0]
#         ax.plot(a_range, prob_pred_1, linewidth=2, color='steelblue')
#         ax.axvline(a_ref, color='red', linestyle='--', linewidth=2, alpha=0.5,
#                    label=f'Median posture ({a_ref:.1f}°)')
#         ax.fill_between(a_range, 0, prob_pred_1, alpha=0.2, color='steelblue')
#         ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
#         ax.set_title('Posture Effect  (holding time & velocity at median)',
#                      fontsize=13, fontweight='bold')
#         ax.legend(fontsize=10)
#         ax.set_ylim(0, 0.3)
    
#     # -------------------------------------------------------------------------
#     # Plot 2: azABS_z effect AS A FUNCTION OF POSTURE
#     # -------------------------------------------------------------------------
#     a_range_symmetric = np.linspace(a_min, a_max, 200)
#     a_z_range_symmetric = (a_range_symmetric - r['a_mu']) / r['a_sd']
#     azABS_z_range_symmetric = (np.abs(a_z_range_symmetric) - r['azABS_mu']) / r['azABS_sd']
    
#     pred_data = pd.DataFrame({
#         'a_z': 0,
#         't_z': t_z_ref,
#         'azABS_z': azABS_z_range_symmetric,
#         'a_t': 0 * t_z_ref,
#         'vel_t': vel_z_ref * t_z_ref
#     })
    
#     X_pred = pred_data[model_vars].copy()
#     X_pred = sm.add_constant(X_pred, has_constant='add')
#     prob_pred_2 = model.predict(X_pred)
    
#     pred_data_storage['abs_posture'] = {
#         'x': a_range_symmetric,
#         'y': prob_pred_2
#     }
    
#     if save_individual or not return_pred_data:
#         ax = axes[0, 1]
#         ax.plot(a_range_symmetric, prob_pred_2, linewidth=2, color='darkgreen')
#         ax.axvline(0, color='gray', linestyle=':', linewidth=1.5, alpha=0.5,
#                    label='Straight posture')
#         ax.fill_between(a_range_symmetric, 0, prob_pred_2, alpha=0.2, color='darkgreen')
#         ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
#         ax.set_title('Absolute Posture Effect  (|z-scored posture| deviation, a_z=0)',
#                      fontsize=13, fontweight='bold')
#         ax.grid(True, alpha=0.3)
#         ax.legend(fontsize=10)
#         ax.set_ylim(0, 0.3)
    
#     # -------------------------------------------------------------------------
#     # Plot 3: Time effect (t_z)
#     # -------------------------------------------------------------------------
#     t_range = np.linspace(t_min, t_max, 200)
#     t_z_range = (t_range - r['t_mu']) / r['t_sd']
#     azABS_z_ref = (np.abs(a_z_ref) - r['azABS_mu']) / r['azABS_sd']
    
#     pred_data = pd.DataFrame({
#         'a_z': a_z_ref,
#         't_z': t_z_range,
#         'azABS_z': azABS_z_ref,
#         'a_t': a_z_ref * t_z_range,
#         'vel_t': vel_z_ref * t_z_range
#     })
    
#     X_pred = pred_data[model_vars].copy()
#     X_pred = sm.add_constant(X_pred, has_constant='add')
#     prob_pred_3 = model.predict(X_pred)
    
#     pred_data_storage['time'] = {
#         'x': t_range,
#         'y': prob_pred_3,
#         'x_ref': t_ref
#     }
    
#     if save_individual or not return_pred_data:
#         ax = axes[0, 2]
#         ax.plot(t_range, prob_pred_3, linewidth=2, color='purple')
#         ax.axvline(t_ref, color='red', linestyle='--', linewidth=2, alpha=0.5,
#                    label=f'Median time ({t_ref:.1f}s)')
#         ax.fill_between(t_range, 0, prob_pred_3, alpha=0.2, color='purple')
#         ax.set_xlabel('Time into IBI (s)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
#         ax.set_title('Time Effect  (holding posture & velocity at median)',
#                      fontsize=13, fontweight='bold')
#         ax.legend(fontsize=10)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0, 0.3)
    
#     # -------------------------------------------------------------------------
#     # Plot 4: a_t interaction - How posture effect changes with time
#     # -------------------------------------------------------------------------
#     if save_individual or not return_pred_data:
#         ax = axes[1, 0]
#         time_slices = np.linspace(t_min, t_max, 5)
#         colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(time_slices)))
        
#         for t_val, color in zip(time_slices, colors):
#             t_z_val = (t_val - r['t_mu']) / r['t_sd']
#             azABS_z_range = (np.abs(a_z_range) - r['azABS_mu']) / r['azABS_sd']
#             pred_data = pd.DataFrame({
#                 'a_z': a_z_range,
#                 't_z': t_z_val,
#                 'azABS_z': azABS_z_range,
#                 'a_t': a_z_range * t_z_val,
#                 'vel_t': vel_z_ref * t_z_val
#             })
#             X_pred = pred_data[model_vars].copy()
#             X_pred = sm.add_constant(X_pred, has_constant='add')
#             prob_pred = model.predict(X_pred)
#             ax.plot(a_range, prob_pred, linewidth=2.5, color=color,
#                     label=f'{t_val:.1f}s', alpha=0.8)
        
#         ax.set_xlabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
#         ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
#         ax.set_title('a_t: How Posture Effect Changes with Time',
#                      fontsize=13, fontweight='bold')
#         ax.legend(title='Time into IBI', fontsize=9, ncol=2)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0, 0.3)
    
#     # -------------------------------------------------------------------------
#     # Plot 5: vel_t interaction - How velocity effect changes with time
#     # -------------------------------------------------------------------------
#     if save_individual or not return_pred_data:
#         ax = axes[1, 1]
#         vel_range = np.linspace(vel_min, vel_max, 200)
#         vel_z_range = (vel_range - r['vel_mu']) / r['vel_sd']
        
#         for t_val, color in zip(time_slices, colors):
#             t_z_val = (t_val - r['t_mu']) / r['t_sd']
#             azABS_z_ref = (np.abs(a_z_ref) - r['azABS_mu']) / r['azABS_sd']
#             pred_data = pd.DataFrame({
#                 'a_z': a_z_ref,
#                 't_z': t_z_val,
#                 'azABS_z': azABS_z_ref,
#                 'a_t': a_z_ref * t_z_val,
#                 'vel_t': vel_z_range * t_z_val
#             })
#             X_pred = pred_data[model_vars].copy()
#             X_pred = sm.add_constant(X_pred, has_constant='add')
#             prob_pred = model.predict(X_pred)
#             ax.plot(vel_range, prob_pred, linewidth=2.5, color=color,
#                     label=f'{t_val:.1f}s', alpha=0.8)
        
#         ax.axvline(0, color='gray', linestyle=':', linewidth=1.5)
#         ax.set_xlabel('Angular Velocity (deg/s)  Nose-down ← 0 → Nose-up',
#                       fontsize=12, fontweight='bold')
#         ax.set_ylabel('Predicted P(swim)', fontsize=12, fontweight='bold')
#         ax.set_title('vel_t: How Velocity Effect Changes with Time  (Early corrections prioritized)',
#                      fontsize=13, fontweight='bold')
#         ax.legend(title='Time into IBI', fontsize=9)
#         ax.grid(True, alpha=0.3)
#         ax.set_ylim(0, 0.3)
    
#     # -------------------------------------------------------------------------
#     # Plot 6: Heatmap - Time into IBI vs Initial Posture
#     # -------------------------------------------------------------------------
#     if save_individual or not return_pred_data:
#         ax = axes[1, 2]
        
#         a_heatmap = np.linspace(a_min, a_max, 100)
#         t_heatmap = np.linspace(t_min, t_max, 100)
#         A_mesh, T_mesh = np.meshgrid(a_heatmap, t_heatmap)
        
#         A_z_mesh = (A_mesh - r['a_mu']) / r['a_sd']
#         T_z_mesh = (T_mesh - r['t_mu']) / r['t_sd']
#         azABS_z_mesh = (np.abs(A_z_mesh) - r['azABS_mu']) / r['azABS_sd']
        
#         n_points = A_mesh.shape[0] * A_mesh.shape[1]
#         pred_data_2d = pd.DataFrame({
#             'a_z': A_z_mesh.flatten(),
#             't_z': T_z_mesh.flatten(),
#             'azABS_z': azABS_z_mesh.flatten(),
#             'a_t': A_z_mesh.flatten() * T_z_mesh.flatten(),
#             'vel_t': vel_z_ref * T_z_mesh.flatten()
#         })
        
#         X_pred_2d = pred_data_2d[model_vars].copy()
#         X_pred_2d = sm.add_constant(X_pred_2d, has_constant='add')
#         prob_pred_2d = model.predict(X_pred_2d).values.reshape(A_mesh.shape)
        
#         im = ax.imshow(prob_pred_2d.T, aspect='auto', cmap='plasma',
#                        origin='lower', extent=[t_min, t_max, a_min, a_max])
        
#         contours = ax.contour(prob_pred_2d.T, levels=8, colors='white',
#                              linewidths=1.5, alpha=0.6,
#                              extent=[t_min, t_max, a_min, a_max])
#         ax.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        
#         ax.set_ylabel('Initial Posture (degrees)', fontsize=12, fontweight='bold')
#         ax.set_xlabel('Time into IBI (s)', fontsize=12, fontweight='bold')
#         ax.set_title('Posture × Time Heatmap  (velocity at median)',
#                      fontsize=13, fontweight='bold')
        
#         cbar = plt.colorbar(im, ax=ax)
#         cbar.set_label('P(swim)', fontsize=11, fontweight='bold')
    
#     # Save individual plot
#     if save_individual:
#         plt.suptitle(f'Partial Dependence Plots: {cond0} dpf | {ztime}  ' +
#                      f'Showing isolated effect of each variable',
#                      fontsize=16, fontweight='bold', y=0.995)
        
#         plt.tight_layout()
#         plt.savefig(os.path.join(fig_dir, f'partial_dependence_{cond0}_{ztime}.pdf'),
#                     format='pdf', dpi=300, bbox_inches='tight')
#         plt.savefig(os.path.join(fig_dir, f'partial_dependence_{cond0}_{ztime}.png'),
#                     dpi=200, bbox_inches='tight')
#         plt.show()
#         print(f"  ✓ Saved partial dependence plots for {cond0}|{ztime}")
    
#     if return_pred_data:
#         return pred_data_storage
    
#     return None


# # Get all unique combinations of conditions
# all_conditions = [(r['cond0'], r['ztime']) for r in results]
# unique_conditions = list(set(all_conditions))

# print(f"Found {len(unique_conditions)} unique condition combinations:")
# for cond0, ztime in sorted(unique_conditions):
#     print(f"  - {cond0} dpf, {ztime}")

# # Generate plots for each condition
# for cond0, ztime in sorted(unique_conditions):
#     plot_partial_dependence(
#         cond0=cond0,
#         ztime=ztime,
#         results=results,
#         boot_hazard_df=boot_hazard_df,
#         model_vars=model_vars,
#         fig_dir=fig_dir,
#         save_individual=True,
#         return_pred_data=False
#     )

# print("\  " + "="*80)
# print("✓ All individual plots generated")
# print("="*80)
#%%