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

#%%
##### Parameters to change #####

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
mask = (
    (IBI_angles["cond1"].isin(["ld"])) 
)
cols_needed = ["yvel", "expNum", "cond0", "cond1", "ztime", "unique_IBI_idx", "time_relative_s", "angVelSmoothed", "ang", "x" ,"y", 'headx', 'heady','swimSpeed']
df_time_filtered = IBI_angles.loc[mask, cols_needed]
# Compute group max (IBI duration) and filter in-place

print("> checkpoint after early filtering")

#%% release memory
del IBI_angles
gc.collect()

#%%
#%% truncate to start from speed threshold
df_time_filtered['swimSpeed_smoothed'] = df_time_filtered.groupby('unique_IBI_idx', observed=True)['swimSpeed'].transform(
    lambda x: savgol_filter(x.values, 11, 3) if len(x) >= 11 else x.values
)

#%%
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

#%%
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

#%% further cleaning: remove IBIs with high velocity after 1s, likely representing translocation
df_valid = df_truncated2.query("cond1 == 'ld'")

# after_1s_mask = df_valid["time_relative_s"] > 1

# df_valid["yvel_sg"] = savgol_filter(df_valid["yvel"].values, 11, 3)
# df_valid["angvel_sg"] = savgol_filter(df_valid["angVelSmoothed"].values, 11, 3)
# df_valid["speed_sg"] = savgol_filter(df_valid["swimSpeed"].values, 11, 3)

# after = df_valid.loc[after_1s_mask]

# #%
# stats = after.groupby("unique_IBI_idx", observed=True).agg(
#     p50_yvel=("yvel_sg", lambda x: np.percentile(x, 50)),
#     p75_angVelSmoothed=("angvel_sg", lambda x: np.percentile(x, 75)),
#     p75_swimSpeed=("speed_sg", lambda x: np.percentile(x, 75)),
#     mad_swimSpeed=("speed_sg", lambda x: np.median(np.abs(x - np.median(x)))),
#     mad_yvel=("yvel_sg", lambda x: np.median(np.abs(x - np.median(x)))),
#     mad_angVelSmoothed=("angvel_sg", lambda x: np.median(np.abs(x - np.median(x))))
# )

# #%
# valid_IBI_idx = stats.index[
#     (stats.p50_yvel < 0.01) &
#     (stats.p75_angVelSmoothed < 0) &
#     (stats.p75_swimSpeed < 0.2) &
#     (stats.mad_swimSpeed < 0.04) &
#     (stats.mad_angVelSmoothed < 1)
# ]  
# # valid_IBI_idx = df_valid.unique_IBI_idx.unique()  # keep all for now
# #%%
before_08s_mask = df_valid["time_relative_s"] < 0.8
before = df_valid.loc[before_08s_mask]
stats2 = before.groupby("unique_IBI_idx", observed=True).agg(
    ang_accel=("angVelSmoothed", lambda x: x.diff().median()),
)
valid_IBI_idx2 = stats2.index[stats2.ang_accel > -0.03]


#%% remove non-contiguous IBIs

# df_filtered = df_valid[df_valid.unique_IBI_idx.isin(np.intersect1d(valid_IBI_idx2, valid_IBI_idx))].copy()
df_passed_QC = df_valid[df_valid.unique_IBI_idx.isin(valid_IBI_idx2)].copy()

from collections import Counter

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
# %%
# calcualte some attributes
grouped = df_filtered.groupby('unique_IBI_idx', observed=True)
IBI_features = grouped.agg(
    IBI_time=('time_relative_s', 'max'),
    IBI_ydispl=('y', lambda x: x.iloc[-1] - x.iloc[0]),
    IBI_xdispl=('x', lambda x: np.abs(x.iloc[-1] - x.iloc[0])),
    IBI_rot=('ang', lambda x: x.iloc[-1] - x.iloc[0]),
    IBI_yvel_avg=('yvel', 'median'),
    IBI_angvel_avg=('angVelSmoothed', 'median'),
    cond0=('cond0', 'first'),
    cond1=('cond1', 'first'),
    expNum=('expNum', 'first'),
    ztime=('ztime', 'first'),
    IBI=('unique_IBI_idx', 'count'),
    aftBout_pitch = ('ang', lambda x: x.iloc[0]),
    bfrBout_pitch = ('ang', lambda x: x.iloc[-1]),
    avg_pitch = ('ang', lambda x: x.median()),
).reset_index()

IBI_features = IBI_features.assign(
    frequency = lambda df: 1 / df['IBI_time']
)
features_toplt =  ['aftBout_pitch','IBI_rot','IBI_yvel_avg','IBI_angvel_avg','IBI_ydispl','bfrBout_pitch','avg_pitch','frequency']
# median per expNum
print(IBI_features.groupby(['cond0','expNum']).size())
median_res = IBI_features.groupby(['cond0','expNum','ztime'])[features_toplt].median().reset_index()
    

# %%
# let's plot something
data_to_plot = median_res

for feature in features_toplt:
    g = plt_categorical_combined_3(
        data=data_to_plot,
        x='cond0',
        y=feature,
        col='ztime',
        order=cond0_all,
        palette=my_palette,
        height=3,
        units='expNum',
    )
# save pdf
    plt.savefig(os.path.join(fig_dir, f'IBI_feature_{feature}_byCond0.pdf'), format='pdf')
    
# %%
# data_to_plot = IBI_features.copy()
# what_x = 'bfrBout_pitch'
# what_y = 'frequency'

# x_range = np.percentile(data_to_plot[what_x], [5,95])

# # determin bins based on what_x
# BIN_WIDTH = 2
# bins = np.linspace(
#     x_range[0],
#     x_range[1] + BIN_WIDTH,
#     math.ceil((x_range[1] - x_range[0]) / BIN_WIDTH) + 1
# )

# all_cond0 = data_to_plot['cond0'].unique().tolist()
# all_cond0.sort()

# df_binned = data_to_plot.groupby(['cond0','expNum','ztime']).apply(
#     lambda group: distribution_binned_average_opt(
#         df=group,
#         bin_col=what_y,
#         by_col=what_x,
#         method="median",
#         bin=bins
#     )
# )
# df_binned.columns = [f'binned_{what_y}', f'binned_{what_x}']
# df_binned = df_binned.reset_index()

# # compute bin centers
# df_binned["bin_center"] = df_binned[f"{what_x}"].apply(
#     lambda interval: (interval.left + interval.right) / 2
# )

# def add_binned_line(data, **kwargs):
#     ax = plt.gca()

#     # identify the cond0 corresponding to this facet
#     cond = data["cond0"].iloc[0]
#     ztime = data["ztime"].iloc[0]

#     # find its index in the ordered cond list
#     idx = all_cond0.index(cond)

#     # pick correct color from palette
#     color = my_palette[idx]

#     # subset df_binned for this cond
#     df_sub = df_binned[df_binned["cond0"] == cond]
#     df_subsub = df_sub[df_sub["ztime"] == ztime]

#     # plot the line
#     sns.lineplot(
#         data=df_subsub,
#         x="bin_center",
#         y=f"binned_{what_x}",
#         ax=ax,
#         color=color,
#         linewidth=2,
#         errorbar='se',
#     )


# # scatter data
# df_scatter = data_to_plot.copy()

# # build the facet grid
# g = sns.FacetGrid(
#     df_scatter,
#     col="cond0",
#     row="ztime",
#     col_order=cond0_all,
#     height=3,
#     sharex=True,
#     sharey='row'
# )

# # scatter layer
# g.map_dataframe(
#     sns.scatterplot,
#     x=what_x,
#     y=what_y,
#     color="grey",
#     alpha=0.02,
#     s=10,
# )

# # line layer: COLOR + ORDER MATCHED VIA all_cond0 + palette
# g.map_dataframe(add_binned_line)

# # y limits
# g.set(
#     ylim=(0, np.percentile(data_to_plot[what_y], 90)),
#     xlim=x_range
# )

# g.tight_layout()


# %%

# %%
df_filtered_ld = df_filtered.copy()
# drop unique_IBI_idx that have the first time_relative_s not equal to 0
df_filtered_ld = df_filtered_ld[df_filtered_ld.groupby('unique_IBI_idx')['time_relative_s'].transform('min') == 0].copy()

# Compute drift per unique_IBI_idx
drift = df_filtered_ld.groupby('unique_IBI_idx')['ang'].transform(lambda x: x.iloc[-1] - x.iloc[0])

# Filter rows where drift < 1
df_negative_drift = df_filtered_ld[drift < 1].copy()

# #%% new filtering
# #%
# tv_df_all_ = []

# for ztime, df_z in df_negative_drift.groupby('ztime', observed=True):
#     print(f"ztime: {ztime}")
#     duration_thresh = np.percentile(df_z['duration'], 90)
#     df = df_z.query("duration <= @duration_thresh").copy()
#     rows = []
#     for ibi, g in df.groupby('unique_IBI_idx', sort=False):
#         # g = g.sort_values('time_relative_s')
#         times = g['time_relative_s'].values
#         angs = g['ang'].values
#         # group-level metadata (assume constant per IBI)
#         meta = g[['expNum', 'cond0', 'cond1','ztime']].iloc[0].to_dict()

#         if len(times) < 2:
#             continue  # cannot make intervals

#         for i in range(len(times)-1):
#             start = times[i]
#             stop = times[i+1]
#             ang_val = angs[i]  # covariate value at interval start
#             event = 1 if i == len(times)-2 else 0  # last interval marks event
#             row = {
#                 'unique_IBI_idx': ibi,
#                 'start': start,
#                 'stop': stop,
#                 'event': event,
#                 'ang': ang_val,
#                 **meta
#             }
#             rows.append(row)

#     tv_df = pd.DataFrame(rows)
#     print("rows:", len(tv_df), "IBIs:", tv_df['unique_IBI_idx'].nunique())

#     # -----------------------------
#     # Standardize covariates safely
#     # -----------------------------

#     # Standardize ang
#     tv_df['ang_z'] = (tv_df['ang'] - tv_df['ang'].mean()) / tv_df['ang'].std(ddof=0)

#     # Add tiny epsilon to avoid log(0)
#     eps = 1e-3
#     tv_df['log_time'] = np.log(tv_df['start'] + eps)

#     # Clip log_time based on observed data to avoid extremes
#     clip_lo = np.floor(tv_df['log_time'].min())
#     clip_hi = np.ceil(tv_df['log_time'].max())
#     tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

#     # Compute interaction and standardize it
#     tv_df['ang_z_logt'] = tv_df['ang_z'] * tv_df['log_time_clipped']
#     tv_df['ang_z_logt'] = (tv_df['ang_z_logt'] - tv_df['ang_z_logt'].mean()) / tv_df['ang_z_logt'].std(ddof=0)
#     # -----------------------------
#     tv_df_all_.append(tv_df)
# tv_df_all = pd.concat(tv_df_all_, ignore_index=True)

# #%%
# results_exp = []

# # Loop over ztime
# for ztime, tv_df in tv_df_all.groupby('ztime', observed=True):
#     print(f"\n=== ztime: {ztime} ===")
    
#     # Group by condition and experimental repeat
#     for (cond, exp), df_e in tv_df.groupby(['cond0', 'expNum'], observed=True):
#         if len(df_e) < 5:  # skip very small repeats
#             continue
        
#         # Fit full logistic model
#         m_full = smf.logit(
#             "event ~ ang_z + log_time_clipped + ang_z_logt",
#             data=df_e
#         ).fit(disp=False)
        
#         # Store coefficients and p-values
#         for param, coef in m_full.params.items():
#             results_exp.append({
#                 'ztime': ztime,
#                 'cond0': cond,
#                 'expNum': exp,
#                 'parameter': param,
#                 'coef': coef,
#                 'pval': m_full.pvalues[param]
#             })

# results_exp_df = pd.DataFrame(results_exp)

# #%%
# g = sns.catplot(
#     data=results_exp_df,
#     x='cond0',
#     y='coef',
#     hue='cond0',
#     row='ztime',
#     col='parameter',
#     kind='point',
#     join=False,
#     capsize=0.1,
#     errwidth=1,
#     palette='tab10',
#     height=3,
#     aspect=0.8,
#     errorbar='se',
#     sharey=False
# )
# plt.tight_layout()
# plt.savefig(os.path.join(fig_dir, "catplot_coef_per_exp.pdf"), format='pdf')

# #%%

# anova_results = {}
# pairwise_results = {}

# for ztime, results_exp_df_cond in results_exp_df.groupby('ztime', observed=True):
#     print(f"\n=== ztime: {ztime} ===")
#     for param in results_exp_df_cond['parameter'].unique():
#         df_param = results_exp_df_cond[results_exp_df_cond['parameter'] == param]
        
#         # One-way ANOVA: coef ~ cond0
#         model = ols('coef ~ C(cond0)', data=df_param).fit()
#         anova_table = sm.stats.anova_lm(model, typ=2)
#         anova_results[param] = anova_table
        
#         # Pairwise comparison (Tukey HSD)
#         tukey = pairwise_tukeyhsd(
#             endog=df_param['coef'],
#             groups=df_param['cond0'],
#             alpha=0.05
#         )
#         pairwise_results[param] = tukey
#     for param in anova_results:
#         print(f"\nParameter: {param}")
#         print("ANOVA:\n", anova_results[param])
#         print("Pairwise Tukey HSD:\n", pairwise_results[param])


#%%

# matching previous behavior
MAX_T = 5.0  # max IBI time (seconds) for censoring, like your previous MAX_T
EPS = 1e-3   # small epsilon to avoid log(0)
MIN_ROWS_PER_EXP = 5  # skip very small repeats

# -----------------------------
# Prepare tv_df_all
# -----------------------------
tv_df_all_ = []

# Filter IBIs shorter than threshold per ztime (e.g., 90th percentile)
for ztime, df_z in df_negative_drift.groupby('ztime', observed=True):
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
    tv_df['log_time'] = np.log(tv_df['start'] + EPS)
    clip_lo = np.floor(tv_df['log_time'].min())
    clip_hi = np.ceil(tv_df['log_time'].max())
    tv_df['log_time_clipped'] = tv_df['log_time'].clip(clip_lo, clip_hi)

    # Interaction (initial posture × log-time)
    tv_df['ang0_z_logt'] = tv_df['ang0_z'] * tv_df['log_time_clipped']
    tv_df['ang0_z_logt'] = (tv_df['ang0_z_logt'] - tv_df['ang0_z_logt'].mean()) / tv_df['ang0_z_logt'].std(ddof=0)

    tv_df_all_.append(tv_df)

tv_df_all = pd.concat(tv_df_all_, ignore_index=True)

# -----------------------------
# Fit logistic regression per cond0 × expNum × ztime
# -----------------------------
results_exp = []

for ztime, tv_df in tv_df_all.groupby('ztime', observed=True):
    print(f"\n=== ztime: {ztime} ===")

    for (cond, exp), df_e in tv_df.groupby(['cond0', 'expNum'], observed=True):
        if len(df_e) < MIN_ROWS_PER_EXP:
            continue

        # Full model
        m_full = smf.logit(
            "event ~ ang0_z + log_time_clipped + ang0_z_logt",
            data=df_e
        ).fit(disp=False)

        # Store coefficients and p-values
        for param, coef in m_full.params.items():
            results_exp.append({
                'ztime': ztime,
                'cond0': cond,
                'expNum': exp,
                'parameter': param,
                'coef': coef,
                'pval': m_full.pvalues[param]
            })

results_exp_df = pd.DataFrame(results_exp)
#%
g = sns.catplot(
    data=results_exp_df,
    x='cond0',
    y='coef',
    hue='cond0',
    row='ztime',
    col='parameter',
    kind='point',
    join=False,
    capsize=0.1,
    errwidth=1,
    palette='tab10',
    height=3,
    aspect=0.8,
    errorbar='se',
    sharey=False
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "catplot_coef_per_exp_byP0.pdf"), format='pdf')


anova_results = {}
pairwise_results = {}

for ztime, results_exp_df_cond in results_exp_df.groupby('ztime', observed=True):
    print(f"\n=== ztime: {ztime} ===")
    for param in results_exp_df_cond['parameter'].unique():
        df_param = results_exp_df_cond[results_exp_df_cond['parameter'] == param]
        
        # One-way ANOVA: coef ~ cond0
        model = ols('coef ~ C(cond0)', data=df_param).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        anova_results[param] = anova_table
        
        # Pairwise comparison (Tukey HSD)
        tukey = pairwise_tukeyhsd(
            endog=df_param['coef'],
            groups=df_param['cond0'],
            alpha=0.05
        )
        pairwise_results[param] = tukey
    for param in anova_results:
        print(f"\nParameter: {param}")
        print("ANOVA:\n", anova_results[param])
        print("Pairwise Tukey HSD:\n", pairwise_results[param])

#%% 
# Pairwise condition comparisons
import itertools
import pandas as pd
import statsmodels.formula.api as smf
import numpy as np

# Prepare container for pairwise comparisons
pairwise_results = []

# Loop through ztime
for ztime, tv_df in tv_df_all.groupby('ztime', observed=True):
    print(f"ztime: {ztime}")

    # Fit full model across all conditions (including cond0 as categorical)
    m_full = smf.logit(
        "event ~ ang0_z + log_time_clipped + ang0_z_logt + C(cond0)",
        data=tv_df
    ).fit(disp=False)

    # Unique condition pairs
    cond_levels = tv_df['cond0'].unique()
    for cond1, cond2 in itertools.combinations(cond_levels, 2):

        # Compute predicted probabilities at mean covariates
        mean_ang0 = tv_df['ang0_z'].mean()
        mean_logt = tv_df['log_time_clipped'].mean()
        mean_inter = tv_df['ang0_z_logt'].mean()

        X_pred = pd.DataFrame([
            {"ang0_z": mean_ang0,
             "log_time_clipped": mean_logt,
             "ang0_z_logt": mean_inter,
             "cond0": cond1},
            {"ang0_z": mean_ang0,
             "log_time_clipped": mean_logt,
             "ang0_z_logt": mean_inter,
             "cond0": cond2}
        ])

        # Predicted probabilities
        p_pred = m_full.predict(X_pred)
        diff = p_pred.iloc[1] - p_pred.iloc[0]  # cond2 - cond1

        # Store results
        pairwise_results.append({
            "ztime": ztime,
            "cond1": cond1,
            "cond2": cond2,
            "prob_diff": diff
        })

# Convert to DataFrame for plotting
pairwise_df = pd.DataFrame(pairwise_results)
#%%
# -----------------------------
# Visualization of predicted curves
# -----------------------------
# # can keep this, but somehow 7 dpf has weorse time compenent

t_grid = np.linspace(0, MAX_T, 100)
t_log = np.log(t_grid + EPS)
cond_levels = tv_df_all['cond0'].unique()
ang_levels = [-1, 0, 1]
colors = {-1: "tab:blue", 0: "tab:gray", 1: "tab:red"}

for ztime, df_z in tv_df_all.groupby('ztime', observed=True):
    clip_lo = np.floor(df_z['log_time'].min())
    clip_hi = np.ceil(df_z['log_time'].max())
    t_log_clip = np.clip(t_log, clip_lo, clip_hi)

    fig, axes = plt.subplots(1, len(cond_levels), figsize=(4*len(cond_levels), 3), sharey=True)
    if len(cond_levels) == 1:
        axes = [axes]

    m_cond = smf.logit("event ~ ang0_z + log_time_clipped + ang0_z_logt + C(cond0)", data=df_z).fit(disp=False)

    for ax, cond in zip(axes, cond_levels):
        for ang in ang_levels:
            X = pd.DataFrame({
                "ang0_z": ang,
                "log_time_clipped": t_log_clip,
                "ang0_z_logt": ang * t_log_clip,
                "cond0": cond,
            })
            p = m_cond.predict(X)
            ax.plot(t_grid, p, color=colors[ang], lw=2, label=f"ang0_z={ang}")

        ax.set_title(f"cond0 = {cond}")
        ax.set_xlabel("Time into IBI (s)")
        ax.grid(False)

    axes[0].set_ylabel("Predicted event probability")
    axes[0].legend(frameon=False)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"IBI_TV_logistic_cond0_ztime{ztime}.pdf"), format="pdf")

#%% likelihood ratio test for cond0 effect
from scipy.stats import chi2

results_lr = []

for ztime, df_z in tv_df_all.groupby('ztime', observed=True):
    print(f"\n=== ztime: {ztime} ===")

    # Null model (no cond0)
    m_null = smf.logit("event ~ ang0_z + log_time_clipped + ang0_z_logt", data=df_z).fit(disp=False)

    # Full model (with cond0)
    m_full = smf.logit("event ~ ang0_z + log_time_clipped + ang0_z_logt + C(cond0)", data=df_z).fit(disp=False)

    # Likelihood ratio test
    LR = -2 * (m_null.llf - m_full.llf)
    df_diff = m_full.df_model - m_null.df_model  # degrees of freedom difference
    pval = chi2.sf(LR, df_diff)

    results_lr.append({
        "ztime": ztime,
        "LR_stat": LR,
        "df": df_diff,
        "pval": pval
    })

results_lr_df = pd.DataFrame(results_lr)
print(results_lr_df)

#%%
# bootstrappppppp

boot_results = []

n_boot = 100

for (ztime, cond), df_cond in tv_df_all.groupby(['ztime', 'cond0'], observed=True):
    # Pre-calculate indices to avoid repeated groupby inside the bootstrap loop
    ibi_groups = df_cond.groupby(['expNum', 'unique_IBI_idx'], observed=True).indices

    # Get a list of unique IBIs per experiment
    exp_to_ibis = {}
    for (exp, ibi), indices in ibi_groups.items():
        if exp not in exp_to_ibis:
            exp_to_ibis[exp] = []
        exp_to_ibis[exp].append(indices)

    exp_list = list(exp_to_ibis.keys())
    
    n_ibi_per_exp = int(np.median([len(ibis) for ibis in exp_to_ibis.values()]))
    
    for b in tqdm(range(n_boot), desc=f"Bootstrapping {ztime}, {cond} with {n_ibi_per_exp} per exp"):
        boot_idx_list = []
        
        for exp in exp_list:
            available_ibis = exp_to_ibis[exp]
            # np.random.choice can take a list of arrays (the indices)
            # Choosing indices directly is faster than choosing IDs and then looking up indices
            sel_idx = np.random.randint(0, len(available_ibis), size=n_ibi_per_exp)
            
            for idx in sel_idx:
                boot_idx_list.append(available_ibis[idx])
        
        # Flatten and locate
        flat_idx = np.concatenate(boot_idx_list)
        df_boot = df_cond.iloc[flat_idx]

        try:
            m_boot = smf.logit(
                "event ~ ang0_z + log_time_clipped + ang0_z_logt",
                data=df_boot
            ).fit(disp=False)

        except Exception:
            continue  # skip failed bootstrap fits

        for param, coef in m_boot.params.items():
            boot_results.append({
                'ztime': ztime,
                'cond0': cond,
                'bootstrap': b,
                'parameter': param,
                'coef': coef
            })

boot_df = pd.DataFrame(boot_results)


#%%

# Plot bootstrap coefficients directly
g = sns.catplot(
    data=boot_df,
    kind='point',
    x='cond0',
    row='ztime',
    y='coef',
    col='parameter',  
    errorbar='sd',          
    height=3,
    aspect=0.7,
    sharey='False',
)
plt.savefig(os.path.join(fig_dir, "catplot_bootstrap_coef_per_cond0_byP0.pdf"), format='pdf')

#%%

boot_wide = boot_df.pivot_table(
    index=['ztime', 'bootstrap', 'parameter'],
    columns='cond0',
    values='coef'
).reset_index()

from itertools import combinations

pairwise_results = []

for ztime, df_z in boot_wide.groupby('ztime'):
    params = df_z['parameter'].unique()
    conds = [c for c in df_z.columns if c not in ['ztime','bootstrap','parameter']]
    
    for param in params:
        df_param = df_z[df_z['parameter'] == param]
        
        for cond1, cond2 in combinations(conds, 2):
            diff = df_param[cond2].values - df_param[cond1].values
            p_val = 2 * min((diff > 0).mean(), (diff < 0).mean())  # two-tailed bootstrap p
            pairwise_results.append({
                'ztime': ztime,
                'parameter': param,
                'cond1': cond1,
                'cond2': cond2,
                'diff_mean': diff.mean(),
                'diff_std': diff.std(),
                'p_val': p_val
            })
pairwise_df = pd.DataFrame(pairwise_results)
print(pairwise_df)

#%% predict event probability
# boot_wide = boot_df.pivot_table(
#     index=['ztime', 'bootstrap', 'parameter'],
#     columns='cond0',
#     values='coef'
# ).reset_index()

MAX_T = 5

t_grid = np.linspace(0.1, MAX_T, 100)
t_log = np.log(t_grid + EPS)
ang_levels = [-1, 0, 1]
rows = []

for ztime, df_z in boot_df.groupby('ztime'):
    cond_levels = df_z['cond0'].unique()
    
    for cond in cond_levels:
        # Extract all bootstraps for this cond
        coefs_cond = df_z[df_z['cond0'] == cond]
        coefs_matrix = coefs_cond.pivot(index='bootstrap', columns='parameter', values='coef').sort_index()
        n_boot = coefs_matrix.shape[0]
        
        for ang in ang_levels:
            # Build design matrix
            X = pd.DataFrame({
                "Intercept": 1,
                "ang0_z": ang,
                "log_time_clipped": t_log,
                "ang0_z_logt": ang * t_log
            })
            X = X[coefs_matrix.columns]  # align columns
            X_mat = X.values  # n_time x n_parameters
            B_mat = coefs_matrix.values.T  # n_parameters x n_boot
            
            # Predict probabilities across all bootstraps
            linpred = X_mat @ B_mat  # n_time x n_boot
            p_boot = 1 / (1 + np.exp(-linpred))
            
            # Flatten into long-form for Seaborn
            n_time, n_boot = p_boot.shape
            for i in range(n_time):
                for b in range(n_boot):
                    rows.append({
                        'ztime': ztime,
                        'cond0': cond,
                        'ang0_z': ang,
                        'time': t_grid[i],
                        'prob': p_boot[i, b]
                    })

# Final DataFrame ready for Seaborn
sim_df = pd.DataFrame(rows)

#%%
g = sns.relplot(
    data=sim_df,
    x='time',
    y='prob',
    hue='ang0',
    row='ztime',
    col='cond0',
    kind='line',
    # palette={-1:"tab:blue", 0:"tab:gray", 1:"tab:red"},
    height=3,
    aspect=1.2,
    errorbar='sd',   
    facet_kws={'sharey': 'row', 'sharex': True}
)
g.set(xlim=(0, MAX_T))
g.set_axis_labels("Time into IBI (s)", "Predicted event probability")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f"IBI_TV_logistic_cond0_ztime_boot_relplot.pdf"), format="pdf")

#%%
EPS = 1e-8
MAX_T = 5  # adjust according to your data
t_grid = np.linspace(0.1, MAX_T, 100)
colors = {-1: "tab:blue", 0: "tab:gray", 1: "tab:red"}  # only for discrete angles, optional

# -----------------------------
# 1. Prepare long-form DataFrame of predicted probabilities
# -----------------------------
rows = []
ang0_grid_points = 10  # number of ang0 points to predict per ztime

for ztime, df_z in boot_df.groupby('ztime', observed=True):
    cond_levels = df_z['cond0'].unique()
    
    # Compute per-ztime mean/std of ang0 for transforming back
    df_tv = tv_df_all[tv_df_all['ztime'] == ztime]
    ang0_mean = df_tv['ang0'].mean()
    ang0_std = df_tv['ang0'].std(ddof=0)
    
    # Interaction term stats for later z-scoring
    def compute_ang0_logt_z(ang0_z_grid, t_log_clip):
        ang0_logt = ang0_z_grid[:, None] * t_log_clip[None, :]
        return (ang0_logt - ang0_logt.mean()) / ang0_logt.std(ddof=0)
    
    for cond in cond_levels:
        coefs_cond = df_z[df_z['cond0'] == cond]
        coefs_matrix = coefs_cond.pivot(index='bootstrap', columns='parameter', values='coef').sort_index()
        n_boot = coefs_matrix.shape[0]
        
        # Generate ang0 grid in original scale
        ang0_grid = np.linspace(-20, 35, ang0_grid_points)
        ang0_z_grid = (ang0_grid - ang0_mean) / ang0_std
        
        # Log-time clipping (same as preprocessing)
        log_time = np.log(t_grid + EPS)
        clip_lo = np.floor(log_time.min())
        clip_hi = np.ceil(log_time.max())
        log_time_clip = np.clip(log_time, clip_lo, clip_hi)
        
        # Compute z-scored interaction term
        ang0_z_logt_grid = compute_ang0_logt_z(ang0_z_grid, log_time_clip)
        
        # Design matrix: shape (n_ang0 x n_time) x n_parameters
        X_mat = np.column_stack([
            np.ones(ang0_z_logt_grid.size),                  # Intercept
            np.repeat(ang0_z_grid, len(t_grid)),            # ang0_z
            np.tile(log_time_clip, len(ang0_z_grid)),       # log_time_clipped
            ang0_z_logt_grid.ravel()                        # ang0_z_logt
        ])
        
        # Ensure columns match the model
        X_df = pd.DataFrame(X_mat, columns=coefs_matrix.columns)
        X_mat = X_df.values
        
        # Bootstrapped coefficient matrix
        B_mat = coefs_matrix.values.T  # n_parameters x n_boot
        
        # Predicted probability: shape (n_ang0 * n_time, n_boot)
        linpred = X_mat @ B_mat
        p_boot = 1 / (1 + np.exp(-linpred))
        
        # Flatten into long-form for Seaborn
        n_ang0 = len(ang0_grid)
        n_time = len(t_grid)
        for i_ang in range(n_ang0):
            for i_time in range(n_time):
                for b in range(n_boot):
                    rows.append({
                        'ztime': ztime,
                        'cond0': cond,
                        'ang0': ang0_grid[i_ang],
                        'time': t_grid[i_time],
                        'prob': p_boot[i_ang*n_time + i_time, b]
                    })

pred_df = pd.DataFrame(rows)

#%%
# Example: relplot with mean ± SD across bootstraps
g = sns.relplot(
    data=pred_df,
    x='time',
    y='prob',
    hue='ang0',
    col='cond0',
    row='ztime',
    kind='line',
    errorbar='sd',
    height=3,
    aspect=1.2,
    facet_kws={'sharex': True, 'sharey': 'row'}
)
g.set(xlim=(0, MAX_T))
g.set_axis_labels("Time into IBI (s)", "Predicted event probability")
g.set_titles("cond0 = {col_name}, ztime = {row_name}")
plt.tight_layout()
plt.show()

#%%
for ztime, df_z in pred_df.groupby('ztime'):
    for cond, df_c in df_z.groupby('cond0'):
        heatmap_data = df_c.groupby(['ang0', 'time'])['prob'].mean().unstack()  # ang0 x time

        plt.figure(figsize=(10,5))
        sns.heatmap(
            heatmap_data,
            cmap='viridis',
            cbar_kws={'label': 'Predicted swim probability'},
            xticklabels=10,
            yticklabels=10
        )
        plt.xlabel('Time into IBI (s)')
        plt.ylabel('Initial angle (ang0)')
        plt.title(f"Predicted swim probability — cond0={cond}, ztime={ztime}")
        plt.tight_layout()
        plt.show()
