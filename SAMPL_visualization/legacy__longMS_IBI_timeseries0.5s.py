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
df_time_filtered["ibi_duration"] = df_time_filtered.groupby(
    group_cols, observed=True
)["time_relative_s"].transform("max")

# # compute IBI duration and filter
min_bin = AVERAGE_BIN[-1]  # upper edge of last bin

# # Keep only rows from valid IBIs
df_valid = df_time_filtered[df_time_filtered["ibi_duration"] >= min_bin].copy()

# 1. Compute bin centers
bin_centers = (AVERAGE_BIN[:-1] + AVERAGE_BIN[1:]) / 2

# 2. Assign each point to a bin index
bin_idx = np.digitize(df_valid["time_relative_s"], AVERAGE_BIN) - 1

# 3. Keep only points that fall within your defined bins
mask = (bin_idx >= 0) & (bin_idx < len(bin_centers))

# 4. Assign the middle value of the bin
df_valid.loc[mask, "time_bin"] = bin_centers[bin_idx[mask]]

df_valid = df_valid.loc[mask].reset_index(drop=True)

print("> checkpoint after final filtering; df_valid ready")


#%% further cleaning: remove IBIs with high velocity after 1s, likely representing translocation
df_unaligned = df_valid.copy()
before_05s_mask = df_unaligned["time_relative_s"] < 0.5
before = df_unaligned.loc[before_05s_mask]
stats2 = before.groupby("unique_IBI_idx", observed=True).agg(
    ang_accel=("angVelSmoothed", lambda x: x.diff().median()),
)
valid_IBI_idx = stats2.index[stats2.ang_accel > -0.03]

df_unaligned_sel = df_unaligned[df_unaligned.unique_IBI_idx.isin(valid_IBI_idx)].copy()

print("> checkpoint after further filtering; df_filtered ready")

#%%
# take the first 0.5 s average as IBI_initial values


IBI_initial = df_unaligned_sel[df_unaligned_sel['time_relative_s'] <= 0.1].copy()

IBI_initial_features = IBI_initial.groupby(['unique_IBI_idx','cond0','ztime','expNum'], observed=True).agg(
    IBIinitial_angVel = ('angVelSmoothed', 'median'),
    IBIinitial_y_vel = ('yvel', 'median'),
    IBIinitial_pitch = ('ang', 'median'),
).reset_index()

IBI_initial_stats = IBI_initial_features.groupby(['cond0','ztime','expNum'])[['IBIinitial_angVel', 'IBIinitial_y_vel', 'IBIinitial_pitch']].median().reset_index()   

#%%
# visualize
for this_feature in ['IBIinitial_angVel', 'IBIinitial_y_vel', 'IBIinitial_pitch']:
    plt_categorical_combined_3(
        data=IBI_initial_stats,
        x='cond0',
        y=this_feature,
        col='ztime',
        hue='cond0',
        palette=my_palette,
        units='expNum',
        errorbar='se',
    )
    plt.savefig(os.path.join(fig_dir, f'{this_feature}_by_cond0.pdf'), format='pdf')
# %%
# check angvel using df_time_filtered, take a 100 ms window after IBI start
#%%

#%%
start_time_s = 0.15
end_time_s = 0.3

# Use the variables in your selection logic
IBI_initial_unaligned = df_unaligned_sel[
    (df_unaligned_sel['time_relative_s'] <= end_time_s) & 
    (df_unaligned_sel['time_relative_s'] > start_time_s)
].copy()
IBI_initial_unaligned_features = IBI_initial_unaligned.groupby(['unique_IBI_idx','cond0','ztime','expNum'], observed=True).agg(
    IBIinitial_angVel = ('angVelSmoothed', 'median'),
).reset_index()

#plot distribution
g = sns.displot(
    data=IBI_initial_unaligned_features,
    x='IBIinitial_angVel',
    hue='cond0',
    kind='kde',
    fill=True,
    hue_order=cond0_all,
    palette=my_palette,
    col='ztime',
    row='cond0',
    row_order=cond0_all,
    height=2,
    common_norm=False,
)
g.set(xlim=(-15, 10))
plt.savefig(os.path.join(fig_dir, f'IBIinitial_angVel 250-400 since swim stops_dist_by_cond0.pdf'), format='pdf')

#%%

TIME_STEPS_s = 0.1
start_s_range = np.arange(0.0, 0.5, TIME_STEPS_s) 

# --- 1. Data Processing Loop (Remains the Same) ---
all_features_list = []

for start_time_s in start_s_range:
    end_time_s = start_time_s + TIME_STEPS_s
    
    # Create a label for the hue legend (e.g., '0-100 ms')
    window_label = f'{int(start_time_s*1000)}-{int(end_time_s*1000)} ms'

    # Select the data for the current time window
    IBI_unaligned_window = df_unaligned_sel[
        (df_unaligned_sel['time_relative_s'] <= end_time_s) & 
        (df_unaligned_sel['time_relative_s'] > start_time_s)
    ].copy()
    
    # Feature aggregation
    IBI_unaligned_features = IBI_unaligned_window.groupby(
        ['unique_IBI_idx','cond0','ztime','expNum'], 
        observed=True
    ).agg(
        IBIinitial_angVel = ('angVelSmoothed', 'median'),
    ).reset_index()

    # Add the time window label column
    IBI_unaligned_features['time_window'] = window_label
    
    all_features_list.append(IBI_unaligned_features)

# Combine all time window data into a single DataFrame
plt_df = pd.concat(all_features_list, ignore_index=True)

# Define the order of the time window labels for the legend
window_order = [f'{int(s*1000)}-{int((s+TIME_STEPS_s)*1000)} ms' for s in start_s_range]

# Plot distribution:
# - hue='time_window': Overlays the different time windows within each facet panel.
# - row='cond0' and col='ztime' still create the panel grid.
g = sns.displot(
    data=plt_df,
    x='IBIinitial_angVel',
    hue='time_window', # <--- KEY CHANGE: Use time_window for overlaying lines
    kind='kde',
    fill=False,        # <--- KEY CHANGE: Use fill=False for cleaner overlay visualization
    hue_order=window_order,
    palette=sns.color_palette('viridis'), # Ensure this palette has enough distinct colors for all time windows
    col='ztime',       # Facet for columns
    row='cond0',       # Facet for rows
    row_order=cond0_all,
    height=2.5,          
    aspect=1,        
    common_norm=False,
    alpha=0.5
)

# Set common x-limits for all subplots
g.set(xlim=(-15, 10))

# Customize titles and save
plt.suptitle(
    f'Distribution of Initial Angular Velocity Overlaid Time Windows', 
    y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, f'IBIinitial_angVel_timewindows_overlaid_by_cond0.pdf'), format='pdf')
#%%
# --- Configuration ---
# Define the 100ms steps (e.g., from 0.0s up to 0.5s)
TIME_STEPS_s = 0.1
start_s_range = np.arange(0.0, 0.5, TIME_STEPS_s) 

# --- 1. Data Processing Loop ---
all_features_list = []

for start_time_s in start_s_range:
    end_time_s = start_time_s + TIME_STEPS_s
    
    # Create a label for the facet title (e.g., '0-100 ms')
    window_label = f'{int(start_time_s*1000)}-{int(end_time_s*1000)} ms'

    # Select the data for the current time window
    IBI_unaligned_window = df_unaligned_sel[
        (df_unaligned_sel['time_relative_s'] <= end_time_s) & 
        (df_unaligned_sel['time_relative_s'] > start_time_s)
    ].copy()
    
    # Feature aggregation
    IBI_unaligned_features = IBI_unaligned_window.groupby(
        ['unique_IBI_idx','cond0','ztime','expNum'], 
        observed=True
    ).agg(
        IBIinitial_angVel = ('angVelSmoothed', 'median'),
    ).reset_index()

    # Add the time window label column
    IBI_unaligned_features['time_window'] = window_label
    
    all_features_list.append(IBI_unaligned_features)

# Combine all time window data into a single DataFrame
all_features_df = pd.concat(all_features_list, ignore_index=True)

# --- 2. Plotting ---

# Define the order of the columns (time windows)
window_order = [f'{int(s*1000)}-{int((s+TIME_STEPS_s)*1000)} ms' for s in start_s_range]

# Plot distribution:
# - row='cond0': Puts the same condition on the same horizontal panel.
# - col='time_window': Separates the panels by the 100ms window.
g = sns.displot(
    data=all_features_df,
    x='IBIinitial_angVel',
    hue='cond0',
    kind='kde',
    fill=True,
    hue_order=cond0_all,
    palette=my_palette,
    row='cond0',       # Existing facet: condition for rows
    row_order=cond0_all,
    height=3,          # Adjusted height
    aspect=0.8,        # Adjusted aspect
    common_norm=False,
)

# Set common x-limits for all subplots
g.set(xlim=(-15, 10))

# Customize titles and save
g.set_titles("{col_name}") 
plt.tight_layout()

# %%

# %%
# plot average time series of IBI angvelSmoothed from 0 to 500ms 
BIN_WIDTH = 0.05
time_bins = np.arange(0, 0.5 + BIN_WIDTH, BIN_WIDTH)
df_time_filtered_truncated = df_unaligned_sel[df_unaligned_sel['time_relative_s'] <= 0.5].copy()
df_time_filtered_truncated['time_bin'] = pd.cut(df_time_filtered_truncated['time_relative_s'], bins=time_bins, right=False, labels=time_bins[:-1] + BIN_WIDTH/2)      
time_series_stats = df_time_filtered_truncated.groupby(['cond0','expNum','time_bin'], observed=True)['angVelSmoothed'].median().reset_index()
# %%
plt.figure(figsize=(4,3))
sns.lineplot(
    data=time_series_stats,
    x='time_bin',
    y='angVelSmoothed',
    hue='cond0',
    palette=my_palette,
    errorbar='se',
)
# %%
