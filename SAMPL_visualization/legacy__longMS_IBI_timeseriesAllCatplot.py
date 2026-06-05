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

after_1s_mask = df_valid["time_relative_s"] > 1

df_valid["yvel_sg"] = savgol_filter(df_valid["yvel"].values, 11, 3)
df_valid["angvel_sg"] = savgol_filter(df_valid["angVelSmoothed"].values, 11, 3)
df_valid["speed_sg"] = savgol_filter(df_valid["swimSpeed"].values, 11, 3)

after = df_valid.loc[after_1s_mask]

#%
stats = after.groupby("unique_IBI_idx", observed=True).agg(
    p50_yvel=("yvel_sg", lambda x: np.percentile(x, 50)),
    p75_angVelSmoothed=("angvel_sg", lambda x: np.percentile(x, 75)),
    p75_swimSpeed=("speed_sg", lambda x: np.percentile(x, 75)),
    mad_swimSpeed=("speed_sg", lambda x: np.median(np.abs(x - np.median(x)))),
    mad_yvel=("yvel_sg", lambda x: np.median(np.abs(x - np.median(x)))),
    mad_angVelSmoothed=("angvel_sg", lambda x: np.median(np.abs(x - np.median(x))))
)

#%
valid_IBI_idx = stats.index[
    (stats.p50_yvel < 0.01) &
    (stats.p75_angVelSmoothed < 0) &
    (stats.p75_swimSpeed < 0.2) &
    (stats.mad_swimSpeed < 0.04) &
    (stats.mad_angVelSmoothed < 1)
]  
# valid_IBI_idx = df_valid.unique_IBI_idx.unique()  # keep all for now
#%%
before_08s_mask = df_valid["time_relative_s"] < 0.8
before = df_valid.loc[before_08s_mask]
stats2 = before.groupby("unique_IBI_idx", observed=True).agg(
    ang_accel=("angVelSmoothed", lambda x: x.diff().median()),
)
valid_IBI_idx2 = stats2.index[stats2.ang_accel > -0.03]

df_filtered = df_valid[df_valid.unique_IBI_idx.isin(np.intersect1d(valid_IBI_idx2, valid_IBI_idx))].copy()

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
    IBI=('unique_IBI_idx', 'count')
).reset_index()

features_toplt =  ['IBI_ydispl','IBI_rot','IBI_yvel_avg','IBI_angvel_avg']
# median per expNum
print(IBI_features.groupby(['cond0','expNum']).size())
median_res = IBI_features.groupby(['cond0','expNum'])[features_toplt].median().reset_index()

#%%

for param in features_toplt:
    plt_categorical_combined_3(
        data=median_res,
        x='cond0',
        y=param,
        hue='cond0',
        palette=my_palette,
        units='expNum',
        errorbar='se',
    )
    plt.savefig(os.path.join(fig_dir, f'{param} .pdf'), format='pdf')
    x_name = 'cond0'
    df_var = median_res[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    

# %%
