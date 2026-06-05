'''
Plot bout features at different peak speeds (use BIN_NUM to specify bin numbers) as a function of time (index range specified by idxRANGE)
with posture neg and posture pos bouts separated
Change all_features for the features to plot
'''

#%%
# import sys
import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
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
from plot_functions.plt_tools import distribution_binned_average_opt

##### Parameters to change #####
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' or 'night', does not support 'all'

##### Parameters to change #####

# %%
# Paste root directory here
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'IBIT1'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')

#%%
IBI_angles, cond0_all, cond1_all= get_timeseriesIBIangles(root, FRAME_RATE, ztime=which_ztime)

# %%
# Convert absTime to datetime and add unique IBI index
IBI_angles["absTime"] = pd.to_datetime(IBI_angles["absTime"])
IBI_angles["unique_IBI_idx"] = IBI_angles["expNum"].astype(str) + "_" + IBI_angles["epochNum"].astype(str) + "_" + IBI_angles["IEI_matchIndex"].astype(str)

# Filter dataset for specified conditions
df_filtered = IBI_angles.query("cond1 in ['dd', 'ld'] and ztime == 'day'").copy()

#%%
# Calculate relative time in seconds and relative y

group_cols = ["unique_IBI_idx", "cond1", "cond0"]
df_filtered["time_relative_s"] = df_filtered.groupby(group_cols, observed=True)["absTime"].transform(lambda x: (x - x.min()).dt.total_seconds())

#%%
parameter = 'ang'

BIN_WIDTH = 0.2
AVERAGE_BIN = np.arange(0,4,BIN_WIDTH)

# Create global bins for all data
df_filtered["time_bin"] = pd.cut(df_filtered["time_relative_s"], bins=AVERAGE_BIN, labels=AVERAGE_BIN[:-1])

# Perform grouping and aggregation at once
binned_df_cond = (
    df_filtered.groupby(group_cols + ["time_bin"], observed=True)[["time_relative_s", parameter]]
    .mean()  # Change to .median() if needed
    .reset_index()
)
#%%

binned_df_cond = binned_df_cond.assign(
    yvel_cond = pd.cut(binned_df_cond.groupby(group_cols)[parameter].transform('first'),bins=[-np.inf, 0, np.inf], labels=['dive','climb'])
)

#%%
g = sns.relplot(
    kind='line',
    data=binned_df_cond,
    x='time_bin',
    y=parameter,
    hue='cond1',
    # errorbar='sd',
    row='yvel_cond',
    aspect=2,
    height=3,
)
plt.hlines(0, 0, 4, color='red', linestyle='--')

# %%
