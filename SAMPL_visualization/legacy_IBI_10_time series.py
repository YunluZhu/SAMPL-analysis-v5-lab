

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
# from SAMPL_visualization.plot_functions.plt_tools import distribution_binned_average_opt

##### Parameters to change #####
pick_data = 'hc' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' or 'night', does not support 'all'

##### Parameters to change #####

#%%
# Paste root directory here
root, FRAME_RATE= get_data_dir(pick_data)


folder_name = f'IBI_time_series_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, os.path.basename(__file__).split('_')[0], folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

#%%
IBI_angles, cond0_all, cond1_all= get_timeseriesIBIangles(root, FRAME_RATE, ztime=which_ztime)

#%%
# Convert absTime to datetime and add unique IBI index
IBI_angles["absTime"] = pd.to_datetime(IBI_angles["absTime"])

IBI_angles["unique_IBI_idx"] = IBI_angles["expNum"].astype(str) + "_" + IBI_angles["epochNum"].astype(str) + "_" + IBI_angles["IEI_matchIndex"].astype(str)

# Filter dataset for specified conditions
df_time_filtered = IBI_angles#.query("cond1 in ['dd', 'ld'] and ztime == 'day'").copy()

#%%
# Calculate relative time in seconds and relative y

group_cols = ["unique_IBI_idx", "cond1", "cond0", "expNum"]
df_time_filtered["time_relative_s"] = df_time_filtered.groupby(group_cols, observed=True)["absTime"].transform(lambda x: (x - x.min()).dt.total_seconds())

#%%

# Define constants
BIN_WIDTH = 0.05   #(s)
AVERAGE_BIN = np.arange(0, 10, BIN_WIDTH)

# Filter data
# if time_threshold != 0:
#     df_filtered = df_time_filtered.loc[df_time_filtered["time_relative_s"] > time_threshold].copy()
# else:
#     df_filtered = df_time_filtered.copy()

df_filtered = df_time_filtered
#%
# Create bins and group data
binned_df_cond = (
    df_filtered
    .assign(time_bin=pd.cut(df_filtered["time_relative_s"], bins=AVERAGE_BIN, labels=AVERAGE_BIN[:-1]))
    .groupby(group_cols + ["time_bin", 'ztime'], observed=True, as_index=False)[["time_relative_s"] + list(set(['yvel','angVelSmoothed']))]
    .mean()  # Change to .median() if needed
)

all_first_yvel = binned_df_cond.groupby(group_cols)['yvel'].apply(lambda y: y.iloc[0:5].mean())
yvel_bins = np.percentile(all_first_yvel, [0,33,66,100])
# Assign yvel_cond category based on the first value in each group
binned_df_cond["yvel_cond"] = pd.cut(
    binned_df_cond.groupby(group_cols)['yvel'].transform(lambda y: y.iloc[0:5].mean()),
    bins=yvel_bins,
    labels=['dive','flat', 'climb']
)

#% average by exp repeat

binned_df_averaged = binned_df_cond.groupby(['cond0', 'cond1', 'expNum','time_bin','yvel_cond'], observed=True)[['yvel','angVelSmoothed']].mean().reset_index()

#%%
ypar = 'yvel'

gridcol = 'cond1'
gridrow = 'yvel_cond'

g = sns.relplot(
    kind='line',
    data=binned_df_averaged,
    x='time_bin',
    y=ypar,
    hue='cond0',
    col=gridcol,
    # errorbar='sd',
    row=gridrow,
    aspect=2,
    height=3,
    facet_kws={'sharey': True},
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.02, 3))
filename = os.path.join(fig_dir,f"IBI time series by exp rep {gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')

col_order = binned_df_cond[gridcol].unique()
col_order.sort()
g = sns.relplot(
    kind='line',
    data=binned_df_cond,
    x='time_bin',
    y=ypar,
    hue='cond0',
    col_order=col_order,
    col=gridcol,
    # errorbar='sd',
    row=gridrow,
    aspect=2,
    height=3,
    facet_kws={'sharey': True},
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.02, 3))
filename = os.path.join(fig_dir,f"IBI time series all bouts {gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')


#%%

gridcol = 'cond0'
gridrow = 'yvel_cond'

g = sns.relplot(
    kind='line',
    data=binned_df_averaged,
    x='time_bin',
    y=ypar,
    hue='cond1',
    col=gridcol,
    # errorbar='sd',
    row=gridrow,
    aspect=2,
    height=3,
    facet_kws={'sharey': True},
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.02, 3))
filename = os.path.join(fig_dir,f"IBI time series by exp rep {gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')


#%%
gridcol = 'cond0'
gridrow = 'yvel_cond'

g = sns.relplot(
    kind='line',
    data=binned_df_cond,
    x='time_bin',
    y='angVelSmoothed',
    hue='cond1',
    col=gridcol,
    row=gridrow,
    aspect=2,
    height=3,
    facet_kws={'sharey': True},
)
for ax in g.axes.flat:
    # draw horizontal line at y = 0 
    ax.axhline(y=0, color='gray', linestyle='--') 
g.set(xlim=(-0.02, 3))
filename = os.path.join(fig_dir,f"IBI time series angVelSmoothed all bouts {gridcol}X{gridrow}.pdf")
plt.savefig(filename,format='PDF')


#%%
# correlation between angvel and yvel?

g = sns.displot(
    kind='hist',
    data=binned_df_cond.query("time_bin > 2"),
    x='angVelSmoothed',
    y='yvel',
    col='cond0',
    row='cond1',
    facet_kws={'xlim':np.percentile(binned_df_cond['angVelSmoothed'],[1,99]),
        'ylim':np.percentile(binned_df_cond['yvel'],[1,99])},
)

#%%
ANG_BIN = np.arange(-20, 25, 4)
binned_byPosture = (
    df_filtered
    .query("swimSpeed > 0.3")
    .assign(posture_bin=pd.cut(df_filtered["ang"], bins=ANG_BIN, labels=ANG_BIN[:-1]))
    .groupby(group_cols + ["posture_bin",'ztime'], observed=True, as_index=False)[['xvel', 'yvel']]
    .mean()  # Change to .median() if needed
)

# calculate 0-360 movement direction using xvel and yvel

binned_byPosture["IBI_insTraj"] = np.degrees(np.arctan2(binned_byPosture["yvel"], np.abs(binned_byPosture["xvel"])))#%
#%%
g = sns.relplot(
    kind='line',
    data=binned_byPosture,
    x='posture_bin',
    y='IBI_insTraj',
    hue='cond1',
    row='cond0',
    aspect=2,
    height=3,
    facet_kws={'sharey': False},
)
plt.show()

# %%
