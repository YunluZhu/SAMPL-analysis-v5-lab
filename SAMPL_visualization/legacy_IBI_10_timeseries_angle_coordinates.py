

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
pick_data = 'meclizine' # name of your dataset to plot as defined in function get_data_dir()
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
# let's calculate angle of the line connecting head x y and x y

# Calculate the angle between (x, y) and (headx, heady)
def calculate_head_body_angle_vectorized(df):
    """Calculates the angle (in degrees) between the body center and head using vectorized operations."""
    dx = np.abs(df['absHeadx'].values - df['absx'].values)
    dy = df['absHeady'].values * -1 - df['absy'].values * -1
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Adjust angles to be within -90 to 90 range using vectorized operations
    angle_deg = np.where(angle_deg > 90, angle_deg - 180, angle_deg)
    angle_deg = np.where(angle_deg <= -90, angle_deg + 180, angle_deg)
    
    return angle_deg

def calculate_distance_between_points_vectorized(df):
    """
    Calculates the Euclidean distance between two sets of points using vectorized operations.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinate columns.
        x1_col (str): Name of the column containing the x-coordinates of the first point.
        y1_col (str): Name of the column containing the y-coordinates of the first point.
        x2_col (str): Name of the column containing the x-coordinates of the second point.
        y2_col (str): Name of the column containing the y-coordinates of the second point.

    Returns:
        np.ndarray: Array of distances between the two sets of points.
    """
    x1_col, y1_col, x2_col, y2_col = 'absHeadx', 'absHeady', 'absx', 'absy'
    dx = df[x2_col].values - df[x1_col].values
    dy = df[y2_col].values - df[y1_col].values
    distance = np.sqrt(dx**2 + dy**2)
    return distance
# Apply the vectorized function
df_time_filtered["head_body_angle"] = calculate_head_body_angle_vectorized(df_time_filtered)
df_time_filtered["head_body_dist"] = calculate_distance_between_points_vectorized(df_time_filtered)

#%%
df_time_filtered['ang_difference'] = df_time_filtered["ang"] - df_time_filtered["head_body_angle"]
df_time_filtered['length_ratio'] = (df_time_filtered["head_body_dist"] / 60) / df_time_filtered["fishLen"]
# average per IBI
df_angDiff_avgEpoch = df_time_filtered.groupby(group_cols)[['ang_difference','length_ratio','ang']].mean()
df_angDiff_avgEpoch = df_angDiff_avgEpoch.reset_index()

# df_angDiff_avgEpoch['box_uid'] = df_angDiff_avgEpoch['expNum'].astype(str).str.cat(df_angDiff_avgEpoch['boxNum'].astype(str), sep='_')

# #%%
# sns.displot(
#     kind='hist',
#     stat='probability',
#     common_norm=False,
#     data = df_angDiff_avgEpoch,
#     x='ang_difference',
#     col='cond0',
#     # row='cond',
#     hue='cond1',
#     height=3,
    
#     facet_kws={'xlim':np.percentile(df_angDiff_avgEpoch['ang_difference'],[0.5,99.5])},
# )
#%%
#%%

df_toplt = df_time_filtered
sns.displot(
    kind='hist',
    element="poly",
    stat='probability',
    common_norm=False,
    data = df_toplt,
    x='ang_difference',
    col='cond0',
    # row='cond',
    hue='cond1',
    hue_order=np.sort(df_toplt.cond1.unique()),
    height=3,
    # log_scale=True,
    facet_kws={'xlim':np.percentile(df_toplt['ang_difference'],[0.01,99.99])}
)
plt.savefig(fig_dir+f"/ecdf on avg per ibi first half.pdf",format='PDF')
#%%

df_toplt = df_angDiff_avgEpoch
sns.displot(
    kind='ecdf',
    # stat='probability',
    # common_norm=False,
    data = df_toplt,
    x='ang_difference',
    col='cond0',
    # row='cond',
    hue='cond1',
    hue_order=np.sort(df_toplt.cond1.unique()),
    height=3,
    # log_scale=True,
    facet_kws={'xlim':[np.percentile(df_toplt['ang_difference'],0.1), df_toplt['ang_difference'].median()],
               'ylim':[0,.5]},
)
plt.savefig(fig_dir+f"/ecdf on avg per ibi first half.pdf",format='PDF')


#%%
BIN_WIDTH = 0.1   #(s)
AVERAGE_BIN = np.arange(0, 10, BIN_WIDTH)

df_time_filtered = (
    df_time_filtered
    .assign(time_bin=pd.cut(df_time_filtered["time_relative_s"], bins=AVERAGE_BIN, labels=AVERAGE_BIN[:-1]))
)
df_tiemBinned = (
    df_time_filtered
    .groupby(group_cols + ["time_bin", 'ztime'], observed=True, as_index=False)[["time_relative_s"] + list(set(['yvel','ang_difference', 'ang', 'head_body_angle']))]
    .mean() 
)
all_first_yvel = df_tiemBinned.groupby(group_cols)['yvel'].apply(lambda y: y.iloc[0:5].mean())
yvel_bins = np.percentile(all_first_yvel, [0,33,66,100])
# Assign yvel_cond category based on the first value in each group
df_tiemBinned["yvel_cond"] = pd.cut(
    df_tiemBinned.groupby(group_cols)['yvel'].transform(lambda y: y.iloc[0:5].mean()),
    bins=yvel_bins,
    labels=['dive','flat', 'climb']
)

#% average by exp repeat

binned_df_averaged = df_tiemBinned.groupby(['cond0', 'cond1', 'expNum','time_bin','yvel_cond'], observed=True)[['yvel','ang_difference', 'ang', 'head_body_angle']].mean().reset_index()

ypar = 'ang_difference'

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

# %%
# check relationship between posture and ang difference
sns.relplot(
    kind='scatter',
    data=df_time_filtered.groupby('cond1').sample(n=5000),
    x='angAccel',
    y='ang_difference',
    hue='cond0',
    col='cond1',
    alpha=0.01,
    facet_kws={
        'xlim': (-20000, 20000),
        'ylim': (-2, 5),
    }
)
# %%
sns.catplot(
    kind='point',
    data=df_time_filtered.groupby(['cond0','cond1','expNum','boxNum'])['ang_difference'].mean().reset_index(),
    x='expNum',
    y='ang_difference',
    col='cond0',
    hue='cond1',
    estimator=np.mean,
    dodge=True,
)
# %%
