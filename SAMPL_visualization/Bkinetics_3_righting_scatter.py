'''
plot Righting fit raw data in scatter and binned average in line.

'''

#%%
# import sys
import os,glob
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_average, day_night_split)
from tqdm import tqdm
import matplotlib as mpl
from plot_functions.get_bout_features import get_bout_features


##### Parameters to change #####
pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
##### Parameters to change #####

# %%
# Paste root directory here
# if_plot_by_speed = True
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'BK3_righting_scatterPlt'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
except:
    pass   
set_font_type()
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.27*FRAME_RATE),peak_idx+round_half_up(0.22*FRAME_RATE)]

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime = 'day')
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# %%
feature = 'pitch_initial'
binned_rotation = 'rot_l_decel'

xname = 'Initial Pitch'
yname = 'Decel. Rotation'

toplt = all_feature_cond

upper = np.percentile(toplt[feature], 99)
lower = np.percentile(toplt[feature], 1)

BIN_WIDTH = 2
AVERAGE_BIN = np.arange(round_half_up(lower),round_half_up(upper),BIN_WIDTH)
binned_df_cond = toplt.groupby(['cond1','cond0']).apply(
    lambda g: distribution_binned_average(g,by_col=feature,bin_col=binned_rotation,bin=AVERAGE_BIN)
)
binned_df = distribution_binned_average(toplt,by_col=feature,bin_col=binned_rotation,bin=AVERAGE_BIN)
binned_df.columns=[xname,yname]
binned_df_cond.columns=[xname,yname]
binned_df_cond = binned_df_cond.reset_index()
# %%
 
####################################
###### Plotting Starts Here ######
####################################

g = sns.relplot(
    kind='scatter',
    data = toplt.sample(frac=0.2),
    x = feature,
    y = binned_rotation,
    # x_ci=95,
    alpha=0.1,
    linewidth = 0,
    color = 'grey',
    height=2.5,
    aspect=1,
    col='cond1',
    row='cond0'
)
g.set(xlim=(lower,upper))
g.set(ylim=(-10,15))

for i , g_row in enumerate(g.axes):
    for j, ax in enumerate(g_row):
        sns.lineplot(data=binned_df_cond.loc[(binned_df_cond['cond0']==all_cond0[i]) & 
                                        (binned_df_cond['cond1']==all_cond1[j])], 
                    x=xname, y=yname, 
                    alpha=1,
                    legend=False,
                    ax=ax)

# %%
