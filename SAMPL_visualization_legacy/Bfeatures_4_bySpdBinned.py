'''
plot mean binned bout features vs. speed bins

note: dive vs climb are separated by pitch initial at 10 deg

'''

#%%
# import sys
import os,glob
# import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
# from astropy.stats import jackknife_resampling
# from astropy.stats import jackknife_stats
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl
from plot_functions.plt_functions import plt_categorical_grid


##### Parameters to change #####
pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
##### Parameters to change #####

# %%
def add_average_peak_speed(grp):
    grp['average_spd_peak'] = grp['spd_peak'].mean()
    return grp

root, FRAME_RATE = get_data_dir(pick_data)
spd_bins = np.arange(5,25,4)

posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'BF4_bySpdBins'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)
# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# mean_data_cond = mean_data_cond.reset_index().sort_values(by='cond1').reset_index(drop=True)


all_feature_cond = all_feature_cond.assign(
    direction = pd.cut(all_feature_cond['pitch_initial'],[-80,10,80],labels=['dive','climb']),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1)),
    # speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(spd_bins)),
)


cat_cols = ['cond1','direction','speed_bins','cond0']
   
all_feature_cond = all_feature_cond.groupby(cat_cols).apply(add_average_peak_speed)
all_feature_cond.drop(columns=['bout_time'],inplace=True)
# %% 
####################################
###### Plotting Starts Here ######
####################################

feature_to_plt = [
    'pitch_initial',
    'pitch_pre_bout',
    'pitch_peak',
    'pitch_post_bout',
    'pitch_end',
    'traj_initial',
    'traj_peak',
    'spd_peak',
    'rot_l_accel',
    'rot_l_decel',
    'rot_to_max_angvel',
    'bout_displ',
    'atk_ang',
    'angvel_chg',
    'depth_chg',
    'additional_depth_chg',
]
x_name = 'cond1'
gridrow = 'cond0' # or 'direction'/'ztime'
gridcol = 'speed_bins'
units = 'expNum'
prename = ''
toplt = all_feature_cond.groupby([x_name, gridcol, gridrow, units]).mean().reset_index()
# feature_to_plt = [c for c in toplt.columns if c not in cat_cols]

for feature in feature_to_plt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey='row',
        height = 3,
        aspect = 1.2
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
