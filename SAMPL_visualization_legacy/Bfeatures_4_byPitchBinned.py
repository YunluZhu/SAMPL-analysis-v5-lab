'''
plot mean binned bout features vs. bins of pitch initial


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
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
from plot_functions.plt_functions import plt_categorical_grid
import matplotlib as mpl


# %%
##### Parameters to change #####

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
bin_by = 'pitch_initial'

##### Parameters to change #####

# %% get data dir and frame rate
def add_average_pitch_initial(grp):
    grp['average_pitch_initial'] = grp['pitch_initial'].mean()
    return grp

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF4_byPitchBins_{bin_by}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)
posture_bins = [-25,-10,-5,0,5,10,15,20,25,50]

# %%
# get data
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE)

# %% tidy data

rolling_windows = pd.DataFrame(data=posture_bins).rolling(2, min_periods=1)
rolling_mean = rolling_windows.mean()

all_feature_cond = all_feature_cond.assign(
    posture_bins = pd.cut(all_feature_cond[bin_by],posture_bins,labels=rolling_mean[1:].astype(str).values.flatten()),
    # speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)
# print("speed buckets:")
# print('--mean')
# print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('mean'))
# print('--min')
# print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('min'))
# print('--max')
# print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('max'))


cat_cols = ['cond1','posture_bins','cond0']

all_feature_cond = all_feature_cond.groupby(cat_cols).apply(add_average_pitch_initial)
all_feature_cond.drop(columns=['bout_time'],inplace=True)
all_feature_cond = all_feature_cond.reset_index(drop=True)

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
gridcol = 'posture_bins'
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
