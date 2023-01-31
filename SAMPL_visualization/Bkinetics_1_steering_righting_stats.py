'''
Statistics on righting and steering gain
Righting and Steering by speed

ONLY Jackknifed
'''

#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from plot_functions.plt_tools import jackknife_list
from plot_functions.plt_functions import plt_categorical_grid


##### Parameters to change #####

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
SAMPLE_NUM = 0 # Bouts drew from each experimental repeat (int.) 0 for no resampling

##### Parameters to change #####
# %%
folder_name = f'BK1_steering_righting'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
spd_bins = np.arange(5,25,4)

root, FRAME_RATE = get_data_dir(pick_data)

all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond1 = get_bout_kinetics(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_ztime)

all_feature_cond = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)

set_font_type()
defaultPlotting()
# %%

####################################
###### Plotting Starts Here ######
####################################

# kinetics by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['jackknife_group','cond1','expNum','dataset','ztime']
all_features = ['steering_gain','righting_gain']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        col = 'cond0',
        hue = 'cond1',
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        err_style='bars',
        errorbar=('ci', 95),
        height=3,
    )
    g.set_xlabels("Peak speed (mm/s)", clear_inner=False)
    g.set(xlim=(4, 20))
    filename = os.path.join(fig_dir,f"{feature_toplt}_bySpd.pdf")
    plt.savefig(filename,format='PDF')
    
df_toplt = kinetics_jackknife
for feature_toplt in ['righting_gain_jack','steering_gain_jack']:
    multi_comp = MultiComparison(df_toplt[feature_toplt], df_toplt['cond0']+"|"+df_toplt['cond1'])
    print(f'* {feature_toplt}')
    print(multi_comp.tukeyhsd().summary())
    # print(multi_comp.tukeyhsd().pvalues)
# %% Compare by condition
toplt = kinetics_jackknife.reset_index(drop=True)
cat_cols = ['jackknife_group','cond1','expNum','dataset','ztime']
all_features = ['steering_gain_jack','righting_gain_jack']

for feature_toplt in (all_features):
    g = plt_categorical_grid(
        data = toplt,
        x_name = 'cond1',
        y_name = feature_toplt,
        gridcol = 'cond0',
        gridrow = 'ztime',
        units = 'jackknife_group',
        aspect=.8,
        height=3,
        )
    filename = os.path.join(fig_dir,f"{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')

    