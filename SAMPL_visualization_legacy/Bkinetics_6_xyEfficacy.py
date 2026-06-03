'''
Check speed distribution
plot y efficacy: slope for depth change vs posture at peak 
plot x posture correlation: corr of x change vs posture
plot y posture correlation: corr of y changeg vs posture
plot lift gain: slope for additional depth change vs depth change
'''

#%%
# import sys
import os,glob
from statistics import mean
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.plt_tools import jackknife_list
from plot_functions.plt_functions import plt_categorical_grid

##### Parameters to change #####
pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day'
DAY_RESAMPLE = 0
NIGHT_RESAMPLE = 0
# if_jackknife = True # only jackknife option for speed specific kinematics
##### Parameters to change #####

# %%
folder_name = f'BK6_xyEfficacy'
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
all_cond0.sort()

all_feature_cond = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)

set_font_type()
sns.set_style("ticks")
    
# %%

####################################
###### Plotting Starts Here ######
####################################

# check speed distribution
toplt = all_feature_cond

# check speed
feature_to_plt = 'spd_peak'
upper = np.percentile(toplt[feature_to_plt], 99.5)
lower = np.percentile(toplt[feature_to_plt], 0.5)
g = sns.FacetGrid(data=toplt,
            col='cond0', 
            hue='cond1',
            sharey =False,
            sharex =True,
            )
g.map(sns.histplot,feature_to_plt,bins = 10, 
                    element="poly",
                    #  kde=True, 
                    stat="probability",
                    pthresh=0.05,
                    fill=False,
                    binrange=(lower,upper),)

g.add_legend()
sns.despine()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')# %%
# %%
toplt = kinetics_bySpd_jackknife
all_features = ['y_efficacy','lift_gain']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        hue = 'cond1',
        col = 'cond0',
        errorbar=('ci', 95),
        err_style='bars',
        marker = True,
    )
    g.figure.set_size_inches(4,2)
    # g.set(xlim=(6, 24))
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    filename = os.path.join(fig_dir,f"depth per peak pitch_bySpeed.pdf")
    plt.savefig(filename,format='PDF')

# %%
# pitch has no correlation with x distance but correlated with y distance
toplt = kinetics_bySpd_jackknife
all_features = ['x_posture_corr','y_posture_corr']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        hue = 'cond1',
        col = 'cond0',
        errorbar=('ci', 95),
        err_style='bars',
        marker = True,
    )
    g.figure.set_size_inches(4,2)
    # g.set(xlim=(6, 24))
    g.set(ylabel=feature_toplt)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    filename = os.path.join(fig_dir,f"{feature_toplt} by spd.pdf")
    plt.savefig(filename,format='PDF')

# %%
