#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd # pandas library
import numpy as np # numpy
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


set_font_type()
defaultPlotting()

# %%
pick_data = 'tau_bkg'
which_zeitgeber = 'day'
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
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond0 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
all_cond0 = pick_data
all_cond0.sort()


all_feature_cond = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)



sns.set_style("ticks")

# %%
# kinetics by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['jackknife_group','cond1','expNum','dataset','ztime']
# all_features = [c for c in toplt.columns if c not in cat_cols]
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
    
# %% Compare by condition
toplt = kinetics_jackknife.reset_index(drop=True)
cat_cols = ['jackknife_group','cond1','expNum','dataset','ztime']
# all_features = [c for c in toplt.columns if c not in cat_cols]
all_features = ['steering_gain_jack','righting_gain_jack']

for feature_toplt in (all_features):
    g = sns.catplot(
        data = toplt,
        col = 'cond0',
        hue = 'cond1',
        x = 'cond1',
        y = feature_toplt,
        linestyles = '',
        kind = 'point',
        # marker = True,
        aspect=.6,
        height=3,
    )
    g.map(sns.lineplot,'cond1',feature_toplt,estimator=None,
      units='jackknife_group',
      data = toplt,
      sort=False,
      color='grey',
      alpha=0.2,)
    g.add_legend()
    # if 'righting' in feature_toplt:
    #     g.set(ylim=(0.04,0.14)) 
    # else:
    #     g.set(ylim=(0.65,0.90)) 

    sns.despine(offset=10, trim=False)
    filename = os.path.join(fig_dir,f"{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')


df_toplt = kinetics_jackknife
for feature_toplt in ['righting_gain_jack','steering_gain_jack']:
    multi_comp = MultiComparison(df_toplt[feature_toplt], df_toplt['cond0']+"|"+df_toplt['cond1'])
    print(f'* {feature_toplt}')
    print(multi_comp.tukeyhsd().summary())
    # print(multi_comp.tukeyhsd().pvalues)
    