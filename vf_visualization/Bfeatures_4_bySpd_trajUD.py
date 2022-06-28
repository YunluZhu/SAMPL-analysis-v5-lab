'''
Do not rely on Bfeatures_4 plots, 2D histogram plots are superior

plot mean binned bout features vs. speed & segmented by up dn traj

zeitgeber time? No
jackknifed? Yes, but not calculated properly
'''

#%%
# import sys
import os,glob
# import time
import pandas as pd # pandas library
import numpy as np # numpy
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

mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'tau_long'
root, FRAME_RATE = get_data_dir(pick_data)
spd_bins = [5,10,15,20,25]
posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'B4_bySpd_trajUD_features'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# mean_data_cond = mean_data_cond.reset_index().sort_values(by='condition').reset_index(drop=True)


all_feature_cond = all_feature_cond.assign(
    direction = pd.cut(all_feature_cond['traj_peak'],[-80,0,80],labels=['dive','climb']),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

# %% Jackknife resampling
cat_cols = ['condition','expNum','direction','speed_bins','dpf']
mean_data = all_feature_cond.groupby(cat_cols).mean().reset_index()
mean_data_jackknife = mean_data.groupby(['condition','direction','speed_bins','dpf']).apply(
    lambda x: jackknife_mean(x)
 ).drop(columns=['dpf']).reset_index()

# calculate the excluded expNum for each jackknifed result
max_exp = mean_data.expNum.max()
mean_data_jackknife['expNum'] = ((max_exp * (max_exp+1))/2 - max_exp * mean_data_jackknife['expNum']).astype(int)
try:
    mean_data_jackknife.drop(columns=['level_3'],inplace=True)
except:
    pass
try:
    mean_data_jackknife.drop(columns=['level_4'],inplace=True)
except:
    pass
mean_data_jackknife.rename(columns={c:c+'_jack' for c in mean_data_jackknife.columns if c not in cat_cols},inplace=True)
# %% mean
# data to plot

# %% Compare Sibs & Tau
toplt = mean_data_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt['expNum'].max())
set_font_type()
defaultPlotting()
ci_dic = {1:None,2:"sd",3:"sd"}

print('Point plot categorized by speed and pitch direction')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x ='condition', y = feature_toplt,
                    row="direction",col='speed_bins', 
                    height=4, aspect=0.8, kind='point',
                    hue='dpf', 
                    hue_order=all_cond1,
                    markers='d',
                    ci=ci_dic[len(all_cond1)],
                    zorder=10,
                    sharey = 'row',
                    )
    if len(all_cond1) == 1: # if only one cond1, plot indivial repeats
        g.map_dataframe(sns.pointplot, 
                        x = "condition", y = feature_toplt,
                        hue='expNum', ci=None,palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')

# %% 
toplt = mean_data_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

print("Plot with long format. as a function of speed. col = time duration")

defaultPlotting()

for feature_toplt in tqdm(['pitch','traj','spd','rot','tsp']):
    wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

    df_toplt = long_data.reset_index()
    g = sns.FacetGrid(df_toplt,
                      row = "direction", 
                      col='feature',
                      hue = 'condition', 
                      height=3, aspect=1.8, 
                      sharey='row',
                      )
    g.map_dataframe(sns.lineplot, 
                    x = 'speed_bins', y = feature_toplt,
                    style='dpf',
                    err_style='band', 
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'speed_bins', y = feature_toplt, 
                    hue ='dpf',
                    ci=None, join=False,
                    # markers=['o','x'],
                    )
    g.add_legend()
    plt.savefig(fig_dir+f"/{pick_data}'s _spd_{feature_toplt}.pdf",format='PDF')
    # plt.clf()

plt.close('all')
 # %%
# %% Plot as a function of speed
# print('As a function of speed')

# toplt = mean_data_jackknife

# defaultPlotting()

# for feature_toplt in tqdm(all_features):
#     g = sns.FacetGrid(toplt,
#                       row = "direction", 
#                       hue = 'condition', 
#                       height=3, aspect=1.8, 
#                       sharey=False,
#                       )
#     g.map_dataframe(sns.lineplot, 
#                     x = 'speed_bins', y = feature_toplt,
#                     err_style='band', 
#                     )
#     g.map_dataframe(sns.pointplot, 
#                     x = 'speed_bins', y = feature_toplt, 
#                     ci=None, join=False,
#                     markers='d')
#     g.add_legend()
#     plt.savefig(fig_dir+f"/{pick_data}'s _spd_{feature_toplt}.pdf",format='PDF')
#     plt.clf()

# plt.close('all')
