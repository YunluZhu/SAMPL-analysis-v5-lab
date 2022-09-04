'''
Plot jackknifed kinetics

righting gain
set point
steering gain
correlation of accel & decel rotation
angvel gain (new)

zeitgeber time? Yes
jackknife? Yes
resampled? No
'''

#%%
# import sys
import os,glob
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_index import get_index
from plot_functions.get_bout_kinetics import get_bout_kinetics
set_font_type()
defaultPlotting()
# %%
pick_data = 'tau_long' # all or specific data
# for day night split
which_zeitgeber = 'day' # day night all
SAMPLE_NUM = 1000
# %%
# def main(pick_data):
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)
# TSP_THRESHOLD = [-np.Inf,-50,50,np.Inf]
# spd_bins = np.arange(3,24,3)

folder_name = f'BK1_z{which_zeitgeber}_sample{SAMPLE_NUM}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber, sample=SAMPLE_NUM)
all_cond1.sort()
all_cond2.sort()
# adjust condition order

# %%
sns.set_style("ticks")

#plot ztime
if which_zeitgeber == 'all':
    cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']

    toplt = kinetics_jackknife
    all_features = [c for c in toplt.columns if c not in cat_cols]
    for feature_toplt in (all_features):
        sns.catplot(data = kinetics_jackknife,
                    x = 'dpf',
                    row = 'condition',
                    hue='ztime',
                    y = feature_toplt,
                    kind='point'
                    )
        filename = os.path.join(fig_dir,f"{feature_toplt}_z{which_zeitgeber}.pdf")
        plt.savefig(filename,format='PDF')

# %%
# by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        row = 'ztime',
        col = 'dpf',
        hue = 'condition',
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"{feature_toplt}_z{which_zeitgeber}_ztime_bySpd.pdf")
    plt.savefig(filename,format='PDF')

# for feature_toplt in (all_features):
#     g = sns.catplot(
#         data = toplt,
#         row = 'ztime',
#         col = 'dpf',
#         hue = 'condition',
#         x = 'speed_bins',
#         y = feature_toplt,
#         kind = 'point',
#         marker = True,
#     )
#     filename = os.path.join(fig_dir,f"{feature_toplt}_ztime.pdf")
#     plt.savefig(filename,format='PDF')

# %% Compare by condition
toplt = kinetics_jackknife
cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']
all_features = [c for c in toplt.columns if c not in cat_cols]
# print('plot jackknife data')

for feature_toplt in (all_features):
    g = sns.catplot(
        data = toplt,
        row = 'ztime',
        hue = 'dpf',
        x = 'condition',
        order=all_cond2,
        y = feature_toplt,
        kind = 'point',
        marker = True,
    )
    g.map(sns.lineplot,'condition',feature_toplt,estimator=None,
      units='jackknife_group',
      hue = 'dpf',
      data = toplt,
      sort=False,
      color='grey',
      alpha=0.2,)
    g.add_legend()

    sns.despine(offset=10, trim=True)
    filename = os.path.join(fig_dir,f"{feature_toplt}_z{which_zeitgeber}_byCondition.pdf")
    plt.savefig(filename,format='PDF')

# %% raw data. no jackknife
# cat_cols = ['expNum','condition','dpf']

# toplt = all_kinetic_cond
# all_features = [c for c in toplt.columns if c not in cat_cols]

# flatui = ["#D0D0D0"] * (toplt.groupby('condition').size().max())

# defaultPlotting()

# # print('plot raw data')

# for feature_toplt in (all_features):
#     g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
#                     order=all_cond2,
#                     height=4, aspect=0.8, kind='point',
#                     hue='dpf', markers='d',sharey=False,
#                     hue_order=all_cond1,
#                     # ci=False, 
#                     zorder=10
#                     )
#     g.map_dataframe(sns.pointplot, 
#                     x = "condition", y = feature_toplt,
#                     order=all_cond2,
#                     hue='expNum', ci=None,
#                     palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    
#     plt.savefig(fig_dir+f"/{pick_data}_{feature_toplt}.pdf",format='PDF')
#     # plt.clf()
# plt.close('all')
# # %% by speed bins
# toplt = kinetics_bySpd_jackknife
# cat_cols = ['speed_bins', 'condition','dpf']
# all_features = [c for c in toplt.columns if c not in cat_cols]

# # print("Plot with long format. as a function of speed. ")

# defaultPlotting()
# toplt = kinetics_bySpd_jackknife
# for feature_toplt in (['righting','set','steering','corr']):
#     wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
#     wide_data['id'] = wide_data.index
#     long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

#     df_toplt = long_data.reset_index()
#     g = sns.FacetGrid(df_toplt,
#                     row = "feature", 
#                     col = 'dpf',
#                     hue = 'condition', 
#                     height=3, aspect=1.8, 
#                     sharey='row',
#                     )
#     g.map_dataframe(sns.lineplot, 
#                     x = 'speed_bins', y = feature_toplt,
#                     err_style='band', 
#                     # ci='sd'
#                     )
#     g.map_dataframe(sns.pointplot, 
#                     x = 'speed_bins', y = feature_toplt, 
#                     ci=None, join=False,
#                     markers='d')
    
#     # if feature_toplt == 'righting':
#     #     g.set(ylim = (0.05,0.19))
#     g.add_legend()
#     plt.savefig(fig_dir+f"/{pick_data}__spd_{feature_toplt}.pdf",format='PDF')
#     # plt.clf()


# %%
