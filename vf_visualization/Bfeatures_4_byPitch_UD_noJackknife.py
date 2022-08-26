'''
Do not rely on Bfeatures_4 plots, 2D histogram plots are superior

plot mean binned bout features vs. pitch initial

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
from plot_functions.get_index import get_index
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm

import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
pick_data = 'wt_fin'
bin_by = 'pitch_initial'
# spd_bins = [3,7,13,18,25]
posture_bins = [-25,-10,-5,0,5,10,15,20,25,50]

# %% get data dir and frame rate
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'B4_byPitch_features_vs_{bin_by}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
# %%
# get data
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE)

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

# %% 
cat_cols = ['condition','posture_bins','dpf']

def add_average_pitch_initial(grp):
    grp['average_pitch_initial'] = grp['pitch_initial'].mean()
    return grp

all_feature_cond = all_feature_cond.groupby(cat_cols).apply(add_average_pitch_initial)
all_feature_cond.drop(columns=['bout_time'],inplace=True)
all_feature_cond = all_feature_cond.reset_index(drop=True)

# %% 
set_font_type()
cat_cols = ['condition','posture_bins','dpf','expNum','ztime']

# plot each feature vs {bin by} 
toplt = all_feature_cond
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt['expNum'].max())
set_font_type()
defaultPlotting()
ci_dic = {1:None,2:"sd",3:"sd"}

print('Point plot categorized by pitch')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data = toplt, x ='condition', y = feature_toplt,
                    # row="direction",
                    col='posture_bins', 
                    height=4, aspect=0.8, kind='point',
                    hue='dpf', 
                    hue_order=all_cond1,
                    markers='d',sharey='row',
                    ci=ci_dic[len(all_cond1)],
                    zorder=10
                    )
    if len(all_cond1) == 1: # if only one cond1, plot indivial repeats
        g.map_dataframe(sns.pointplot, 
                        x = "condition", y = feature_toplt,
                        hue='expNum', ci=None,palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')

# %% 
# Plot with long format. as a function of speed. col = time duration
cat_cols = ['condition','posture_bins','dpf']

toplt = all_feature_cond
all_features = [c for c in toplt.columns if c not in cat_cols]

print("Plot with long format. as a function of speed. col = time duration")

defaultPlotting()

for feature_toplt in tqdm([
    'pitch','traj','angvel','rot','tsp',
    'spd',
    'bout','atk'
    ]):
    wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

    df_toplt = long_data.reset_index()
    g = sns.FacetGrid(df_toplt,
                    #   row = "direction", 
                      col='feature',
                      hue = 'condition', 
                      height=3, aspect=1.8, 
                      sharey='row',
                      )
    g.map_dataframe(sns.lineplot, 
                    x = 'posture_bins', y = feature_toplt,
                    style='dpf',
                    err_style='band', 
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'posture_bins', y = feature_toplt, 
                    hue ='dpf',
                    ci=None, join=False,
                    )
    g.add_legend()
    plt.savefig(fig_dir+f"/{pick_data}'s _initialPitch_{feature_toplt}.pdf",format='PDF')
    # plt.clf()
# %%
