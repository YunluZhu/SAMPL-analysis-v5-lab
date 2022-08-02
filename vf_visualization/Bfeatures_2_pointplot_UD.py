'''
Do not rely on Bfeatures_4 plots, 2D histogram plots are superior

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
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm

import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'sfld_combined'
# segment_by = 'bout_traj'
segment_by = 'pitch_initial'
root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'B2_by{segment_by}_features_compare'
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
    direction = pd.cut(all_feature_cond[segment_by],[-80,0,80],labels=['dive','climb']),
)

# %% Jackknife resampling
cat_cols = ['condition','expNum','direction','dpf']
mean_data = all_feature_cond.groupby(cat_cols).mean().reset_index()
mean_data_jackknife = mean_data.groupby(['condition','direction','dpf']).apply(
    lambda x: jackknife_mean(x)
 )
try:
    mean_data_jackknife.drop(columns=['dpf'],inplace=True)
except:
    pass
mean_data_jackknife = mean_data_jackknife.reset_index()

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
# %% 
# data to plot

# %% 

toplt = mean_data_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

set_font_type()
defaultPlotting()
ci_dic = {1:None,2:"sd",3:"sd"}

print('Point plot categorized by speed and pitch direction')
for feature_toplt in tqdm(all_features):
    g = sns.catplot(data=toplt, 
                    y = feature_toplt,
                    x='condition',
                    col="dpf", row="direction",col_order=all_cond1,hue='condition',
                    sharey=False,
                    kind='point', 
                    # marker=['d','d'],
                    aspect=.8,
                )
    (g.map(sns.lineplot,'condition',feature_toplt,
          estimator=None,
          units='expNum',
          data = toplt,
        #   hue='expNum',
          color='grey',
          alpha=0.2,))
    sns.despine(offset=10, trim=False)
    plt.savefig(fig_dir+f"/{pick_data}'s {feature_toplt}.pdf",format='PDF')
    plt.clf()
plt.close('all')
# %%
