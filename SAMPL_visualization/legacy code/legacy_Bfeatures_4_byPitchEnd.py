'''
Do not rely on Bfeatures_4 plots, 2D histogram plots are superior

plot mean binned bout features vs. pitch end

zeitgeber time? No
jackknifed? Yes, but not calculated properly

'''

#%%
# import sys
import os,glob
# import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
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
pick_data = 'tau_long'
bin_by = 'pitch_end'
spd_bins = [3,7,13,18,25]
posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

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
all_feature_cond, all_cond0, all_cond0 = get_bout_features(root, FRAME_RATE)

# %% tidy data

rolling_windows = pd.DataFrame(data=posture_bins).rolling(2, min_periods=1)
rolling_mean = rolling_windows.mean()

all_feature_cond = all_feature_cond.assign(
    posture_bins = pd.cut(all_feature_cond[bin_by],posture_bins,labels=rolling_mean[1:].astype(str).values.flatten()),

    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),

)
print("speed buckets:")
print('--mean')
print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('mean'))
print('--min')
print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('min'))
print('--max')
print(all_feature_cond.groupby('speed_bins')['spd_peak'].agg('max'))

# %% Jackknife resampling
cat_cols = ['cond1','expNum','posture_bins','cond0']
mean_data = all_feature_cond.groupby(cat_cols).mean().reset_index()
mean_data_jackknife = mean_data.groupby(['cond1','cond0','posture_bins']).apply(
    lambda x: jackknife_mean(x)
 )
mean_data_jackknife = mean_data_jackknife.drop(columns=['cond0']).reset_index()

# calculate the excluded expNum for each jackknifed result
max_exp = mean_data.expNum.max()
mean_data_jackknife['expNum'] = ((max_exp * (max_exp+1))/2 - max_exp * mean_data_jackknife['expNum']).astype(int)
try:
    mean_data_jackknife.drop(columns=['level_2'],inplace=True)
except:
    pass
try:
    mean_data_jackknife.drop(columns=['level_3'],inplace=True)
except:
    pass

mean_data_jackknife.rename(columns={c:c+'_jack' for c in mean_data_jackknife.columns if c not in cat_cols},inplace=True)
# %% some othre features
toplt = mean_data_jackknife
all_features = ['angvel_initial_phase_jack',
                'angvel_prep_phase_jack',
                'angvel_post_phase_jack',
                'angvel_chg_jack',
                'bout_traj_jack',
                'atk_ang_jack']

print("Plot with long format. as a function of pitch. col = time duration")

defaultPlotting()

for feature_toplt in all_features:
    g = sns.FacetGrid(toplt,
                      row = 'cond0', 
                    #   col='feature',
                      hue = 'cond1', 
                      height=5, aspect=.8, 
                      sharey=False,
                      )
    g.map_dataframe(sns.lineplot, 
                    x = 'posture_bins', y = feature_toplt,
                    err_style='band', 
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'posture_bins', y = feature_toplt, 
                    ci=None, join=False,
                    markers='d')
    g.add_legend()
    # plt.savefig(fig_dir+f"/{pick_data}_{feature_toplt}_vs_{bin_by}.pdf",format='PDF')
    # plt.clf()

# plt.close('all')
# %%
