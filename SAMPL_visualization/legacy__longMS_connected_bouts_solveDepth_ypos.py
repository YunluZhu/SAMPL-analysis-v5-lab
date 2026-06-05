'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import random
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_connected_bouts
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scipy.stats as st
import statsmodels.api as sm
import statsmodels.robust.norms as norms
from sklearn.metrics import r2_score
from scipy.stats import theilslopes
from tqdm import tqdm

#%%

##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'night' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
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
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)

# %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# tidy bout uid
all_features = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)

# select dataset
all_features = all_features.query('cond1 == "ld"')
#%%
list_of_features = [
    'WHM',
    'pre_IBI_time',
    'post_IBI_time',
    # 'pitch_initial',
    # 'pitch_end',
    # 'rot_total',
    'y_initial',
    'y_end',
    # 'x_initial',
    # 'x_end',
    'depth_chg_fullBout',
    # 'atk_ang',
    # 'lift_distance',
    'traj_peak',
    'pitch_peak',
    # 'x_chg_fullBout',
    'bout_time',
    'spd_peak'
    
                    ]

# %% associate consecutive bouts

# %%
# consecutive bouts vs depth change in total

max_lag = 3
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)

#%%
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond1','cond0','id','lag','ztime']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    bouts = sel_consecutive_bouts['lag'] + 1
)

# Compare current y_initial with next bout's y_initial
sel_consecutive_bouts['bout_direction'] = sel_consecutive_bouts.apply(
    lambda row: 'climb' if row['y_initial'] < row['y_end'] else 'dive',
    axis=1
)

selected_data = (
    sel_consecutive_bouts
    .groupby(["cond1",  'cond0', "ztime", "expNum","id"])
    .apply(lambda group: group.assign(
        preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        
    ), include_groups=False)
    .reset_index()  # Reset index after apply()
)

#%%

selected_data = selected_data.assign(
    ypos_afterBout_diff = selected_data['y_end'] - selected_data['y_end_first'],
    time_lapse = (selected_data['bout_time'] - selected_data['bout_time_first']).dt.total_seconds(),
)

selected_data = selected_data.assign(
    ypos_cumu = selected_data.groupby(["cond1", "cond0", "ztime", "expNum", "id"], as_index=False)['ypos_afterBout_diff'].cumsum().values
)

# separate by traj_peak direction
selected_data = selected_data.assign(
    traj_cat = pd.cut(selected_data['traj_peak_first'], bins=[-90, 0, 90], labels=['dive', 'climb']),
    speed_cat = pd.cut(selected_data['spd_peak'], bins=[0, selected_data['spd_peak'].median(), np.inf], labels=['slow', 'fast']),
)

#%%
# calculate average by expNum
bout_series_avg = selected_data.groupby(["cond1", "cond0", "ztime", "expNum", "traj_cat", "lag"], as_index=True, observed=True).agg(
    # avg_ypos_afterBout_diff = ('ypos_afterBout_diff', 'mean'),
    avg_ypos_cumu = ('ypos_cumu', 'median'),
    # avg_traj_peak = ('traj_peak_first', 'mean'),
).reset_index()

#%%

climb_df = bout_series_avg.query('traj_cat == "climb" and lag > 0')

g = sns.relplot(
    kind='line',
    data=climb_df,
    x='lag',
    hue='cond0',
    y='avg_ypos_cumu',
    palette=my_palette,
    errorbar='se',
    estimator='mean',
    height=3,
    aspect=1
)
g.set(xticks=range(1,5))
plt.savefig(os.path.join(fig_dir, f'climb_boutSeries_avgYposCumu_byCond0.pdf'), bbox_inches='tight')
#%%
# anova stats
param = 'avg_ypos_cumu'
x_name = 'cond0'
print(f"\n--- ANOVA for {param} ---")
# 1. One-way ANOVA
for bouts_num in sorted(climb_df['lag'].unique()):
    print(f"\nBouts: {bouts_num}")
    bout_series_sub = climb_df.query('lag == @bouts_num')
    model = ols(f"{param} ~ C(cond0)", data=bout_series_sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=bout_series_sub[param],
        groups=bout_series_sub["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())

