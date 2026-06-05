# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average, get_2sd)
from plot_functions.plt_functions import plt_categorical_combined
import scipy.stats as st
from scipy.stats import (norm, median_test,ttest_rel)
from statsmodels.stats.weightstats import ztest
from plot_functions.plt_stats import *
import random
from functools import partial


def extract_bout_series(df, col_toplt, col_togroup, max_lag):
    long_form_shifted = pd.DataFrame()
    shift_df = pd.concat([df[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
    df_to_corr = shift_df.groupby(df[col_togroup], group_keys=False).apply(
        lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
    )
    return long_form_shifted, df_to_corr

# %%
pick_data = 'nMLF' # all or specific data
which_zeitgeber = 'all'
BOOT_REP = 100

fig_folder = get_figure_dir(os.path.basename(__file__))
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, fig_folder)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass

set_font_type()

spd_bins = np.arange(5,25,4)


root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_connected_bouts(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond['dataset'] = pick_data

all_feature_cond = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) 
)

#%%

# % dir change diff abs avg
max_lag = 4
feature_toplt = 'traj_peak'
col = 'expNum'
bts_df_shifted = pd.DataFrame()
bts_corr_res = pd.DataFrame()

for (cond0, condition, ztime), group in all_feature_cond.groupby(['cond0', 'cond1','ztime'], group_keys=False):
    _, this_df_tocorr = extract_bout_series(group, feature_toplt, 'epoch_uid', max_lag)
    bts_df_shifted = pd.concat([bts_df_shifted,
                                this_df_tocorr.assign(
                                    cond0 = cond0,
                                    cond1 = condition,
                                    ztime = ztime,
                                    exp_uid = group['exp_uid'].values
                                )],ignore_index=True)

sel_bouts = bts_df_shifted.loc[bts_df_shifted[f'{feature_toplt}_{max_lag}'].notna()]

sel_bouts = sel_bouts.assign(
    bout_series_uid = sel_bouts.index
)
long_df = pd.wide_to_long(sel_bouts, stubnames=feature_toplt, sep='_', j='lag', i=['bout_series_uid','exp_uid']).reset_index()
long_df = long_df.loc[long_df['lag']<=max_lag]
long_df = long_df.sort_values(by=['cond0','cond1','ztime','bout_series_uid','lag',]).reset_index(drop=True)
long_df['first_bout_val'] = long_df.groupby(['cond0', 'cond1','bout_series_uid'])[feature_toplt].transform('first')
long_df['dir_bins'] = pd.cut(long_df['first_bout_val'], bins=[-90,0, 20,90], labels=['dive','flat','climb'])

#%%
feature_toplt = 'traj_peak'

df_avg_chg = long_df.groupby(['cond0', 'cond1','ztime','dir_bins','exp_uid'] + ['bout_series_uid'])[feature_toplt].apply(
    lambda v: v.diff().abs().sum()/max_lag
    ).reset_index()
df_avg_chg = df_avg_chg.dropna()
    
#%% plot and stats on all bouts
feature_toplt = 'traj_peak'

plt_median_ci = partial(boot_ci, stat='median', boot_rep=BOOT_REP, confidence_level=0.95)

df_toplt = df_avg_chg

g = plt_categorical_combined(
    data=df_toplt,
    x='cond1',
    y=feature_toplt,
    row='cond0',
    col='ztime',
    units='bout_series_uid',
    sharey=True,
    related=False,
    errorbar=plt_median_ci,
    overlay_func=None,
    alpha=0.05,
    estimator=np.nanmedian
)

plt.savefig(fig_dir+f"/avgChg_consecBouts {feature_toplt}.pdf",format='PDF')
print(f"*{max_lag+1} bouts")
print(feature_toplt)

# for cond0 in df_toplt.cond0.unique():
#     to_compare = df_toplt.loc[df_toplt['cond0']==cond0]
#     print(cond0)
#     ctrl_df = to_compare.loc[to_compare['cond1'] == to_compare.cond1.unique()[0], feature_toplt]
#     print("ctrl")
#     print_values(ctrl_df, if_normal=False, feature_name=cond0+feature_toplt+to_compare.cond1.unique()[0])
#     cond_df = to_compare.loc[to_compare['cond1'] == to_compare.cond1.unique()[1], feature_toplt]
#     print("cond")
#     print_values(cond_df, if_normal=False, feature_name=cond0+feature_toplt+to_compare.cond1.unique()[1])
#     # se = np.sqrt(np.std(ctrl_df)**2 / len(ctrl_df) + np.std(cond_df)**2 / len(cond_df))
#     # mean_diff = np.mean(cond_df) -  np.mean(ctrl_df)
#     # zscore = mean_diff/se
#     # pval = norm.sf(np.abs(zscore))*2
#     # print(f'Manual z test: {pval}') 
#     # print(ztest(cond_df, ctrl_df, value=0))
#     # print(pval)
#     print(median_test(ctrl_df, cond_df))
#     # print(cohensd_alt(ctrl_df, cond_df))



#%%


#%% plot and statistics on dir cat
feature_toplt = 'traj_peak'

df_toplt = df_avg_chg
g = plt_categorical_combined(
    data=df_toplt,
    x='cond1',
    y=feature_toplt,
    row='ztime',
    units='bout_series_uid',
    sharey=False,
    related=False,
    errorbar=plt_median_ci,
    overlay_func=None,
    col ='dir_bins',
    alpha=0.05,
    col_order=['dive','flat','climb'],
    estimator=np.nanmedian
)

plt.savefig(fig_dir+f"/avgDiff_consecBouts Boot byDir{feature_toplt}.pdf",format='PDF')

print(feature_toplt)