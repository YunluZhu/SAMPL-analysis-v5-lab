
# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average, get_2sd)
from plot_functions.get_bout_kinetics import get_bout_kinetics
import scipy.stats as st
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest
from plot_functions.plt_stats import *
import random



from plot_functions.plt_tools import jackknife_list


def extract_bout_series(df, col_toplt, col_togroup, max_lag):
    long_form_shifted = pd.DataFrame()
    shift_df = pd.concat([df[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
    df_to_corr = shift_df.groupby(df[col_togroup], group_keys=False).apply(
        lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
    )
    return long_form_shifted, df_to_corr

# %%
pick_data = 'hc' # all or specific data
which_zeitgeber = 'day'

fig_folder = get_figure_dir(os.path.basename(__file__))
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, fig_folder)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass


root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_zeitgeber)

all_feature_cond = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)
    
#%%
list_of_features = [
                    'traj_peak', 
                    'pitch_peak', 
                    'spd_peak',
                    'y_end',
                    # 'pitch_end', 'pitch_initial',
                    'ydispl_swim',
                    # 'y_pre_swim','y_post_swim',
                    # 'y_initial','y_end'
                    ]

df_input = all_feature_cond.groupby(['epoch_uid']).filter(lambda g: len(g)>1)

sel_features = df_input

# %%
max_lag = 5
consecutive_bout_features = pd.DataFrame()

for feature_toplt in list_of_features:
    shifted_df = pd.DataFrame()
    autoCorr_res = pd.DataFrame()
    df_tocorr = pd.DataFrame()

    grouped = sel_features.groupby(['cond0', 'cond1','ztime'],group_keys=False)
    col_toplt = feature_toplt
    for (cond0, cond1,ztime), group in grouped:
        shift_df = pd.concat([group[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
        this_df_tocorr = shift_df.groupby(group['epoch_uid'],group_keys=False).apply(
            lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
        )
        this_df_tocorr = this_df_tocorr.assign(
            cond1=cond1, 
            cond0=cond0,
            exp_uid=group['exp_uid'],
            ztime=ztime
            )
        df_tocorr = pd.concat([df_tocorr, this_df_tocorr], ignore_index=True)

    sel_bouts = df_tocorr
    sel_bouts = sel_bouts.loc[sel_bouts[f'{feature_toplt}_{max_lag}'].notna()]
    sel_bouts = sel_bouts.assign(
        first_bout = sel_bouts[f'{feature_toplt}_0'],
    )
    sel_bouts['id'] = sel_bouts.index
    long_df = pd.wide_to_long(sel_bouts, stubnames=feature_toplt, sep='_', j='lag', i='id').reset_index()
    long_df = long_df.loc[long_df['lag']<=max_lag]
    
    long_df = long_df.rename(columns={'first_bout': f'{feature_toplt}_first'})
    
    if consecutive_bout_features.empty:
        consecutive_bout_features = long_df
    else:
        consecutive_bout_features = consecutive_bout_features.merge(long_df, on=['id', 'lag','cond0','cond1','exp_uid','ztime'])


# %% Cumulative y displ. on Y axis after X bouts hue by bin of first bout traj


sel_consecutive_bouts = consecutive_bout_features.copy()
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    cumu_swim_ydispl = sel_consecutive_bouts.groupby(['cond1','id'],group_keys=False)['ydispl_swim'].apply(np.cumsum),
    traj_peak_bins = pd.cut(sel_consecutive_bouts['traj_peak_first'], bins=[-90,-0,20, 90], labels=['dive','flat','climb']),
    traj_peak_bins2 = pd.cut(sel_consecutive_bouts['traj_peak_first'], bins=[-90,-0, 90], labels=['dive','climb']),
    pitch_peak_bins = pd.cut(sel_consecutive_bouts['pitch_peak_first'], bins=[-90,-0,20, 90], labels=['dive','flat','climb']),
    bouts = sel_consecutive_bouts['lag'] + 1,
    y_end_relative = sel_consecutive_bouts.groupby(['cond0','cond1','id'])['y_end'].transform(lambda g: g - g.iloc[0]),
)
# %%
sns.relplot(
    data=sel_consecutive_bouts,
    y='y_end_relative',
    x='bouts',
    kind='line',
    col='cond0',
    row='traj_peak_bins2',
    hue='cond1',
    errorbar=('ci', 95),
    height=3
)
plt.savefig(os.path.join(fig_dir, f"bouts cumu yend.pdf"),format='PDF')

# %%
sel_consecutive_wide = sel_consecutive_bouts.pivot(
    index='id', columns='lag',values='y_end_relative'
)
sel_consecutive_wide = sel_consecutive_wide.merge(sel_consecutive_bouts.groupby('id').head(1)[['cond0','cond1','traj_peak_bins2','id','exp_uid','ztime']], on='id',how='inner')
# %%
sel_consecutive_wide = sel_consecutive_wide.assign(
    y_0to1 = sel_consecutive_wide[1] - sel_consecutive_wide[0],
    y_2to3 = sel_consecutive_wide[3] - sel_consecutive_wide[2],
    y_3to4 = sel_consecutive_wide[4] - sel_consecutive_wide[3],
    y_4to5 = sel_consecutive_wide[5] - sel_consecutive_wide[4]
)

y_0to1_bin_val = np.arange(-2,3,0.2)
sel_consecutive_wide = sel_consecutive_wide.assign(
    y_0to1_bins = pd.cut(sel_consecutive_wide['y_0to1'], bins=y_0to1_bin_val, labels=(y_0to1_bin_val[1:]+y_0to1_bin_val[:-1])/2),
    y_0to1_dir = pd.cut(sel_consecutive_wide['y_0to1'], bins=[-90,0,90], labels=['total_dive','total_climb'])
)
#%%
yfit = 'y_3to4'
sel_consecutive_wide_avg = sel_consecutive_wide.groupby(['cond0','cond1','y_0to1_bins','ztime'],observed=True)[[
    'y_0to1', yfit, 0,1,2,3,4,5
]].median().reset_index()
# %%
g = sns.relplot(
    kind='line',
    data=sel_consecutive_wide_avg,
    y=yfit,
    x='y_0to1',
    row='cond1',
    hue='ztime',
    height=3,
)
# g.set(ylim=[-0.5,0.6])
# g.set(xlim=[-1.5,1.5])

# %%
g = sns.displot(
    data=sel_consecutive_wide,
    x='y_4to5',
    hue='cond1',
    col='traj_peak_bins2',
    row='cond0',
    kind='kde',
    common_norm=False
)

#%% Filter

sel_consecutive_wide_f = sel_consecutive_wide.loc[(sel_consecutive_wide['y_0to1'] < np.percentile(sel_consecutive_wide.y_0to1,99)) &
                                                  (sel_consecutive_wide['y_0to1'] > np.percentile(sel_consecutive_wide.y_0to1,1)),:]


#%%
# #%%
g = sns.displot(
    kind='kde',
    data=sel_consecutive_wide_f,
    y=yfit,
    x='y_0to1',
    row='cond1',
    col='y_0to1_dir',
    common_norm=False,
    hue='ztime',
    height=3,
)

# g.set(xlim=[np.percentile(sel_consecutive_wide['y_0to1'],0.1), np.percentile(sel_consecutive_wide['y_0to1'],99.9)])
g.set(ylim=[np.percentile(sel_consecutive_wide[yfit],0.5), np.percentile(sel_consecutive_wide[yfit],99.5)])

# %%
g = sns.relplot(
    kind='scatter',
    data=sel_consecutive_wide_f,
    y='y_4to5',
    x='y_0to1',
    row='cond1',
    col='ztime',
    height=3,
    alpha=0.01,
)
g.set(ylim=[np.percentile(sel_consecutive_wide[yfit],0.5), np.percentile(sel_consecutive_wide[yfit],99.5)])

#%%
g = sns.lmplot(
    data=sel_consecutive_wide_f,
    y='y_3to4',
    x='y_0to1',
    row='cond1',
    hue='ztime',
    scatter=False,
    col='traj_peak_bins2',
    height=3,
)

# %%
