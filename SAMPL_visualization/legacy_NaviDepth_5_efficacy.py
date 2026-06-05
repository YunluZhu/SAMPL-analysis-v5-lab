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
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.plt_functions import plt_categorical_combined
import scipy.stats as st
from scipy.stats import norm
from statsmodels.stats.weightstats import ztest
from plot_functions.plt_stats import *
import random

set_font_type()
# %%
pick_data = 'wt_dl' # all or specific data
which_zeitgeber = 'all'
BOOT_REP = 20

fig_folder = get_figure_dir(os.path.basename(__file__))
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, fig_folder)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass



all_feature_cond = pd.DataFrame()


root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_zeitgeber)

all_feature_cond = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)

# %% 
list_of_features = ['traj_peak', 'pitch_peak', 'spd_peak',
                    'pitch_end', 'pitch_initial',
                    'ydispl_swim', 'rot_full_decel',
                    # 'y_pre_swim','y_post_swim',
                    'post_IBI_time',
                    'y_initial','y_end',
                    'x_initial','x_end',
                    'xdispl_swim'
                    ]

df_input = all_feature_cond.groupby(['epoch_uid']).filter(lambda g: len(g)>1)

sel_features = df_input.copy()

# %%
max_lag = 5
consecutive_bout_features = pd.DataFrame()

for feature_toplt in list_of_features:
    shifted_df = pd.DataFrame()
    autoCorr_res = pd.DataFrame()
    df_tocorr = pd.DataFrame()

    grouped = sel_features.groupby(['cond0','cond1', 'ztime'])
    col_toplt = feature_toplt
    for (cond0, cond1, ztime), group in grouped:
        shift_df = pd.concat([group[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
        this_df_tocorr = shift_df.groupby(group['epoch_uid'], group_keys=False).apply(
            lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
        )
        this_df_tocorr = this_df_tocorr.assign(
            cond0=cond0,
            cond1=cond1, 
            ztime=ztime, 
            exp_uid=group['exp_uid']
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
        consecutive_bout_features = consecutive_bout_features.merge(long_df, on=['id', 'lag','cond0','cond1','ztime','exp_uid'], how='left')

        
#%%
cat_col = ['id','lag',	'cond1', 'cond0', 'ztime']
which_plot_x = 'traj_peak'
name_of_x = 'first_dir'
# which_sum_y = 'ydispl_swim'
sel_consecutive_bouts = consecutive_bout_features.copy()
sel_consecutive_bouts = sel_consecutive_bouts[cat_col +	['ydispl_swim' ,which_plot_x,'exp_uid']].dropna()
df_cumu = sel_consecutive_bouts.groupby(['cond1','cond0','ztime','id','exp_uid'])['ydispl_swim'].sum().reset_index()
first_dir = sel_consecutive_bouts.groupby(['cond1','cond0','ztime','id','exp_uid']).head(1)
df_cumu = df_cumu.merge(first_dir[['id',which_plot_x]], on='id')
df_cumu.rename(mapper={which_plot_x:name_of_x}, axis=1, inplace=True)
df_cumu['dir_bins'] = pd.cut(df_cumu[name_of_x], bins=[-90,-10,20,90], labels=['dive','flat','climb'])

#%% bootstrap

res = pd.DataFrame()
for (cond0, cond1, ztime), group in df_cumu.groupby(['cond0','cond1', 'ztime']):
    bst_input = group.id.unique()
    bst_output = [np.array(random.choices(bst_input, k=len(bst_input))) for i in np.arange(BOOT_REP)]
    for bts, ind in enumerate(bst_output):
        this_df_to_corr = group.loc[group.id.isin(ind)]
        this_x = this_df_to_corr[name_of_x].values
        this_y = this_df_to_corr['ydispl_swim'].values
        this_slope, this_intercept, this_r, this_p, this_se = st.linregress(this_x, this_y)
        this_res = pd.DataFrame(
            data={
                'slope':this_slope,
                'intercept':this_intercept,
                'r':this_r,
                'bts_rep':bts,
                'cond0':cond0,
                'cond1':cond1,
                'ztime':ztime,
                'dir_bins':dir,
            }, index=[0])
        res = pd.concat([res, this_res], ignore_index=True)

#%
#%%
g = plt_categorical_combined(
    data=res,
    x='cond1',
    y='slope',
    row='cond0',
    # col='dir_bins',
    units='bts_rep',
    height=3,
    col='ztime',
    aspect=0.7,
    sharey=True,
    related=False,
    errorbar=('sd'),
    overlay_func=sns.stripplot,
    alpha=0.1
)
plt.savefig(os.path.join(fig_dir, f"Depth efficacy boot.pdf"),format='PDF')


# %%
