
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
pick_data = 'tau_long' # all or specific data
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
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid']
) 
    


#%%

# % dir change diff abs avg
max_lag = 5

feature_toplt = 'traj_peak'
col = 'expNum'
# jackknife_mean = pd.DataFrame()
bts_df_shifted = pd.DataFrame()
bts_corr_res = pd.DataFrame()

for (cond0, cond1, ztime, expNum), group in all_feature_cond.groupby(['cond0', 'cond1', 'ztime','expNum'], group_keys=False):
    _, this_df_tocorr = extract_bout_series(group, feature_toplt, 'epoch_uid', max_lag)
    bts_df_shifted = pd.concat([bts_df_shifted,
                                this_df_tocorr.assign(
                                    cond0 = cond0,
                                    cond1 = cond1,
                                    ztime = ztime,
                                    expNum = expNum,
                                )],ignore_index=True)

sel_bouts = bts_df_shifted.loc[bts_df_shifted[f'{feature_toplt}_{max_lag}'].notna()]

sel_bouts = sel_bouts.assign(
    bout_series_uid = sel_bouts.index
)
long_df = pd.wide_to_long(sel_bouts, stubnames=feature_toplt, sep='_', j='lag', i='bout_series_uid').reset_index()
long_df = long_df.loc[long_df['lag']<=max_lag]
long_df = long_df.sort_values(by=['cond0','cond1','ztime','expNum','bout_series_uid','lag',]).reset_index(drop=True)

    
#%% calculate slope for bootstrap groups

col_toplt = 'traj_peak'
bts_corr_res = pd.DataFrame()
for (cond0, cond1, ztime), df_to_corr in bts_df_shifted.groupby(['cond0','cond1','ztime']):
    bst_input = df_to_corr.index
    bst_output = [np.array(random.choices(bst_input, k=len(bst_input))) for i in np.arange(100)]
    for bts, ind in enumerate(bst_output):
        this_df_to_corr = df_to_corr.loc[ind]
        
        slope = []
        slope_err=[]
        corrres = []
        pearsonRci = []
        lag = []
        n = []
        intercept = []

        for j in np.arange(1,max_lag+1):
            this_df = this_df_to_corr.iloc[:,[0,j]].dropna(axis='rows')
            if len(this_df[this_df[f'{col_toplt}_{j}'].notna()]) >= 10:
                x = this_df[f'{col_toplt}_0']
                y = this_df[f'{col_toplt}_{j}']
                this_corr = st.pearsonr(x, y)
                this_slope, this_intercept, this_r, this_p, this_se = st.linregress(x, y)
                intercept.append(this_intercept)
                slope.append(this_slope)
                slope_err.append(this_se)
                corrres.append(this_corr[0])
                pearsonRci.append(this_corr.confidence_interval())
                lag.append(j)
                n.append(len(this_df[this_df[f'{col_toplt}_{j}'].notna()]))
                this_df.columns = ['ori', 'shifted']
        this_output = pd.DataFrame(data={
            'slope': slope,
            'slope_err': slope_err,
            'intercept': intercept,
            f'autocorr_{col_toplt}': corrres,
            'lag': lag,
            'ci': [[np.abs(ci[0] - corrres[i]), ci[1] - corrres[i]] for i, ci in enumerate(pearsonRci)],
            'n': n,
        }).assign(
            cond0 = cond0, cond1 = cond1, ztime=ztime, bts_rep = bts
        )
        bts_corr_res = pd.concat([bts_corr_res, this_output], ignore_index=True)


# %% Consecutive direction consistency

# plot auto corr ** obsolete - plot by expnum to estimate erros
df_toplt = bts_corr_res
fig, ax = plt.subplots(figsize=(4,4))
g = sns.relplot(
    kind='line',
    data=df_toplt,
    x='lag',
    y='slope',
    hue='cond1',
    col='cond0',
    row='ztime',
    errorbar='sd',
)
ax.legend()
plt.ylabel(f'Consistency {feature_toplt}')
plt.xlabel('Lag (bout)')
plt.savefig(fig_dir+f"/Consistency Boot {feature_toplt}.pdf",format='PDF')


# %%
df_toplt = bts_corr_res
fig, ax = plt.subplots(figsize=(4,4))
g = sns.relplot(
    kind='line',
    data=df_toplt,
    x='lag',
    y='slope',
    hue='ztime',
    col='cond0',
    row='cond1',
    errorbar='sd',
)
ax.legend()
plt.ylabel(f'Consistency {feature_toplt}')
plt.xlabel('Lag (bout)')
plt.savefig(fig_dir+f"/Consistency ztime {feature_toplt}.pdf",format='PDF')


#%% DS grant

col_toplt = 'traj_peak'
byExp_coorres = pd.DataFrame()
for (cond0, cond1, expNum), df_to_corr in bts_df_shifted.groupby(['cond0','cond1','expNum']):
    slope = []
    slope_err=[]
    corrres = []
    pearsonRci = []
    lag = []
    n = []
    intercept = []

    for j in np.arange(1,max_lag+1):
        this_df = df_to_corr.iloc[:,[0,j]].dropna(axis='rows')
        if len(this_df[this_df[f'{col_toplt}_{j}'].notna()]) >= 10:
            x = this_df[f'{col_toplt}_0']
            y = this_df[f'{col_toplt}_{j}']
            this_corr = st.pearsonr(x, y)
            this_slope, this_intercept, this_r, this_p, this_se = st.linregress(x, y)
            intercept.append(this_intercept)
            slope.append(this_slope)
            slope_err.append(this_se)
            corrres.append(this_corr[0])
            pearsonRci.append(this_corr.confidence_interval())
            lag.append(j)
            n.append(len(this_df[this_df[f'{col_toplt}_{j}'].notna()]))
            this_df.columns = ['ori', 'shifted']
    this_output = pd.DataFrame(data={
        'slope': slope,
        'slope_err': slope_err,
        'intercept': intercept,
        f'autocorr_{col_toplt}': corrres,
        'lag': lag,
        'ci': [[np.abs(ci[0] - corrres[i]), ci[1] - corrres[i]] for i, ci in enumerate(pearsonRci)],
        'n': n,
    }).assign(
        cond0 = cond0, cond1 = cond1, expNum=expNum
    )
    byExp_coorres = pd.concat([byExp_coorres, this_output], ignore_index=True)



df_touse = byExp_coorres.query("cond1 == '1sibs'")
# filter lag 1-3
df_touse = df_touse.query("lag <= 3")
# take average per bts_rep
df_toplot = df_touse.groupby(['cond0', 'expNum',])['slope'].mean().reset_index()
# %%
sns.relplot(
    kind='line',
    data=df_toplot,
    x='cond0',
    y='slope',
    # kind='point',
    # errorbar='sd',
    units='expNum',
    estimator=None,
    height=3,
    aspect=1.2,
)
plt.savefig(fig_dir+f"/Consistency cond0 by expNum.pdf",format='PDF')

# %%
