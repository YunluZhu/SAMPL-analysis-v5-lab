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
    'atk_ang',
    # 'lift_distance',
    'traj_peak',
    'pitch_peak',
    'x_chg_fullBout',
    
                    ]

# %% associate consecutive bouts

# %%
# consecutive bouts vs depth change in total

max_lag = 4
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
bout_series_df = (
    selected_data
    .groupby(["cond1", "cond0", "ztime", "expNum", "id"], as_index=False)
    .agg(
        avgIBI_y=("preIBI_y_displ", "mean"),
        avgIBITime=("pre_IBI_time", "mean"), 
        avgBout_y=("depth_chg_fullBout", lambda x: x.iloc[1:].mean()),
        avgTraj=("traj_peak", "mean"), 
        avgAtkAng=("atk_ang", "mean"), 
        avgWHM=("WHM", lambda x: x.iloc[0:].mean()),
        veering = ('traj_peak', lambda x: x.diff().abs().sum()/max_lag),
    )
)

bout_series_df = bout_series_df.assign(
    IBI_duration = pd.cut(bout_series_df['avgIBITime'], 
                          bins=[0,np.quantile(bout_series_df['avgIBITime'],.25),np.inf], 
                          labels=['short IBI','long IBI'])
    )

bout_series_df = bout_series_df.assign(
    WHM_duration = pd.cut(bout_series_df['avgWHM'], 
                          bins=[0,0.1,np.inf], 
                          labels=['short bout','long bout']),
    y_residual = bout_series_df['avgIBI_y'] + bout_series_df['avgBout_y'] ,
    frequency = 1/bout_series_df['avgIBITime']
    )

#%%

g = sns.relplot(
    kind='scatter',
    data=bout_series_df.groupby(['cond0']).sample(1000, replace=True),
    x='avgIBITime',
    y='veering',
    col='cond0',
    # row='WHM_duration',
    alpha=0.1,
    color='black',
    hue='avgTraj',
    palette=sns.color_palette("viridis", as_cmap=True)
)
#%%
plt.figure(figsize=(3,3))
g = sns.histplot(
    data=bout_series_df,
    x='avgIBI_y',
    hue='cond0',
    stat='probability',
    element='poly',
    common_norm=False,
    bins='scott',
    palette=my_palette,
    fill=False,
)
#%%
g = sns.displot(
    data=bout_series_df,
    x='avgIBI_y',
    y='frequency',
    col='cond0',
    common_norm=False,
)
g.set(xlim=(-1,1), ylim=(0,2))
# # %%
# xval='preIBI_y_displ'
# yval='atk_ang'

# g = sns.displot(
#     data=selected_data,
#     x=xval,
#     y=yval,
#     col='cond0',
#     common_norm=False,
# )
# g.set(xlim=np.percentile(selected_data.dropna()[xval],[1,99]), ylim=np.percentile(selected_data.dropna()[yval],[1,99]))


# #%%
# max_lag = 3
# #####################
# consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)

# #%
# sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond1','cond0','id','lag','ztime']).reset_index(drop=True)
# sel_consecutive_bouts = sel_consecutive_bouts.assign(
#     bouts = sel_consecutive_bouts['lag'] + 1
# )

# selected_data = (
#     sel_consecutive_bouts
#     .groupby(["cond1",  'cond0', "ztime", "expNum","id"])
#     .apply(lambda group: group.assign(
#         preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
#         postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        
#     ), include_groups=False)
#     .reset_index()  # Reset index after apply()
# )

# df_cumu = (
#     selected_data.groupby(["cond0", "id", "exp_conduid"])[[
#         "depth_chg_fullBout",
#         'x_chg_fullBout'
#     ]]
#     .sum()
#     .reset_index()
# )
# df_cumu['displ_fullBout'] = np.sqrt(
#     df_cumu['depth_chg_fullBout']**2 +
#     df_cumu['x_chg_fullBout']**2
# )

# df_cumu['IBI_ydispl'] = selected_data.groupby(["cond0", "id", "exp_conduid"])['preIBI_y_displ'].apply(lambda x: x[1:].sum()).values

# first_dir = selected_data.groupby(['cond0','id','exp_conduid']).head(1)
# df_cumu = df_cumu.merge(first_dir[['id','traj_peak']], on='id')
# df_cumu.rename(mapper={'traj_peak':'first_dir'}, axis=1, inplace=True)
# df_cumu['ydispl_ratio'] = df_cumu['depth_chg_fullBout']/df_cumu['displ_fullBout']
# df_cumu.dropna(inplace=True)

# #%%

# sns.relplot(
#     data=df_cumu.groupby('cond0').sample(1000, replace=True),
#     x='first_dir',
#     y='ydispl_ratio',
#     col='cond0',
#     alpha=0.1   
# )

# sns.relplot(
#     data=df_cumu.groupby('cond0').sample(1000, replace=True),
#     x='IBI_ydispl',
#     y='displ_fullBout',
#     col='cond0',
#     alpha=0.1   
# )

# #%%
# sns.lmplot(
#     data=df_cumu,
#     x='first_dir',
#     y='ydispl_ratio',
#     palette=my_palette,
#     hue='cond0',
#     scatter_kws={'alpha':0.01},
# )

# #%
# res_rep = pd.DataFrame()
# for (cond0, exp_conduid), group in df_cumu.groupby(["cond0",'exp_conduid'],observed=True):

#     this_df_to_corr = group
#     this_x = this_df_to_corr['first_dir'].values
#     this_y = this_df_to_corr["ydispl_ratio"].values
#     # this_slope, this_intercept, this_r, this_p, this_se = st.linregress(
#     #     this_x, this_y,
#     # )
#     this_slope, this_intercept, lo_slope, hi_slope = theilslopes(this_y, this_x)

#     this_res = pd.DataFrame(
#         data={
#             "slope": this_slope,
#             "intercept": this_intercept,
#             # "r": this_r,
#             "bts_rep": bts,
#             "cond0": cond0,
#             "exp_conduid": exp_conduid,
#         },
#         index=[0],
#     )
#     res_rep = pd.concat([res_rep, this_res], ignore_index=True)


# #% plot bootstrap regression results
# sns.pointplot(
#     data=res_rep,
#     x='cond0',
#     y='slope',
#     hue='cond0',
# )

#%% consecutive 
# feature_AutoCorrelation = 'depth_chg_fullBout' # select a feature here
feature_AutoCorrelation = 'traj_peak' # select a feature here
# feature_AutoCorrelation = 'x_chg_fullBout' # select a feature here

df_toplt = consecutive_bout_features.query("lag == 1").sort_values(by=['id']).reset_index(drop=True)
df_toplt.columns = [f'{feature_AutoCorrelation}_N' if x==f'{feature_AutoCorrelation}_first' else x for x in df_toplt.columns]
df_toplt.columns = [f'{feature_AutoCorrelation}_N+1' if x==f'{feature_AutoCorrelation}' else x for x in df_toplt.columns]

xmin = np.percentile(df_toplt[f'{feature_AutoCorrelation}_N'].values,0.1)
xmax = np.percentile(df_toplt[f'{feature_AutoCorrelation}_N'].values,99.9)

g = sns.FacetGrid(
    data=df_toplt, 
    # x=f'{feature_AutoCorrelation}_N', 
    # y=f'{feature_AutoCorrelation}_N+1', 
    col='cond1',
    row='cond0',
    # kind="scatter", 
    # alpha=0.05, 
    aspect = 1, 
    height = 3,
    # linewidths=0
    ylim = (xmin, xmax),
    xlim = (xmin, xmax),
    )

for (row_val, col_val), ax in g.axes_dict.items():

    this_cond_data = df_toplt.query("cond0 == @row_val & cond1 == @col_val")

    xval = this_cond_data[f'{feature_AutoCorrelation}_N'].values
    yval = this_cond_data[f'{feature_AutoCorrelation}_N+1'].values

    # --- Theil–Sen regression ---
    ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(yval, xval)

    # --- For plotting regression line ---
    X_plot = np.linspace(xmin, xmax, 1000)
    Y_plot = ts_slope * X_plot + ts_intercept

    # --- Robust correlation is optional; you kept Pearson r ---
    this_r = np.corrcoef(xval, yval)[0, 1]

    # sample down for plotting
    if len(this_cond_data) > 2000:
        this_cond_data = this_cond_data.sample(2000)

    sns.scatterplot(
        data=this_cond_data,
        x=f'{feature_AutoCorrelation}_N',
        y=f'{feature_AutoCorrelation}_N+1',
        alpha=0.05,
        linewidths=0,
        ax=ax
    )
    sns.lineplot(x=X_plot, y=Y_plot, color='r', ax=ax)
    ax.text(
        0.05, 0.9,
        f'slope = {ts_slope:.3f}',
        transform=ax.transAxes,
        fontsize=9
    )

    ax.text(
        0.05, 0.8,
        f'r = {this_r:.3f}',
        transform=ax.transAxes,
        fontsize=9
    )
plt.savefig(fig_dir+f"/{feature_AutoCorrelation} N+1 vs N scatter.pdf",format='PDF')

#%%
# combine 2 conditions
df_toplt2 = df_toplt.query("cond0 in ['04', '14']")

g = sns.displot(
    kind='kde',
    data=df_toplt2,#.groupby('cond0').sample(1200), 
    x=f'{feature_AutoCorrelation}_N', 
    y=f'{feature_AutoCorrelation}_N+1',  
    # linewidths=0,
    hue='cond0',
    height=2.5,
    aspect=1,
    thresh=0.1,
    levels=4,
    common_norm=False,
    fill=False,
    )
g.set(
    xlim=[xmin,xmax],
    ylim=[xmin,xmax],
)
plt.savefig(fig_dir+f"/{feature_AutoCorrelation} N+1 vs N kde.pdf",format='PDF')

# # # access the single axes
# # ax = g.axes[0, 0]

# # # get color mapping used by seaborn for consistency
# # # (same order as df_toplt2['cond0'].unique())
# # palette = sns.color_palette(n_colors=df_toplt2['cond0'].nunique())

# # for (cond, color) in zip(df_toplt2['cond0'].unique(), palette):

# #     df_sub = df_toplt2[df_toplt2['cond0'] == cond]

# #     x = df_sub[f'{feature_AutoCorrelation}_N'].values
# #     y = df_sub[f'{feature_AutoCorrelation}_N+1'].values

# #     # robust regression
# #     slope, intercept, lo, hi = theilslopes(y, x)

# #     # line for drawing
# #     Xp = np.linspace(xmin, xmax, 500)
# #     Yp = slope * Xp + intercept

# #     # draw line using hue color
# #     ax.plot(Xp, Yp, color=color, lw=2)

# #     # optional annotation
# #     ax.text(
# #         0.02, 0.95 - 0.07 * list(df_toplt2['cond0'].unique()).index(cond),
# #         f"{cond}: slope={slope:.3f}",
# #         transform=ax.transAxes,
# #         color=color,
# #         fontsize=9,
# #     )

# # %%  autocorrelation-----------------
# col = 'expNum'

# corr_list_ = []

# for (cond0, cond1,e), group in all_features.groupby(['cond0', 'cond1', 'expNum']):
#     this_corr_res, _, df_to_corr = cal_autocorrelation_feature(group, feature_AutoCorrelation, 'epoch_uid', max_lag)
#     this_corr_res = this_corr_res.assign(
#         cond1 = cond1,
#         cond0 = cond0,
#         expNum = e
#     )
#     corr_list_.append(this_corr_res)
    
# autoCorr_res = pd.concat(corr_list_, ignore_index=True)

# autoCorr_res = autoCorr_res.assign(
#     r_sq = autoCorr_res[f'autocorr_{feature_AutoCorrelation}'] ** 2
# )

# # %%  Autocorrelation

# g = sns.relplot(
#     data=autoCorr_res,
#     x='lag',
#     y='slope',
#     hue='cond0',
#     errorbar='se',
#     # col='cond0',
#     # row='cond1',
#     kind='line',
#     height=3
# )
# plt.savefig(fig_dir+f"/slope {feature_AutoCorrelation}.pdf",format='PDF')

# #%%
# g = sns.relplot(
#     data=autoCorr_res,
#     x='lag',
#     y='r_sq',
#     hue='cond0',
#     errorbar='se',
#     # col='cond0',
#     # row='cond1',
#     kind='line',
#     height=3,
#     palette=my_palette,
#     units='expNum',
#     estimator=None,
#     alpha=0.2,
# )
# #%
# g.map_dataframe(
#     sns.lineplot,
#     x='lag',
#     y='r_sq',
#     hue='cond0',
#     alpha=1,
#     legend=False,
#     palette=my_palette,
#     # errorbar=None,
#     err_style='bars',

# )
    
# plt.savefig(fig_dir+f"/R_square {feature_AutoCorrelation}.pdf",format='PDF')

# # run stats on r_sq at lag 1 -4
# for lag_i in range(1, max_lag+1):
#     print(f'-----lag {lag_i}-----')
#     df_lag = autoCorr_res.query("lag == @lag_i")
#     model = ols('r_sq ~ C(cond0)', data=df_lag).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(anova_table)
#     tukey = pairwise_tukeyhsd(endog=df_lag['r_sq'], groups=df_lag['cond0'], alpha=0.05)
#     print(tukey)

# #%%

# # %%  autocorrelation-------CLIMB only----------
# col = 'expNum'

# corr_list_ = []

# for (cond0, cond1,e), group in all_features.groupby(['cond0', 'cond1', 'expNum']):
#     _, _, df_to_corr = cal_autocorrelation_feature(group, feature_AutoCorrelation, 'epoch_uid', max_lag)
    
#     df_to_corr_sel = df_to_corr.loc[df_to_corr[f'{feature_AutoCorrelation}_0']>0, :]
    
#     corrres = []
#     pearsonRci = []
#     slope = []
#     intercept = []
#     slope_err = []
#     lag = []
#     n = []

#     for j in np.arange(1, max_lag+1):

#         this_df = df_to_corr_sel.iloc[:, [0, j]].dropna(axis='rows')

#         if len(this_df[this_df[f'{feature_AutoCorrelation}_{j}'].notna()]) >= 10:

#             x = this_df[f'{feature_AutoCorrelation}_0']
#             y = this_df[f'{feature_AutoCorrelation}_{j}']

#             # --- correlation ---
#             this_corr = st.pearsonr(x, y)
#             corrres.append(this_corr[0])
#             pearsonRci.append(this_corr.confidence_interval())

#             # --- Theil–Sen regression ---
#             ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(y, x)

#             slope.append(ts_slope)
#             intercept.append(ts_intercept)

#             # symmetric error from CI
#             slope_err.append((hi_slope - lo_slope) / 2)

#             # bookkeeping
#             lag.append(j)
#             n.append(len(this_df))
            
#     # --- output dataframe ---
#     output = pd.DataFrame({
#         'slope': slope,
#         'slope_err': slope_err,
#         'intercept': intercept,
#         f'autocorr_{feature_AutoCorrelation}': corrres,
#         'lag': lag,
#         'ci': [[np.abs(ci[0] - corr), ci[1] - corr] 
#             for corr, ci in zip(corrres, pearsonRci)],
#         'n': n,
#     })
    
#     this_corr_res = output.assign(
#         cond1 = cond1,
#         cond0 = cond0,
#         expNum = e
#     )
#     corr_list_.append(this_corr_res)
    
# autoCorr_res = pd.concat(corr_list_, ignore_index=True)

# autoCorr_res = autoCorr_res.assign(
#     r_sq = autoCorr_res[f'autocorr_{feature_AutoCorrelation}'] ** 2
# )


# g = sns.relplot(
#     data=autoCorr_res,
#     x='lag',
#     y='slope',
#     hue='cond0',
#     errorbar='se',
#     # col='cond0',
#     # row='cond1',
#     kind='line',
#     height=3
# )
# plt.savefig(fig_dir+f"/slope {feature_AutoCorrelation}.pdf",format='PDF')

# g = sns.relplot(
#     data=autoCorr_res,
#     x='lag',
#     y='r_sq',
#     hue='cond0',
#     errorbar='se',
#     # col='cond0',
#     # row='cond1',
#     kind='line',
#     height=3,
#     palette=my_palette,
#     units='expNum',
#     estimator=None,
#     alpha=0.2,
# )
# #%
# g.map_dataframe(
#     sns.lineplot,
#     x='lag',
#     y='r_sq',
#     hue='cond0',
#     alpha=1,
#     legend=False,
#     palette=my_palette,
#     # errorbar=None,
#     err_style='bars',

# )
    
# plt.savefig(fig_dir+f"/R_square {feature_AutoCorrelation}.pdf",format='PDF')

# # run stats on r_sq at lag 1 -4
# for lag_i in range(1, max_lag+1):
#     print(f'-----lag {lag_i}-----')
#     df_lag = autoCorr_res.query("lag == @lag_i")
#     model = ols('r_sq ~ C(cond0)', data=df_lag).fit()
#     anova_table = sm.stats.anova_lm(model, typ=2)
#     print(anova_table)
#     tukey = pairwise_tukeyhsd(endog=df_lag['r_sq'], groups=df_lag['cond0'], alpha=0.05)
#     print(tukey)

#%%
# auto correlation on all data with bootstrap sampling
feature_AutoCorrelation = 'traj_peak' # select a feature here

max_lag_corr = 5
col = 'expNum'

results = []

for (cond0, cond1), group in all_features.groupby(["cond0", "cond1"]):

    if ("epoch_conduid" not in group.columns):
        group = group.assign(
            epoch_conduid = group['cond0'] + group['cond1'] + group['expNum'].astype(str) + group['epoch_uid'],
            exp_conduid = group['cond0'] + group['cond1'] + group['expNum'].astype(str),
        )

    # -------- vectorized autocorrelation matrix construction --------
    feature_vals = group[feature_AutoCorrelation].values
    epoch_ids = group["epoch_conduid"].values

    shift_df = pd.concat([group[feature_AutoCorrelation].shift(-i).rename(f'{feature_AutoCorrelation}_{i}') for i in range(max_lag_corr+1)], axis=1)
    df_to_corr = shift_df.groupby(group["epoch_conduid"]).apply(
        lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag_corr, len(g))], np.zeros((len(g), max(1+max_lag_corr-len(g),0))).astype(bool)), axis=1))
    )

    # select climbs only
    df_to_corr_sel = df_to_corr.loc[df_to_corr[f'{feature_AutoCorrelation}_0']>0, :]
    
    # ---------------------------
    # Bootstrap iterations
    # ---------------------------
    N = len(df_to_corr_sel)
    col0 = f"{feature_AutoCorrelation}_0"

    for b in tqdm(range(100)):

        # fast bootstrap sample from df_to_corr_sel directly
        boot_idx = np.random.randint(0, N, N)
        boot_df = df_to_corr_sel.iloc[boot_idx]

        # keep only positive col0 rows
        df = boot_df.loc[boot_df[col0] > 0]

        if df.empty:
            continue

        x0 = df[col0].values
        for j in range(1, max_lag_corr+1):
            y = df[f"{feature_AutoCorrelation}_{j}"].values
            mask = ~np.isnan(x0) & ~np.isnan(y)

            if mask.sum() < 10:
                continue

            xv = x0[mask]
            yv = y[mask]

            # fast correlation
            r = np.corrcoef(xv, yv)[0, 1]
            ci = st.pearsonr(xv, yv).confidence_interval()

            # # --- Theil–Sen regression ---
            # # returns: slope, intercept, slope_low, slope_high
            # ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(yv, xv)

            # # symmetric error from CI
            # slope_err = (hi_slope - lo_slope) / 2

            # # optional intercept CI — Theil–Sen doesn't give intercept CI
            # # so we leave intercept_err = None
            # intercept_err = None

            results.append(
                {
                    "lag": j,
                    f"autocorr_{feature_AutoCorrelation}": r,
                    "ci": [abs(ci[0] - r), ci[1] - r],
                    "n": mask.sum(),
                    "bootstrap_iter": b,
                    "cond0": cond0,
                    "cond1": cond1,
                    'shuffle': False,

                    # "slope": ts_slope,
                    # "slope_err": slope_err,
                    # "intercept": ts_intercept,
                    # "intercept_err": intercept_err,
                }
            )
    
    # add shuffle control
    # -------- vectorized autocorrelation matrix construction --------
    shuffled = group.copy()
    shuffled[feature_AutoCorrelation] = np.random.permutation(shuffled[feature_AutoCorrelation].values)
    
    feature_vals = shuffled[feature_AutoCorrelation].values
    epoch_ids = shuffled["epoch_conduid"].values

    shift_df = pd.concat([shuffled[feature_AutoCorrelation].shift(-i).rename(f'{feature_AutoCorrelation}_{i}') for i in range(max_lag_corr+1)], axis=1)
    df_to_corr = shift_df.groupby(shuffled["epoch_conduid"]).apply(
        lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag_corr, len(g))], np.zeros((len(g), max(1+max_lag_corr-len(g),0))).astype(bool)), axis=1))
    )
    
    for b in tqdm(range(100)):

        # fast bootstrap sample from df_to_corr directly
        boot_idx = np.random.randint(0, N, N)
        boot_df = df_to_corr.iloc[boot_idx]

        # keep only positive col0 rows
        df = boot_df.loc[boot_df[col0] > 0]

        if df.empty:
            continue

        x0 = df[col0].values
        for j in range(1, max_lag_corr+1):
            y = df[f"{feature_AutoCorrelation}_{j}"].values
            mask = ~np.isnan(x0) & ~np.isnan(y)

            if mask.sum() < 10:
                continue

            xv = x0[mask]
            yv = y[mask]

            # fast correlation
            r = np.corrcoef(xv, yv)[0, 1]
            ci = st.pearsonr(xv, yv).confidence_interval()

            # # --- Theil–Sen regression ---
            # # returns: slope, intercept, slope_low, slope_high
            # ts_slope, ts_intercept, lo_slope, hi_slope = theilslopes(yv, xv)

            # # symmetric error from CI
            # slope_err = (hi_slope - lo_slope) / 2

            # # optional intercept CI — Theil–Sen doesn't give intercept CI
            # # so we leave intercept_err = None
            # intercept_err = None

            results.append(
                {
                    "lag": j,
                    f"autocorr_{feature_AutoCorrelation}": r,
                    "ci": [abs(ci[0] - r), ci[1] - r],
                    "n": mask.sum(),
                    "bootstrap_iter": b,
                    "cond0": cond0,
                    "cond1": cond1,
                    'shuffle': True,

                    # "slope": ts_slope,
                    # "slope_err": slope_err,
                    # "intercept": ts_intercept,
                    # "intercept_err": intercept_err,
                }
            )
    
    
autoCorr_res = pd.DataFrame(results)
autoCorr_res["r_sq"] = autoCorr_res[f"autocorr_{feature_AutoCorrelation}"] ** 2

#%%

# g = sns.relplot(
#     data=autoCorr_res,
#     x='lag',
#     y='slope',
#     hue='cond0',
#     errorbar='sd',
#     # col='cond0',
#     # row='cond1',
#     kind='line',
#     height=3
# )
# plt.savefig(fig_dir+f"/slope {feature_AutoCorrelation}.pdf",format='PDF')

g = sns.relplot(
    data=autoCorr_res,
    x='lag',
    y=f'autocorr_{feature_AutoCorrelation}',
    hue='cond0',
    errorbar='se',
    col='shuffle',
    # row='cond1',
    kind='line',
    height=3,
    palette=my_palette,
    units='bootstrap_iter',
    estimator=None,
    alpha=0.02,
)
#%
g.map_dataframe(
    sns.lineplot,
    x='lag',
    y= f'autocorr_{feature_AutoCorrelation}',
    hue='cond0',
    alpha=1,
    legend=False,
    palette=my_palette,
    errorbar='sd',
    err_style='bars',

)
    
plt.savefig(fig_dir+f"/r {feature_AutoCorrelation}.pdf",format='PDF')

# run stats on r_sq at lag 1 -4
for lag_i in range(1, max_lag_corr+1):
    print(f'-----lag {lag_i}-----')
    df_lag = autoCorr_res.query("lag == @lag_i and shuffle == False")
    model = ols(f'autocorr_{feature_AutoCorrelation} ~ C(cond0)', data=df_lag).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    tukey = pairwise_tukeyhsd(endog=df_lag[f'autocorr_{feature_AutoCorrelation}'], groups=df_lag['cond0'], alpha=0.05)
    print(tukey)


#%%
list_of_features = [
    'WHM',
    'pre_IBI_time',
    'post_IBI_time',
    'pitch_initial',
    'pitch_end',
    # 'rot_total',
    'y_initial',
    'y_end',
    # 'x_initial',
    # 'x_end',
    'depth_chg_fullBout',
    'atk_ang',
    # 'lift_distance',
    'traj_peak',
    'pitch_peak',
    
                    ]

max_lag = 2
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
        preIBI_drift=group["pitch_initial"]-group["pitch_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        postIBI_drift=group["pitch_initial"].shift(-1) - group["pitch_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        
    ), include_groups=False)
    .reset_index()  # Reset index after apply()
)


data = selected_data.query("lag==1")
col1 = 'preIBI_y_displ'
col2 = 'postIBI_y_displ'

# r, p = stats.pearsonr(data[col1], data[col2])
# print(f"Pearson r = {r:.3f},  p = {p:.3e}")
g = sns.relplot(
    data=data.groupby('cond0').sample(1000, replace=True),
    col='cond0',
    x=col1,
    y=col2,
    alpha=0.1,
    height=3
)
g.set(xlim=np.percentile(data.dropna()[col1],[1,99]), ylim=np.percentile(data.dropna()[col2],[1,99]))



col1 = 'preIBI_drift'
col2 = 'postIBI_drift'

# r, p = stats.pearsonr(data[col1], data[col2])
# print(f"Pearson r = {r:.3f},  p = {p:.3e}")
g = sns.relplot(
    data=data.groupby('cond0').sample(1000, replace=True),
    col='cond0',
    x=col1,
    y=col2,
    alpha=0.1,
    height=3
)
g.set(xlim=np.percentile(data.dropna()[col1],[1,99]), ylim=np.percentile(data.dropna()[col2],[1,99]))


#%%

