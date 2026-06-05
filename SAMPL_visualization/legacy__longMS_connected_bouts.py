'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_connected_bouts
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.get_bout_consecutive_features import extract_consecutive_bout_features
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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
    'pitch_initial',
    'pitch_end',
    'rot_total',
    'y_initial',
    'y_end',
    'x_initial',
    'x_end',
                    ]

# %% associate consecutive bouts

#####################
max_lag = 1
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
    .groupby(["cond1",  "ztime", "expNum","id"], as_index=False)
    .apply(lambda group: group.assign(
        preIBI_y_displ=group["y_initial"]-group["y_end"].shift(1)  ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        preIBI_x_displ=np.abs(group["x_initial"]-group["x_end"].shift(1)) ,  # preIBI_y_displ = y end from last bout - y initial from current bout
        # postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
        preIBI_rot=group["pitch_initial"] - group["pitch_end"].shift(1),
        # postIBI_rot=group["pitch_initial"].shift(-1) - group["pitch_end"]
    ), include_groups=False)
    .reset_index(drop=True)  # Reset index after apply()
)

# %%
data_to_plot = selected_data.query('bouts == 2')    
data_to_plot = data_to_plot.query("pre_IBI_time > 0.5")  # only consider pre IBI within 20s
data_to_plot = data_to_plot.query("pre_IBI_time < 20")  # only consider pre IBI within 20s
# data_to_plot['angAccel'] = data_to_plot['preIBI_rot'] / data_to_plot['pre_IBI_time']
data_to_plot['expNum'] = data_to_plot['exp_conduid'].apply(lambda x: x[-1])
# data_to_plot = data_to_plot.query("angAccel > -10")

data_to_plot['bout_freq'] = 1 / data_to_plot['pre_IBI_time']
#%%
# calculate angle of displacement
data_to_plot['IBI_displAng'] = np.degrees(np.arctan2(data_to_plot['preIBI_y_displ'], data_to_plot['preIBI_x_displ']))
data_to_plot['IBI_displ'] = np.sqrt(data_to_plot['preIBI_y_displ']**2 + data_to_plot['preIBI_x_displ']**2)

#%% Polar plot is commented out

# x_val = 'IBI_displAng'
# y_val = 'IBI_displ'

# # bin_edges = np.percentile(data_to_plot[x_val],[1,99])
# # bins = np.linspace(bin_edges[0], bin_edges[1], 11)


# # generate dataframed binned by preIBI_rot and preIBI_y_displ
# binned_data_rot_ = []
# for cond, group in data_to_plot.groupby('cond0'):
#     bins = np.percentile(group[x_val], np.linspace(0.5,99.5,9))
#     bin_centers = (bins[:-1] + bins[1:]) / 2
#     for expNum, subrep_group in group.groupby('expNum'):
#         binned_data_rot = distribution_binned_average_opt(
#             df=subrep_group,
#             by_col=x_val,
#             bin_col = y_val,
#             bin = bins,
#             method='median'
#         )
#         binned_data_rot_.append(
#             binned_data_rot.assign(cond0=cond, 
#                                 expNum=expNum, 
#                                 bin_centers=bin_centers,
#                                 )
#             )
# binned_data_rot = pd.concat(binned_data_rot_, ignore_index=True)

# #%%
# # --- 1. Aggregate Data Across 'expNum' with Mean and SEM ---

# # -------------------- Aggregation -------------------- #
# # 1. Within-experiment mean for each bin
# binned_per_exp = (
#     binned_data_rot
#     .groupby(['cond0', 'expNum', 'bin_centers'], observed=True)
#     .agg(
#         IBI_displ_mean=('IBI_displ', 'mean')
#     )
#     .reset_index()
# )

# # 2. Across-experiment stats for each condition + bin
# average_binned_data = (
#     binned_per_exp
#     .groupby(['cond0', 'bin_centers'], observed=True)
#     .agg(
#         IBI_displ_mean=('IBI_displ_mean', 'mean'),
#         IBI_displ_std=('IBI_displ_mean', 'std'),
#         count=('IBI_displ_mean', 'count')
#     )
#     .reset_index()
# )

# # 3. SEM
# average_binned_data['IBI_displ_sem'] = (
#     average_binned_data['IBI_displ_std'] / np.sqrt(average_binned_data['count'])
# )
# # -------------------- Plotting -------------------- #

# conditions = average_binned_data['cond0'].unique()
# palette = my_palette

# plt.figure(figsize=(5, 5))
# ax = plt.subplot(111, projection='polar')

# for color, (this_cond, this_data) in zip(palette, average_binned_data.groupby('cond0')):

#     this_data = this_data.sort_values('bin_centers')

#     theta = np.radians(this_data['bin_centers'].values)
#     mean_r = this_data['IBI_displ_mean'].values
#     sem_r = this_data['IBI_displ_sem'].values

#     upper = mean_r + sem_r
#     lower = mean_r - sem_r

#     # Mean line
#     ax.plot(
#         theta,
#         mean_r,
#         marker='o',
#         linewidth=2,
#         markersize=0,
#         color=color,
#         label=this_cond
#     )

#     # SEM shading
#     ax.fill_between(
#         theta,
#         lower,
#         upper,
#         color=color,
#         alpha=0.2
#     )

# # Cosmetics
# ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
# plt.tight_layout()
# plt.show()

    
#     # Add a legend if desired
#     # ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5))    
# #%%

# g = sns.displot(kind='kde',data=data_to_plot.query("preIBI_y_displ < 0"),x='preIBI_y_displ', hue='cond0',
#                 # row='expNum',
#                 log_scale=False, common_norm=False,
#                 height=3)
# g.set(xlim=(-2, 1.8))
#%%

# histogram
g = sns.histplot(
    data=data_to_plot,#.query("preIBI_y_displ < 0"),
    y='preIBI_y_displ',
    hue='cond0',
    stat='probability',
    common_norm=False,
    element='poly',
    fill=False,
    bins='scott',
    alpha=0.7,
)
g.set(ylim=(-2, 2))


#%%
median_res = data_to_plot.groupby(['cond0','exp_conduid'])[['preIBI_rot','preIBI_y_displ','IBI_displAng']].median().reset_index()

param = 'preIBI_y_displ'
plt_categorical_combined_3(
    data=median_res,
    x='cond0',
    y=param,
    hue='cond0',
    palette=my_palette,
    units='exp_conduid',
    errorbar='se',
)
plt.savefig(os.path.join(fig_dir, f'{param}_by_cond0.pdf'), format='pdf')
x_name = 'cond0'
df_var = median_res[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
if df_var["cond0"].nunique() < 2:
    print(f"  Skipped (only one condition for {param})")
# 1. One-way ANOVA
model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. Tukey’s HSD for post hoc comparison
tukey = pairwise_tukeyhsd(
    endog=df_var[param],
    groups=df_var["cond0"],
    alpha=0.05
)
print("\nTukey HSD:")
print(tukey.summary())

#%% split data by 4 quadrants of preIBI_y_displ and plot for each group
data_to_plot['quadrant'] = (
    data_to_plot
    .groupby(['cond0', 'expNum'])['IBI_displAng']
    .transform(lambda x: pd.cut(x, bins=[-np.inf, 0, np.inf], labels=['neg','pos']))
)

#%
sns.catplot(
    data=data_to_plot.groupby(['cond0','expNum','quadrant'])['IBI_displAng'].median().reset_index(),
    x='cond0',
    y='IBI_displAng',
    hue='quadrant',
    kind='point',
)

#%%
median_res = data_to_plot.groupby(['cond0','exp_conduid'])[['preIBI_rot','preIBI_y_displ','IBI_displAng']].median().reset_index()

for param in ['preIBI_rot','preIBI_y_displ','IBI_displAng']:
    plt_categorical_combined_3(
        data=median_res,
        x='cond0',
        y=param,
        hue='cond0',
        palette=my_palette,
        units='exp_conduid',
        errorbar='se',
    )
    plt.savefig(os.path.join(fig_dir, f'{param}_by_cond0.pdf'), format='pdf')
    x_name = 'cond0'
    df_var = median_res[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)

    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    
#%%
x_val = 'preIBI_y_displ'
y_val = 'preIBI_x_displ'

g = sns.displot(
    data=data_to_plot,
    kind='hist',
    stat='probability',
    x=x_val,
    y=y_val,
    col='cond0',
    height=2.5,
    aspect=0.8,
    common_norm=False,
)
g.set(xlim=np.percentile(data_to_plot[x_val], [0.5,99.5]))
g.set(ylim=np.percentile(data_to_plot[y_val], [0.5,99.5]))

#%%
plt.figure(figsize=(4,3))
g = sns.kdeplot(
    data=data_to_plot,
    x=x_val,
    y=y_val,
    hue='cond0',
    common_norm=False,
    alpha=0.5,
)
g.set(xlim=np.percentile(data_to_plot[x_val], [0.5,99.5]))
g.set(ylim=np.percentile(data_to_plot[y_val], [0.5,99.5]))


# bin_edges = np.percentile(data_to_plot[x_val],[1,99])
# bins = np.linspace(bin_edges[0], bin_edges[1], 11)
# # set bins with equal data points
# bin_centers = (bins[:-1] + bins[1:]) / 2

# # generate dataframed binned by preIBI_rot and preIBI_y_displ
# binned_data_rot_ = []
# for cond, group in data_to_plot.groupby(['cond0','expNum']):
#     binned_data_rot = distribution_binned_average_opt(
#         df=group,
#         by_col=x_val,
#         bin_col = y_val,
#         bin = bins,
#         method='median'
#     )
#     print(len(binned_data_rot))
#     binned_data_rot_.append(
#         binned_data_rot.assign(cond0=cond[0], 
#                                expNum=cond[1], 
#                                bin_centers=bin_centers,
#                                )
#         )
# binned_data_rot = pd.concat(binned_data_rot_, ignore_index=True)


# plt.figure(figsize=(4,3))
# # plot binned data
# sns.lineplot(
#     data=binned_data_rot,
#     x='bin_centers',
#     y=y_val,
#     hue='cond0',
#     palette=my_palette,
#     errorbar='se',
# )
# %%
