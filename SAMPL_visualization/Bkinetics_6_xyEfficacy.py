'''

'''

#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from plot_functions.plt_tools import jackknife_list


set_font_type()
defaultPlotting()

# %%
pick_data = 'tau_bkg'
which_zeitgeber = 'day'
folder_name = f'BK6_xyEfficacy'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
spd_bins = np.arange(5,25,4)

root, FRAME_RATE = get_data_dir(pick_data)
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond0 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
all_cond0 = pick_data
all_cond0.sort()


all_feature_cond = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)



sns.set_style("ticks")
    
# %%
# check speed distribution
toplt = all_feature_cond

# check speed
feature_to_plt = 'spd_peak'
upper = np.percentile(toplt[feature_to_plt], 99.5)
lower = np.percentile(toplt[feature_to_plt], 0.5)
g = sns.FacetGrid(data=toplt,
            col='cond0', 
            hue='cond1',
            sharey =False,
            sharex =True,
            )
g.map(sns.histplot,feature_to_plt,bins = 10, 
                    element="poly",
                    #  kde=True, 
                    stat="probability",
                    pthresh=0.05,
                    fill=False,
                    binrange=(lower,upper),)

g.add_legend()
sns.despine()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')# %%

# %%
# jackknife std
col = 'expNum'
# jackknife_mean = pd.DataFrame()
jackknife_std = pd.DataFrame()
for (dpf, condition), group in all_feature_cond.groupby(['cond0','cond1']):
    exp_df = group.groupby(col).size()
    jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    output = pd.DataFrame()
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group[col].isin(exp_group),:]
        # this_jackknife_mean = this_group_data.mean(numeric_only=True).to_frame().T
        this_jackknife_std = this_group_data.std(numeric_only=True).to_frame().T
        # jackknife_mean = pd.concat([jackknife_mean,
        #                             this_jackknife_mean.assign(
        #                                 cond0 = dpf,
        #                                 cond1 = condition,
        #                                 jakknife_group = j
        #                             )],ignore_index=True)
        jackknife_std = pd.concat([jackknife_std,
                                    this_jackknife_std.assign(
                                        cond0 = dpf,
                                        cond1 = condition,
                                        jakknife_group = j
                                    )],ignore_index=True)
# %%
toplt = kinetics_bySpd_jackknife
all_features = ['y_efficacy']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        hue = 'cond1',
        col = 'cond0',
        errorbar=('ci', 95),
        err_style='bars',
        marker = True,
    )
    g.figure.set_size_inches(4,2)
    # g.set(xlim=(6, 24))
    g.set(ylabel="Depth/posture")
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    filename = os.path.join(fig_dir,f"depth per peak pitch_bySpeed.pdf")
    plt.savefig(filename,format='PDF')
    
# for sel_dpf in ['otog','tan']:
#     df_toplt = toplt.query("cond0 == @sel_dpf")
#     for feature_toplt in ['y_efficacy']:
#         multi_comp = MultiComparison(df_toplt[feature_toplt], df_toplt['cond1']+"|"+df_toplt['speed_bins'].astype('str'))
#         print(f'* {feature_toplt}')
#         print(multi_comp.tukeyhsd().summary())
        # print(multi_comp.tukeyhsd().pvalues)
# %%
# pitch has no correlation with x distance but correlated with y distance
toplt = kinetics_bySpd_jackknife
all_features = ['x_posture_corr','y_posture_corr']

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        hue = 'cond1',
        col = 'cond0',
        errorbar=('ci', 95),
        err_style='bars',
        marker = True,
    )
    g.figure.set_size_inches(4,2)
    # g.set(xlim=(6, 24))
    g.set(ylabel=feature_toplt)
    sns.move_legend(g, "upper left", bbox_to_anchor=(1, 1))
    filename = os.path.join(fig_dir,f"{feature_toplt} by spd.pdf")
    plt.savefig(filename,format='PDF')
    
# for sel_dpf in ['otog','tan']:
#     df_toplt = toplt.query("cond0 == @sel_dpf")
#     for feature_toplt in ['y_efficacy']:
#         multi_comp = MultiComparison(df_toplt[feature_toplt], df_toplt['cond1']+"|"+df_toplt['speed_bins'].astype('str'))
#         print(f'* {feature_toplt}')
#         print(multi_comp.tukeyhsd().summary())
#         # print(multi_comp.tukeyhsd().pvalues)
# %%
