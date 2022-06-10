'''
Plot averaged features (pitch, inst_traj...) categorized bt pitch up/down and speed bins
Results are jackknifed mean results across experiments (expNum)

Change all_features for the features to plot

Definition of time duration picked for averaging:
prep: bout preperation phase, -200 to -100 ms before peak speed
dur: during bout, -25 to 25 ms
post: +100 to 200 ms 
see idx_bins

Todo: bin by initial posture

NOTE
righting rotation: 0-100ms!
'''

#%%
# import sys
import os,glob
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_index import get_index
from plot_functions.get_bout_kinetics import get_bout_kinetics
set_font_type()
# %%
pick_data = 'tau_long' # all or specific data
# for day night split
which_zeitgeber = 'all'

# %%
# def main(pick_data):
root, FRAME_RATE = get_data_dir(pick_data)
peak_idx , total_aligned = get_index(FRAME_RATE)
# TSP_THRESHOLD = [-np.Inf,-50,50,np.Inf]
# spd_bins = np.arange(3,24,3)

folder_name = f'B_kinetics_{which_zeitgeber}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond1, all_cond2 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber)

# %%
#plot ztime
cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']

toplt = kinetics_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]
for feature_toplt in (all_features):
    sns.catplot(data = kinetics_jackknife,
                x = 'dpf',
                row = 'condition',
                hue='ztime',
                y = feature_toplt,
                kind='point'
                )
    filename = os.path.join(fig_dir,f"{feature_toplt}_ztime.pdf")
    plt.savefig(filename,format='PDF')

# %%
# by speed bins
toplt = kinetics_bySpd_jackknife
for feature_toplt in (['righting','set','steering','corr']):
    wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

    df_toplt = long_data.reset_index()
    g = sns.FacetGrid(df_toplt,
                    row = "feature", 
                    col = 'dpf',
                    hue = 'condition', 
                    height=3, aspect=1.8, 
                    sharey='row',
                    )
    g.map_dataframe(sns.lineplot, 
                    x = 'speed_bins', y = feature_toplt,
                    err_style='band', 
                    # ci='sd'
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'speed_bins', y = feature_toplt, 
                    ci=None, join=False,
                    markers='d')
    
    # if feature_toplt == 'righting':
    #     g.set(ylim = (0.05,0.19))
    g.add_legend()
    # plt.savefig(fig_dir+f"/{pick_data}__spd_{feature_toplt}.pdf",format='PDF')
  
    
    
    
    
    
    
    # fig, ax = plt.subplots(1, figsize=[3.2,4])
    # sns.pointplot(
    #             x = "condition", y = feature_toplt, data = toplt,
    #             order=all_cond2,
    #             hue='jackknife_group', ci=None,
    #             palette=sns.color_palette(flatui), scale=0.5,zorder=1,
    #             ax=ax)
    # g = sns.pointplot(data = toplt, x = 'condition', y = feature_toplt,
    #                 order=all_cond2,
    #                 linewidth=0,
    #                 hue='dpf', markers='d',
    #                 hue_order=all_cond1,
    #                 # ci=False, 
    #                 zorder=100,
    #                 ax=ax,
    #                 )
    # # if feature_toplt == 'righting_gain_jack':
    # #     g.set_ylim(0.13,0.16)
    # # g.legend_.remove()
    # plt.savefig(fig_dir+f"/{pick_data}_{feature_toplt}.pdf",format='PDF')
    # plt.clf()

















# %%
# calculate kinetics by speed bins
kinetics_bySpd_jackknife = pd.DataFrame()
for name, group in all_feature_cond.groupby(['condition','dpf']):
    kinetics_all_speed = pd.DataFrame()
    for speed_bin in set(group.speed_bins):
        if pd.notna(speed_bin):
            this_speed_data = group.loc[group['speed_bins']==speed_bin,:]
            this_speed_kinetics = jackknife_kinetics(this_speed_data,'expNum')
            this_speed_kinetics = this_speed_kinetics.assign(speed_bins=speed_bin)
            kinetics_all_speed = pd.concat([kinetics_all_speed,this_speed_kinetics],ignore_index=True)
    kinetics_all_speed = kinetics_all_speed.assign(
        condition = name[0],
        dpf = name[1]
        )   
    kinetics_bySpd_jackknife = pd.concat([kinetics_bySpd_jackknife, kinetics_all_speed],ignore_index=True)
kinetics_bySpd_jackknife = kinetics_bySpd_jackknife.sort_values(by=['condition','jackknife_group','dpf']).reset_index(drop=True)
# for condition in set(all_feature_cond.condition):
#     this_cond_data = all_feature_cond.loc[all_feature_cond['condition']==condition,:]
#     kinetics_all_speed = pd.DataFrame()
#     for speed_bin in set(this_cond_data.speed_bins):
#         if pd.notna(speed_bin):
#             this_speed_data = this_cond_data.loc[this_cond_data['speed_bins']==speed_bin,:]
#             this_speed_kinetics = jackknife_kinetics(this_speed_data)
#             this_speed_kinetics = this_speed_kinetics.assign(speed_bins=speed_bin)
#             kinetics_all_speed = pd.concat([kinetics_all_speed,this_speed_kinetics],ignore_index=True)
#     kinetics_all_speed = kinetics_all_speed.assign(condition = condition)    
#     kinetics_bySpd_jackknife = pd.concat([kinetics_bySpd_jackknife, kinetics_all_speed],ignore_index=True)
# kinetics_bySpd_jackknife = kinetics_bySpd_jackknife.sort_values(by=['condition','jackknife_group','dpf']).reset_index(drop=True)

# %%








# %% Compare Sibs & Tau
cat_cols = ['jackknife_group','condition','expNum','dpf','ztime']

toplt = kinetics_jackknife
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('condition').size().max())
defaultPlotting()

# print('plot jackknife data')

for feature_toplt in (all_features):
    fig, ax = plt.subplots(1, figsize=[3.2,4])
    sns.pointplot(
                x = "condition", y = feature_toplt, data = toplt,
                order=all_cond2,
                hue='jackknife_group', ci=None,
                palette=sns.color_palette(flatui), scale=0.5,zorder=1,
                ax=ax)
    g = sns.pointplot(data = toplt, x = 'condition', y = feature_toplt,
                    order=all_cond2,
                    linewidth=0,
                    hue='dpf', markers='d',
                    hue_order=all_cond1,
                    # ci=False, 
                    zorder=100,
                    ax=ax,
                    )
    # if feature_toplt == 'righting_gain_jack':
    #     g.set_ylim(0.13,0.16)
    # g.legend_.remove()
    plt.savefig(fig_dir+f"/{pick_data}_{feature_toplt}.pdf",format='PDF')
    # plt.clf()
plt.close('all')

# %% raw data. no jackknife
cat_cols = ['expNum','condition','dpf']

toplt = all_kinetic_cond
all_features = [c for c in toplt.columns if c not in cat_cols]

flatui = ["#D0D0D0"] * (toplt.groupby('condition').size().max())

defaultPlotting()

# print('plot raw data')

for feature_toplt in (all_features):
    g = sns.catplot(data = toplt, x = 'condition', y = feature_toplt,
                    order=all_cond2,
                    height=4, aspect=0.8, kind='point',
                    hue='dpf', markers='d',sharey=False,
                    hue_order=all_cond1,
                    # ci=False, 
                    zorder=10
                    )
    g.map_dataframe(sns.pointplot, 
                    x = "condition", y = feature_toplt,
                    order=all_cond2,
                    hue='expNum', ci=None,
                    palette=sns.color_palette(flatui), scale=0.5,zorder=-1)
    
    plt.savefig(fig_dir+f"/{pick_data}_{feature_toplt}.pdf",format='PDF')
    # plt.clf()
plt.close('all')
# %% by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['speed_bins', 'condition','dpf']
all_features = [c for c in toplt.columns if c not in cat_cols]

# print("Plot with long format. as a function of speed. ")

defaultPlotting()
toplt = kinetics_bySpd_jackknife
for feature_toplt in (['righting','set','steering','corr']):
    wide_data = toplt.loc[:,cat_cols + [col for col in all_features if f'{feature_toplt}' in col]]
    wide_data['id'] = wide_data.index
    long_data = pd.wide_to_long(wide_data, stubnames=feature_toplt, i='id', j='feature', sep='_', suffix='\w+')

    df_toplt = long_data.reset_index()
    g = sns.FacetGrid(df_toplt,
                    row = "feature", 
                    col = 'dpf',
                    hue = 'condition', 
                    height=3, aspect=1.8, 
                    sharey='row',
                    )
    g.map_dataframe(sns.lineplot, 
                    x = 'speed_bins', y = feature_toplt,
                    err_style='band', 
                    # ci='sd'
                    )
    g.map_dataframe(sns.pointplot, 
                    x = 'speed_bins', y = feature_toplt, 
                    ci=None, join=False,
                    markers='d')
    
    # if feature_toplt == 'righting':
    #     g.set(ylim = (0.05,0.19))
    g.add_legend()
    plt.savefig(fig_dir+f"/{pick_data}__spd_{feature_toplt}.pdf",format='PDF')
    # plt.clf()

# plt.close('all')

# # %%
# if __name__ == '__main__':
#     mpl.rcParams['pdf.fonttype'] = 42
#     mpl.rc('figure', max_open_warning = 0)

#     if which_data == 'all':
#         all_data = ['ori','hets','lesion','s','master','4d','ld']
#         for pick_data in tqdm(all_data):
#             print(f"Plotting {pick_data} data:")
#             main(pick_data)
#     else:
#         print(f"Plotting {which_data} data:")
#         main(which_data)        

    


# %%
