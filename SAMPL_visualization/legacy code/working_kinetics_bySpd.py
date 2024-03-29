
#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from plot_functions.get_bout_kinetics import get_bout_kinetics

from tqdm import tqdm
import matplotlib as mpl

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'otog'
which_ztime = 'all'
root, FRAME_RATE = get_data_dir(pick_data)
folder_name = pick_data

folder_name = f'kinetics_bySpd'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond1 = get_bout_kinetics(root, FRAME_RATE, ztime=which_ztime)

# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# all_ibi_cond = all_ibi_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
#                                     direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,7.5,90],labels=['dive','climb']))

one_kinetics_bycond0 = all_feature_cond.groupby(['cond0']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
pitch_bins = np.arange(-20,42,12)
spd_bins = np.arange(5,25,4)
# set_point = one_kinetics.loc[0,'set_point']
all_feature_UD = pd.DataFrame()
set_point = all_kinetic_cond.groupby(['cond1','ztime','cond0'])['set_point'].mean().reset_index()
for (cond,ztime,dpf),group in all_feature_cond.groupby(['cond1','ztime','cond0']):
    this_set_point = set_point.loc[(set_point['cond1']==cond) &
                                  (set_point['ztime']==ztime) &
                                  (set_point['cond0']==dpf),'set_point'].values[0]
    group = group.assign(
        direction = pd.cut(group['pitch_pre_bout'],bins=[-90,this_set_point,90],labels=['down','up']),
        set_point = this_set_point,
    )
    all_feature_UD = pd.concat([all_feature_UD,group.assign(
        cond1=cond,
        ztime=ztime,
        dpf=dpf
    )
    ])
all_feature_UD = all_feature_UD.assign(
    pre_bout_bins = pd.cut(all_feature_UD['pitch_pre_bout'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    speed_bins = pd.cut(all_feature_UD['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

all_feature_UD = all_feature_UD.dropna().reset_index(drop=True)

# %%
# # get kinetics by spd, same as bkinetics_ztime
# by speed bins
toplt = kinetics_bySpd_jackknife
cat_cols = ['jackknife_group','cond1','expNum','cond0','ztime']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        row = 'ztime',
        col = 'cond0',
        hue = 'cond1',
        x = 'average_speed',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"{feature_toplt}_z{which_ztime}_ztime_bySpd.pdf")
    plt.savefig(filename,format='PDF')
# %%
# Plots
# %% 
# distribution of pre pitch as speed 
# NOTE: only one of dpf vs ztime can be ploted

toplt = all_feature_UD
feature_to_plt = 'pitch_pre_bout'
g = sns.FacetGrid(data=toplt,
            col='cond1', row="ztime",
            # col_order=all_cond0,
            hue='speed_bins',
            sharey =False,
            )
g.map(sns.kdeplot,feature_to_plt,common_norm=False)
g.add_legend()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')
# %%
# distribution of pre pitch as speed 
toplt = all_feature_UD
feature_to_plt = 'tsp_peak'
g = sns.FacetGrid(data=toplt,
            col='cond1', row="ztime",
            # col_order=all_cond0,
            hue='speed_bins',
            sharey =False,
            )
g.map(sns.kdeplot,feature_to_plt,common_norm=False)
g.add_legend()
g.set(xlim=(-25,25))
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')

# %%
# distribution of pre pitch as speed 
# only plot corrective bouts

# toplt = all_feature_UD.loc[np.abs(all_feature_UD['pitch_post_bout']-setpoint)<np.abs(all_feature_UD['pitch_pre_bout']-setpoint)]

all_feature_UD = all_feature_UD.assign(
    deviationFromSet = np.abs(all_feature_UD['pitch_pre_bout']-set_point),
    mean_speed_byDir = all_feature_UD.groupby('direction')['spd_peak'].transform('mean'),
    absRightingRot = np.abs(toplt['rot_l_decel'])
)
all_feature_UD = all_feature_UD.assign(
    speed_2bin = 'slow'
)
all_feature_UD.loc[all_feature_UD['spd_peak']>all_feature_UD['mean_speed_byDir'],'speed_2bin'] = 'fast'


# %%
# check speed
mean_spd_per_dir = all_feature_UD.groupby('direction')['spd_peak'].mean()

toplt = all_feature_UD
feature_to_plt = 'spd_peak'
g = sns.FacetGrid(data=toplt,
            col='cond0', row='cond1',
            col_order=all_cond0,
            hue='direction',
            sharey =False,
            )
g.map(sns.kdeplot,feature_to_plt,common_norm=False)
plt.axvline(x=mean_spd_per_dir.iloc[0],
            ymin=0,
            ymax=0.04)
plt.axvline(x=mean_spd_per_dir.iloc[1],
            ymin=0,
            ymax=0.04)
g.add_legend()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')# %%

# %%

toplt = all_feature_UD
toplt = all_feature_UD.loc[all_feature_UD['deviationFromSet']>5]

g = sns.FacetGrid(data=toplt,
            col="direction", 
            row='cond1',
            # col="speed_bins",
            # col_order=all_cond0,
            hue='speed_2bin',
            # hue='direction',
            sharey=False,
            aspect=0.8
            )
g.map(sns.regplot,'deviationFromSet','absRightingRot',x_bins=6)
# g.map(sns.scatterplot,'pitch_pre_bout','percent_corrected',alpha=0.1)
g.set(xlim=(5,22))

g.add_legend()
plt.savefig(fig_dir+f"/absRighting vs deviation from set UP DN speed.pdf",format='PDF')

# %%
toplt = all_feature_UD
toplt = all_feature_UD.loc[all_feature_UD['deviationFromSet']>5]

g = sns.FacetGrid(data=toplt,
            col="direction", 
            row='speed_2bin',
            # col="speed_bins",
            # col_order=all_cond0,
            hue='cond1',
            # hue='direction',
            sharey=True,
            aspect=0.8
            )
g.map(sns.regplot,'deviationFromSet','absRightingRot',x_bins=6)
# g.map(sns.scatterplot,'pitch_pre_bout','percent_corrected',alpha=0.1)
g.set(xlim=(5,22))

g.add_legend()
plt.savefig(fig_dir+f"/absRighting vs deviation from set UP DN speed_cond.pdf",format='PDF')

# %%
