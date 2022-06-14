'''
Plot averaged features (pitch, inst_traj...) categorized bt pitch up/down and speed bins
Results are jackknifed mean results across experiments (expNum)

Change all_features for the features to plot

Definition of time duration picked for averaging:
prep: bout preperation phase, -200 to -100 ms before peak speed
dur: during bout, -25 to 25 ms
post: +100 to 200 ms 
see idx_bins

'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'tau_long'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'B3_feature_distribution_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztim=which_ztime)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

# get kinetics for separating up and down
all_kinetics = all_feature_cond.groupby(['dpf']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
all_feature_UD = pd.DataFrame()
all_feature_cond = all_feature_cond.assign(direction=np.nan)
for key, group in all_feature_cond.groupby(['dpf']):
    this_setvalue = all_kinetics.loc[all_kinetics['dpf']==key,'set_point'].to_list()[0]
    group['direction'] = pd.cut(group['pitch_initial'],
                                bins=[-91,this_setvalue,91],
                                labels=['dn','up'])
    all_feature_UD = pd.concat([all_feature_UD,group])
    
    

# # %% histogram
# toplt = all_feature_UD
# feature_to_plt = 'pitch_initial'
# sns.displot(data=toplt,x=feature_to_plt, kde=True, 
#             col="dpf", row="condition",col_order=all_cond1,hue='condition',
#             facet_kws={'sharey':False})
# plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')

# %% kde, separate up dn
toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','bout_traj']

for feature in feature_to_plt:
    g = sns.FacetGrid(data=toplt, 
                col="dpf", row="direction",col_order=all_cond1,hue='condition',
                sharey=False
                )
    g.map(sns.kdeplot,feature)
    plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')

# %%
#mean
toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','bout_traj']

for feature in feature_to_plt:
    g = sns.catplot(data=toplt, 
                    y = feature,
                    x='condition',
                col="dpf", row="direction",col_order=all_cond1,hue='condition',
                sharey=False,
                kind='point'
                )
    # plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')



#%%
# 2D distribution plot
# # ['pitch_initial', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
#        'pitch_end', 'traj_initial', 'traj_pre_bout', 'traj_peak',
#        'traj_post_bout', 'traj_end', 'spd_peak', 'angvel_prep_phase',
#        'pitch_prep_phase', 'rot_total', 'rot_pre_bout', 'rot_l_accel',
#        'rot_l_decel', 'rot_early_accel', 'rot_late_accel', 'rot_early_decel',
#        'rot_late_decel', 'bout_traj', 'atk_ang', 'expNum', 'dpf', 'condition'],

toplt = all_feature_cond
g = sns.displot(data=toplt, 
                y='traj_peak',x='pitch_peak',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
plt.savefig(fig_dir+"/traj_peak v pitch_peak.pdf",format='PDF')

g = sns.displot(data=toplt, 
                y='tsp_peak',x='pitch_initial',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
g.set(ylim=(-30,30))
plt.savefig(fig_dir+"/tsp_peak v pitch_initial.pdf",format='PDF')

g = sns.displot(data=toplt, 
                y='pitch_initial',x='rot_total',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
plt.savefig(fig_dir+"/pitch_initial v rot_total.pdf",format='PDF')

# %%
g = sns.displot(data=toplt, 
                y='rot_l_decel',x='pitch_prep_phase',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
# plt.savefig(fig_dir+"/rot_l_decel v pitch_pre_bout.pdf",format='PDF')

 # %%
 # fin body ratio filtering
toplt = all_feature_cond
spd_bins = [3,7,100]

toplt = toplt.assign(
    spd_bins = pd.cut(toplt['spd_peak'],bins=spd_bins,labels=['slow','fast'])
)
toplt = toplt.loc[toplt['spd_bins']=='fast',:]

g = sns.displot(data=toplt, 
                y='atk_ang',x='rot_pre_bout', 
                col="dpf",col_order=all_cond1, row="condition",hue='condition')
g.set(ylim=(-75,75),xlim=(-5,10))

plt.savefig(fig_dir+"/atk_ang v rot_pre_bout _speed filtered.pdf",format='PDF')


# %% 2d distribution

# %%
toplt = all_feature_cond
g = sns.displot(data=toplt, 
                x='pitch_prep_phase',y='angvel_prep_phase',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
g.set(xlim=(-15, 30),ylim=(-10,10))

# %%
toplt = all_ibi_cond
g = sns.displot(data=toplt, 
                x='propBoutIEI_pitch',y='propBoutIEI_angVel',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
g.set(xlim=(-15, 30),ylim=(-10,10))
# %%
# # %%
# toplt = all_feature_cond
# g = sns.FacetGrid(toplt, col="dpf", row="condition",col_order=all_cond1,hue='condition')
# g.map(sns.kdeplot,'pitch_initial','traj_initial',fill=True,thresh=0.2)
# g.set(xlim=(-30, 40))

# # %%
# toplt = all_feature_cond
# g = sns.FacetGrid(toplt, col="dpf", row="condition",col_order=all_cond1,hue='condition')
# g.map(sns.kdeplot,'pitch_initial','angvel_prep_phase',fill=True,thresh=0.2)
# g.set(xlim=(-15, 30),ylim=(-10,10))


# toplt = all_ibi_cond
# g = sns.FacetGrid(toplt, col="dpf", row="condition",col_order=all_cond1,hue='condition')
# g.map(sns.kdeplot,'propBoutIEI_pitch','propBoutIEI_angVel',fill=True,thresh=0.2)
# g.set(xlim=(-15, 30),ylim=(-10,10))
# %%
 # %%
 
 # # ['pitch_initial', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
#        'pitch_end', 'traj_initial', 'traj_pre_bout', 'traj_peak',
#        'traj_post_bout', 'traj_end', 'spd_peak', 'angvel_prep_phase',
#        'pitch_prep_phase', 'rot_total', 'rot_pre_bout', 'rot_l_accel',
#        'rot_l_decel', 'rot_early_accel', 'rot_late_accel', 'rot_early_decel',
#        'rot_late_decel', 'bout_traj', 'atk_ang', 'expNum', 'dpf', 'condition'],

toplt = all_feature_cond
g = sns.displot(data=toplt, 
                y='pitch_initial',x='rot_total',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
# %%
