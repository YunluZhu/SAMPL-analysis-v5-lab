'''
Test code... identical to Bfeatures_3_distribution but with options to segment data by speed

Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

zeitgeber time? Yes
jackknifed? No
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

folder_name = f'B3_feature_distribution_z{which_ztime}_selSpd'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
                                    direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,7.5,90],labels=['dive','climb']))

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
    print(this_setvalue)
    group['direction'] = pd.cut(group['pitch_initial'],
                                bins=[-91,this_setvalue,91],
                                labels=['dn','up'])
    all_feature_UD = pd.concat([all_feature_UD,group])
    
all_feature_sel = all_feature_UD.loc[(all_feature_UD['spd_peak']>7)&(all_feature_UD['spd_peak']<21),:]
# all_feature_sel = all_feature_UD.loc[(all_feature_UD['spd_peak']>15),:]

# %%
# Plots

# %% kde, separate up dn
toplt = all_feature_sel
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','bout_traj']

for feature in feature_to_plt:
    g = sns.FacetGrid(data=toplt, 
                col="dpf", row="direction",col_order=all_cond1,hue='condition',
                sharey=False,
                )
    g.map(sns.kdeplot,feature)
    plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')

# %%
# IBI posture
g = sns.FacetGrid(data=all_ibi_cond, 
                col="dpf", 
                row="ztime",
                col_order=all_cond1,hue='condition',
                sharey=False,
                )
g.map(sns.kdeplot,'propBoutIEI_pitch')
g.set(xlim=(-30, 40))
plt.savefig(fig_dir+f"/propBoutIEI_pitch distribution.pdf",format='PDF')
# %%
#mean
toplt = all_feature_sel
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

toplt = all_feature_sel
# steering
g = sns.displot(data=toplt, 
                y='traj_peak',x='pitch_peak',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
plt.savefig(fig_dir+"/traj_peak v pitch_peak.pdf",format='PDF')

# set point
g = sns.displot(data=toplt, 
                y='pitch_initial',x='rot_total',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
# g.set(ylim=(-30,30))
plt.savefig(fig_dir+"/pitch_initial v rot_total.pdf",format='PDF')

#righting
g = sns.displot(data=toplt, 
                y='rot_l_decel',x='pitch_pre_bout',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
plt.savefig(fig_dir+"/rot_l_decel v pitch_pre_bout.pdf",format='PDF')

#corr
g = sns.displot(data=toplt, 
                y='rot_l_decel',x='rot_late_accel',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
plt.savefig(fig_dir+"/rot_l_decel v rot_late_accel.pdf",format='PDF')

# %%
# atk angle - pitch
toplt = all_feature_cond

# g = sns.displot(data=toplt, 
#                 y='atk_ang',x='pitch_peak',
#                 col="dpf", row="condition",col_order=all_cond1,
#                 hue='condition',
#                 kind='kde'
#                 )

# g.set(ylim=(-50,50),
#       xlim=(-30,40))
# plt.savefig(fig_dir+"/atk_ang v pitch_initial.pdf",format='PDF')
g = sns.jointplot(data=toplt.loc[toplt['dpf']=='07',:], 
                y='atk_ang',x='pitch_peak',
                hue='condition',
                kind='kde',
                xlim=[-40,50],
                ylim=[-30,40],
                )
plt.savefig(fig_dir+"/atk_ang v pitch_initial 7dpf.pdf",format='PDF')

# %%
# angvel change
toplt = all_feature_UD
# bins = [3,5,10,100]
# bins = [-50,-5,10,20,100]
# toplt = toplt.assign(
#     bins = pd.cut(toplt['pitch_peak'],bins=bins,labels=['0','1','2','3'])
# )
# toplt = toplt.loc[toplt['bins']=='0',:]


g = sns.displot(data=toplt, 
                y='angvel_chg',x='angvel_initial_phase',
                col="dpf", row="condition",col_order=all_cond1,
                hue='condition',
                )
g.set(ylim=(-20,20),
      xlim=(-20,20))
plt.savefig(fig_dir+"/angvel_chg v angvel_initial_phase.pdf",format='PDF')

# %%
# post angvel vs prep pitch
toplt = all_feature_UD

g = sns.displot(data=toplt, 
                y='angvel_post_phase',
                x='pitch_initial',
                col="dpf", 
                row="condition",col_order=all_cond1,
                hue='condition',
                )
g.set(ylim=(-30,30),
      xlim=(-25,40))
plt.savefig(fig_dir+"/angvel_post_phase v pitch_initial.pdf",format='PDF')

# post angvel vs end pitch
toplt = all_feature_UD

g = sns.displot(data=toplt, 
                y='angvel_post_phase',
                x='pitch_end',
                col="dpf", 
                row="condition",col_order=all_cond1,
                hue='condition',
                )
g.set(ylim=(-30,30),
      xlim=(-25,40))
plt.savefig(fig_dir+"/angvel_post_phase v pitch_end.pdf",format='PDF')

# %% try fit
toplt = all_feature_UD

p = sns.FacetGrid(
    data = toplt,
    col='dpf',col_order=all_cond1,
    hue='condition',
)
p.map(sns.regplot,"pitch_end", "angvel_post_phase",
      x_bins=8,
      ci=95,
      
      scatter_kws={"s": 10,},
      )
p.set(ylim=(-1,8),
      xlim=(-12,30))
plt.savefig(fig_dir+"/angvel_post_phase v pitch_end fit.pdf",format='PDF')

# %%
# toplt = all_feature_UD

# # p = sns.FacetGrid(
# #     data = toplt,
# #     col='dpf',col_order=all_cond1,
# #     hue='condition',
# # )
# p = sns.jointplot(data=toplt.loc[toplt['dpf']=='04',:],
#                   x="pitch_initial", 
#                   y="angvel_post_phase",
#                   hue='condition',
#                     # scatter_kws={"s": 10,},
#                   kind='kde',
#                   xlim=[-30,50],
#                   ylim=[-20,20]
#                   )

#---
  # %%
# angvel percent recover
plt.close('all')
toplt = all_feature_cond
toplt = toplt.assign(
    angvel_percent_recover = toplt['angvel_chg']/toplt['angvel_initial_phase']
)
toplt = toplt.loc[toplt['angvel_percent_recover']<0,:]
toplt = toplt.loc[toplt['condition']==all_cond2[0],:]
g = sns.displot(data=toplt, 
                y='angvel_percent_recover',
                x='angvel_initial_phase',
                col="dpf", col_order=all_cond1,
                # row="condition",
                # hue='condition',
                )
g.set(
    ylim=(-2,0),
      xlim=(-40,60)
      )
plt.savefig(fig_dir+"/angvel_percent_recover v angvel_initial_phase.pdf",format='PDF')

# %%
# pitch percent recover
plt.close('all')
toplt = all_feature_cond
toplt = toplt.assign(
    pitch_percent_recover = toplt['rot_total']/toplt['pitch_initial']
)
# toplt = toplt.loc[toplt['pitch_percent_recover']<0,:]
toplt = toplt.loc[toplt['condition']==all_cond2[0],:]
g = sns.displot(data=toplt.sample(n=1000), 
                y='pitch_percent_recover',
                x='pitch_initial',
                col="dpf", col_order=all_cond1,
                # row="condition",
                # hue='condition',
                )
g.set(
    ylim=(-2,2),
    xlim=(-30,30)
    )
plt.savefig(fig_dir+"/pitch_percent_recover v pitch_initial.pdf",format='PDF')

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
# IBI features
toplt = all_ibi_cond
g = sns.displot(data=toplt, 
                x='propBoutIEI_pitch',y='propBoutIEI_angVel',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
g.set(xlim=(-15, 30),ylim=(-10,10))
# %%
# IBI frequency
toplt = all_ibi_cond
g = sns.displot(data=toplt, 
                x='propBoutIEI_pitch',y='y_boutFreq',
                col="dpf", row="condition",col_order=all_cond1,hue='condition')
g.set(xlim=(-20, 40),ylim=(0,4))
plt.savefig(fig_dir+"IEI freq pitch.pdf",format='PDF')

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
