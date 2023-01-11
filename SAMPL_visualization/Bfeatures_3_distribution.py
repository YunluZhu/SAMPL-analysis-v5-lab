'''
Very useful new function!
Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

'''

#%%
# import sys
import os
import pandas as pd
from plot_functions.plt_tools import round_half_up 
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
pick_data = 'tmp'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF3_distribution_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond0, all_cond0 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
# %% tidy data
all_feature_cond['direction'] = pd.cut(all_feature_cond['pitch_initial'],[-90,10,90],labels=['DN','UP'])

all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
                                    direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,10,90],labels=['DN','UP']))

# get kinetics for separating up and down
all_kinetics = all_feature_cond.groupby(['cond0']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
# all_feature_UD = pd.DataFrame()
# # all_feature_cond = all_feature_cond.assign(direction=np.nan)
# # for key, group in all_feature_cond.groupby(['cond0']):
# #     this_setvalue = all_kinetics.loc[all_kinetics['cond0']==key,'set_point'].to_list()[0]
# #     print(this_setvalue)
# #     group['direction'] = pd.cut(group['pitch_initial'],
# #                                 bins=[-91,this_setvalue,91],
# #                                 labels=['dn','up'])
# #     all_feature_UD = pd.concat([all_feature_UD,group])

# all_feature_cond = all_feature_cond.assign(direction=np.nan)
# for key, group in all_feature_cond.groupby(['cond0']):
#     group['direction'] = pd.cut(group['pitch_initial'],
#                                 bins=[-91,10,91],
#                                 labels=['dn','up'])
#     all_feature_UD = pd.concat([all_feature_UD,group])

# %%
# Plots
# %% histogram
all_feature_UD = all_feature_cond
toplt = all_feature_UD
feature_to_plt = 'pitch_initial'
sns.displot(data=toplt,x=feature_to_plt, kde=True,
            col='cond0', row='cond1',col_order=all_cond0,hue='cond1',
            facet_kws={'sharey':False})
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')

# %% kde
feature_to_plt = 'pitch_initial'
g = sns.FacetGrid(data=all_feature_UD,
                col='cond0',
                row="ztime",
                col_order=all_cond0,hue='cond1',
                sharey=False,
                )
g.map(sns.kdeplot,feature_to_plt)
g.set(xlim=(-30, 40))
g.add_legend()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')


# %% kde, separate up dn
toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','traj_peak']

for feature in feature_to_plt:
    g = sns.FacetGrid(data=toplt,
                col='cond0', row="direction",col_order=all_cond0,hue='cond1',
                sharey=False,
                )
    g.map(sns.kdeplot,feature)
    g.add_legend()
    plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')

toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','traj_peak']
# %%
# IBI duration
toplt = all_feature_UD
feature_to_plt = 'propBoutIEI'
sns.displot(data=all_ibi_cond,
            x=feature_to_plt,
            log_scale=True,
            # kde=True,
            col='cond0',
            # stat='density',
            # row='cond1',
            col_order=all_cond0,
            hue='cond1',
            kind='ecdf',
            # facet_kws={'sharey':False}
            )
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')
# %%
# IBI posture
g = sns.FacetGrid(data=all_ibi_cond,
                col='cond0',
                row="ztime",
                col_order=all_cond0,hue='cond1',
                sharey=False,
                )
g.map(sns.kdeplot,'propBoutIEI_pitch')
g.set(xlim=(-30, 40))
g.add_legend()
plt.savefig(fig_dir+f"/propBoutIEI_pitch distribution.pdf",format='PDF')
# %%
#mean
toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','traj_peak']

for feature in feature_to_plt:
    g = sns.catplot(data=toplt,
                    y = feature,
                    x='cond1',
                col='cond0', row="direction",col_order=all_cond0,hue='cond1',
                sharey=False,
                kind='point'
                )
    g.add_legend()

    # plt.savefig(fig_dir+f"/{feature} distribution.pdf",format='PDF')



#%%
# 2D distribution plot

# %%
# TEST MODULE
# 'pitch_initial', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
#        'pitch_end', 'traj_initial', 'traj_pre_bout', 'traj_peak',
#        'traj_post_bout', 'traj_end', 'spd_peak', 'angvel_initial_phase',
#        'angvel_prep_phase', 'pitch_prep_phase', 'angvel_post_phase',
#        'rot_total', 'rot_pre_bout', 'rot_l_accel', 'rot_l_decel',
#        'rot_early_accel', 'rot_late_accel', 'rot_early_decel',
#        'rot_late_decel', 'traj_peak', 'atk_ang', 'tsp_pre', 'tsp_peak',
#        'angvel_chg',
toplt = all_feature_UD

# toplt = toplt.loc[(toplt['rot_pre_bout']>-2) & (toplt['rot_pre_bout']<1),:]
xname = 'traj_peak'
yname = 'atk_ang'
if len(toplt) > 10000:
    toplt = toplt.sample(n=10000)
g = sns.displot(data= toplt,
                y=yname,x=xname,
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
g.add_legend()

plt.savefig(fig_dir+f"/{yname} v {xname}.pdf",format='PDF')

# %%
# p = sns.lmplot(
#     data = toplt.sample(n=10000),
#     col='cond0',col_order=all_cond0,
#     hue='cond1',
#     x = xname, y = yname,
#     x_bins=8,
#      robust=True,
#     #   ci=95,
#     scatter_kws={"s": 10,},
#     )
# p.set(
#     ylim=(-5,10),
#     xlim=(-10,25)
#     )
# plt.savefig(fig_dir+f"/{yname} v {xname} fit.pdf",format='PDF')

# %%
toplt = all_feature_cond
# steering
g = sns.displot(data=toplt,
                y='traj_peak',x='pitch_peak',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
g.add_legend()

plt.savefig(fig_dir+"/traj_peak v pitch_peak.pdf",format='PDF')

# set point
g = sns.displot(data=toplt,
                y='pitch_pre_bout',x='rot_total',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
g.add_legend()

g.set(ylim=(-40,50))
plt.savefig(fig_dir+"/pitch_initial v rot_total.pdf",format='PDF')

#righting
g = sns.displot(data=toplt,
                y='rot_l_decel',x='pitch_pre_bout',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
g.add_legend()

plt.savefig(fig_dir+"/rot_l_decel v pitch_pre_bout.pdf",format='PDF')

# g = sns.relplot(data=toplt,
#                 y='rot_l_decel',x='pitch_pre_bout',
#                 col='cond0', row='cond1',col_order=all_cond0,hue='spd_peak', kind='scatter', alpha=0.2)
# g.add_legend()
# plt.savefig(fig_dir+"/rot_l_decel v pitch_pre_bout scatter.pdf",format='PDF')


#corr
g = sns.displot(data=toplt,
                y='rot_l_decel',x='rot_late_accel',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
g.add_legend()

plt.savefig(fig_dir+"/rot_l_decel v rot_late_accel.pdf",format='PDF')

# %%
# atk angle - pitch
toplt = all_feature_cond

# g = sns.displot(data=toplt,
#                 y='atk_ang',x='pitch_peak',
#                 col='cond0', row='cond1',col_order=all_cond0,
#                 hue='cond1',
#                 kind='kde'
#                 )

# g.set(ylim=(-50,50),
#       xlim=(-30,40))
# plt.savefig(fig_dir+"/atk_ang v pitch_initial.pdf",format='PDF')
g = sns.jointplot(data=toplt.loc[toplt['cond0']=='07',:],
                y='atk_ang',x='pitch_peak',
                hue='cond1',
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
                col='cond0', row='cond1',col_order=all_cond0,
                hue='cond1',
                )
g.set(ylim=(-20,20),
      xlim=(-20,20))
g.add_legend()

plt.savefig(fig_dir+"/angvel_chg v angvel_initial_phase.pdf",format='PDF')

# %%
# post angvel vs prep pitch
toplt = all_feature_UD

g = sns.displot(data=toplt,
                y='angvel_post_phase',
                x='pitch_initial',
                col='cond0',
                row='cond1',col_order=all_cond0,
                hue='cond1',
                )
g.set(ylim=(-30,30),
      xlim=(-30,40))
g.add_legend()

plt.savefig(fig_dir+"/angvel_post_phase v pitch_initial.pdf",format='PDF')

# post angvel vs prepp pitch
toplt = all_feature_UD

g = sns.displot(data=toplt,
                y='angvel_post_phase',
                x='pitch_end',
                col='cond0',
                row='cond1',col_order=all_cond0,
                hue='cond1',
                )
g.set(ylim=(-30,30),
      xlim=(-30,40))
g.add_legend()

plt.savefig(fig_dir+"/angvel_post_phase v pitch_end.pdf",format='PDF')

# %% try fit
toplt = all_feature_UD

p = sns.FacetGrid(
    data = toplt,
    col='cond0',col_order=all_cond0,
    hue='cond1',
)
p.map(sns.regplot,"pitch_end", "angvel_post_phase",
      x_bins=8,
      ci=95,

      scatter_kws={"s": 10,},
      )
p.set(ylim=(-5,8),
      xlim=(-12,30))
g.add_legend()

plt.savefig(fig_dir+"/angvel_post_phase v pitch_end fit.pdf",format='PDF')

# %% try fit
toplt = all_feature_UD

p = sns.FacetGrid(
    data = toplt,
    col='cond0',col_order=all_cond0,
    hue='cond1',
)
p.map(sns.regplot,"angvel_initial_phase", "angvel_chg",
      x_bins=8,
      ci=95,

      scatter_kws={"s": 10,},
      )
p.set(ylim=(0,10),
      xlim=(-10,0))
g.add_legend()

plt.savefig(fig_dir+"/angvel_chg v angvel_initial_phase fit.pdf",format='PDF')



# %% 2d distribution
toplt = all_feature_cond
spd_bins = [3,7,100]

toplt = toplt.assign(
    spd_bins = pd.cut(toplt['spd_peak'],bins=spd_bins,labels=['slow','fast'])
)
toplt = toplt.loc[toplt['spd_bins']=='fast',:]
# toplt = toplt.groupby('cond0').sample(n=30000)
g = sns.displot(data=toplt,
                y='atk_ang',x='rot_pre_bout',
                col='cond0',col_order=all_cond0, row='cond1',hue='cond1')
g.set(ylim=(-40,40),xlim=(-5,10))
plt.savefig(fig_dir+"/atk_ang v rot_pre_bout _speed filtered.pdf",format='PDF')

# %%
toplt = all_feature_cond
g = sns.displot(data=toplt,
                x='pitch_end',y='angvel_post_phase',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
# g.set(xlim=(-15, 30),ylim=(-10,10))
plt.savefig(fig_dir+"/pitch_end v angvel_post_phase.pdf",format='PDF')

# %%
toplt = all_feature_cond
g = sns.displot(data=toplt,
                x='pitch_peak',y='traj_peak', hue='direction',
                col='cond0', row='cond1',col_order=all_cond0,
                kind='kde'
                )
plt.savefig(fig_dir+"/traj_peak v pitch_peak.pdf",format='PDF')

# g.set(xlim=(-15, 30),ylim=(-10,10))

# %%
# IBI features
toplt = all_ibi_cond
g = sns.displot(data=toplt,
                x='propBoutIEI_pitch',y='propBoutIEI_angVel',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
# g.set(xlim=(-15, 30),ylim=(-10,10))
# %%
# IBI frequency
toplt = all_ibi_cond
g = sns.displot(data=toplt,
                x='propBoutIEI_pitch',y='y_boutFreq',
                col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
# g.set(xlim=(-20, 40),ylim=(0,4))
plt.savefig(fig_dir+"/IEI freq pitch.pdf",format='PDF')

# %%
# 3D navigation
spd_bins = np.arange(5,25,4)

toplt = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)
# %%
g = sns.lmplot(
    data = toplt,
    col = 'cond1',
    hue = 'speed_bins',
    x = 'depth_chg',
    y = 'pitch_peak',
    height=3,
    x_bins=8,
)
g.set(xlim=(-1,2),
      ylim=(-20,40))
plt.savefig(fig_dir+"/pitch_peak vs depth_chg.pdf",format='PDF')

p = sns.lmplot(
    data = toplt,
    col = 'cond1',
    hue = 'speed_bins',
    x = 'depth_chg',
    y = 'atk_ang',
    height=3,
    x_bins=8,
)
p.set(xlim=(-1,2),
      ylim=(-20,30))
plt.savefig(fig_dir+"/atk_ang vs depth_chg.pdf",format='PDF')
# %%
#---
#   # %%
# # angvel percent recover
# plt.close('all')
# toplt = all_feature_cond
# toplt = toplt.assign(
#     angvel_percent_recover = toplt['angvel_chg']/toplt['angvel_initial_phase']
# )
# toplt = toplt.loc[toplt['angvel_percent_recover']<0,:]
# toplt = toplt.loc[toplt['cond1']==all_cond0[0],:]
# g = sns.displot(data=toplt,
#                 y='angvel_percent_recover',
#                 x='angvel_initial_phase',
#                 col='cond0', col_order=all_cond0,
#                 # row='cond1',
#                 # hue='cond1',
#                 )
# g.set(
#     ylim=(-2,0),
#       xlim=(-40,60)
#       )
# g.add_legend()

# plt.savefig(fig_dir+"/angvel_percent_recover v angvel_initial_phase.pdf",format='PDF')

# # %%
# # pitch percent recover
# plt.close('all')
# toplt = all_feature_cond
# toplt = toplt.assign(
#     pitch_percent_recover = toplt['rot_total']/toplt['pitch_initial']
# )
# # toplt = toplt.loc[toplt['pitch_percent_recover']<0,:]
# toplt = toplt.loc[toplt['cond1']==all_cond0[0],:]
# g = sns.displot(data=toplt.sample(n=1000),
#                 y='pitch_percent_recover',
#                 x='pitch_initial',
#                 col='cond0', col_order=all_cond0,
#                 # row='cond1',
#                 # hue='cond1',
#                 )
# g.set(
#     ylim=(-2,2),
#     xlim=(-30,30)
#     )
# g.add_legend()

# plt.savefig(fig_dir+"/pitch_percent_recover v pitch_initial.pdf",format='PDF')

# # %%

# g = sns.displot(data=toplt,
#                 y='rot_l_decel',x='pitch_prep_phase',
#                 col='cond0', row='cond1',col_order=all_cond0,hue='cond1')
# # plt.savefig(fig_dir+"/rot_l_decel v pitch_pre_bout.pdf",format='PDF')

#  # %%
#  # fin body ratio filtering
# toplt = all_feature_cond
# spd_bins = [3,7,100]

# toplt = toplt.assign(
#     spd_bins = pd.cut(toplt['spd_peak'],bins=spd_bins,labels=['slow','fast'])
# )
# toplt = toplt.loc[toplt['spd_bins']=='fast',:]

# g = sns.displot(data=toplt,
#                 y='atk_ang',x='rot_pre_bout',
#                 col='cond0',col_order=all_cond0, row='cond1',hue='cond1')
# g.set(ylim=(-75,75),xlim=(-5,10))

# plt.savefig(fig_dir+"/atk_ang v rot_pre_bout _speed filtered.pdf",format='PDF')

# %%