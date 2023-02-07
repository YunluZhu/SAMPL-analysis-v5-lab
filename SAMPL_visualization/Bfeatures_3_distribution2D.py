'''
Very useful new function!
Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

refer to the 2D distribution plot sections for custimization.
'''

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
from plot_functions.plt_tools import (set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

##### Parameters to change #####

pick_data = 'tmp' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF3_distribution2D_z{which_ztime}'
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
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
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
all_feature_UD = all_feature_cond


# %% 2D distribution plot

####################################
###### Plotting Starts Here ######
####################################

# %% here's an example

toplt = all_feature_UD
xname = 'traj_peak'
yname = 'atk_ang'
if len(toplt) > 10000:
    toplt = toplt.sample(n=10000)
g = sns.displot(data= toplt,
                y=yname,x=xname,
                col='cond0', row='cond1',hue='cond1')
g.add_legend()
plt.savefig(fig_dir+f"/{yname} v {xname}.pdf",format='PDF')

# %% batch ploting
toplt = all_feature_cond

features_to_plt = [
    ('traj_peak', 'pitch_peak'),  # steering
    ('pitch_initial', 'rot_l_decel'),  # righting
    ('rot_to_max_angvel','atk_ang'),  # coordination
    ('pitch_peak','depth_chg'),
    ('additional_depth_chg', 'depth_chg'), # lift efficacy
    ('angvel_initial_phase', 'angvel_chg'),
    ('pitch_end', 'angvel_post_phase'),

]
for features in features_to_plt:
    xname, yname = features
    g = sns.displot(data=toplt,
                    y=yname,x=xname,
                    col='cond0', row='cond1',hue='cond1')
    g.add_legend()
    g.set(ylim=(np.percentile(toplt[yname],0.5),np.percentile(toplt[yname],99.5)),
      xlim=(np.percentile(toplt[xname],0.5),np.percentile(toplt[xname],99.5)))
    plt.savefig(fig_dir+f"/{yname} v {xname}.pdf",format='PDF')

# %% 
#### fin-body ratio specific
toplt = all_feature_cond
spd_bins = [3,7,100]

toplt = toplt.assign(
    spd_bins = pd.cut(toplt['spd_peak'],bins=spd_bins,labels=['slow','fast'])
)
toplt = toplt.loc[toplt['spd_bins']=='fast',:]
# toplt = toplt.groupby('cond0').sample(n=30000)
g = sns.displot(data=toplt,
                y='atk_ang',x='rot_pre_bout',
                col='cond0', row='cond1',hue='cond1')
g.set(ylim=(-40,40),xlim=(-5,10))
plt.savefig(fig_dir+"/atk_ang v rot_pre_bout _speed filtered.pdf",format='PDF')

# %%
#### IBI frequency
toplt = all_ibi_cond
g = sns.displot(data=toplt,
                x='propBoutIEI_pitch',y='y_boutFreq',
                col='cond0', row='cond1',hue='cond1')
# g.set(xlim=(-20, 40),ylim=(0,4))
plt.savefig(fig_dir+"/IEI freq pitch.pdf",format='PDF')

# %%
#### depth navigation related plots
spd_bins = np.arange(5,25,4)

toplt = all_feature_cond.assign(
    speed_bins = pd.cut(all_feature_cond['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)

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
