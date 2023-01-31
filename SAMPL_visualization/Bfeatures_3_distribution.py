'''
Very useful new function!
Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

refer to the Plot section for custimization.
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

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF3_distribution_z{which_ztime}'
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


# %%

####################################
###### Plotting Starts Here ######
####################################


# Here's an example

feature_to_plt = ['pitch_initial','pitch_peak','pitch_end','traj_peak','spd_peak','rot_l_accel','rot_l_decel','atk_ang','depth_chg','additional_depth_chg'] # just to name a few
for feature in feature_to_plt:
    # kde
    toplt = all_feature_UD
    sns.displot(data=toplt,x=feature, kde=True,
                col='cond0', row='cond1',hue='cond1',
                facet_kws={'sharey':False})
    plt.savefig(fig_dir+f"/{feature} histogram.pdf",format='PDF')

    # kde
    g = sns.FacetGrid(data=all_feature_UD,
                    col='cond0',
                    row="ztime",
                    hue='cond1',
                    sharey=False,
                    )
    g.map(sns.kdeplot,feature)
    g.set(xlim=(np.percentile(all_feature_UD[feature],1), np.percentile(all_feature_UD[feature],99)))
    g.add_legend()
    plt.savefig(fig_dir+f"/{feature} kde.pdf",format='PDF')

# %% kde, separate up dn
toplt = all_feature_UD
feature_to_plt = ['rot_late_accel','pitch_peak','pitch_initial','rot_l_decel','atk_ang','traj_peak']

for feature in feature_to_plt:
    g = sns.FacetGrid(data=toplt,
                col='cond0', row="direction",hue='cond1',
                sharey=False,
                )
    g.map(sns.kdeplot,feature)
    g.add_legend()
    plt.savefig(fig_dir+f"/{feature} UD kde.pdf",format='PDF')

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
                hue='cond1',
                sharey=False,
                )
g.map(sns.kdeplot,'propBoutIEI_pitch')
g.set(xlim=(-30, 40))
g.add_legend()
plt.savefig(fig_dir+f"/IEIpitch distribution.pdf",format='PDF')
