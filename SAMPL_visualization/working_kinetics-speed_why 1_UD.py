'''
Very useful new function!
Plots histogram/kde of bout/IBI features. Plots 2D distribution of features.

If there are specific features you're interested in, just change the x and y in the plot functions

zeitgeber time? Yes
jackknifed? No
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
pick_data = 'for_paper'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'why_faster_more_righting'
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
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
all_ibi_cond = all_ibi_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
                                    direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,7.5,90],labels=['dive','climb']))

# get kinetics for separating up and down
all_kinetics = all_feature_cond.groupby(['cond0']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
pitch_bins = np.arange(-20,42,12)
spd_bins = np.arange(5,25,4)

all_feature_UD = pd.DataFrame()
all_feature_UD = all_feature_cond.assign(
    pre_bout_bins = pd.cut(all_feature_cond['pitch_pre_bout'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    direction = pd.cut(all_feature_cond['pitch_initial'], 
                       bins=[-51,10,91],labels=['dn','up']),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

all_feature_UD = all_feature_UD.dropna().reset_index(drop=True)

# for key, group in all_feature_cond.groupby(['cond0']):
#     this_setvalue = all_kinetics.loc[all_kinetics['cond0']==key,'set_point'].to_list()[0]
#     print(this_setvalue)
#     group['direction'] = pd.cut(group['pitch_initial'],
#                                 bins=[-91,this_setvalue,91],
#                                 labels=['dn','up'])
#     all_feature_UD = pd.concat([all_feature_UD,group])
    
# assign speed bin

# %%
# get kinetics
df_kinetics = all_feature_UD.groupby(['speed_bins','cond1']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# Plots
# %% 
# distribution of pre pitch as speed 
toplt = all_feature_UD
feature_to_plt = 'pitch_pre_bout'
g = sns.FacetGrid(data=toplt,
            col='cond0', row='cond1',
            col_order=all_cond0,
            hue='speed_bins',
            sharey =False,
            )
g.map(sns.kdeplot,feature_to_plt,common_norm=False)
g.add_legend()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')

# %%
# distribution of rot as speed 
toplt = all_feature_UD
feature_to_plt = 'rot_l_decel'
g = sns.FacetGrid(data=toplt,
            col='cond0', row='cond1',
            col_order=all_cond0,
            hue='speed_bins',
            sharey =False,
            )
g.map(sns.kdeplot,feature_to_plt,common_norm=False)
g.add_legend()
plt.savefig(fig_dir+f"/{feature_to_plt} distribution.pdf",format='PDF')

#### wider distribution of rotation, especially towards the neg, as speed increases
# %% 
# joint
toplt_07dpf  = all_feature_UD.loc[all_feature_UD['cond1']=='07dpf']
sns.jointplot(data=toplt, x="pitch_pre_bout", y="rot_l_decel", kind="kde",
              hue = 'speed_bins'
              )
# %% 
# reg separated
toplt_07cond0 = all_feature_UD.loc[all_feature_UD['cond1']=='07dpf']
sns.lmplot(data=toplt, x="pitch_pre_bout", y="rot_l_decel",
              col = 'speed_bins', markers='o'
              )
plt.savefig(fig_dir+f"/{feature_to_plt} 07dpf reg spd bins.pdf",format='PDF')

# %% 
# different ratio of pitch direction in fast vs slow?
toplt_07cond0 = all_feature_UD.loc[all_feature_UD['cond1']=='07dpf']

sns.lmplot(data=toplt_07dpf, x="pitch_pre_bout", y="rot_l_decel",
              col = 'direction', markers='o'
              )
plt.savefig(fig_dir+f"/{feature_to_plt} 07dpf reg initial dir.pdf",format='PDF')


sns.jointplot(data=toplt_07dpf, x="pitch_pre_bout", y="rot_l_decel", kind="kde",
              hue = 'direction'
              )
plt.savefig(fig_dir+f"/{feature_to_plt} 07dpf joint initial dir.pdf",format='PDF')

# %% 
# what are the ratio of up and down bout in different speed bins?
dir_count = toplt_07dpf.groupby(['speed_bins','direction']).size()
dir_count = dir_count.reset_index()
dir_count.columns = ['speed_bins','direction','count']

dir_count = dir_count.assign(
    flag = dir_count['speed_bins'].astype('int').diff(),
    total = dir_count.groupby('speed_bins')['count'].cumsum()
)
dir_count = dir_count.assign(
    percent = dir_count['count'] / dir_count['total']
)
up_percent = dir_count.loc[dir_count['flag']==0]
# %%
# up bout ratio does increase hmmm. let's see if randomly generated dataset gives the same result
# generate an artificial dataset and assign speed bins
down_bouts = toplt_07dpf.loc[toplt_07dpf['direction']=='dn']
up_bouts = toplt_07dpf.loc[toplt_07dpf['direction']=='up']
total_dir_bout_num = min(len(down_bouts), len(up_bouts))

artificial_bout_per_spd = total_dir_bout_num * 0.5  # no specific reason to choose 0.5, may want to play with the number

artificial_kinetics = pd.DataFrame()
for i in ['try1','try2','try3']:
    artificial_df = pd.DataFrame()
    for idx, row in up_percent.iterrows():
        this_df = pd.concat([
            up_bouts.sample(n=round_half_up(artificial_bout_per_spd * row['percent'])),
            down_bouts.sample(n=round_half_up(artificial_bout_per_spd * (1-row['percent']))),
            ])
        this_df = this_df.assign(speed_bins = row['speed_bins'])
        artificial_df = pd.concat(
            [artificial_df,this_df]
        )
    # get kinetics
    this_kinetics = artificial_df.groupby(['speed_bins']).apply(
                            lambda x: get_kinetics(x)
                            ).reset_index()
    this_kinetics = this_kinetics.assign(type=f'artificial_{i}')
    artificial_kinetics = pd.concat([artificial_kinetics,this_kinetics])

# %%
ori_kinetics = df_kinetics.loc[df_kinetics['cond1']=='07dpf'].assign(type='original')
# %%
kinetics_toplt = pd.concat([artificial_kinetics,ori_kinetics]).reset_index(drop=True)
# %%
toplt = kinetics_toplt
cat_cols = ['type','cond1','speed_bins']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        hue = 'type',
        x = 'speed_bins',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"{feature_toplt}_ztime_bySpd_compare.pdf")
    plt.savefig(filename,format='PDF')
    
    
    
    
    
# %%%%%%%%%%%%%





# faster bouts increase their deviation from horizontal during acceleration
# while slower bouts start correction earlier than the righting rotation
# should separate by pre_bout posture
# different ratio of pitch direction in fast vs slow?
toplt_07cond0 = all_feature_UD.loc[all_feature_UD['cond1']=='07dpf']

# %%
# what are the ratio of pre-pitch bins  in different speed bins?
pitchbin_count = toplt_07dpf.groupby(['speed_bins','pre_bout_bins']).size()
pitchbin_count = pitchbin_count.reset_index()
pitchbin_count.columns = ['speed_bins','pre_bout_bins','count']
total_bins = list(set(pitchbin_count['pre_bout_bins'].values))

pitchbin_count = pitchbin_count.assign(
    # flag = pitchbin_count['pre_bout_bins'].astype('int').diff(),
    total = pitchbin_count.groupby('speed_bins')['count'].cumsum()
)

total = pitchbin_count.groupby('speed_bins')['total'].max()

pitchbin_count['total'] = np.repeat(total,len(total_bins)).values
pitchbin_count['pre_bout angle'] = [-14,-2,10,22,34] * (len(total_bins)-1)
pitchbin_count = pitchbin_count.assign(
    percent = pitchbin_count['count'] / pitchbin_count['total']
)

# %%
sns.lineplot(data=pitchbin_count,
             x='pre_bout angle',
             y='percent',
             hue='speed_bins')
filename = os.path.join(fig_dir,f"byPreBoutBins_percentage_spd.pdf")
plt.savefig(filename,format='PDF')

# %%
# up bout ratio does increase hmmm. let's see if randomly generated dataset gives the same result
# generate an artificial dataset and assign speed bins


artificial_kinetics_byPreBout = pd.DataFrame()
for i in ['try1','try2','try3']:
    artificial_df = pd.DataFrame()
    for idx, row in pitchbin_count.iterrows():
        which_bin = row['pre_bout_bins']
        total = row['total']
        count = row['count']
        this_df = row['count']
        sel_bouts = toplt_07dpf.loc[
            (toplt_07dpf['pre_bout_bins']==which_bin)
            ].sample(n=round_half_up(count))
        artificial_df = pd.concat(
            [artificial_df,sel_bouts]
        )
    # get kinetics
    this_kinetics = artificial_df.groupby(['speed_bins']).apply(
                            lambda x: get_kinetics(x)
                            ).reset_index()
    this_kinetics = this_kinetics.assign(type=f'artificial_{i}')
    artificial_kinetics_byPreBout = pd.concat([artificial_kinetics_byPreBout,this_kinetics])

# %%
kinetics_toplt = pd.concat([artificial_kinetics_byPreBout,ori_kinetics]).reset_index(drop=True)

toplt = kinetics_toplt
cat_cols = ['type','cond1','speed_bins']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        data = toplt,
        hue = 'type',
        x = 'speed_bins',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"byPreBoutBins_{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')
# %%
