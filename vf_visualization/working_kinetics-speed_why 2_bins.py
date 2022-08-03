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
pitch_bins = np.arange(-20,42,12)
spd_bins = np.arange(5,25,4)

all_feature_UD = pd.DataFrame()
all_feature_UD = all_feature_cond.assign(
    pre_bout_bins = pd.cut(all_feature_cond['pitch_pre_bout'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    initial_bins = pd.cut(all_feature_cond['pitch_initial'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)


df_kinetics = all_feature_UD.groupby(['speed_bins','condition']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
toplt_07dpf = all_feature_UD.loc[all_feature_UD['condition']=='07dpf']
ori_kinetics = df_kinetics.loc[df_kinetics['condition']=='07dpf'].assign(type='original')

# %%
# what are the ratio of pre-pitch bins  in different speed bins?
pitchbin_count = toplt_07dpf.groupby(['speed_bins','initial_bins']).size()
pitchbin_count = pitchbin_count.reset_index()
pitchbin_count.columns = ['speed_bins','initial_bins','count']
total_bins = list(set(pitchbin_count['initial_bins'].values))

pitchbin_count = pitchbin_count.assign(
    total = pitchbin_count.groupby('speed_bins')['count'].cumsum()
)

total = pitchbin_count.groupby('speed_bins')['total'].max()

pitchbin_count['total'] = np.repeat(total,len(total_bins)).values
pitchbin_count['Initial angle'] = [-14,-2,10,22,34] * (len(total_bins)-1)
pitchbin_count = pitchbin_count.assign(
    percent = pitchbin_count['count'] / pitchbin_count['total']
)

# %%
plt.figure()
sns.lineplot(data=pitchbin_count,
             x='Initial angle',
             y='percent',
             hue='speed_bins')
filename = os.path.join(fig_dir,f"byInitialBins_percentage_spd.pdf")
plt.savefig(filename,format='PDF')


plt.figure()
sns.lineplot(data=toplt_07dpf,
             x='initial_bins',
             y='spd_peak',
            #  hue='speed_bins'
             )
filename = os.path.join(fig_dir,f"byInitialBins_raw_speed_as_pitch.pdf")
plt.savefig(filename,format='PDF')

# %%

artificial_kinetics_byPreBout = pd.DataFrame()
for i in ['try1','try2','try3']:
    artificial_df = pd.DataFrame()
    for idx, row in pitchbin_count.iterrows():
        which_bin = row['initial_bins']
        total = row['total']
        count = row['count']
        this_df = row['count']
        sel_bouts = toplt_07dpf.loc[
            (toplt_07dpf['initial_bins']==which_bin)
            ].sample(n=int(count))
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
cat_cols = ['type','condition','speed_bins']
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
    filename = os.path.join(fig_dir,f"byInitialBins_{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')
# %%

    
# %%%%%%%%%%%%%





# faster bouts increase their deviation from horizontal during acceleration
# while slower bouts start correction earlier than the righting rotation
# should separate by pre_bout posture
# different ratio of pitch direction in fast vs slow?
toplt_07dpf = all_feature_UD.loc[all_feature_UD['condition']=='07dpf']

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
            ].sample(n=int(count))
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
cat_cols = ['type','condition','speed_bins']
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
