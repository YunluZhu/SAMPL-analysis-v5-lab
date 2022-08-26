'''
finer pitch interval
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
defaultPlotting()
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
# all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)
# %% tidy data
if len(all_cond1) > 1:
    print(all_cond1)
    j = input("pick a condition: ")
    all_feature_cond = all_feature_cond.loc[all_feature_cond['dpf']==j]
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_ibi_cond = all_ibi_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
#                                     direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,7.5,90],labels=['dive','climb']))

# get kinetics 
all_kinetics = all_feature_cond.groupby(['condition']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
pitch_bins = np.arange(-18,39,8)
# pitch_bins = [-20,-5,0,10,25,40]
spd_bins = np.arange(5,25,4)
all_mid_angles = (np.add(pitch_bins[:-1],pitch_bins[1:]))/2

all_feature_UD = pd.DataFrame()
all_feature_UD = all_feature_cond.assign(
    pre_bout_bins = pd.cut(all_feature_cond['pitch_pre_bout'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    initial_bins = pd.cut(all_feature_cond['pitch_initial'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)
if all_feature_UD.groupby(['speed_bins','condition']).size().min() < 20:
    spd_bins = np.arange(spd_bins.min(),spd_bins.max(),int(np.mean(np.diff(spd_bins))))
    all_feature_UD['speed_bins'] = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1))

all_feature_UD = all_feature_UD.dropna().reset_index(drop=True)

df_kinetics = all_feature_UD.groupby(['speed_bins','condition']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
df_kinetics = df_kinetics.assign(type='original')

# %%
# what are the ratio of pre-pitch bins  in different speed bins?
pitchbin_count = all_feature_UD.groupby(['speed_bins','initial_bins','condition']).size()
pitchbin_count = pitchbin_count.reset_index()
pitchbin_count.columns = ['speed_bins','initial_bins','condition','count']
total_bins = list(set(pitchbin_count['initial_bins'].values))
pitchbin_count = pitchbin_count.sort_values(by=['condition','speed_bins','initial_bins']).reset_index(drop=True)

pitchbin_count = pitchbin_count.assign(
    total = pitchbin_count.groupby(['speed_bins','condition'])['count'].cumsum()
)
total = pitchbin_count.groupby(['speed_bins','condition'])['total'].max()
total = total.reset_index().sort_values(by=['condition','speed_bins'])
pitchbin_count['total'] = np.repeat(total['total'],len(total_bins)).values


# pitch bin value average
pitch_bin_average = all_feature_UD.groupby(['condition','speed_bins','initial_bins'])['pitch_initial'].mean().reset_index()
pitch_bin_average.columns = ['condition','speed_bins','initial_bins','mean_pitch']
pitch_bin_average = pitch_bin_average.sort_values(by=['condition','speed_bins'])

pitchbin_count['Initial angle'] = pitch_bin_average['mean_pitch'].values
pitchbin_count = pitchbin_count.assign(
    percent = pitchbin_count['count'] / pitchbin_count['total']
)

pitch_bin_mapping = all_feature_UD.groupby(['initial_bins'])['pitch_initial'].mean().to_dict()
all_feature_UD = all_feature_UD.assign(
    pitch_bin_mapped = all_feature_UD['initial_bins'].map(pitch_bin_mapping)
)
# %%
plt.figure()
sns.relplot(data=pitchbin_count,
            col='condition',
            kind='line',
             x='Initial angle',
             y='percent',
             hue='speed_bins')
filename = os.path.join(fig_dir,f"byInitialBins_percentage_spd.pdf")
plt.savefig(filename,format='PDF')


plt.figure()
sns.relplot(data=all_feature_UD,
            col='condition',
            kind='line',
             x='pitch_bin_mapped',
             y='spd_peak',
            #  hue='speed_bins'
             )
filename = os.path.join(fig_dir,f"byInitialBins_raw_speed_as_pitch.pdf")
plt.savefig(filename,format='PDF')

# %%

artificial_kinetics_byPreBout = pd.DataFrame()
artificial_bouts = pd.DataFrame()
for i in ['try1','try2','try3']:
    artificial_df = pd.DataFrame()
    for idx, row in pitchbin_count.iterrows():
        which_bin = row['initial_bins']
        total = row['total']
        count = row['count']
        this_df = row['count']
        cond = row['condition']
        sel_bouts = all_feature_UD.loc[
            (all_feature_UD['initial_bins']==which_bin) & (all_feature_UD['condition']==cond)
            ].sample(n=int(count))
        artificial_df = pd.concat(
            [artificial_df,sel_bouts]
        )
    # get kinetics
    this_kinetics = artificial_df.groupby(['speed_bins','condition']).apply(
                            lambda x: get_kinetics(x)
                            ).reset_index()
    this_kinetics = this_kinetics.assign(
        type=i,
        )
    artificial_kinetics_byPreBout = pd.concat([artificial_kinetics_byPreBout,this_kinetics])

    artificial_df = artificial_df.assign(
        type = i,
    )
    artificial_bouts = pd.concat([artificial_bouts,artificial_df])
# %%
all_feature_UD = all_feature_UD.assign(type = 'original')
bouts_combined = pd.concat([artificial_bouts,all_feature_UD]).reset_index(drop=True)
kinetics_toplt = pd.concat([artificial_kinetics_byPreBout,df_kinetics]).reset_index(drop=True)

# %%
toplt = kinetics_toplt
cat_cols = ['type','condition','speed_bins']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        col='condition',
        data = toplt,
        hue = 'type',
        x = 'speed_bins',
        y = feature_toplt,
        kind = 'line',
        marker = True,
    )
    filename = os.path.join(fig_dir,f"byInitialBins_{feature_toplt}_compare.pdf")
    plt.savefig(filename,format='PDF')

# %%%%%%%%%%%%%
# does speed distribution remain the same?
plt.figure()
sns.relplot(data=bouts_combined,
            col='condition',
            hue='type',
            kind='line',
             x='initial_bins',
             y='spd_peak',
            #  hue='speed_bins'
             )
filename = os.path.join(fig_dir,f"artificial_byInitialBins_raw_speed_as_pitch.pdf")
plt.savefig(filename,format='PDF')

# %%
# so faster more positive pitches. What about decel rotation

# %%
