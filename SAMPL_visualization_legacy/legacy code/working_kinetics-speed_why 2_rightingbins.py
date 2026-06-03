'''
finer pitch interval
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
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting)
from tqdm import tqdm
import matplotlib as mpl

set_font_type()
defaultPlotting()
mpl.rc('figure', max_open_warning = 0)

colors = ["#28e5ab", "#29c4cc", "#2e7acc", "#cc23bc"]
sns.set_palette(sns.color_palette())
my_palette = sns.color_palette(colors)


# %%
# Select data and create figure folder
pick_data = '7dd_all'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
# rot_bins = np.arange(-6,9,3)
rot_bins = [-6,0,8]

spd_bins = np.arange(4,24,4)

folder_name = f'why_faster_more_righting 2 decel_rotation'
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
# %% tidy data
if len(all_cond0) > 1:
    print(all_cond0)
    j = input("pick a condition: ")
    all_feature_cond = all_feature_cond.loc[all_feature_cond['cond0']==j]
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# all_ibi_cond = all_ibi_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# all_ibi_cond = all_ibi_cond.assign(y_boutFreq=1/all_ibi_cond['propBoutIEI'],
#                                     direction = pd.cut(all_ibi_cond['propBoutIEI_pitch'],[-90,7.5,90],labels=['dive','climb']))

# get kinetics 
all_kinetics = all_feature_cond.groupby(['cond1']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
# %%
# assign up and down
# rot_bins = np.arange(-20,42,12)

all_mid_angles = (np.add(rot_bins[:-1],rot_bins[1:]))/2

all_feature_UD = pd.DataFrame()
all_feature_UD = all_feature_cond.assign(
    # pre_bout_bins = pd.cut(all_feature_cond['pitch_pre_bout'],rot_bins,labels=np.arange(len(rot_bins)-1)),
    rightingRot_bins = pd.cut(all_feature_cond['rot_l_decel'],rot_bins,labels=np.arange(len(rot_bins)-1)),
    speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)
if all_feature_UD.groupby(['speed_bins','cond1']).size().min() < 20:
    spd_bins = np.arange(spd_bins.min(),spd_bins.max(),round_half_up(np.mean(np.diff(spd_bins))))
    all_feature_UD['speed_bins'] = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1))

all_feature_UD = all_feature_UD.dropna().reset_index(drop=True)

df_kinetics = all_feature_UD.groupby(['speed_bins','cond1']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
df_kinetics = df_kinetics.assign(type='original')

# %%
# what are the ratio of pre-pitch bins  in different speed bins?
rotbin_count = all_feature_UD.groupby(['speed_bins','rightingRot_bins','cond1']).size()
rotbin_count = rotbin_count.reset_index()
rotbin_count.columns = ['speed_bins','rightingRot_bins','cond1','count']
total_bins = list(set(rotbin_count['rightingRot_bins'].values))
rotbin_count = rotbin_count.sort_values(by=['cond1','speed_bins','rightingRot_bins']).reset_index(drop=True)

rotbin_count = rotbin_count.assign(
    total = rotbin_count.groupby(['speed_bins','cond1'])['count'].cumsum()
)
total = rotbin_count.groupby(['speed_bins','cond1'])['total'].max()
total = total.reset_index().sort_values(by=['cond1','speed_bins'])
rotbin_count['total'] = np.repeat(total['total'],len(total_bins)).values
rotbin_count['righting rotation'] = all_mid_angles[0:len(spd_bins)+2].tolist() * len(rotbin_count.groupby(['speed_bins','cond1']).size())
rotbin_count = rotbin_count.assign(
    percent = rotbin_count['count'] / rotbin_count['total']
)

# %%
plt.figure()
sns.catplot(data=rotbin_count,
            col='cond1',
            kind='bar',
             x='righting rotation',
             y='percent',
             palette = my_palette,
             hue='speed_bins')
filename = os.path.join(fig_dir,f"byInitialBins_percentage_spd.pdf")
plt.savefig(filename,format='PDF')


plt.figure()
sns.relplot(data=all_feature_UD,
            col='cond1',
            kind='line',
             x='rightingRot_bins',
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
    for idx, row in rotbin_count.iterrows():
        which_bin = row['rightingRot_bins']
        total = row['total']
        count = row['count']
        this_df = row['count']
        cond = row['cond1']
        sel_bouts = all_feature_UD.loc[
            (all_feature_UD['rightingRot_bins']==which_bin) & (all_feature_UD['cond1']==cond)
            ].sample(n=round_half_up(count))
        artificial_df = pd.concat(
            [artificial_df,sel_bouts]
        )
    # get kinetics
    this_kinetics = artificial_df.groupby(['speed_bins','cond1']).apply(
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
all_feature_UD = all_feature_UD.assign(type = 'original')
bouts_combined = pd.concat([artificial_bouts,all_feature_UD]).reset_index(drop=True)
kinetics_toplt = pd.concat([artificial_kinetics_byPreBout,df_kinetics]).reset_index(drop=True)

# %%
toplt = kinetics_toplt
cat_cols = ['type','cond1','speed_bins']
all_features = [c for c in toplt.columns if c not in cat_cols]

for feature_toplt in (all_features):
    g = sns.relplot(
        col='cond1',
        data = toplt,
        hue = 'type',
        palette = my_palette,
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
            col='cond1',
            hue='type',
            palette = my_palette,
            kind='line',
             x='rightingRot_bins',
             y='spd_peak',
            #  hue='speed_bins'
             )
filename = os.path.join(fig_dir,f"artificial_byInitialBins_raw_speed_as_pitch.pdf")
plt.savefig(filename,format='PDF')

# so faster more positive pitches. What about decel rotation
