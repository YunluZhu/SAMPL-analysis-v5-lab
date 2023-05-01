'''
Plot depth change of consecutive bout during bout phase
Plot depth change of consecutive bout during Inter-swim/bout phase
separated in hues by initial pitch angle 


NOTE variables to keep an eye on:

pick_data = 'wt_fin' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 6 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 


'''

# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.plt_tools import (set_font_type)

set_font_type()

##### Parameters to change #####

pick_data = 'hc' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 6 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 

##### Parameters to change #####

# %%

root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'Navi_cumuDepth_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()


# %
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE)
all_features = all_features.assign(
    epoch_uid = all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_uid = all_features['cond1'] + all_features['expNum'].astype(str),
)
    

# %% std of directions of consecutive bouts
list_of_features = ['traj_peak', 'pitch_peak', 'spd_peak',
                    'pitch_end', 'pitch_initial',
                    'ydispl_swim','y_pre_swim','y_post_swim',
                    'y_initial','y_end'
                    ]

df_input = all_features.groupby(['epoch_uid']).filter(lambda g: len(g)>1)

sel_features = df_input

# %% Connect consecutive bouts
max_lag = consecutive_bout_num - 1
consecutive_bout_features = pd.DataFrame()

for feature_toplt in list_of_features:
    shifted_df = pd.DataFrame()
    autoCorr_res = pd.DataFrame()
    df_tocorr = pd.DataFrame()

    grouped = sel_features.groupby(['cond1','cond0'])
    col_toplt = feature_toplt
    for (cond1, cond0), group in grouped:
        shift_df = pd.concat([group[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
        this_df_tocorr = shift_df.groupby(group['epoch_uid']).apply(
            lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
        )
        this_df_tocorr = this_df_tocorr.assign(
            cond1=cond1, 
            cond0=cond0,
            exp_uid=group['exp_uid']
            )
        df_tocorr = pd.concat([df_tocorr, this_df_tocorr], ignore_index=True)

    sel_bouts = df_tocorr
    sel_bouts = sel_bouts.loc[sel_bouts[f'{feature_toplt}_{max_lag}'].notna()]
    sel_bouts = sel_bouts.assign(
        first_bout = sel_bouts[f'{feature_toplt}_0'],
    )
    sel_bouts['id'] = sel_bouts.index
    long_df = pd.wide_to_long(sel_bouts, stubnames=feature_toplt, sep='_', j='lag', i='id').reset_index()
    long_df = long_df.loc[long_df['lag']<=max_lag]
    
    long_df = long_df.rename(columns={'first_bout': f'{feature_toplt}_first'})
    
    if consecutive_bout_features.empty:
        consecutive_bout_features = long_df
    else:
        consecutive_bout_features = consecutive_bout_features.merge(long_df, on=['id', 'lag','cond1','cond0','exp_uid'], how='left')

# %%
pitch_bins = [-90,0,20, 60]

sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond1','cond0','id','lag']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    cumu_swim_ydispl = sel_consecutive_bouts.groupby(['cond1','cond0','id'])['ydispl_swim'].apply(np.cumsum),
    pitch_peak_bins = pd.cut(sel_consecutive_bouts['pitch_peak_first'], bins=pitch_bins, labels=['dive','flat','climb']),
    bouts = sel_consecutive_bouts['lag'] + 1
)
# 
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    IBI_swim_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values, np.nan),
    IBI_bout_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['y_end'].values, np.nan),
)
last_bout_num = sel_consecutive_bouts['lag'].unique().max()
remove_last_bout = sel_consecutive_bouts.query("lag<@last_bout_num")
remove_last_bout = remove_last_bout.assign(
    cumu_ISIydispl = remove_last_bout.groupby(['cond1','cond0','id'])['IBI_swim_ydispl'].apply(np.cumsum),
    cumu_IBIydispl = remove_last_bout.groupby(['cond1','cond0','id'])['IBI_bout_ydispl'].apply(np.cumsum),
)

sns.relplot(
    data=sel_consecutive_bouts,
    y='cumu_swim_ydispl',
    x='bouts',
    kind='line',
    col='cond1',
    row='cond0',
    hue='pitch_peak_bins',
    height=3
)
plt.savefig(os.path.join(fig_dir, f"bouts cumu ydispl.pdf"),format='PDF')

# %% IBI Cumulative y displ. on Y axis after X bouts hue by bin of first bout traj

sns.relplot(
    data=remove_last_bout,
    y='cumu_ISIydispl',
    x='bouts',
    kind='line',
    col='cond1',
    row='cond0',
    hue='pitch_peak_bins',
    height=3,
    
)
plt.savefig(os.path.join(fig_dir, f"ISI cumu ydispl.pdf"),format='PDF')

sns.relplot(
    data=remove_last_bout,
    y='cumu_IBIydispl',
    x='bouts',
    kind='line',
    col='cond1',
    row='cond0',
    hue='pitch_peak_bins',
    height=3
)
plt.savefig(os.path.join(fig_dir, f"IBI cumu ydispl.pdf"),format='PDF')

# %%
