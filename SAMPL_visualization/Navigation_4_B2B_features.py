# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_connected_bouts)
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid

set_font_type()

# %%
pick_data = 'hc'

folder_name = f'Navi4_B2B_features' 
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
    
    
    
root, FRAME_RATE= get_data_dir(pick_data)

# get consecutive bouts
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE)

# tidy bout uid
all_features = all_features.assign(
    epoch_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str),
)
    

# %% std of directions of consecutive bouts
list_of_features = ['traj_peak', 'pitch_peak', 
                    'spd_peak',
                    'pitch_end', 'pitch_initial',
                    # 'ydispl_swim', 
                    'y_pre_swim', 'y_post_swim',
                    'x_pre_swim', 'x_post_swim',
                    'y_initial','y_end'
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts
# will be wrapped into a function

#####################
max_lag = 1 
# NOTE Here you can select the number of consecutive bouts to look at. 1-7 is recommended. Let's do 1 first to get as many data as possible
#####################

consecutive_bout_features = pd.DataFrame()

for feature_toplt in list_of_features:
    shifted_df = pd.DataFrame()
    autoCorr_res = pd.DataFrame()
    df_tocorr = pd.DataFrame()

    grouped = df_input.groupby(['cond0', 'cond1'], group_keys=False)
    col_toplt = feature_toplt
    for (cond0, cond1), group in grouped:
        shift_df = pd.concat([group[col_toplt].shift(-i).rename(f'{col_toplt}_{i}') for i in range(max_lag+1)], axis=1)
        this_df_tocorr = shift_df.groupby(group['epoch_uid'], group_keys=False).apply(
            lambda g: g.where(np.concatenate((np.flip(np.tri(len(g)), axis=0).astype(bool)[:,:min(1+max_lag, len(g))], np.zeros((len(g), max(1+max_lag-len(g),0))).astype(bool)), axis=1))
        )
        this_df_tocorr = this_df_tocorr.assign(
            cond0=cond0,
            cond1=cond1, 
            exp_uid=group['exp_uid'],
            expNum = group['expNum'],
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
        consecutive_bout_features = consecutive_bout_features.merge(long_df, on=['id', 'lag','cond0', 'cond1','exp_uid', 'expNum'])
        

# %% IBI Cumulative y displ. on Y axis after X bouts hue by bin of first bout traj
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond0', 'cond1','id']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    B2B_posture = np.append((sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values+sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values)/2, np.nan),
    B2B_swim_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values, np.nan),
    B2B_swim_xdispl = np.append(sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values, np.nan),
    B2B_bout_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['y_end'].values, np.nan),
    bouts = sel_consecutive_bouts['lag'] + 1
)
last_bout_num = sel_consecutive_bouts['lag'].unique().max()

# because the last bout in each bout series does not have 
remove_last_bout = sel_consecutive_bouts.query("lag<@last_bout_num")
remove_last_bout = remove_last_bout.assign(
#     cumu_ISIydispl = remove_last_bout.groupby(['cond0', 'cond1','id'], group_keys=False)['IBI_swim_ydispl'].apply(np.cumsum),
#     cumu_IBIydispl = remove_last_bout.groupby(['cond0', 'cond1','id'], group_keys=False)['IBI_bout_ydispl'].apply(np.cumsum),
    traj_peak_bins = pd.cut(remove_last_bout['traj_peak_first'], bins=[-90,0,20, 90], labels=['dive','flat','climb']),
    pitch_peak_bins = pd.cut(remove_last_bout['pitch_peak_first'], bins=[-90,0,20, 90], labels=['dive','flat','climb']),
)

# %% Let's plot!

# B2B_bout_xxx is calculated using the position of the fish at the end of the current bout vs the initial of the next bout
# B2B_swim_xxx is calculated using the position of the fish from the time when its speed is lower than swim threshold post current bout to when it swims faster than the threshold during the next bout

feature_to_plt = ['B2B_posture', 'B2B_swim_ydispl', 'B2B_swim_xdispl', 'B2B_bout_ydispl']

x_name = 'cond1'
gridrow = 'pitch_peak_bins'
gridcol = 'cond0'
units = 'expNum'


df_averaged = remove_last_bout.groupby([x_name, gridrow, gridcol, units]).mean().reset_index()
toplt = df_averaged
for feature in feature_to_plt:
    g = plt_categorical_grid(
        data = toplt,
        x_name = x_name,
        y_name = feature,
        gridrow = gridrow,
        gridcol = gridcol,
        units = units,
        sharey=False,
        height = 3,
        aspect = 1,
        )
    filename = os.path.join(fig_dir,f"{feature}__by{x_name}__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
# plot kde distribution 
for feature in feature_to_plt:
    g = sns.FacetGrid(remove_last_bout, 
                    # row=gridrow,
                    col=gridcol,
                    hue=x_name,
                    # sharex=False,
                    height = 3,
                    aspect = 1,
                    )
    upper = np.percentile(remove_last_bout[feature], 99.5)
    lower = np.percentile(remove_last_bout[feature], 0.5)
    g.map(sns.kdeplot, feature, alpha=0.5, clip=[lower, upper], common_norm=False, )
    g.add_legend()
    filename = os.path.join(fig_dir,f"{feature}__kde__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
    
# %%
for feature in feature_to_plt:
    upper = np.percentile(remove_last_bout[feature], 99.5)
    lower = np.percentile(remove_last_bout[feature], 0.5)
    
    p = sns.displot(data=remove_last_bout, 
                     x=feature, 
                     bins = 18, 
                     element="poly",
                     #  kde=True, 
                     stat="probability",
                     hue = x_name,
                    #  pthresh=0.05,
                     binrange=(lower,upper),
                     color='grey',
                     col=gridcol,
                     height = 3,
                     aspect = 1,
                     common_norm=False,  
                    )
    filename = os.path.join(fig_dir,f"{feature}__dis__{gridcol}X{gridrow}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %%
