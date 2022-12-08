# %%
from cmath import exp
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
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from tqdm import tqdm

'''
Plots single epoch that contains one or more bouts
Input directory needs to be a folder containing analyzed dlm data.
'''
# %%
FRAME_RATE = 166

which_ztime = 'day'
spd_bins = np.arange(5,30,5)

folder_dir = get_figure_dir('Fig_1')
fig_dir = os.path.join(folder_dir, folder_dir)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    print('Notes: re-writing old figures')


def extract_epochs(root):
    # below are all the properties can be plotted. 
    all_features = {
        'ang':'pitch', # (deg)',
        # 'absy':'y position (mm)'
        # 'deltaT', 
        'x':'x',
        'y':'y',
        'headx':'headx',# (mm)',
        'heady':'heady',# (mm)',
        # 'centeredAng':'centered angle (deg)',
        # 'xvel', 
        # 'yvel', 
        'dist':'distance', # (mm)',
        # 'displ':'displacement (mm)',
        'angVel':'angvel', #(deg*s-1)',
        # 'angVelSmoothed', 
        # 'angAccel':'ang accel (deg*s-2)',
        'swimSpeed':'speed',# (mm*s-1)',
        'velocity':'velocity',# (mm*s-1)'
    }

    # for each sub-folder, get the path
    all_dir = [ele[0] for ele in os.walk(root)]
    if len(all_dir) > 1:
        all_dir = all_dir[1:]

    epoch_info_all = pd.DataFrame()
    epoch_data_all = pd.DataFrame()
    for exp_num, exp_path in enumerate(all_dir):
        # get pitch                
        all_data = pd.read_hdf(f"{exp_path}/all_data.h5", key='grabbed_all')

        exp_data = all_data.loc[:,all_features.keys()]
        exp_data = exp_data.rename(columns=all_features)
        exp_data = exp_data.assign(
            exp_num = exp_num,
            epochNum = all_data['epochNum'].values,
            deltaT = all_data['deltaT'].values
        )
        
        epoch_info = exp_data.groupby('epochNum').size().reset_index()

        epoch_info = epoch_info.rename(columns={
            'epochNum':'epoch_num',
            0:'frame_num',
        })
        epoch_info.reset_index(drop=True)
        epoch_info = epoch_info.assign(
            idx = np.arange(0,len(epoch_info))+1,
            duration = epoch_info['frame_num']/FRAME_RATE,
            exp_num = exp_num,
        )
        epoch_info_all = pd.concat([epoch_info_all,epoch_info], ignore_index=True)
        epoch_data_all = pd.concat([epoch_data_all,exp_data], ignore_index=True)
        
    epoch_info_all = epoch_info_all.sort_values(by='duration',ascending=False)
    epoch_info_all = epoch_info_all.reset_index(drop=True)
    print(f'{len(epoch_info_all)} epochs detected. Sorted from long to short.')
    return epoch_data_all, epoch_info_all
# %%

data_dir = {
    "1":"/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_07dpf/7dd_AB/ab_r1",
    "2":"/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_07dpf/7dd_AB/ab_r2",
    "3":"/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_07dpf/7dd_AB/ab_r3",
}

# %%
pick_data = '1'
root = data_dir[pick_data]
epoch_data_all, epoch_info_all = extract_epochs(root)

yupper = epoch_data_all['y'].max()
ylower = epoch_data_all['y'].min()
xupper = epoch_data_all['x'].max()
xlower = epoch_data_all['x'].min()

epochs_list = np.arange(50,80)

for epoch_num in epochs_list:
    which_toplt = epoch_num
    toplt = epoch_info_all.loc[which_toplt-1,:]
    data_toplt = epoch_data_all.loc[(epoch_data_all['exp_num']==toplt['exp_num']) & (epoch_data_all['epochNum']==round_half_up(toplt['epoch_num'])), :]

    data_toplt = data_toplt.assign(
        time_s = np.cumsum(data_toplt['deltaT'])
    )

    set_font_type()
    plt.figure(figsize=(4,4))

    # for feature_toplt in tqdm(list(all_features.values())):
    p = sns.scatterplot(
        data = data_toplt, x = 'x', y = 'y', alpha = 0.2, size=2, linewidths=0,
        )
    plt.vlines(data_toplt['x'].values[0],data_toplt['y'].values[0]+1,data_toplt['y'].values[0]-1)
    plt.hlines(data_toplt['y'].values[0],data_toplt['x'].values[0]+1,data_toplt['x'].values[0]-1)

    # p.set(xlim=(xlower,xupper),
    #       ylim=(ylower,yupper))
    plt.savefig(os.path.join(fig_dir, f"{pick_data}_xy_{which_toplt}_raw.pdf"),format='PDF')
    plt.close()
# %%
pick_data = '1'
root = data_dir[pick_data]
epoch_data_all, epoch_info_all = extract_epochs(root)

downsampled_df = epoch_data_all.iloc[::10, :]

epochs_list = [4,29,40,52, 55, 69, 72]
for epoch_num in epochs_list:
    toplt = epoch_info_all.loc[epoch_num-1,:]
    data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num']) & (downsampled_df['epochNum']==round_half_up(toplt['epoch_num'])), :]

    data_toplt = data_toplt.assign(
        time_s = np.cumsum(data_toplt['deltaT'])
    )
    if epoch_num == 4:
        data_toplt = data_toplt.iloc[np.argmin(data_toplt['y']):,:]
        data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
    set_font_type()
    # plt.figure(figsize=(4,4))
    if data_toplt['x'].mean() < 0:
        data_toplt['x'] = data_toplt['x'] * -1
    p = sns.scatterplot(
        data = data_toplt, x = 'x', y = 'y', alpha = 0.2, size=2, linewidths=0,
        )
    plt.vlines(data_toplt['x'].values[0],data_toplt['y'].values[0]+1,data_toplt['y'].values[0]-1)
    plt.hlines(data_toplt['y'].values[0],data_toplt['x'].values[0]+1,data_toplt['x'].values[0]-1)
    plt.legend(labels=epoch_num)
    p.set(xlim=(xlower,xupper),
          ylim=(ylower,yupper))
plt.savefig(os.path.join(fig_dir, f"xy_combined1.pdf"),format='PDF')

# %%
pick_data = '2'
root = data_dir[pick_data]
epoch_data_all, epoch_info_all = extract_epochs(root)

downsampled_df = epoch_data_all.iloc[::10, :]

epochs_list = [7,8,10,14]
for epoch_num in epochs_list:
    toplt = epoch_info_all.loc[epoch_num-1,:]
    data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num']) & (downsampled_df['epochNum']==round_half_up(toplt['epoch_num'])), :]

    data_toplt = data_toplt.assign(
        time_s = np.cumsum(data_toplt['deltaT'])
    )
    # if epoch_num == 4:
    #     data_toplt = data_toplt.iloc[np.argmin(data_toplt['y']):,:]
    #     data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
    set_font_type()
    # plt.figure(figsize=(4,4))
    if data_toplt['x'].mean() < 0:
        data_toplt['x'] = data_toplt['x'] * -1
    p = sns.scatterplot(
        data = data_toplt, x = 'x', y = 'y', alpha = 0.2, size=2, linewidths=0,
        )
    plt.vlines(data_toplt['x'].values[0],data_toplt['y'].values[0]+1,data_toplt['y'].values[0]-1)
    plt.hlines(data_toplt['y'].values[0],data_toplt['x'].values[0]+1,data_toplt['x'].values[0]-1)

    p.set(xlim=(xlower,xupper),
          ylim=(ylower,yupper))
plt.savefig(os.path.join(fig_dir, f"xy_combined2.pdf"),format='PDF')

# %%
pick_data = '1'
root = data_dir[pick_data]
epoch_data_all, epoch_info_all = extract_epochs(root)

downsampled_df = epoch_data_all.iloc[::10, :]

epochs_list = [40]
for epoch_num in epochs_list:
    toplt = epoch_info_all.loc[epoch_num-1,:]
    data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num']) & (downsampled_df['epochNum']==round_half_up(toplt['epoch_num'])), :]

    data_toplt = data_toplt.assign(
        time_s = np.cumsum(data_toplt['deltaT'])
    )
    # if epoch_num == 4:
    #     data_toplt = data_toplt.iloc[np.argmin(data_toplt['y']):,:]
    #     data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
    set_font_type()
    # plt.figure(figsize=(4,4))
    if data_toplt['x'].mean() < 0:
        data_toplt['x'] = data_toplt['x'] * -1
    p = sns.scatterplot(
        data = data_toplt, x = 'x', y = 'y', alpha = 0.2, size=2, linewidths=0,
        )
    plt.vlines(data_toplt['x'].values[0],data_toplt['y'].values[0]+1,data_toplt['y'].values[0]-1)
    plt.hlines(data_toplt['y'].values[0],data_toplt['x'].values[0]+1,data_toplt['x'].values[0]-1)

    p.set(xlim=(xlower,xupper),
          ylim=(ylower,yupper))
plt.savefig(os.path.join(fig_dir, f"xy_combined2.pdf"),format='PDF')
# %%
