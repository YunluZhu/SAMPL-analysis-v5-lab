# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, )
from tqdm import tqdm
from scipy.signal import savgol_filter


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
        'xvel':'xvel', 
        'yvel':'yvel', 
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
    return epoch_data_all, epoch_info_all

# %%

# %%
root = '/Volumes/LabDataPro/SAMPL_data_v5/WT_daylight_2025 fullepoch/LD epoch'
set_font_type()
epoch_data_all, epoch_info_all = extract_epochs(root)
# %%

sec5frames = int(166 * 5 /5)

downsampled_df = epoch_data_all.iloc[::5, :]

#%%



epoch_combined = pd.DataFrame()
for epoch_num in [17449, 27299, 18694,9487]:
    toplt = epoch_info_all.loc[epoch_info_all['epoch_num']==epoch_num,:]
    data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num'].values[0]) & (downsampled_df['epochNum']==epoch_num), :]

    if epoch_num == 17449:
        data_toplt = data_toplt.iloc[
            40:40+sec5frames
        ]
        data_toplt['x'] = data_toplt['x'] * -1
        
    
    if epoch_num == 27299:
        i0 = 10
        print(len(data_toplt))
        data_toplt = data_toplt.iloc[
            i0:i0+sec5frames
        ]        
        data_toplt['x'] = data_toplt['x'] * -1
        
    
    if epoch_num == 9487:
        i0 = 10
        data_toplt = data_toplt.iloc[
            i0:i0+sec5frames
        ]        
        data_toplt['x'] = data_toplt['x'] * -1

    if epoch_num == 18694:
        i0 = 70
        data_toplt = data_toplt.iloc[
            i0:i0+sec5frames
        ]        

    data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
    data_toplt = data_toplt.assign(
        epoch_num = epoch_num
    )
    epoch_combined = pd.concat([epoch_combined,data_toplt],ignore_index=True)



plt.figure(figsize=(4,4))

p = sns.scatterplot(
    data = epoch_combined, x = 'x', y = 'y', alpha = 0.5, linewidths=0,legend='full',
    s=10,
    # hue='epoch_num',
    )
plt.axis('equal')
p.set(
    xlim=(-2,12),
    ylim=(-6,10),
)
plt.savefig(os.path.join(fig_dir, f"xy_combined LD.pdf"),format='PDF')

# %%

# %%
root = '/Volumes/LabDataPro/SAMPL_data_v5/WT_daylight_2025 fullepoch/DD epoch'
set_font_type()
epoch_data_all, epoch_info_all = extract_epochs(root)
# %%

sec5frames = int(166 * 5 /5)

downsampled_df = epoch_data_all.iloc[::5, :]

#%%

#  [62133, 58213, 36133, 3257]
# epochs_list = [4040, 36133, 17216, 3257,30892]


epoch_combined = pd.DataFrame()
for epoch_num in [1154, 171,4846, 482]:
    toplt = epoch_info_all.loc[epoch_info_all['epoch_num']==epoch_num,:]
    data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num'].values[0]) & (downsampled_df['epochNum']==epoch_num), :]

    
    if epoch_num == 1154:
        i0 = 220
        if i0+sec5frames < len(data_toplt):
            data_toplt = data_toplt.iloc[
                i0:i0+sec5frames
            ]     
        else:
            break
            
    if epoch_num == 171:
        i0 = 80
        if i0+sec5frames < len(data_toplt):
            data_toplt = data_toplt.iloc[
                i0:i0+sec5frames
            ]     
        else:
            break
    if epoch_num == 4846:
        i0 = 240
        if i0+sec5frames < len(data_toplt):
            data_toplt = data_toplt.iloc[
                i0:i0+sec5frames
            ]     
        else:
            break
    if epoch_num == 482:
        i0 = 20
        if i0+sec5frames < len(data_toplt):
            data_toplt = data_toplt.iloc[
                i0:i0+sec5frames
            ]     
        else:
            break

    data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
    data_toplt = data_toplt.assign(
        epoch_num = epoch_num
    )
    epoch_combined = pd.concat([epoch_combined,data_toplt],ignore_index=True)

plt.figure(figsize=(4,4))

p = sns.scatterplot(
    data = epoch_combined, x = 'x', y = 'y', alpha = 0.5, linewidths=0,legend='full',
    s=10,
    # hue='epoch_num',
    )
plt.axis('equal')
p.set(
    xlim=(-2,12),
    ylim=(-6,10),
)
plt.savefig(os.path.join(fig_dir, f"xy_combined DD.pdf"),format='PDF')

########################
#%%
epoch_i = 1200
epoch_info_all = epoch_info_all.sort_values(by='duration',ascending=False)
epoch_info_all = epoch_info_all.reset_index(drop=True)
epochs_list = epoch_info_all.epoch_num.unique()

# %%
epoch_i += 1
epoch_num = epochs_list[epoch_i]
print(epoch_num)
toplt = epoch_info_all.loc[epoch_info_all['epoch_num']==epoch_num,:]
data_toplt = downsampled_df.loc[(downsampled_df['exp_num']==toplt['exp_num'].values[0]) & (downsampled_df['epochNum']==epoch_num), :]

data_toplt = data_toplt.assign(
    time_s = np.cumsum(data_toplt['deltaT'])
)

data_toplt[['x','y']] = data_toplt[['x','y']] - data_toplt[['x','y']].values[0]
data_toplt = data_toplt.assign(
    epoch_num = epoch_num
)

p = sns.scatterplot(
    data = data_toplt, x = 'x', y = 'y', alpha = 0.2, linewidths=0,legend=None,
    s=10,
    )
plt.axis('equal')
p.set(
    xlim=(-2,12),
    ylim=(-10,10),
)
print(f"total time {data_toplt.deltaT.sum() * 4}")
# %%

# %%

# %%

# %%
