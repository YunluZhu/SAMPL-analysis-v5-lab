'''
Plot averaged features (pitch, inst_traj...) categorized bt pitch up/down and speed bins
Results are jackknifed mean results across experiments (expNum)

Change all_features for the features to plot

Definition of time duration picked for averaging:
prep: bout preperation phase, -200 to -100 ms before peak speed
dur: during bout, -25 to 25 ms
post: +100 to 200 ms 
see idx_bins

'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import signal
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.get_index import get_index

from plot_functions.plt_tools import (jackknife_mean,set_font_type, day_night_split, defaultPlotting)
import matplotlib as mpl

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %%
# Select data and create figure folder
pick_data = 'tau_long'
which_ztime = 'day'
root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'Test_corr_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get bout features and IBI data
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)

# %% get aligned time series data
HEADING_LIM = 90
BIN_NUM = 4
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-int(0.3*FRAME_RATE),peak_idx+int(0.2*FRAME_RATE)]
initial_idx = peak_idx-int(0.25*FRAME_RATE)

all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()
# all_cond1 = []
# all_cond2 = []

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs(),
                                            yvel = exp_data['propBoutAligned_y'].diff()*FRAME_RATE,
                                            xvel = exp_data['propBoutAligned_x'].diff()*FRAME_RATE,
                                            linear_accel = exp_data['propBoutAligned_speed'].diff(),
                                            ang_accel_of_SMangVel = exp_data['propBoutAligned_angVel'].diff(),
                                           )
                # assign frame number, total_aligned frames per bout
                exp_data = exp_data.assign(idx=int(len(exp_data)/total_aligned)*list(range(0,total_aligned)),
                                           expNum = expNum,
                                           exp_id = condition_idx*100+expNum)
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                
                if which_ztime == 'all':
                    # day_rows = []
                    night_rows = []
                    # day_bouts = day_night_split(bout_time,'aligned_time',ztime='day').index
                    night_bouts = day_night_split(bout_time,'aligned_time',ztime='night').index
                    for i in bout_time.index:
                        rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    for i in night_bouts:
                        night_rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    # generate the ztime column
                    exp_data_ztime = exp_data.loc[rows,:].assign(ztime='day')
                    exp_data_ztime.loc[night_rows,'ztime'] = 'night'
                else:
                    for i in day_night_split(bout_time,'aligned_time',ztime=which_ztime).index:
                        rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    # generate the ztime column
                    exp_data_ztime = exp_data.loc[rows,:].assign(ztime=which_ztime)
                    
                around_peak_data = pd.concat([around_peak_data,exp_data_ztime])
  
            # # combine data from different conditions
            cond1 = all_conditions[condition_idx].split("_")[0]
            # all_cond1.append(cond1)
            cond2 = all_conditions[condition_idx].split("_")[1]
            # all_cond2.append(cond2)
            all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(dpf=cond1,
                                                                                            condition=cond2)])
all_around_peak_data = all_around_peak_data.assign(time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000)

# %% tidy feature data
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)

# get kinetics separated by dpf
all_kinetics = all_feature_cond.groupby(['dpf','condition']).apply(
                        lambda x: get_kinetics(x)
                        ).reset_index()
ctrl_kinetics = all_kinetics.loc[all_kinetics['condition']==all_cond2[0],:]


# %% tidy aligned data
# speed by peak speed
all_around_peak_data = all_around_peak_data.reset_index(drop=True)
peak_speed = all_around_peak_data.loc[all_around_peak_data.idx==peak_idx,'propBoutAligned_speed'],
pitch_initial = all_around_peak_data.loc[all_around_peak_data.idx==initial_idx,'propBoutAligned_pitch'],

grp = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))
all_around_peak_data = all_around_peak_data.assign(
                                    pitch_initial = np.repeat(pitch_initial,(idxRANGE[1]-idxRANGE[0])),
                                    peak_speed = np.repeat(peak_speed,(idxRANGE[1]-idxRANGE[0])),
                                    bout_number = grp.ngroup(),
                                )
all_around_peak_data = all_around_peak_data.assign(
                                    speed_bin = pd.cut(all_around_peak_data['peak_speed'],BIN_NUM,labels = np.arange(BIN_NUM))
                                )

# direction by set point
all_aligned_UD = pd.DataFrame()
all_around_peak_data = all_around_peak_data.assign(direction=np.nan)
for key, group in all_around_peak_data.groupby(['dpf']):
    this_setvalue = ctrl_kinetics.loc[ctrl_kinetics['dpf']==key,'set_point'].to_list()[0]
    group['direction'] = pd.cut(group['pitch_initial'],
                                bins=[-91,this_setvalue,91],
                                labels=['dn','up'])
    all_aligned_UD = pd.concat([all_aligned_UD,group])
    

# %%
# time series correlation
all_correlated = pd.DataFrame()
# df_toana = all_aligned_UD.loc[all_aligned_UD['condition']==all_cond2[0],:]
# df_toana = df_toana.loc[df_toana['dpf']==all_cond1[1],:]
df_toana = all_aligned_UD


for i, (key, group) in enumerate(df_toana.groupby(['bout_number'])):
    if i%101 == 0: # cross correlation is slow, so only calculate 1 of 10 bouts
        a = group['propBoutAligned_pitch']
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = group['propBoutAligned_instHeading']
        b = (b - np.mean(b)) / (np.std(b))
        corr = signal.correlate(a, b, 
                                mode='same')
        corr_data = pd.concat([pd.DataFrame(data={'corr':corr}), 
                            group.reset_index(drop=True)], axis=1)
        all_correlated = pd.concat([all_correlated, corr_data])
all_correlated = all_correlated.reset_index(drop=True)
all_correlated['time_ms'] = (all_correlated['idx']-peak_idx)/FRAME_RATE*1000
# %%
# if plotting below is too slow, calculate average fist
cat_col = ['direction','idx','condition','ztime','dpf']
mean_correlated = all_correlated.groupby(cat_col).mean().reset_index()
mean_correlated['time_ms'] = (mean_correlated['idx']-peak_idx)/FRAME_RATE*1000

# %%
df_to_plt = all_correlated
p = sns.relplot(
    data = df_to_plt, x = 'time_ms', y = 'corr', 
    row = 'direction',
    col='dpf',
    hue = 'condition',
    hue_order=all_cond2,
    style = 'dpf',
    style_order=all_cond1,
    # ci='sd',
    kind = 'line',aspect=3, height=4
)
p.map(plt.axvline, x=0, linewidth=1, color=".5", zorder=0)
plt.savefig(fig_dir+f"/crossCorr traj-pitch.pdf",format='PDF')

# %%
