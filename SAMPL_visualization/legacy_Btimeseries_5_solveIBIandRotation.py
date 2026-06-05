'''
plot correlation of features by time
1. corr of angular vel at each timpoint with preBoutPitch / atkAngle / trajectory deviation
2. corr of ins. trajectory at each timepoint with bout trajectory
3. other correlations
trajectory deviation (trajecgtory residual) is defined as (bout_trajecgtory - pitch_pre_bout)

NOTE bout direction and speed are not separated
'''

#%%
# import sys
import os,glob
from pickle import FRAME
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from plot_functions.get_bout_features import extract_bout_features_v5
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from plot_functions.plt_tools import (set_font_type, defaultPlotting, distribution_binned_average, day_night_split)
from tqdm import tqdm
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)
import matplotlib as mpl
from scipy.signal import savgol_filter

##### Parameters to change #####
pick_data = 'wt_dl' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' or 'night', does not support 'all'
if_strict_daynight = True
##### Parameters to change #####

# %%
def corr_calc(df, grp_cols, col1, col2, name):
    corr_calc = df.groupby(grp_cols).apply(
            lambda y: stats.pearsonr(
                y[col1].values,y[col2].values)[0]
                )
    corr_calc.name = name
    output = corr_calc.to_frame()
    return output

# %%
# Paste root directory here
root, FRAME_RATE= get_data_dir(pick_data)

folder_name = f'BT5_IBIandRotation'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('Notes: re-writing old figures')
set_font_type()
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.30*FRAME_RATE),peak_idx+round_half_up(0.22*FRAME_RATE)]
spd_bins = np.arange(5,25,4)

# %% features for plotting
all_features = [
    'propBoutAligned_speed', 
    'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    'ang_speed',
    'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    # 'xvel', 'yvel',
]

# %%
# CONSTANTS
# %%
T_INITIAL = -0.25 #s
T_PREP_200 = -0.2
T_PREP_150 = -0.15
T_PRE_BOUT = -0.10 #s
T_POST_BOUT = 0.1 #s
T_post_150 = 0.15
T_END = 0.2
T_MID_ACCEL = -0.05
T_MID_DECEL = 0.05
idxRANGE_features = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.25*FRAME_RATE)]


idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = round_half_up(peak_idx + T_POST_BOUT * FRAME_RATE)
idx_mid_accel = round_half_up(peak_idx + T_MID_ACCEL * FRAME_RATE)
idx_mid_decel = round_half_up(peak_idx + T_MID_DECEL * FRAME_RATE)
idx_end = round_half_up(peak_idx + T_END * FRAME_RATE)

idx_dur250ms = round_half_up(250/1000*FRAME_RATE)
idx_dur275ms = round_half_up(275/1000*FRAME_RATE)
# %%
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

all_around_peak_data = pd.DataFrame()
all_bout_features = pd.DataFrame()
all_cond0 = []
all_cond1 = []

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data_ = []
            bout_features_ = []
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch                

                ## get attributes
                all_attributes = pd.read_hdf(f"{exp_path}/bout_data.h5", key='bout_attributes')
                attributes = all_attributes[all_attributes['if_align']]
                attributes = attributes.assign(exp_uid = (condition_idx+1)*100+(expNum+1))
                # Identify where the epochNum changes
                attributes['epochChange'] = (attributes['epochNum'] != attributes['epochNum'].shift()).cumsum()

                # Create a unique epoch ID that considers these breaks
                attributes['uniqueEpoch'] = attributes.groupby('epochChange').ngroup()

                # Drop the helper column if not needed
                attributes.drop(columns=['epochChange'], inplace=True)
                
                attributes = attributes.assign(
                    bout_uid = attributes['exp_uid'].astype('int').astype('str')+\
                        '_'+\
                            attributes['uniqueEpoch'].astype('int').astype('str')+\
                        '_'+\
                            attributes.index.astype('str'),
                    epoch_uid = attributes['exp_uid'].astype('int').astype('str')+\
                        '_'+\
                            attributes['uniqueEpoch'].astype('int').astype('str')
                ).reset_index(drop=True)
                
                
                ## get aligned time series
                exp_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')
                exp_data = exp_data.assign(ang_speed=exp_data['propBoutAligned_angVel'].abs(),
                                            yvel = exp_data['propBoutAligned_y'].diff()*FRAME_RATE,
                                            xvel = exp_data['propBoutAligned_x'].diff()*FRAME_RATE,
                                            # linear_accel = exp_data['propBoutAligned_speed'].diff(),
                                            ang_accel = np.diff(savgol_filter(exp_data['propBoutAligned_angVel'],11,3),prepend=np.array([np.nan]))*FRAME_RATE,
                                            tsp = exp_data['propBoutAligned_instHeading'] - exp_data['propBoutAligned_pitch']
                                           )
                
                # assign frame number, total_aligned frames per bout
                exp_data = exp_data.assign(idx=round_half_up(len(exp_data)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                ori_bout_data = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,:]

                
                # for i in bout_time.index:
                # # if only need day or night bouts:
                index_for_aligned_rows = day_night_split(ori_bout_data,'aligned_time',ztime=which_ztime, narrow_bin=if_strict_daynight).index
                for i in index_for_aligned_rows:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                    
                    
                exp_data_to_keep = exp_data.loc[rows,:]
                attributes_daynight_split = day_night_split(attributes,'propBout_time',ztime=which_ztime, narrow_bin=if_strict_daynight)
                
                exp_data_to_keep = exp_data_to_keep.assign(
                    bout_time = np.repeat(ori_bout_data.loc[index_for_aligned_rows,'aligned_time'].values,idxRANGE[1]-idxRANGE[0]),
                    expNum = expNum,
                    bout_uid = np.repeat(attributes_daynight_split['bout_uid'].values, idxRANGE[1]-idxRANGE[0]),
                    epoch_uid = np.repeat(attributes_daynight_split['epoch_uid'].values, idxRANGE[1]-idxRANGE[0]),
                    exp = exp,
                )
                
                around_peak_data_.append(exp_data_to_keep)
                
     
                ###################### get connected bouts

                # what's the next bout (row)
                to_bout_list = attributes.loc[1:,'bout_uid'].values
                to_bout_list = np.append(to_bout_list, None)
                # whether the next row is in a new epoch
                if_between_epochs = attributes.epochNum.diff().astype('bool')
                if_last_in_epoch = np.append(if_between_epochs[1:], True)
                # dis connect bouts between epochs by setting the last epoch's next bout to None
                to_bout_list[if_last_in_epoch] = None
                
                IBI_swim_after = (attributes.loc[1:,'swim_start_idx'].values - attributes.loc[:attributes.index.max()-1,'swim_end_idx'].values)/FRAME_RATE
                IBI_swim_after = np.append(IBI_swim_after, np.nan)
                IBI_swim_after[if_last_in_epoch] = None
                IBI_swim_before = np.append(np.nan, IBI_swim_after)[:-1]
                    
                IBI_align_after = (attributes.loc[1:,'peak_idx'].values - attributes.loc[:attributes.index.max()-1,'peak_idx'].values)/FRAME_RATE - 0.45 # 450 is the aligned bout duration
                IBI_align_after = np.append(IBI_align_after, np.nan)
                IBI_align_after[if_last_in_epoch] = None
                IBI_align_before = np.append(np.nan, IBI_align_after)[:-1]                    
                ###################### get bout features
                rows_features = []
                for i in ori_bout_data.aligned_time.index:
                    rows_features.extend(list(range(i*total_aligned+round_half_up(idxRANGE_features[0]),i*total_aligned+round_half_up(idxRANGE_features[1]))))
                # # assign bout numbers
                trunc_exp_data = exp_data.loc[rows_features,:]
                trunc_exp_data = trunc_exp_data.assign(
                    bout_num = trunc_exp_data.groupby(np.arange(len(trunc_exp_data))//(idxRANGE_features[1]-idxRANGE_features[0])).ngroup()
                )
                this_exp_features = extract_bout_features_v5(trunc_exp_data,peak_idx,FRAME_RATE)
                
                this_exp_features = this_exp_features.assign(
                    bout_time = ori_bout_data.aligned_time.values,
                    expNum = expNum,
                    bout_uid = attributes['bout_uid'].values,
                    epoch_uid = attributes['epoch_uid'].values,
                    to_bout = to_bout_list,
                    post_IBI_time = IBI_swim_after,
                    pre_IBI_time = IBI_swim_before,
                    post_IBI_align_time = IBI_align_after,
                    pre_IBI_align_time = IBI_align_before,
                    exp = exp,
                )
                # day night split. also assign ztime column
                this_ztime_exp_features = day_night_split(this_exp_features,'bout_time',narrow_bin=if_strict_daynight, ztime=which_ztime)
                
                bout_features_.append(this_ztime_exp_features)
                
                
    # combine data from different conditions
    cond0 = all_conditions[condition_idx].split("_")[0]
    all_cond0.append(cond0)
    cond1 = all_conditions[condition_idx].split("_")[1]
    all_cond1.append(cond1)
    
    all_around_peak_data = pd.concat([all_around_peak_data, pd.concat(around_peak_data_, ignore_index=True).
                                      assign(cond0=cond0,cond1=cond1)],ignore_index=True)
    all_bout_features = pd.concat([all_bout_features, pd.concat(bout_features_, ignore_index=True).
                                      assign(cond0=cond0,cond1=cond1)],ignore_index=True)

    
all_around_peak_data = all_around_peak_data.assign(
    time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000,
)
# %% tidy data
all_cond0 = list(set(all_cond0))
all_cond0.sort()
all_cond1 = list(set(all_cond1))
all_cond1.sort()

all_features = all_bout_features.assign(
    bout_uuid = all_bout_features['cond0'] + all_bout_features['cond1'] + all_bout_features['expNum'].astype(str) + all_bout_features['epoch_uid'] + all_bout_features['bout_uid'],
    epoch_uid = all_bout_features['cond0'] + all_bout_features['cond1'] + all_bout_features['expNum'].astype(str) + all_bout_features['epoch_uid'],
    exp_uid = all_bout_features['cond0'] + all_bout_features['cond1'] + all_bout_features['expNum'].astype(str),
)
all_aligned = all_around_peak_data.assign(
    bout_uuid = all_around_peak_data['cond0'] + all_around_peak_data['cond1'] + all_around_peak_data['expNum'].astype(str) + all_around_peak_data['epoch_uid'] + all_around_peak_data['bout_uid'],
    epoch_uid = all_around_peak_data['cond0'] + all_around_peak_data['cond1'] + all_around_peak_data['expNum'].astype(str) + all_around_peak_data['epoch_uid'],
    exp_uid = all_around_peak_data['cond0'] + all_around_peak_data['cond1'] + all_around_peak_data['expNum'].astype(str),
)

# %%
# select bouts that has pre_IBI_time not na
all_features_sel = all_features.loc[all_features['post_IBI_time'].notna() & all_features['pre_IBI_time'].notna(),:]
all_aligned_sel = all_aligned.loc[all_aligned['bout_uuid'].isin(all_features_sel['bout_uuid']),:]
#%%

#####################
max_lag = 2
#####################

list_of_features = [
    'traj_peak',
    # 'bout_trajectory_Pre2Post',
    # 'y_post_swim','y_pre_swim', 'ydispl_swim', 
    # 'y_end','y_initial',
    # 'post_IBI_align_time',
    'post_IBI_time',
    # 'pre_IBI_align_time',
    'pre_IBI_time',
    'spd_peak',
    'pitch_end', 
    'pitch_initial',
    'rot_total',
    'rot_full_accel',
    'rot_l_decel',
    'bout_uuid',
    'epoch_uid',
    'exp_uid',
                    ]

consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)

# %
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond0', 'cond1','ztime','id']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    # rotation
    post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),
    bouts = sel_consecutive_bouts['lag'] + 1,
)
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)
middle_bout_df = middle_bout_df.loc[middle_bout_df['ztime'].isin(['day','night'])].reset_index(drop=True)

middle_bout_df = middle_bout_df.assign(
    traj_deviation = middle_bout_df['traj_peak']- middle_bout_df['pitch_initial'],
)


#%%
# comine bout features
all_aligned_sel_feature = all_aligned_sel.merge(
    middle_bout_df[['bout_uuid','pre_B2B_rot','post_B2B_rot','bouts','pre_IBI_time','traj_deviation','pitch_initial','rot_total','rot_full_accel','rot_l_decel']],
    on=['bout_uuid'],
    how='inner'
)

all_aligned_sel_feature['IBI_threshold'] = 1.7
all_aligned_sel_feature['IBI_cat'] = 'long_IBI'
all_aligned_sel_feature.loc[all_aligned_sel_feature[f'pre_IBI_time'] <= all_aligned_sel_feature['IBI_threshold'], 'IBI_cat'] = 'short_IBI'

sel_all_aligned_sel_feature = all_aligned_sel_feature.loc[all_aligned_sel_feature['IBI_cat'].isin(['long_IBI'])]
sel_all_aligned_sel_feature = sel_all_aligned_sel_feature.assign(
    trajDev_cat = pd.cut(sel_all_aligned_sel_feature['traj_deviation'], bins=[-np.inf,0,10, np.inf], labels=['dev_down','dev_flat','dev_up']),
    pitchInitial_cat = pd.cut(sel_all_aligned_sel_feature['pitch_initial'], bins=[-np.inf,-10, 0, np.inf], labels=['initial_dn','initial_slight_dn', 'initial_up']),
    boutRot_cat = pd.cut(sel_all_aligned_sel_feature['rot_total'], bins=[-np.inf,0, 12, np.inf], labels=['rot_dn','rot_slight_up', 'rot_up']),
    IBIRot_cat = pd.cut(sel_all_aligned_sel_feature['pre_B2B_rot'], bins=[-np.inf,-11, np.inf], labels=['largeIBI_noseDN','smallIBI']),
    rot_consistency = pd.cut(
        sel_all_aligned_sel_feature['rot_full_accel']*sel_all_aligned_sel_feature['rot_l_decel'], bins=[-np.inf,0, np.inf], labels=['diff_dir','same_dir']),

)


#%%

corr_dict = {
    'Steering correlation':['traj_deviation','propBoutAligned_angVel'],
    'Righting correlation':['pitch_initial','propBoutAligned_angVel'],
    'IBI correlation':['pre_B2B_rot','propBoutAligned_angVel'],
    # 'angVel_corr_postIBIrot':['post_B2B_rot','propBoutAligned_angVel'],
}

corr_all_ = []

for (cond0, cond1, expNum, IBI_cat), group in all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum','IBI_cat']):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            IBI_cat = IBI_cat,
        )
        corr_all_.append(corr_res)
    
corr_all = pd.concat(corr_all_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

# %%

####################################
###### Plotting Starts Here ######
####################################

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_all,
    kind='line',
    col='cat',
    row='cond1',
    hue='IBI_cat',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/all correlations.pdf",format='PDF')
plt.show()



#%%


which_cat = 'trajDev_cat'

corr_sel_ = []

for (cond0, cond1, expNum, some_cat), group in sel_all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum',which_cat]):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            by_category = some_cat,
        )
        corr_sel_.append(corr_res)
    
corr_sel = pd.concat(corr_sel_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_sel,
    kind='line',
    col='cat',
    row='cond1',
    hue='by_category',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/long IBI corr by {which_cat}.pdf",format='PDF')
plt.show()


#%%

which_cat = 'pitchInitial_cat'

corr_sel_ = []

for (cond0, cond1, expNum, some_cat), group in sel_all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum',which_cat]):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            by_category = some_cat,
        )
        corr_sel_.append(corr_res)
    
corr_sel = pd.concat(corr_sel_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_sel,
    kind='line',
    col='cat',
    row='cond1',
    hue='by_category',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/long IBI corr by {which_cat}.pdf",format='PDF')
plt.show()

#%%

which_cat = 'boutRot_cat'

corr_sel_ = []

for (cond0, cond1, expNum, some_cat), group in sel_all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum',which_cat]):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            by_category = some_cat,
        )
        corr_sel_.append(corr_res)
    
corr_sel = pd.concat(corr_sel_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_sel,
    kind='line',
    col='cat',
    row='cond1',
    hue='by_category',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/long IBI corr by {which_cat}.pdf",format='PDF')
plt.show()
# %%
#%%

which_cat = 'IBIRot_cat'

corr_sel_ = []

for (cond0, cond1, expNum, some_cat), group in sel_all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum',which_cat]):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            by_category = some_cat,
        )
        corr_sel_.append(corr_res)
    
corr_sel = pd.concat(corr_sel_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_sel,
    kind='line',
    col='cat',
    row='cond1',
    hue='by_category',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/long IBI corr by {which_cat}.pdf",format='PDF')
plt.show()
# %%

which_cat = 'rot_consistency'

corr_sel_ = []

for (cond0, cond1, expNum, some_cat), group in sel_all_aligned_sel_feature.groupby(['cond0', 'cond1','expNum',which_cat]):
    cat_cols = ['cond1','cond0']
    grp_cols = cat_cols + ['time_ms']
    for i, name in enumerate(corr_dict):
        [col1, col2] = corr_dict[name]
        corr_res = group.groupby('time_ms').apply(
            lambda y: pd.Series([stats.pearsonr(y[col1].values,y[col2].values)[0],
                                 stats.linregress(y[col1].values,y[col2].values)[0]],),
                ).reset_index()
        corr_res.columns = ['time_ms', 'r', 'slope']
        corr_res = corr_res.assign(
            cat = name,
            expNum = expNum,
            cond0 = cond0,
            cond1 = cond1,
            by_category = some_cat,
        )
        corr_sel_.append(corr_res)
    
corr_sel = pd.concat(corr_sel_, ignore_index=True).reset_index(drop=True)
# corr_bySpd = corr_bySpd.reset_index(drop=True)

g = sns.relplot(
    x='time_ms',
    y='r',
    data=corr_sel,
    kind='line',
    col='cat',
    row='cond1',
    hue='by_category',
    errorbar='sd',
    aspect=1.2,
    height=3
    )
g.set(xlim=(-300,200))
plt.savefig(fig_dir+f"/long IBI corr by {which_cat}.pdf",format='PDF')
plt.show()
# %%
