# %%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_connected_bouts)
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_combined
from plot_functions.get_bout_consecutive_features import (cal_autocorrelation_feature, extract_consecutive_bout_features)


set_font_type()

# %%
pick_data = 'wt_ld'

folder_name = f'Navi4_B2B_correlation' 
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
all_features, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, day_light_narrow_bin=True, ztime='day')

# tidy bout uid
all_features = all_features.assign(
    epoch_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str) + all_features['epoch_uid'],
    exp_uid = all_features['cond0'] + all_features['cond1'] + all_features['expNum'].astype(str),
)
    

# %% std of directions of consecutive bouts
list_of_features = [
    # 'traj_peak',
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
    # grab x parameters
    # 'x_pre_swim','x_post_swim',
    # 'x_end','x_initial',
    # 'xdispl_swim',
    # 'atk_ang'
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts

#####################
max_lag = 2
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)
# take absolute value of x parameters
consecutive_bout_features = consecutive_bout_features.assign(
    # xdispl_swim = np.abs(consecutive_bout_features['xdispl_swim']),
    # x_pre_swim = np.abs(consecutive_bout_features['x_pre_swim']),
    # x_post_swim = np.abs(consecutive_bout_features['x_post_swim']),
    # x_initial = np.abs(consecutive_bout_features['x_initial']),
    # x_end = np.abs(consecutive_bout_features['x_end']),
)
# %% 
sel_consecutive_bouts = consecutive_bout_features.sort_values(by=['cond0', 'cond1','ztime','id']).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    # swim to swim, 5 mm/s
    # post_S2S_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values, np.nan),
    # pre_S2S_ydispl = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['y_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['y_post_swim'].values),
    # bout to bout, aligned, using next initial - previous end
    # post_B2B_ydispl = np.append(sel_consecutive_bouts.iloc[1:,:]['y_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['y_end'].values, np.nan),
    # ydispl_bout = sel_consecutive_bouts.iloc[:,:]['y_end'].values - sel_consecutive_bouts.iloc[:,:]['y_initial'].values,
    
    # x displacement
    # post_S2S_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values, np.nan)),
    # pre_S2S_xdispl = np.abs(np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['x_pre_swim'].values - sel_consecutive_bouts.iloc[:-1,:]['x_post_swim'].values)),
    # post_B2B_xdispl = np.abs(np.append(sel_consecutive_bouts.iloc[1:,:]['x_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['x_end'].values, np.nan)),
    # xdispl_bout = np.abs(sel_consecutive_bouts.iloc[:,:]['x_end'].values - sel_consecutive_bouts.iloc[:,:]['x_initial'].values),
    
    # rotation
    post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),

    bouts = sel_consecutive_bouts['lag'] + 1,
)
# IMPORTANT: because did the above calculation in the messy way, we can only pick the middle bouts. Drop the first and last bout of each series
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)
middle_bout_df = middle_bout_df.loc[middle_bout_df['ztime'].isin(['day','night'])].reset_index(drop=True)

#%%
xval='pre_IBI_time'
yval='post_IBI_time'
sns.displot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    common_norm=False,
    height=3,
    bins=300,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"{yval} {xval}.pdf"),format='PDF')


xval='pre_IBI_time'
yval='pre_B2B_rot'
sns.displot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    common_norm=False,
    height=3,
    bins=300,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"{yval} {xval}.pdf"),format='PDF')

#%%

xval='pre_B2B_rot'
yval='rot_total'
sns.displot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    common_norm=False,
    height=3,
    bins=500,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"{yval} {xval}.pdf"),format='PDF')

xval='post_B2B_rot'
yval='rot_total'
sns.displot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    common_norm=False,
    height=3,
    bins=500,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"{yval} {xval}.pdf"),format='PDF')

#%%

xval='pre_B2B_rot'
yval='rot_total'
sns.lmplot(
    data=middle_bout_df,
    scatter_kws={'alpha':0.01},
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"fit {yval} {xval}.pdf"),format='PDF')

xval='pre_B2B_rot'
yval='rot_full_accel'
sns.lmplot(
    data=middle_bout_df,
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    },
    scatter_kws={'alpha':0.01}
)
plt.savefig(os.path.join(fig_dir, f"fit {yval} {xval}.pdf"),format='PDF')


xval='post_B2B_rot'
yval='rot_total'
sns.lmplot(
    data=middle_bout_df,
    scatter_kws={'alpha':0.01},
    x=xval,
    y=yval,
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={
        'xlim': np.percentile(middle_bout_df[xval].dropna(),[.5, 99.5]),
        'ylim': np.percentile(middle_bout_df[yval].dropna(),[.5, 99.5])
    }
)
plt.savefig(os.path.join(fig_dir, f"fit {yval} {xval}.pdf"),format='PDF')

#%%
# calculate correlation and slope across conditions and repeats

corr_res = middle_bout_df.groupby(['cond0','cond1','ztime','expNum']).apply(
    lambda x: pd.Series({
        'slope_postRot': np.polyfit(x['post_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[0],
        'slopeError_postRot': np.polyfit(x['post_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[0] / np.sqrt(len(x['post_B2B_rot'].dropna())),
        'intercept_postRot': np.polyfit(x['post_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[1],
        'r_postRot': np.corrcoef(x['post_B2B_rot'].dropna(), x['rot_total'].dropna())[0,1],
        'p_postRot': np.polyfit(x['post_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[1],
        'slope_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[0],
        'slopeError_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[0] / np.sqrt(len(x['pre_B2B_rot'].dropna())),
        'intercept_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[1],
        'r_preRot': np.corrcoef(x['pre_B2B_rot'].dropna(), x['rot_total'].dropna())[0,1],
        'p_preRot':np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_total'].dropna(), 1)[1],
        
        'accelSlope_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_full_accel'].dropna(), 1)[0],
        'accelSlopeError_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_full_accel'].dropna(), 1)[0] / np.sqrt(len(x['pre_B2B_rot'].dropna())),
        'accelIntercept_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_full_accel'].dropna(), 1)[1],
        'accelR_preRot': np.corrcoef(x['pre_B2B_rot'].dropna(), x['rot_full_accel'].dropna())[0,1],
        'accelP_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_full_accel'].dropna(), 1)[1],
        'decelSlope_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_l_decel'].dropna(), 1)[0],
        'decelSlopeError_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_l_decel'].dropna(), 1)[0] / np.sqrt(len(x['pre_B2B_rot'].dropna())),
        'decelIntercept_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_l_decel'].dropna(), 1)[1],
        'decelR_preRot': np.corrcoef(x['pre_B2B_rot'].dropna(), x['rot_l_decel'].dropna())[0,1],
        'decelP_preRot': np.polyfit(x['pre_B2B_rot'].dropna(), x['rot_l_decel'].dropna(), 1)[1],
        
    })
).reset_index()
corr_res = corr_res.assign(
    rSquare_postRot = corr_res['r_postRot']**2,
    rSquare_preRot = corr_res['r_preRot']**2,
    accelRSquare_preRot = corr_res['accelR_preRot']**2,
    decelRSquare_preRot = corr_res['decelR_preRot']**2,
)


#%% let's plot the correlation
g = sns.relplot(
    kind='scatter',
    data=corr_res,
    y='slope_postRot',
    x='slope_preRot',
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={'xlim':[-1,0],'ylim':[-1,0]},
)
# plot a line at y=x
for ax in g.axes.flat:
    ax.plot([-1,0], [-1,0], color='red', linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim(-1,0)
    ax.set_ylim(-1,0)
    ax.set_xlabel('Slope of pre-IBI rotation')
    ax.set_ylabel('Slope of post-IBI rotation')
plt.savefig(os.path.join(fig_dir, f"scatter compare slope.pdf"),format='PDF')

g = sns.relplot(
    kind='scatter',
    data=corr_res,
    y='rSquare_postRot',
    x='rSquare_preRot',
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={'xlim':[0,1],'ylim':[0,1]},
)
# plot a line at y=x
for ax in g.axes.flat:
    ax.plot([0,1], [0,1], color='red', linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_xlabel('Slope of pre-IBI rotation')
    ax.set_ylabel('Slope of post-IBI rotation')
plt.savefig(os.path.join(fig_dir, f"scatter compare slope.pdf"),format='PDF')


# repeat for r
g = sns.relplot(
    kind='scatter',
    data=corr_res,
    y='r_postRot',
    x='r_preRot',
    col='cond1',
    row='ztime',
    height=3,
    facet_kws={'xlim':[-1,0],'ylim':[-1,0]},
)
# plot a line at y=x    
for ax in g.axes.flat:
    ax.plot([-1,0], [-1,0], color='red', linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim(-1,0)
    ax.set_ylim(-1,0)
    ax.set_xlabel('r of pre-IBI rotation')
    ax.set_ylabel('r of post-IBI rotation')
plt.savefig(os.path.join(fig_dir, f"scatter compare r.pdf"),format='PDF')

#%%
# categorical plot of slopes
# wide to long for r and slope
corr_res_long = pd.wide_to_long(
    corr_res,
    stubnames=['slope', 'r', 'p', 'slopeError', 'intercept','rSquare','accelSlope','accelR','decelSlope','decelR','accelRSquare','accelRSquare'],
    i=['cond0','cond1','ztime','expNum'],
    j='condition',
    sep='_',
    suffix='.+'
).reset_index()

#%%
plt_categorical_combined(
    data=corr_res_long,
    x='condition',
    y='slope',
    col='cond1',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,

)
plt.savefig(os.path.join(fig_dir, f"cat compare slope.pdf"),format='PDF')


#%%
plt_categorical_combined(
    data=corr_res_long,
    x='condition',
    y='rSquare',
    col='cond1',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,
)
plt.savefig(os.path.join(fig_dir, f"cat compare r2.pdf"),format='PDF')

plt_categorical_combined(
    data=corr_res,
    x='cond1',
    y='rSquare_preRot',
    col='cond0',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,
)
plt.savefig(os.path.join(fig_dir, f"cat compare preRot r2 by condition.pdf"),format='PDF')

# %%
#%%
plt_categorical_combined(
    data=corr_res,
    x='cond1',
    y='accelRSquare_preRot',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,

)
plt.savefig(os.path.join(fig_dir, f"cat compare accel rot RSquare.pdf"),format='PDF')

plt_categorical_combined(
    data=corr_res,
    x='cond1',
    y='decelRSquare_preRot',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,

)
plt.savefig(os.path.join(fig_dir, f"cat compare decel rot RSquare.pdf"),format='PDF')

plt_categorical_combined(
    data=corr_res,
    x='cond1',
    y='accelSlope_preRot',
    row='ztime',
    units='expNum',
    height=3,
    aspect=0.8,

)
plt.savefig(os.path.join(fig_dir, f"cat compare accel slope.pdf"),format='PDF')

# %%
