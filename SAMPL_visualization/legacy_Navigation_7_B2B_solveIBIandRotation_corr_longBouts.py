'''


'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,distribution_binned_average_opt)
from plot_functions.get_bout_consecutive_features import extract_consecutive_bout_features
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel
from lmfit import Model
import scipy.stats as st


set_font_type()

# %%
##### Parameters to change #####
pick_data = 'blind' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' 'night', or 'all'
if_day_light_narrow_bin = True
##### Parameters to change #####

print(pick_data,which_ztime)
# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'Navi7_B2B_IBIrotation_corr_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass

root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime, day_light_narrow_bin=if_day_light_narrow_bin)

# tidy bout uid
all_features = all_feature_cond.assign(
    epoch_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + all_feature_cond['epoch_uid'],
    exp_uid = all_feature_cond['cond0'] + all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str),
)
    
all_features = all_features.loc[all_features['ztime'].isin(['day','night'])].reset_index(drop=True)
# %% std of directions of consecutive bouts
list_of_features = [
    'traj_peak',
    'post_IBI_time',
    'pre_IBI_time',
    
    'spd_peak',
    
    'pitch_end', 
    'pitch_initial',
    
    'rot_total',
    'rot_full_accel',
    'rot_l_decel',
    'rot_l_accel',
                    ]

df_input = all_features.groupby(['epoch_uid'], group_keys=False).filter(lambda g: len(g)>1)

# %% associate consecutive bouts

#####################
max_lag = 1
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
    # post_B2B_rot = np.append(sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values, np.nan),
    pre_B2B_rot = np.append(np.nan, sel_consecutive_bouts.iloc[1:,:]['pitch_initial'].values - sel_consecutive_bouts.iloc[:-1,:]['pitch_end'].values),

    bouts = sel_consecutive_bouts['lag'] + 1,
)
#%%
# IMPORTANT: let's grab the second bout if there're only 2 consecutive bouts
# IMPORTANT: because did the above calculation in the messy way, we can only pick the middle bouts. Drop the first and last bout of each series
middle_bout_df = sel_consecutive_bouts.loc[sel_consecutive_bouts['lag']==1].reset_index(drop=True)
middle_bout_df = middle_bout_df.loc[middle_bout_df['ztime'].isin(['day','night'])].reset_index(drop=True)

middle_bout_df = middle_bout_df.assign(
    traj_deviation = middle_bout_df['traj_peak']- middle_bout_df['pitch_initial'],
)
which_IBI = 'pre'

# IBI_threshold = middle_bout_df.groupby(['cond0','cond1'])[f'{which_IBI}_IBI_time'].transform(np.percentile, 50)
middle_bout_df['IBI_threshold'] = 1.7
middle_bout_df['IBI_cat'] = 'long_IBI'
middle_bout_df.loc[middle_bout_df[f'{which_IBI}_IBI_time'] < middle_bout_df['IBI_threshold'], 'IBI_cat'] = 'short_IBI'

middle_bout_df['traj_cat'] = pd.cut(middle_bout_df['traj_peak'], bins=[-np.inf,0, np.inf], labels=['dive','climb'])
middle_bout_df['initialPitch_cat'] = pd.cut(middle_bout_df['pitch_initial'], bins=[-np.inf,0, np.inf], labels=['initial_DN','initial_UP'])

#%%
which_cat = 'pitch_initial'

df_toplt = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI']
xval=f'pre_B2B_rot'
yval='rot_total'

# scatter
sns.relplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',
    row='cond1',
    col='cond0',
    kind='scatter', 
    height=3,
    alpha=0.03,
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.2, 99.8])
    }
    # cmap
)
plt.savefig(os.path.join(fig_dir, f"longIBI scatter by pitchInitial {yval} {xval} by {which_cat}.pdf"),format='PDF')

#%%
which_cat = 'pitch_initial'

df_toplt = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI']
xval=f'pre_B2B_rot'
yval='rot_total'

# scatter
sns.lmplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',
    hue='cond1',
    col='cond0',
    height=3,
    # alpha=0.03,
    scatter_kws={'alpha':0.03},
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.2, 99.8])
    }
    # cmap
)
plt.savefig(os.path.join(fig_dir, f"longIBI lm by pitchInitial {yval} {xval} by {which_cat}.pdf"),format='PDF')

#%%

which_cat = 'pitch_initial'

df_toplt = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI']
df_toplt['pos_cat'] = pd.cut(df_toplt['pitch_initial'], bins=[-90,-15,-5,0,5,90])
xval=f'pre_B2B_rot'
yval='rot_total'

# scatter
sns.relplot(
    data=df_toplt,
    x=xval,
    y=yval,
    # palette='mako',
    row='cond1',
    hue='pos_cat',
    # col='cond1',
    kind='scatter', 
    height=3,
    # color code by posture initial
    col='pos_cat',
    alpha=0.03,
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.2, 99.8]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.2, 99.8])
    }
    # cmap
)
plt.savefig(os.path.join(fig_dir, f"longIBI scatter by pitchInitial {yval} {xval} by {which_cat}.pdf"),format='PDF')


#%%

which_cat = None

xval=f'pre_B2B_rot'
yval='rot_total'

sns.displot(
    kind='kde',
    data=df_toplt,
    x=xval,
    y=yval,
    # col='traj_cat',
    hue=which_cat,
    row='cond1',
    common_norm=False,
    height=3,
    levels=8,
    facet_kws={
        'xlim': np.percentile(df_toplt[xval].dropna(),[.2, 99.5]),
        'ylim': np.percentile(df_toplt[yval].dropna(),[.5, 99.8])
    }
)
plt.savefig(os.path.join(fig_dir, f"longIBI hist {yval} {xval} by {which_cat}.pdf"),format='PDF')

# # %% check bouts with long IBI and negative trajectory deviation
# longIBI_trajDevDN = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI'].loc[middle_bout_df['traj_deviation']< 0]

# #%
# xval=f'rot_full_accel'

# sns.displot(
#     kind='hist',
#     stat='probability',
#     element='poly',
#     data=longIBI_trajDevDN,
#     x=xval,
#     # col='traj_cat',
#     row='cond1',
#     common_norm=False,
#     height=2,
#     facet_kws={
#         'xlim': np.percentile(longIBI_trajDevDN[xval].dropna(),[.2, 99.8]),
#     }
# )
# plt.savefig(os.path.join(fig_dir, f"longIBI_trajDevDN hist {xval} by.pdf"),format='PDF')


# # %% check bouts with long IBI and negative trajectory deviation and postive initlal pitch
# largeIBIrot_trajDevDN_initialPitchUP = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI'].loc[middle_bout_df['traj_deviation']< 0].loc[middle_bout_df['initialPitch_cat']=='initial_UP'].loc[middle_bout_df['pre_B2B_rot']<-5]

# #%
# xval=f'rot_total'

# sns.displot(
#     kind='hist',
#     stat='probability',
#     element='poly',
#     data=largeIBIrot_trajDevDN_initialPitchUP,
#     x=xval,
#     # col='traj_cat',
#     row='cond1',
#     common_norm=False,
#     height=2,
#     facet_kws={
#         'xlim': np.percentile(largeIBIrot_trajDevDN_initialPitchUP[xval].dropna(),[.2, 99.8]),
#     }
# )
# plt.savefig(os.path.join(fig_dir, f"largeIBIrot_trajDevDN_initialPitchUP hist {xval} by.pdf"),format='PDF')


# # %% check bouts with long IBI and negative trajectory deviation and postive initlal pitch
# largeIBIrot_trajDevDN_initialPitchDN = middle_bout_df.loc[middle_bout_df['IBI_cat']=='long_IBI'].loc[middle_bout_df['traj_deviation']< 0].loc[middle_bout_df['initialPitch_cat']=='initial_DN'].loc[middle_bout_df['pre_B2B_rot']<-5]

# #%
# xval=f'rot_total'

# sns.displot(
#     kind='hist',
#     stat='probability',
#     element='poly',
#     data=largeIBIrot_trajDevDN_initialPitchDN,
#     x=xval,
#     # col='traj_cat',
#     row='cond1',
#     common_norm=False,
#     height=2,
#     facet_kws={
#         'xlim': np.percentile(largeIBIrot_trajDevDN_initialPitchDN[xval].dropna(),[.2, 99.8]),
#     }
# )
# plt.savefig(os.path.join(fig_dir, f"largeIBIrot_trajDevDN_initialPitchDN hist {xval} by.pdf"),format='PDF')


# #%%

# which_cat = None



# sns.displot(
#     kind='kde',
#     data=longIBI_trajDevDN,
#     x=xval,
#     y=yval,
#     # col='traj_cat',
#     hue=which_cat,
#     row='cond1',
#     common_norm=False,
#     height=3,
#     levels=8,
#     facet_kws={
#         'xlim': np.percentile(longIBI_trajDevDN[xval].dropna(),[.2, 99.8]),
#         'ylim': np.percentile(longIBI_trajDevDN[yval].dropna(),[.2, 99.8])
#     }
# )
# plt.savefig(os.path.join(fig_dir, f"longIBI_trajDevDN hist {yval} {xval} by {which_cat}.pdf"),format='PDF')


#%%###############

#################
#################
#################

# now, let's make the master scatter plot of bout rotation vs IBI rotation


# %%
from statsmodels.api import OLS, add_constant

df_toana = df_toplt.query("cond1 == 'dd'").reset_index(drop=True)
rot_total_resid = OLS(df_toana['rot_total'], add_constant(df_toana['pitch_initial'])).fit().resid
IBI_rot_resid = OLS(df_toana['pre_B2B_rot'], add_constant(df_toana['pitch_initial'])).fit().resid

#%%
g = sns.regplot(x=IBI_rot_resid, y=rot_total_resid, scatter_kws={'alpha':0.03}, line_kws={'color':'red'},
)
g.set_xlim(np.percentile(IBI_rot_resid.dropna(),[.2, 99.8]))
g.set_ylim(np.percentile(rot_total_resid.dropna(),[.2, 99.8]))
plt.xlabel("Residual IBIrot (controlling for pitch_initial)")
plt.ylabel("Residual totalrot (controlling for pitch_initial)")
plt.title("IBIrot vs totalrot correlation, posture-controlled")
plt.savefig(os.path.join(fig_dir, "longIBI IBIrot vs totalrot corr posture controlled.pdf"),format='PDF')

# %%
sns.lmplot(
    data=df_toplt,
    x='pitch_initial',
    y='rot_total',
    scatter_kws={'alpha':0.03},
    line_kws={'color':'red'},
    facet_kws={
        'xlim': np.percentile(df_toplt['pitch_initial'].dropna(),[.2, 99.8]),
        'ylim': np.percentile(df_toplt['rot_total'].dropna(),[.2, 99.8])
    },
)

# %%
