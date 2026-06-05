# %%
from cmath import exp
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.plt_functions import plt_categorical_grid
import scipy.stats as st
from tqdm import tqdm
from scipy.stats import ttest_rel
from sklearn.linear_model import LinearRegression
from plot_functions.plt_tools import jackknife_list
import math
from scipy import stats


pick_data = 'otog_lightR'  # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

##### Parameters to change #####

# %%

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'BF8_obliqueness_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
# all_ibi_cond, _, _  = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)

# %% tidy data
col_to_adj = 'pitch_peak' 
# origin for x is on the right. To make it North0East, use preX-postX to correct posture calculation 
all_feature_cond['N0E_' + col_to_adj] = (all_feature_cond.x_pre_swim - all_feature_cond.x_post_swim)/(all_feature_cond.x_post_swim - all_feature_cond.x_pre_swim).abs() * (90 + all_feature_cond[col_to_adj]) + 180
# col_to_adj = 'propBoutIEI_pitch'
# all_ibi_cond = all_ibi_cond['N0E_' + col_to_adj] = (all_ibi_cond.x_post_swim - all_ibi_cond.x_pre_swim)/(all_ibi_cond.x_post_swim - all_ibi_cond.x_pre_swim).abs() * (90 + all_ibi_cond[col_to_adj]) + 180
    
# %% 
col_toplt = 'N0E_' + col_to_adj
# col_toplt = col_to_adj
df_toplt = all_feature_cond#.query("cond0 == @all_cond0[1]")

min_val = 0
max_val = 360
step = (max_val-min_val)/100
bins = np.arange(min_val,max_val+step,step)

angle_counts = df_toplt.groupby(['cond0','cond1']).apply(
    lambda g: np.histogram(g[col_toplt], bins)[0]/len(g)
)

bin_mid = (bins[1:] + bins[:-1])/2
# %
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)

for i, cond1 in enumerate(df_toplt.cond1.unique()):
    for cond0 in df_toplt.query("cond1 == @cond1").cond0.unique():
        ax.plot(np.radians(bin_mid), angle_counts[cond0, cond1])
        print(cond0, cond1)

plt.savefig(os.path.join(fig_dir, f"bout direction hist.pdf"),format='PDF')

# %%
# quantify obliqueness
hist_df = angle_counts.reset_index().explode(0)
hist_df.columns = ['cond0', 'cond1', 'probab']
hist_df = hist_df.assign(
    bin_mid = np.tile(bin_mid, int(len(hist_df)/len(bin_mid)))
)
hist_df.reset_index(drop=True, inplace=True)

hist_df = hist_df.assign(
    xcor = np.sin(np.radians(hist_df['bin_mid'])) * hist_df['probab'].astype('float')*100,
    ycor = np.cos(np.radians(hist_df['bin_mid'])) * hist_df['probab'].astype('float')*100,
)
hist_df = hist_df.query("probab > 0")
# %% calc menan of angle (something's wrong)

# df_toplt = all_feature_cond.copy().assign(
#     xcor = np.sin(np.radians(df_toplt['N0E_' + col_to_adj])),
#     ycor = np.cos(np.radians(df_toplt['N0E_' + col_to_adj])) 
# )
# df_toplt['N0E_180' + col_to_adj] = df_toplt['N0E_' + col_to_adj]
# df_toplt.loc[df_toplt['N0E_' + col_to_adj] > 180, 'N0E_180' + col_to_adj] = df_toplt.loc[df_toplt['N0E_' + col_to_adj] > 180, 'N0E_' + col_to_adj] - 360
# df_toplt.loc[df_toplt['N0E_' + col_to_adj] < 180, 'N0E_180' + col_to_adj] = df_toplt.loc[df_toplt['N0E_' + col_to_adj] < 180, 'N0E_' + col_to_adj] * -1

# df_toplt.groupby(['cond0','cond1'])['N0E_180' + col_to_adj].apply(np.nanmean)
# %%
sns.lmplot(
    data=hist_df,
    x='xcor',
    y='ycor',
    hue='cond1',
    col='cond0'
)

# %%
hist_df.groupby(['cond0', 'cond1']).apply(
    lambda group: np.polyfit(x=group['xcor'], y=group['ycor'], deg=1)[0]
)
# %% linear regression
data_tocalc = hist_df[['cond0','cond1','xcor','ycor']].dropna()
# slope, intercept, r_value, p_value, std_err = stats.linregress(data_tocalc['xcor'].values,data_tocalc['ycor'].values)
reg_res_stacked = data_tocalc.groupby(['cond0', 'cond1']).apply(
    lambda group: stats.linregress(group['xcor'],group['ycor'])
)
reg_res = reg_res_stacked.reset_index()
reg_res[['slope', 'intercept', 'r_value', 'p_value', 'std_err']] = pd.DataFrame(reg_res[0].tolist(), index=reg_res.index)

# %%
# resample 
reg_res_onSample = pd.DataFrame()

data_tocalc = all_feature_cond

for n in np.arange(100):
    df_toplt_sampled = data_tocalc.groupby(['cond0','cond1']).sample(frac=0.8, replace=True)
    angle_counts_onSampled = df_toplt_sampled.groupby(['cond0','cond1']).apply(
        lambda g: np.histogram(g[col_toplt], bins)[0]/len(g)
    )
    hist_df_ThisSampled = angle_counts_onSampled.reset_index().explode(0)
    hist_df_ThisSampled.columns = ['cond0', 'cond1', 'probab']
    hist_df_ThisSampled = hist_df_ThisSampled.assign(
        bin_mid = np.tile(bin_mid, int(len(hist_df_ThisSampled)/len(bin_mid)))
    )
    hist_df_ThisSampled.reset_index(drop=True, inplace=True)

    hist_df_ThisSampled = hist_df_ThisSampled.assign(
        xcor = np.sin(np.radians(hist_df_ThisSampled['bin_mid'])) * hist_df_ThisSampled['probab'].astype('float')*100,
        ycor = np.cos(np.radians(hist_df_ThisSampled['bin_mid'])) * hist_df_ThisSampled['probab'].astype('float')*100,
    )
    hist_df_ThisSampled = hist_df_ThisSampled.query("probab > 0")
    hist_df_ThisSampled = hist_df_ThisSampled[['cond0','cond1','xcor','ycor']].dropna()
    reg_res_stacked_thisSampled = hist_df_ThisSampled.groupby(['cond0', 'cond1']).apply(
        lambda group: stats.linregress(group['xcor'],group['ycor'])
    )
    reg_res_thisSampled= reg_res_stacked_thisSampled.reset_index()
    reg_res_thisSampled[['slope', 'intercept', 'r_value', 'p_value', 'std_err']] = pd.DataFrame(reg_res_thisSampled[0].tolist(), index=reg_res_thisSampled.index)
    reg_res_thisSampled = reg_res_thisSampled.assign(
        bootstrapNum = n
    )
    reg_res_onSample = pd.concat([reg_res_onSample, reg_res_thisSampled], ignore_index=True)

#%%
reg_res_onSample = reg_res_onSample.assign(
    cond0cond1 = reg_res_onSample['cond0']+reg_res_onSample['cond1']
)
reg_res_onSample['slope_corrected'] = reg_res_onSample['slope'] * -1
reg_res_onSample = reg_res_onSample.assign(
    deg_from_hori = np.rad2deg(np.arctan(reg_res_onSample['slope_corrected'].values))
)
reg_res_onSample.groupby(['cond0','cond1'])['deg_from_hori'].apply(np.nanmean)


sns.catplot(
    data=reg_res_onSample,
    x='cond0cond1',
    y='slope_corrected',
    kind='point',
    errorbar='sd',
    height=2.5,
    markers='d'
)
plt.savefig(os.path.join(fig_dir, f"slope_compare.pdf"),format='PDF')

# %%
