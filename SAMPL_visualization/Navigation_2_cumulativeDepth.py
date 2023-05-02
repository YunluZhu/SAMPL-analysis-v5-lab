'''
Plot depth change of consecutive bout during bout phase
Plot depth change of consecutive bout during Inter-swim/bout phase separated in hues by initial pitch angle
Plot sum depth change of consecutive bouts as a function of posture at peak speed during first bout
Plot slope of sum depth change VS posture by speed

NOTE variables to keep an eye on:

pick_data = 'wt_fin' # name of your cond0 to plot as defined in function get_data_dir()
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
from plot_functions.get_bout_consecutive_features import (extract_consecutive_bout_features)
from plot_functions.plt_tools import (set_font_type, get_2sd)
from plot_functions.plt_tools import jackknife_list
import scipy.stats as st


##### Parameters to change #####

pick_data = 'hc' # name of your cond0 to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'
consecutive_bout_num = 6 # number of consecutive bouts to extract. bout series with fewer consecutive bouts will be excluded. determined according to Navigation_1 
if_jackknife = True
##### Parameters to change #####

# %%

root, FRAME_RATE = get_data_dir(pick_data)
# spd_bins = [5,10,15,20,25]
# posture_bins = [-50,-20,-10,-5,0,5,10,15,20,25,50]

folder_name = f'Navi2_cumuDepth_z{which_ztime}'
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

# %% std of directions of consecutive bouts
list_of_features = ['traj_peak', 'pitch_peak', 'spd_peak',
                    'pitch_end', 'pitch_initial',
                    'ydispl_swim','y_pre_swim','y_post_swim',
                    'y_initial','y_end'
                    ]



# %% Connect consecutive bouts
max_lag = consecutive_bout_num - 1

consecutive_bout_features, _ = extract_consecutive_bout_features(all_features, list_of_features, max_lag)
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

# %% ###################################################
# ############## Depth change VS posture ##############

cat_col = ['id',	'lag',	'cond1', 'cond0']
which_plot_x = 'pitch_peak'
name_of_x = 'first_pitch'
df_sum = sel_consecutive_bouts.groupby(['cond1','cond0','id','exp_uid'])['ydispl_swim'].sum().reset_index()
df_sum = df_sum.assign(
    first_pitch = sel_consecutive_bouts.groupby(['cond1','cond0','id','exp_uid'])[which_plot_x].head(1).values,
    swim_speed = sel_consecutive_bouts.groupby(['cond1','cond0','id','exp_uid'])['spd_peak'].mean().values,
)

# plot scatter

x_name = 'first_pitch'
y_name = 'ydispl_swim'

opt_alpha = int(max(50000/len(df_sum), 10))/100
g = sns.relplot(data=df_sum, x=x_name, y=y_name,alpha=opt_alpha,row='cond0', col='cond1', hue='cond1', kind='scatter', height=3
                )
plt.savefig(os.path.join(fig_dir, f"cumudispl-{name_of_x}_scatter.pdf"),format='PDF')

# %% calculate slope by speed

speed_list = df_sum['swim_speed'].values
spd_bins = [np.percentile(speed_list, perct) for perct in [0, 25, 50, 75, 100]]
df_toplt = df_sum.assign(
    spd_bins = pd.cut(df_sum['swim_speed'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

res = pd.DataFrame()
for (cond0, cond1, spd), group in df_toplt.groupby(['cond0','cond1','spd_bins']):
    exp_df = group.groupby('exp_uid').size()
    if if_jackknife:
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
    else:
        jackknife_exp_matrix = [[item] for item in exp_df.index]
    for j, exp_group in enumerate(jackknife_exp_matrix):
        this_group_data = group.loc[group['exp_uid'].isin(exp_group),:]    
        this_x = this_group_data[name_of_x].values
        this_y = this_group_data['ydispl_swim'].values
        this_slope, this_intercept, this_r, this_p, this_se = st.linregress(this_x, this_y)
        this_res = pd.DataFrame(
            data={
                'slope':this_slope,
                'intercept':this_intercept,
                'r':this_r,
                'exp_uid':j,
                'spd_bin':spd,
                'cond0':cond0,
                'cond1':cond1,
            }, index=[0])
        res = pd.concat([res, this_res], ignore_index=True)
    

# get mean speed 
res = res.assign(
    spd_bin = res.cond0 + res.spd_bin.astype(str)
)
spd_bin_speed = df_toplt.groupby(['cond0','spd_bins']).mean()['swim_speed'].reset_index()
spd_bin_speed = spd_bin_speed.assign(
    spd_bin = spd_bin_speed.cond0 + spd_bin_speed.spd_bins.astype(str)
)
spd_bin_dict = spd_bin_speed.set_index('spd_bin')['swim_speed'].to_dict()
res = res.assign(
    mean_speed = res['spd_bin'].map(spd_bin_dict)
)

# plot
g = sns.relplot(
    data=res,
    x='mean_speed',
    y='slope',
    kind='line',
    col='cond0',
    hue='cond1',
    err_style="bars",
    # errorbar=get_2sd,
    height=3,
    aspect=1,
    facet_kws={'sharey': True, 'sharex': True}
)
plt.savefig(os.path.join(fig_dir, f"cumudispl-{name_of_x}_slope.pdf"),format='PDF')

