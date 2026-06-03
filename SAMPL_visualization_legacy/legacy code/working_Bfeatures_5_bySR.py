'''

'''

#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from plot_functions.plt_tools import jackknife_list


set_font_type()
defaultPlotting()

# %%
def calc_jackknife_histogram(df_tocalc, feature, jackknife_col):
    BIN_NUM = round_half_up((3 + math.log2(len(df_tocalc)/2)))
    feature_data = df_tocalc[feature]
    lower, upper = np.percentile(feature_data,0.1), np.percentile(feature_data,99.9)
    interval = (upper-lower)/BIN_NUM
    feature_bins = np.arange(lower, upper+interval*2,interval)
    mid_bins = feature_bins[0:-1] + interval/2
    
    jackknife_res = pd.DataFrame()

    for condition, df_cond in df_tocalc.groupby('cond1'):
        exp_df = df_cond.groupby(jackknife_col).size()
        jackknife_exp_matrix = jackknife_list(list(exp_df.index))
        for j, exp_group in enumerate(jackknife_exp_matrix):
            this_group_data = df_cond.loc[df_cond[jackknife_col].isin(exp_group),:]
            
            binned_ratio = binned_proportion(df=this_group_data, col=feature, bin=feature_bins).values
            binned_res = pd.DataFrame(data={
                'yval':binned_ratio
            })
            binned_res = binned_res.assign(
                jackknife_rep = j,
                feature = mid_bins,
                cond1 = condition
            )
            binned_res.columns = ['probability','jackknife_rep',feature,'cond1']
            jackknife_res = pd.concat([jackknife_res, binned_res], ignore_index=True)
            
    # print(f"{feature}:\n - bin num = {BIN_NUM}; bootstrap N = {bootstrap_n}")
    # ctrl_data = df_tocalc.query("cond1 == @all_cond0[0]")[feature]
    # cond_data = df_tocalc.query("cond1 == @all_cond0[1]")[feature]
    # ksres = st.ks_2samp(ctrl_data, cond_data)
    # print(f" - K-S p value = {ksres.pvalue}")
    return jackknife_res

def binned_proportion(df, col, bin):
    bins = pd.cut(df[col], bin)
    grp = df.groupby(bins)
    counts = grp.size()
    proportion = counts/counts.sum()
    return proportion

# %%
data_list = ['otog','tan','vs'] # all or specific data
which_zeitgeber = 'day'
folder_name = f'otog TAN kinetics'
folder_dir = get_figure_dir('Fig_5')
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    
spd_bins = np.arange(5,25,4)

df_features_combined = pd.DataFrame()
df_bySpd_combined = pd.DataFrame()
df_kinetics_combined = pd.DataFrame()
for pick_data in data_list:
    root, FRAME_RATE = get_data_dir(pick_data)
    all_kinetic_cond, kinetics_jackknife, kinetics_bySpd_jackknife, all_cond0, all_cond1 = get_bout_kinetics(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond, _, all_conditions = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
    all_cond0 = pick_data
    all_cond0.sort()
    kinetics_bySpd_jackknife['cond0'] = pick_data
    kinetics_jackknife['cond0'] = pick_data
    all_feature_cond['cond0'] = pick_data
    df_bySpd_combined = pd.concat([df_bySpd_combined,kinetics_bySpd_jackknife], ignore_index=True)
    df_kinetics_combined = pd.concat([df_kinetics_combined,kinetics_jackknife], ignore_index=True)
    df_features_combined = pd.concat([df_features_combined,all_feature_cond], ignore_index=True)
    

df_features_combined = df_features_combined.assign(
    speed_bins = pd.cut(df_features_combined['spd_peak'],bins=spd_bins,labels=np.arange(len(spd_bins)-1))
)
df_bySpd_combined.rename(columns={'cond0':'dataset'},inplace=True)
df_kinetics_combined.rename(columns={'cond0':'dataset'},inplace=True)
df_features_combined.rename(columns={'cond0':'dataset'},inplace=True)

df_features_combined['cond1'] = df_features_combined['cond1'].map(
                                            {'1ctrl':'1ctrl',
                                             '2cond':'2cond',
                                             'hets':'1ctrl',
                                             'otog':'2cond',
                                             'ctrl':'1ctrl',
                                             'lesion':'2cond'})
# %%
sns.set_style("ticks")

# %%
# probability plot
feature_list = ['pitch_post_bout','traj_peak']
for dataset in data_list:
    df = df_features_combined.query("dataset == @dataset")
    for feature in feature_list:
        probab = calc_jackknife_histogram(df, feature, 'expNum')
        control_probab = probab.query('cond1 == @all_conditions[0]').sort_values(by=['jackknife_rep',feature]).reset_index(drop=True)
        condition_probab = probab.query('cond1 == @all_conditions[1]').sort_values(by=['jackknife_rep',feature]).reset_index(drop=True)
        probab_diff = condition_probab
        probab_diff['probability'] = condition_probab['probability'] - control_probab['probability']
        
        # plot distribution
        plt.figure(figsize=(3,2))
        g = sns.lineplot(
            data = probab_diff,
            y = 'probability',
            x = feature,
            errorbar = 'sd',
            # errorbar=('ci',95),
            # hue = 'cond1',
            # palette = ['black', 'red']
        )
        sns.despine()
        # sns.move_legend(g, 'upper left',bbox_to_anchor=(1, 1))
        fig_name = f"{dataset}_{feature}_probabbility diff.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name),format='PDF')
        
# %%
# less exploratory
feature_list = ['depth_chg']
for dataset in data_list:
    df = df_features_combined.query("dataset == @dataset")
    for feature in feature_list:
        probab = calc_jackknife_histogram(df, feature, 'expNum')        
        # plot distribution
        plt.figure(figsize=(3,2))
        g = sns.lineplot(
            data = probab,
            y = 'probability',
            x = feature,
            errorbar = 'sd',
            hue = 'cond1',
            palette = ['black', 'red']
        )
        sns.despine()
        # sns.move_legend(g, 'upper left',bbox_to_anchor=(1, 1))
        fig_name = f"{dataset}_{feature}_probabbility diff.pdf"
        plt.savefig(os.path.join(fig_dir, fig_name),format='PDF')
        

# %%
# SR_dir = df_features_combined['rot_full_accel'] * df_features_combined['rot_l_decel']
R_dir = pd.cut(df_features_combined.rot_l_decel, bins=[-np.inf, 0, np.inf], labels=['Rn','Rp']).to_list()
S_dir = pd.cut(df_features_combined.rot_full_accel, bins=[-np.inf, 0, np.inf], labels=['Sn','Sp']).to_list()

df_SRcat = df_features_combined.assign(
    abs_depth_chg = df_features_combined['depth_chg'].abs(),
    SR_dir = [a + b for a, b in zip(R_dir, S_dir)]
)

df_SRcat.groupby(['dataset','SR_dir','cond1']).mean()['traj_peak']
df_kinetics_combined.groupby('dataset')['set_point_jack'].mean()

df_SRcat.sort_values(by=['dataset','cond1'], inplace=True)
 # %%
feature = 'rot_l_accel'
g = sns.catplot(
    data = df_SRcat,
    col = 'dataset',
    row = 'SR_dir',
    hue = 'cond1',
    kind = 'point',
    x = 'cond1',
    y = feature,
    sharey=False,
    height=3
)
# %%
