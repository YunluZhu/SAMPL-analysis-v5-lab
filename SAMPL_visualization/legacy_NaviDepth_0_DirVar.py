'''

'''

#%%
# import sys
import os,glob
from statistics import mean
# import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
import math
# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average,get_2sd,jackknife_mean_by_col, MAD)
from plot_functions.get_bout_kinetics import get_bout_kinetics
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_functions import plt_categorical_combined
from plot_functions.plt_stats import print_values
from scipy.stats import norm
from plot_functions.plt_tools import jackknife_list
import random
from statsmodels.stats.weightstats import ztest
set_font_type()
defaultPlotting()

# %%
data_list = ['gtau'] # all or specific data
fd = '_'.join(data_list)

which_zeitgeber = 'day'
folder_name = f'{fd} '

folder_dir = get_figure_dir(os.path.basename(__file__).split('_')[0])

fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

df_features_combined = pd.DataFrame()

for pick_data in data_list:
    root, FRAME_RATE = get_data_dir(pick_data)
    all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_zeitgeber)
    all_feature_cond['dataset'] = pick_data
    df_features_combined = pd.concat([df_features_combined,all_feature_cond], ignore_index=True)

df_features_combined.rename(columns={'dpf':'lightcond'},inplace=True)

sns.set_style("ticks")

#%% bootstrap
feature_toplt = ['traj_peak']
df_MAD_res = pd.DataFrame()

for (dataset, cond), gorup in df_features_combined.groupby(['lightcond','condition']):
    bst_input = gorup.index
    bst_output = [np.array(random.choices(bst_input, k=len(bst_input))) for i in np.arange(500)]
    for bts, ind in enumerate(bst_output):
        this_df = gorup.loc[ind, feature_toplt].apply(MAD).to_frame().T.assign(
            lightcond=dataset, condition=cond, bts_rep=bts
        )
        df_MAD_res = pd.concat([df_MAD_res, this_df], ignore_index=True)
        

# %%
print(f"---Compare MAD---")
toplt = df_MAD_res
toplt['condition'] = toplt['condition'].map({'1ctrl':'1ctrl',
                                            '2cond':'2cond',
                                            'hets':'1ctrl',
                                            'otog':'2cond',
                                            'ctrl':'1ctrl',
                                            'lesion':'2cond'})

for feature in feature_toplt:
    plt_categorical_combined(
        data=toplt,
        x='condition',
        y=feature,
        col='lightcond',
        row=None,
        units='bts_rep',
        related=False,
        sharey=False,
        errorbar='sd',
        overlay_func=sns.stripplot,
        alpha=0.05
    )
    plt.savefig(fig_dir+f"/{feature} compare MAD boot.pdf",format='PDF')
    
    print(f"* {feature}")
    for cond in set(toplt['lightcond']):
        this_df = toplt.loc[toplt['lightcond']==cond]
        ctrl_df = this_df.loc[this_df['condition']=='1ctrl',feature].values
        cond_df = this_df.loc[this_df['condition']=='2cond',feature].values
        se = np.sqrt(np.std(ctrl_df)**2 / len(ctrl_df) + np.std(cond_df)**2 / len(cond_df))
        mean_diff = np.mean(cond_df) -  np.mean(ctrl_df)
        zscore = mean_diff/se
        print(cond)
        pval = norm.sf(np.abs(zscore))*2
        print(f'Manual z test: {pval}') 
        print(ztest(cond_df, ctrl_df, value=0))
        
        print('Control:')
        print_values(ctrl_df, if_normal=True)
        print('Condition:')
        print_values(cond_df, if_normal=True)


# %%
from pprint import pprint
for cond in set(toplt['lightcond']):
    this_df = toplt.loc[toplt['lightcond']==cond]
    ctrl_df = this_df.loc[this_df['condition']=='1ctrl',feature].values
    cond_df = this_df.loc[this_df['condition']=='2cond',feature].values
    # se = np.sqrt(np.std(ctrl_df)**2 / len(ctrl_df) + np.std(cond_df)**2 / len(cond_df))
    # mean_diff = np.mean(cond_df) -  np.mean(ctrl_df)
    # zscore = mean_diff/se
    print(cond)
    pprint([(ztest(ctrl_df[:i], cond_df[:i])[1]) for i in range(2,len(cond_df),5)])
# %%
