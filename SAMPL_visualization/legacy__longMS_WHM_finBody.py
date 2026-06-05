'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_functions import plt_categorical_combined_3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from plot_functions.plt_tools import distribution_binned_average
# %%
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# %%

cat_cols = ['cond1','expNum','cond0','bout_time','ztime','exp','direction']
feature_to_plt = [c for c in all_feature_cond.columns if c not in cat_cols]
all_feature_cond['xdispl_swim'] = all_feature_cond['xdispl_swim'].abs()
feature_for_comp = feature_to_plt

cat_cols = ['cond0','cond1','expNum','ztime']
avg_data = all_feature_cond.groupby(cat_cols)[feature_to_plt].median()
avg_data = avg_data.reset_index()

cat_cols = ['cond0','cond1','expNum','ztime']

# %%
feature_to_plt.sort()

# %% tidy data

####################################
###### depth!!!! ######
####################################


df_to_plt = all_feature_cond.query("cond1=='ld' and ztime=='night'")

#%%
sns.displot(
    data=df_to_plt,
    x='WHM',
    col='cond0',
    palette=my_palette,
    stat='probability',
    # element='poly',
    common_norm=False,
    # bins='scott',
    fill=True,
    height=3,
    multiple="dodge",
)
#%%
sns.displot(
    kind='kde',
    data=df_to_plt,
    x='WHM',
    hue='cond0',
    palette=my_palette,
    # stat='probability',
    # element='poly',
    common_norm=False,
    # bins='scott',
    # fill=True,
    height=3,
    # multiple="dodge",
)
plt.savefig(os.path.join(fig_dir, 'WHM kde.pdf'), format='pdf')

