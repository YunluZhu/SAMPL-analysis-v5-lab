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

# %%
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = f'MultiComp_BF1_median_z{which_ztime}'
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
feature_for_comp = feature_to_plt

cat_cols = ['cond0','cond1','expNum','ztime']
avg_data = all_feature_cond.groupby(cat_cols)[feature_to_plt].median()
avg_data = avg_data.reset_index()

cat_cols = ['cond0','cond1','expNum','ztime']

# %%
feature_to_plt.sort()
####################################
###### Plotting Starts Here ######
####################################

toplt = avg_data.loc[avg_data['ztime'].isin(['day','night'])]
x_name = 'cond0'
# gridrow = 'direction'
gridrow = 'cond1'
gridcol = None
hue = 'ztime'
units = 'expNum'
prename = ''

for feature in feature_to_plt:
    g = sns.catplot(
        kind='point',
        data = toplt,
        x = x_name,
        y = feature,
        col = None,
        row = gridrow,
        hue=  hue,
        sharey=True,
        height = 2.5,
        aspect = 2,
        errorbar='sd',
        )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%
