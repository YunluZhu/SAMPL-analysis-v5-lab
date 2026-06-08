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
from plot_functions.plt_functions import plt_categorical_combined_3
import matplotlib as mpl
import seaborn as sns

# %%
##### Parameters to change #####
pick_data = 'nMLF' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'

# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','') + f'_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True) # type: ignore
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# %%
cat_cols = ['cond1','expNum','cond0','bout_time','ztime','exp','boxNum']
features_to_plt = [c for c in all_feature_cond.columns if c not in cat_cols]
# features_to_plt = [
#     ]

#%%
median_df = all_feature_cond.groupby(['cond0','cond1','expNum','ztime'])[features_to_plt].median().reset_index()
for this_feature in features_to_plt:
    plt_categorical_combined_3(
            data=median_df,
            x='cond1',
            y=this_feature,
            col='ztime',
            units='expNum',
            errorbar='se',
            height=2.5,
        )
    plt.savefig(os.path.join(fig_dir, f'pointPlot_{this_feature}.pdf'), format='pdf')
    plt.show()
    
# %%
df_toplt = all_feature_cond
for this_feature in features_to_plt:
    g = sns.displot(
        data=df_toplt,
        stat='probability',
        element='poly',
        x=this_feature,
        col='cond1',
        kind='hist',
        fill=True,
        common_norm=False,
        bins='scott',
        height=2,
    )
    plt.savefig(os.path.join(fig_dir, f'Distribution_{this_feature}.pdf'), format='pdf')
    plt.show()

# %%
