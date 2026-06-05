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
pick_data = 'xxxxxx' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day', 'night', or 'all'

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
all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# %%
avg_length = all_feature_cond.groupby(['cond0','cond1','expNum'])['fish_length'].median().reset_index()
#%%
# # plot average
# plt.figure(figsize=(3,3))
# sns.pointplot(data=avg_length, x='cond0', y='fish_length', hue='cond0', errorbar='se')
# plt.title('Average fish length')
# plt.ylabel('Fish Length (mm)')
# plt.yscale('log')
# plt.savefig(os.path.join(fig_dir, 'Average_fish_length.pdf'), format='pdf')

#%%
from plot_functions.plt_functions import plt_categorical_combined_3
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
plt_categorical_combined_3(
        data=avg_length.query("cond1 == 'ld'"),
        x='cond0',
        y='fish_length',
        hue='cond0',
        units='expNum',
        errorbar='se',
        palette=my_palette,
        height=2.5,
    )
plt.savefig(os.path.join(fig_dir, 'Average_fish_length.pdf'), format='pdf')
plt.show()