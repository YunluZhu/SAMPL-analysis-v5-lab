'''
This version uses "prop_Bout_IEI2" in the "prop_bout_IEI_pitch" file
which includes mean of body angles during IEI
Output: plots of distribution of body angles across different conditioins

paired T test results for standard deviation of posture  if number of conditions per age == 2
multiple comparison results for standard deviation of posture if number of conditions per age > 2

Conditions and age (dpf) are soft-coded
recognizable folder names (under root directory): xAA_abcdefghi
conditions (tau/lesion/control/sibs) are taken from folder names after underscore (abcde in this case)
age info is taken from the first character in folder names (x in this case, does not need to be number)
AA represents light dark conditions (LD or DD or LL...), not used.
'''

#%%
import sys
import os,glob
from matplotlib import style
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.plt_tools import (day_night_split,defaultPlotting, set_font_type, jackknife_mean, distribution_binned_average)
from plot_functions.get_IBIangles import get_IBIangles
set_font_type()
# %%
# Paste root directory here
pick_data = 'tau_long'
which_zeitgeber = 'all'

# %%
# ztime_dict = {}

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'IBI_pitch_z{which_zeitgeber}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created:{folder_name}')
except:
    print('fig folder already exist')

# %%
# main function
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)

bins = list(range(-90,95,5))

IBI_angles_cond, cond1, cond2 = get_IBIangles(root, FRAME_RATE, ztime=which_zeitgeber)
IBI_angles_cond = IBI_angles_cond.loc[:,['propBoutIEI_pitch','ztime','expNum','dpf','condition']]
IBI_angles_cond.columns = ['IBI_pitch','ztime','expNum','dpf','condition']
IBI_angles_cond.reset_index(drop=True,inplace=True)
cond_cols = ['ztime','dpf','condition']
all_ztime = list(set(IBI_angles_cond.ztime))
all_ztime.sort()
# %%
# plot kde of all
g = sns.FacetGrid(IBI_angles_cond, 
                  row="ztime", row_order=all_ztime,
                  col='dpf', col_order=cond1,
                  hue='condition',
                  )
g.map(sns.kdeplot, "IBI_pitch",alpha=0.5,
      )
filename = os.path.join(fig_dir,"IBI pitch kde.pdf")
plt.savefig(filename,format='PDF')

# %% 
# jackknife std
plt.close()
IBI_std_cond = IBI_angles_cond.groupby(['ztime','dpf','condition','expNum']).std().reset_index()
std_jackknife = IBI_std_cond.groupby(cond_cols).apply(
    lambda x: jackknife_mean(x)
)
std_jackknife = std_jackknife.loc[:,'IBI_pitch'].reset_index()

g = sns.catplot(data=std_jackknife,
                col='condition',
                x='ztime', y='IBI_pitch',
                hue='dpf',
                kind='point')
filename = os.path.join(fig_dir,"std of IBI pitch.pdf")
plt.savefig(filename,format='PDF')
