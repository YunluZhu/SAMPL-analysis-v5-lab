'''

'''

#%%
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from astropy.stats import jackknife_resampling
from scipy.optimize import curve_fit
from plot_functions.get_data_dir import (get_data_dir,get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.plt_tools import (jackknife_mean, set_font_type, defaultPlotting, distribution_binned_average)

set_font_type()
# defaultPlotting()
# %%
pick_data = 'tau_bkg'
which_zeitgeber = 'day' # Day only!!!!!!!
DAY_RESAMPLE = 1000
NIGHT_RESAMPLE = 500

# %%
# Select data and create figure folder
root, FRAME_RATE = get_data_dir(pick_data)

X_RANGE = np.arange(-30,40,0.1)
BIN_WIDTH = 1
AVERAGE_BIN = np.arange(min(X_RANGE),max(X_RANGE),BIN_WIDTH)

folder_name = f'BK5_righting_z{which_zeitgeber}_sample{DAY_RESAMPLE}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime = which_zeitgeber)

# %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# all_feature_cond.drop(all_feature_cond[all_feature_cond['spd_peak']<7].index, inplace=True)
all_feature_cond = all_feature_cond.sort_values(by=['condition','expNum']).reset_index(drop=True)
# one_kinetics = all_feature_cond.groupby(['dpf']).apply(
#                         lambda x: get_kinetics(x)
#                         ).reset_index()
# assign up and down
pitch_bins = np.arange(-20,42,12)
# set_point = one_kinetics.loc[0,'set_point']
set_point = 15
all_feature_UD = pd.DataFrame()
all_feature_UD = all_feature_cond.assign(
    pre_bout_bins = pd.cut(all_feature_cond['pitch_pre_bout'],pitch_bins,labels=np.arange(len(pitch_bins)-1)),
    direction = pd.cut(all_feature_cond['pitch_pre_bout'], 
                       bins=[-90,set_point,91],labels=['dn','up']),
    # speed_bins = pd.cut(all_feature_cond['spd_peak'],spd_bins,labels=np.arange(len(spd_bins)-1)),
)

all_feature_UD = all_feature_UD.dropna().reset_index(drop=True)

# %% fit sigmoid - master

# speed bins
speed_separated_by = 'mean_speed'

all_feature_UD = all_feature_UD.assign(
    deviationFromSet = np.abs(set_point  - all_feature_UD['pitch_pre_bout']),
    mean_speed_byDir = all_feature_UD.groupby('direction')['spd_peak'].transform('median'),
    mean_speed = all_feature_UD['spd_peak'].mean(),
    absRightingRot = np.abs(all_feature_UD['rot_l_decel'])
)
all_feature_UD = all_feature_UD.assign(
    speed_2bin = 'slow'
)
all_feature_UD.loc[all_feature_UD['spd_peak']>all_feature_UD[speed_separated_by],'speed_2bin'] = 'fast'

# %%
# absolutely righting rotation by speed
df = all_feature_UD
bins = np.arange(0,34,5)
bins_middle = (bins[:-1] + bins[1:])/2

df = df.assign(
    posture_deviation_bins = pd.cut(df.deviationFromSet,bins=bins,labels=bins_middle)
)
df.dropna(inplace=True)
df = df.assign(
    # average_deviation = df.groupby(['direction','speed_2bin','posture_deviation_bins'])['deviationFromSet'].transform('mean')
    average_deviation = df.groupby(['direction','posture_deviation_bins'])['deviationFromSet'].transform('mean')

)
g = sns.relplot(kind='line',
                col='direction',
                row='dpf',
                data = df,
                hue='condition',
                x='average_deviation',
                y='absRightingRot',
                markers=False,
                # style='speed_2bin',
                err_style = 'bars',
                # hue_order=['slow','fast'],
                # style_order = ['slow','fast'],
                # facet_kws={'sharey': True, 'sharex': False}
                )
# g.set_xlabel('Posture deviation from set point')
# plt.savefig(fig_dir+f"/absRighting vs deviation from set binned.pdf",format='PDF')
# %%

