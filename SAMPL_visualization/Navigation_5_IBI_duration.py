'''


'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (jackknife_mean,set_font_type, defaultPlotting,distribution_binned_average,jackknife_list)
from plot_functions.get_bout_kinetics import get_bout_kinetics
import matplotlib as mpl
from sklearn.metrics import r2_score
# from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from lmfit.models import ExpressionModel
from lmfit import Model

set_font_type()

# %%
##### Parameters to change #####
pick_data = 'hc' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day' 'night', or 'all'
##### Parameters to change #####

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'Navi5_IBItime_z{which_ztime}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {fig_dir}')
except:
    pass

root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime)

# # %%

# all_feature_cond = all_feature_cond.assign(
#     pitch_peak_abs = all_feature_cond['pitch_peak'].abs(),
#     yspd_peak = all_feature_cond['spd_peak']*np.sin(all_feature_cond['traj_peak']* np.pi / 180),
#     xspd_peak = all_feature_cond['spd_peak']*np.cos(all_feature_cond['traj_peak']* np.pi / 180),
#     lift_ratio = all_feature_cond['additional_depth_chg']/all_feature_cond['depth_chg'],
# )
# all_feature_cond = all_feature_cond.assign(
#     yposture_spd_peak = all_feature_cond['xspd_peak']*np.tan(all_feature_cond['pitch_peak']* np.pi / 180),
# )
# all_feature_cond = all_feature_cond.assign(
#     lift_spd = all_feature_cond['yspd_peak'] - all_feature_cond['yposture_spd_peak']
# )

# all_feature_cond = all_feature_cond.assign(
#     atk_ang_bins = pd.cut(all_feature_cond.atk_ang, bins=[-50,-20,-2,3,20,50]),
# )

# %% 
x_name = 'pre_IBI_time'

df_toplt_filtered = all_feature_cond.dropna(subset=[x_name])
df_toplt_filtered = df_toplt_filtered.sort_values(by=['cond0', 'cond1', 'ztime']).reset_index(drop=True)

# %% IBI duraion (idle time) histogram

g = sns.displot(
    df_toplt_filtered,
    x=x_name,
    col='cond0',
    row='ztime',
    height=3,
    hue='cond1',
    palette='gray',
    stat='density',
    log_scale=True,
    element = 'poly',
    bins=80,
    common_norm=False,
)
g.set(
    xlim=[np.percentile(df_toplt_filtered[x_name],0.5), np.percentile(df_toplt_filtered[x_name],99.5)],
)   
plt.savefig(fig_dir+f"/pre_IBI dist.pdf",format='PDF')


# %%

# # %% idle time and swim parameters

# for condition0 in all_feature_cond['cond0'].unique():
#     data_toplt = all_feature_cond.query("cond0 == @condition0")
    
#     x_name = 'pitch_initial'
#     y_name = 'pre_IBI_time'
    
#     df_toplt_filtered = data_toplt.dropna(subset=[x_name, y_name])

#     g = sns.displot(
#         df_toplt_filtered,
#         x=x_name,
#         y=y_name,
#         col='cond1',
#         row='cond0',
#         height=3,
#         cbar=True, 
#         hue='cond1',
#         pthresh=.05,
#         palette='gray',
#         stat='count',
#         bins=[200,200],
#         common_norm=False,
#     )
#     g.set(
#         xlim=[np.percentile(df_toplt_filtered[x_name],1), np.percentile(df_toplt_filtered[x_name],99)],
#         ylim=[np.percentile(df_toplt_filtered[y_name],1), np.percentile(df_toplt_filtered[y_name],99)],
#     )   
#     plt.savefig(fig_dir+f"/dis_{condition0} {y_name} vs {x_name}.pdf",format='PDF')

#     ###
    
#     x_name = 'pitch_end'
#     y_name = 'post_IBI_time'

#     df_toplt_filtered = data_toplt.dropna(subset=[x_name, y_name])

#     g = sns.displot(
#         df_toplt_filtered,
#         x=x_name,
#         y=y_name,
#         col='cond1',
#         row='cond0',
#         height=3,
#         cbar=True, 
#         hue='cond1',
#         pthresh=.05,
#         palette='gray',
#         stat='count',
#         bins=[200,200],
#         common_norm=False,
#     )
#     g.set(
#         xlim=[np.percentile(df_toplt_filtered[x_name],1), np.percentile(df_toplt_filtered[x_name],99)],
#         ylim=[np.percentile(df_toplt_filtered[y_name],1), np.percentile(df_toplt_filtered[y_name],99)],
#     )   
#     plt.savefig(fig_dir+f"/dis_{condition0} {y_name} vs {x_name}.pdf",format='PDF')

# %%
# for condition0 in all_feature_cond['cond0'].unique():
#     data_toplt = all_feature_cond.query("cond0 == @condition0")

#     x_name = 'pitch_peak'
#     y_name = 'rot_full_accel'

#     # x_range = np.arange(-35,40,0.5)

#     df_toplt_filtered = data_toplt.dropna(subset=[x_name, y_name])
#     # df_toplt_filtered = df_toplt_filtered.query("pre_IBI_time > 4")

#     binned_angles_raw = pd.DataFrame()

#     g = sns.displot(
#         df_toplt_filtered,
#         x=x_name,
#         y=y_name,
#         col='cond1',
#         row='ztime',
#         height=3,
#         cbar=True, 
#         hue='cond1',
#         pthresh=.05,
#         palette='gray',
#         stat='count',
#         bins=[200,200],
#         common_norm=False,
#     )
#     g.set(
#         xlim=[np.percentile(df_toplt_filtered[x_name],1), np.percentile(df_toplt_filtered[x_name],99)],
#         ylim=[np.percentile(df_toplt_filtered[y_name],1), np.percentile(df_toplt_filtered[y_name],99)],
#     )   
#     plt.savefig(fig_dir+f"/dis_{condition0} {y_name} vs {x_name}.pdf",format='PDF')


