"""
For multiple comparisons across conditions and day night

"""

# %%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import get_data_dir, get_figure_dir
from plot_functions.get_bout_features import get_connected_bouts
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_tools import (
    set_font_type,
    defaultPlotting,
    distribution_binned_average,
    distribution_binned_average_opt,
)
from plot_functions.plt_functions import plt_categorical_combined_3
from plot_functions.get_bout_consecutive_features import (
    extract_consecutive_bout_features,
)
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

import scipy.stats as st
import statsmodels.api as sm
import statsmodels.robust.norms as norms
from sklearn.metrics import r2_score

# %%

##### Parameters to change #####
pick_data = "wt_light_long"  # name of your dataset to plot as defined in function get_data_dir()
which_ztime = "day"  # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split("/")[-1].replace(".py", "")
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f"fig folder created: {folder_name}")
except:
    print("Notes: re-writing old figures")

set_font_type()
mpl.rc("figure", max_open_warning=0)

# %% get features
all_feature_cond, all_cond0, all_cond1 = get_connected_bouts(
    root,
    FRAME_RATE,
    ztime=which_ztime,
    if_strict_DayNightSplit=True,
)

# %% tidy data
# all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)
# tidy bout uid
all_features = all_feature_cond.assign(
    epoch_uid=all_feature_cond["cond0"]
    + all_feature_cond["cond1"]
    + all_feature_cond["expNum"].astype(str)
    + all_feature_cond["epoch_uid"],
    exp_uid=all_feature_cond["cond0"]
    + all_feature_cond["cond1"]
    + all_feature_cond["expNum"].astype(str),
)

# select dataset
all_features = all_features.query('cond1 == "ld"')
# %%
list_of_features = [
    "displ_swim",
    'bout_time',
    "y_initial",
    "y_end",
    "x_initial",
    "x_end",
]

# %% associate consecutive bouts

#####################
max_lag = 4
#####################
consecutive_bout_features, _ = extract_consecutive_bout_features(
    all_features, list_of_features, max_lag
)

# %%
sel_consecutive_bouts = consecutive_bout_features.sort_values(
    by=["cond1", "cond0", "id", "lag", "ztime"]
).reset_index(drop=True)
sel_consecutive_bouts = sel_consecutive_bouts.assign(
    bouts=sel_consecutive_bouts["lag"] + 1
)
sel_consecutive_bouts['x_initial_first'] = sel_consecutive_bouts['x_initial_first'].abs()
sel_consecutive_bouts['x_initial'] = sel_consecutive_bouts['x_initial'].abs()

selected_data = (
    sel_consecutive_bouts.groupby(["cond1", "ztime", "expNum", "id"], as_index=False)
    .apply(
        lambda group: group.assign(
            time_elapsed=lambda df: df["bout_time"] - df["bout_time_first"],
            total_displ=lambda df: df["displ_swim"].cumsum(),
            position_change_y=lambda df: df["y_initial"] - df["y_initial_first"],
            position_change_x=lambda df: df["x_initial"] - df["x_initial_first"],
        ),
        include_groups=False,
    )
    .reset_index(drop=True)  # Reset index after apply()
)
selected_data = selected_data.assign(
    avg_speed=lambda df: df["total_displ"] / df["time_elapsed"].dt.total_seconds(),
    position_change_total=lambda df: np.sqrt(
        df["position_change_y"] ** 2 + df["position_change_x"] ** 2
    ),
)
selected_data = selected_data.assign(
    position_chg_per_sec=lambda df: df["position_change_total"]
    / df["time_elapsed"].dt.total_seconds()
)
# %%
avg_data = selected_data.groupby(
    ["cond0", "exp_conduid",'bouts'], as_index=False).agg(
    time_dur=("time_elapsed", "median"),
    total_displ=("total_displ", "median"),
    avg_speed=("avg_speed", "median"),
    position_change_total=("position_change_total", "median"),
    position_chg_per_sec=("position_chg_per_sec", "median"),
    position_change_y=("position_change_y", "median"),
    position_change_x=("position_change_x", "median"),
)

# %%
plt_categorical_combined_3(
    data=avg_data.query("bouts==5"),
    x="cond0",
    y="total_displ",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"total_displacement_bouts5.pdf"), format='PDF')

plt_categorical_combined_3(
    data=avg_data.query("bouts==5"),
    x="cond0",
    y="time_dur",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"time_duration_bouts5.pdf"), format='PDF')


plt_categorical_combined_3(
    data=avg_data.query("bouts==5"),
    x="cond0",
    y="position_change_total",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"position_change_total_bouts5.pdf"), format='PDF')

plt_categorical_combined_3(
    data=avg_data.query("bouts==5"),
    x="cond0",
    y="position_change_y",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"position_change_y_bouts5.pdf"), format='PDF')

plt_categorical_combined_3(
    data=avg_data.query("bouts==5"),
    x="cond0",
    y="position_change_x",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"position_change_x_bouts5.pdf"), format='PDF')
#%% average speed

avg_data_toplt = avg_data.query("bouts==5")

plt_categorical_combined_3(
    data=avg_data_toplt,
    x="cond0",
    y="avg_speed",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"average_speed_bouts5.pdf"), format='PDF')

plt_categorical_combined_3(
    data=avg_data_toplt,
    x="cond0",
    y="position_chg_per_sec",
    hue="cond0",
    units="exp_conduid",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(os.path.join(fig_dir, f"position_chg_per_sec_bouts5.pdf"), format='PDF')
# %%
