""" """

# %%
# import sys
import os

# import time
import numpy as np  # numpy
import seaborn as sns
import matplotlib.pyplot as plt

# from statsmodels.stats.multicomp import (pairwise_tukeyhsd, MultiComparison)
from plot_functions.get_data_dir import get_data_dir, get_figure_dir
from plot_functions.plt_tools import set_font_type, defaultPlotting, MAD
from plot_functions.get_bout_features import get_bout_features
from plot_functions.plt_functions import plt_categorical_combined

set_font_type()
defaultPlotting()


pick_data = (
    "sldp2025"  # name of your dataset to plot as defined in function get_data_dir()
)
which_ztime = "day"  # 'day', 'night', or 'all'

# %%
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f"bf2_MAD_{which_ztime}"
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f"fig folder created: {folder_name}")
except:
    print("Notes: re-writing old figures")

set_font_type()


root, FRAME_RATE = get_data_dir(pick_data)
all_feature_cond, _, _ = get_bout_features(root, FRAME_RATE, ztime=which_ztime)
all_feature_cond["dataset"] = pick_data

sns.set_style("ticks")

# %%
dtype_sel = all_feature_cond.dtypes == "float64"
columns = all_feature_cond.columns
columns = np.array(columns)
feature_toplt = columns[dtype_sel.values]

df_MAD_res = (
    all_feature_cond.groupby(["cond0", "cond1", "expNum"])[feature_toplt]
    .apply(lambda x: MAD(x))
    .reset_index()
)


# %%

# %%
print("---Compare MAD---")
toplt = df_MAD_res

# %%
for feature in feature_toplt:
    plt_categorical_combined(
        data=toplt,
        x="cond1",
        y=feature,
        col="cond0",
        row=None,
        units="expNum",
        related=True,
        sharey=True,
        errorbar="se",
        alpha=0.5,
    )
    plt.savefig(fig_dir + f"/{feature} compare median.pdf", format="PDF")


# %%
from scipy.stats import zscore

listoffeatures = [
    # "rot_full_accel",
    # "rot_full_decel",
    # "angvel_post_phase",
    "pitch_peak",
    "pitch_initial",
    "pitch_end",
]
zdf = zscore(toplt[listoffeatures]).assign(
    cond0=toplt.cond0,
    cond1=toplt.cond1,
    expNum=toplt.expNum,
)
zdf["avg"] = zdf[listoffeatures].mean(1)
for feature in ["avg"]:
    sns.catplot(
        kind="point",
        data=zdf,
        x="cond1",
        y=feature,
        col="cond0",
        row=None,
        hue="expNum",
        join=True,
    )
# %%
