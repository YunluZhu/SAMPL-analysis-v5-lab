"""
This script is used to plot the time series of the angle between the body and head direction.
Only compatible with the data analyzed using script after march 2025
"""

# %%
# import sys
import os, glob
import pandas as pd
from plot_functions.plt_tools import round_half_up
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math
from plot_functions.get_data_dir import get_data_dir, get_figure_dir
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from plot_functions.plt_tools import set_font_type, defaultPlotting, day_night_split
from tqdm import tqdm

##### Parameters to change #####
pick_data = (
    "meclizine"  # name of your dataset to plot as defined in function get_data_dir()
)
which_ztime = "day"  # 'day' or 'night', does not support 'all'
BIN_NUM = 4  # number of speed bins
##### Parameters to change #####

# %%
# Paste root directory here
root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f"BT4_angleCoordinates"
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f"fig folder created:{folder_name}")
except:
    print("Notes: re-writing old figures")


peak_idx, total_aligned = get_index(FRAME_RATE)
idxRANGE = [
    peak_idx - round_half_up(0.3 * FRAME_RATE),
    peak_idx + round_half_up(0.2 * FRAME_RATE),
]
set_font_type()

# %%
# CONSTANTS
SMOOTH = 11
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != ".":
        folder_paths.append(root + "/" + folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()
all_cond0 = []
all_cond1 = []

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch
                raw = pd.read_hdf(
                    f"{exp_path}/bout_data.h5", key="prop_bout_aligned"
                )  # .loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                raw = raw.assign(
                    ang_speed=raw["propBoutAligned_angVel"].abs(),
                    yvel=raw["propBoutAligned_y"].diff() * FRAME_RATE,
                    xvel=raw["propBoutAligned_x"].diff() * FRAME_RATE,
                    linear_accel=raw["propBoutAligned_speed"].diff(),
                    ang_accel_of_SMangVel=raw["propBoutAligned_angVel"].diff(),
                )
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(
                    idx=round_half_up(len(raw) / total_aligned)
                    * list(range(0, total_aligned))
                )

                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(
                    f"{exp_path}/bout_data.h5", key="prop_bout2"
                ).loc[:, ["aligned_time"]]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(
                    bout_time, "aligned_time", ztime=which_ztime
                ).index:
                    rows.extend(
                        list(
                            range(
                                i * total_aligned + idxRANGE[0],
                                i * total_aligned + idxRANGE[1],
                            )
                        )
                    )
                exp_data = raw.loc[rows, :]
                exp_data = exp_data.assign(
                    expNum=expNum, exp_id=condition_idx * 100 + expNum
                )
                grp = exp_data.groupby(
                    np.arange(len(exp_data)) // (idxRANGE[1] - idxRANGE[0])
                )
                angvel_smoothed = grp["propBoutAligned_angVel"].apply(
                    lambda x: savgol_filter(x, 11, 3)
                )
                # exp_data = exp_data.assign(
                #     # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                #     traj_cur=angvel_smoothed / exp_data["propBoutAligned_speed"] * math.pi/ 180
                # )
                around_peak_data = pd.concat([around_peak_data, exp_data])
    # combine data from different conditions
    cond0 = all_conditions[condition_idx].split("_")[0]
    all_cond0.append(cond0)
    cond1 = all_conditions[condition_idx].split("_")[1]
    all_cond1.append(cond1)
    all_around_peak_data = pd.concat(
        [all_around_peak_data, around_peak_data.assign(cond0=cond0, cond1=cond1)]
    )
all_around_peak_data = all_around_peak_data.assign(
    time_ms=(all_around_peak_data["idx"] - peak_idx) / FRAME_RATE * 1000
)
# %% tidy data
all_cond0 = list(set(all_cond0))
all_cond0.sort()
all_cond1 = list(set(all_cond1))
all_cond1.sort()

all_around_peak_data = all_around_peak_data.reset_index(drop=True)
peak_speed = (
    all_around_peak_data.loc[
        all_around_peak_data.idx == peak_idx, "propBoutAligned_speed"
    ],
)

all_around_peak_data = all_around_peak_data.assign(
    heading_sub_pitch=all_around_peak_data["propBoutAligned_instHeading"]
    - all_around_peak_data["propBoutAligned_pitch"],
)

grp = all_around_peak_data.groupby(
    np.arange(len(all_around_peak_data)) // (idxRANGE[1] - idxRANGE[0])
)
all_around_peak_data = all_around_peak_data.assign(
    peak_speed=np.repeat(peak_speed, (idxRANGE[1] - idxRANGE[0])),
    bout_number=grp.ngroup(),
)
speed_bins = all_around_peak_data["peak_speed"].quantile(np.arange(0, 1, 1 / BIN_NUM))
all_around_peak_data = all_around_peak_data.assign(
    speed_bin=pd.cut(
        all_around_peak_data["peak_speed"],
        bins=np.append(speed_bins.values, [np.inf]),
        labels=np.arange(BIN_NUM),
    )
)
print("speed buckets:")
print("--mean")
print(all_around_peak_data.groupby("speed_bin")["peak_speed"].agg("mean"))
print("--min")
print(all_around_peak_data.groupby("speed_bin")["peak_speed"].agg("min"))
print("--max")
print(all_around_peak_data.groupby("speed_bin")["peak_speed"].agg("max"))

# %%
# Peak data and pitch segmentation
T_INITIAL = -0.25  # s
T_PREP_200 = -0.2
T_PREP_150 = -0.15
T_PRE_BOUT = -0.10  # s
T_POST_BOUT = 0.1  # s
T_END = 0.2
T_MID_ACCEL = -0.05
T_MID_DECEL = 0.05
idx_initial = round_half_up(peak_idx + T_INITIAL * FRAME_RATE)
idx_pre_bout = round_half_up(peak_idx + T_PRE_BOUT * FRAME_RATE)
idx_post_bout = round_half_up(peak_idx + T_POST_BOUT * FRAME_RATE)

peak_data = all_around_peak_data.loc[
    all_around_peak_data["idx"] == peak_idx
].reset_index(drop=True)
peak_data = peak_data.assign(
    pitch_pre_bout=all_around_peak_data.loc[
        all_around_peak_data["idx"] == idx_pre_bout, "propBoutAligned_pitch"
    ].values,
    pitch_post_bout=all_around_peak_data.loc[
        all_around_peak_data["idx"] == idx_post_bout, "propBoutAligned_pitch"
    ].values,
    pitch_initial=all_around_peak_data.loc[
        all_around_peak_data["idx"] == idx_initial, "propBoutAligned_pitch"
    ].values,
)

yy = (
    all_around_peak_data.loc[
        all_around_peak_data["idx"] == idx_post_bout, "propBoutAligned_y"
    ].values
    - all_around_peak_data.loc[
        all_around_peak_data["idx"] == idx_pre_bout, "propBoutAligned_y"
    ].values
)
absxx = np.absolute(
    (
        all_around_peak_data.loc[
            all_around_peak_data["idx"] == idx_post_bout, "propBoutAligned_x"
        ].values
        - all_around_peak_data.loc[
            all_around_peak_data["idx"] == idx_pre_bout, "propBoutAligned_x"
        ].values
    )
)
epochBouts_trajectory = np.degrees(
    np.arctan(yy / absxx)
)  # direction of the bout, -90:90
peak_data = peak_data.assign(
    traj_deviation=peak_data["propBoutAligned_instHeading"]
    - peak_data["pitch_pre_bout"].values,
    rot_l_decel=peak_data["pitch_post_bout"] - peak_data["propBoutAligned_pitch"],
    rot_full_accel=peak_data["propBoutAligned_pitch"] - peak_data["pitch_initial"],
)


peak_grp = peak_data.groupby(["expNum", "cond1"], as_index=False)

# assign by pitch
neg_pitch_bout_num = peak_data.loc[peak_data["pitch_pre_bout"] < 10, "bout_number"]
pos_pitch_bout_num = peak_data.loc[peak_data["pitch_pre_bout"] > 10, "bout_number"]
all_around_peak_data = all_around_peak_data.assign(pitch_dir="neg_pitch")
all_around_peak_data.loc[
    all_around_peak_data["bout_number"].isin(pos_pitch_bout_num.values), "pitch_dir"
] = "pos_pitch"

# assign by traj deviation
neg_trajDev_bout_num = peak_data.loc[peak_data["traj_deviation"] < 0, "bout_number"]
pos_trajDev_bout_num = peak_data.loc[peak_data["traj_deviation"] > 0, "bout_number"]
all_around_peak_data = all_around_peak_data.assign(
    traj_deviation_dir="neg_traj_deviation"
)
all_around_peak_data.loc[
    all_around_peak_data["bout_number"].isin(pos_trajDev_bout_num.values),
    "traj_deviation_dir",
] = "pos_traj_deviation"


SpRp = peak_data.loc[
    (peak_data.rot_full_accel > 0) & (peak_data.rot_l_decel > 0), "bout_number"
]
SnRp = peak_data.loc[
    (peak_data.rot_full_accel < 0) & (peak_data.rot_l_decel > 0), "bout_number"
]
SpRn = peak_data.loc[
    (peak_data.rot_full_accel > 0) & (peak_data.rot_l_decel < 0), "bout_number"
]
SnRn = peak_data.loc[
    (peak_data.rot_full_accel < 0) & (peak_data.rot_l_decel < 0), "bout_number"
]

all_around_peak_data = all_around_peak_data.assign(SR_category="SpRp")
all_around_peak_data.loc[
    all_around_peak_data["bout_number"].isin(SnRp.values), "SR_category"
] = "SnRp"
all_around_peak_data.loc[
    all_around_peak_data["bout_number"].isin(SpRn.values), "SR_category"
] = "SpRn"
all_around_peak_data.loc[
    all_around_peak_data["bout_number"].isin(SnRn.values), "SR_category"
] = "SnRn"

# %%


# Calculate the angle between (x, y) and (headx, heady)
def calculate_head_body_angle_vectorized(df):
    """Calculates the angle (in degrees) between the body center and head using vectorized operations."""
    dx = np.abs(df["propBoutAligned_headx"].values - df["propBoutAligned_x"].values)
    dy = df["propBoutAligned_heady"].values - df["propBoutAligned_y"].values
    angle_rad = np.arctan2(dy, dx)
    angle_deg = np.degrees(angle_rad)

    # Adjust angles to be within -90 to 90 range using vectorized operations
    angle_deg = np.where(angle_deg > 90, angle_deg - 180, angle_deg)
    angle_deg = np.where(angle_deg <= -90, angle_deg + 180, angle_deg)

    return angle_deg


def calculate_distance_between_points_vectorized(df):
    """
    Calculates the Euclidean distance between two sets of points using vectorized operations.

    Args:
        df (pd.DataFrame): DataFrame containing the coordinate columns.
        x1_col (str): Name of the column containing the x-coordinates of the first point.
        y1_col (str): Name of the column containing the y-coordinates of the first point.
        x2_col (str): Name of the column containing the x-coordinates of the second point.
        y2_col (str): Name of the column containing the y-coordinates of the second point.

    Returns:
        np.ndarray: Array of distances between the two sets of points.
    """
    x1_col, y1_col, x2_col, y2_col = (
        "propBoutAligned_headx",
        "propBoutAligned_heady",
        "propBoutAligned_x",
        "propBoutAligned_y",
    )
    dx = df[x2_col].values - df[x1_col].values
    dy = df[y2_col].values - df[y1_col].values
    distance = np.sqrt(dx**2 + dy**2)
    return distance


# Apply the vectorized function
all_around_peak_data["head_body_angle"] = calculate_head_body_angle_vectorized(
    all_around_peak_data
)
all_around_peak_data["head_body_dist"] = calculate_distance_between_points_vectorized(
    all_around_peak_data
)

# %%
all_around_peak_data["ang_difference"] = (
    all_around_peak_data["propBoutAligned_pitch"]
    - all_around_peak_data["head_body_angle"]
)
all_around_peak_data["ang_difference_adj"] = (
    all_around_peak_data["ang_difference"]
    - all_around_peak_data["ang_difference"].mean()
) * -1
# average per IBI
df_angDiff_avgEpoch = all_around_peak_data.groupby(
    ["cond0", "cond1", "expNum", "bout_number"]
)["ang_difference"].mean()
df_angDiff_avgEpoch = df_angDiff_avgEpoch.reset_index()

# df_angDiff_avgEpoch['box_uid'] = df_angDiff_avgEpoch['expNum'].astype(str).str.cat(df_angDiff_avgEpoch['boxNum'].astype(str), sep='_')

# %%
df_toplt = df_angDiff_avgEpoch
sns.displot(
    kind="ecdf",
    # stat='probability',
    # common_norm=False,
    data=df_toplt,
    x="ang_difference",
    col="cond0",
    # row='cond',
    hue="cond1",
    height=3,
    # log_scale=True,
    facet_kws={
        "xlim": [
            np.percentile(df_toplt["ang_difference"], 0.1),
            np.percentile(df_toplt["ang_difference"], 99.9),
        ],
        "ylim": [0, 1],
    },
)
plt.savefig(fig_dir + f"/ecdf on avg per bout first half.pdf", format="PDF")

# %%

sns.displot(
    kind="hist",
    stat="probability",
    common_norm=False,
    data=all_around_peak_data,
    x="ang_difference",
    col="cond0",
    # row='cond',
    hue="cond1",
    height=3,
    element="poly",
    facet_kws={
        "xlim": np.percentile(all_around_peak_data["ang_difference"], [0.1, 99.9])
    },
)


# %%
sns.displot(
    kind="ecdf",
    data=all_around_peak_data,
    x="ang_difference",
    col="cond0",
    # row='cond',
    hue="cond1",
    hue_order=np.sort(all_around_peak_data.cond1.unique()),
    height=3,
    facet_kws={
        "xlim": [
            np.percentile(all_around_peak_data["ang_difference"], 0.1),
            all_around_peak_data["ang_difference"].median(),
        ],
        "ylim": [0, 0.5],
    },
)
plt.savefig(fig_dir + f"/ecdf all timepoints bout first half.pdf", format="PDF")

# %%
df_prebout = all_around_peak_data.loc[all_around_peak_data["time_ms"] < -100, :]
df_postbout = all_around_peak_data.loc[all_around_peak_data["time_ms"] > 100, :]
df_duringbout = all_around_peak_data.loc[
    (all_around_peak_data["time_ms"] < 100) & (all_around_peak_data["time_ms"] > -100),
    :,
]
df_prebout_avg = (
    df_prebout.groupby(["cond0", "cond1", "expNum", "bout_number"])["ang_difference"]
    .apply(np.nanmean)
    .reset_index()
)
df_postbout_avg = (
    df_postbout.groupby(["cond0", "cond1", "expNum", "bout_number"])["ang_difference"]
    .apply(np.nanmean)
    .reset_index()
)

# %%
df_toplt = df_duringbout  # .loc[df_postbout_avg['ang_difference']<df_postbout_avg['ang_difference'].median(),:]
sns.displot(
    kind="ecdf",
    # stat='probability',
    # common_norm=False,
    data=df_toplt,
    x="ang_difference",
    col="cond0",
    # row='cond',
    hue="cond1",
    height=3,
    # log_scale=True,
    facet_kws={
        "xlim": [
            np.percentile(df_toplt["ang_difference"], 0.1),
            df_toplt["ang_difference"].median(),
        ],
        "ylim": [0, 0.5],
    },
)


# %% time series

toplt = all_around_peak_data

feature_toplt = "ang_difference_adj"
p = sns.relplot(
    data=toplt,
    x="time_ms",
    y=feature_toplt,
    row="speed_bin",
    hue="cond1",
    hue_order=all_cond1,
    style="cond0",
    style_order=all_cond0,
    # ci='sd',
    kind="line",
    aspect=2.4,
    height=3,
)
plt.savefig(fig_dir + f"/timeseries of ang diff.pdf", format="PDF")

# #%% look at those that flip / corkscrew during bouts
# all_around_peak_data["ang_diff_assert"] = all_around_peak_data.groupby(
#     np.arange(len(all_around_peak_data)) // (idxRANGE[1] - idxRANGE[0])
# )['ang_difference'].transform(lambda x: x.iloc[18:82].mean())

# df_flip = all_around_peak_data.query("ang_diff_assert < -0")

# #%%
# toplt = df_flip.loc[df_flip.ang_diff_assert.abs()<5]

# feature_toplt = "ang_difference"
# p = sns.relplot(
#     data=toplt,
#     x="time_ms",
#     y=feature_toplt,
#     row="speed_bin",
#     hue="cond1",
#     hue_order=all_cond1,
#     style="cond0",
#     style_order=all_cond0,
#     # ci='sd',
#     kind="line",
#     aspect=2.4,
#     height=3,
# )


# # %% by speed and rotation. of control
# toplt = all_around_peak_data.query("cond1 == 'ctrl'")

# df_reconstruct = pd.concat(
#    [ toplt.rename(columns={"ang_difference_adj": "value"}).assign(feature='ang_difference_adj')[['time_ms','value','feature','SR_category','cond0','cond1','speed_bin']],
#     toplt.rename(columns={"propBoutAligned_angVel": "value"}).assign(feature='angVel')[[ 'time_ms','value','feature','SR_category','cond0','cond1','speed_bin']],
#     ],ignore_index=True
#     )

# p = sns.relplot(
#     data=df_reconstruct,
#     x="time_ms",
#     y='value',
#     col="SR_category",
#     row="feature",
#     hue='speed_bin',
#     kind="line",
#     aspect=2.4,
#     height=2,
#     facet_kws={'sharey':'row'},
# )
# plt.savefig(fig_dir + f"/timeseries ctrl ang diff angvel.pdf", format="PDF")


# %% any chance we can detect single undulation effects
toplt = all_around_peak_data

df_reconstruct = pd.concat(
    [
        toplt.rename(columns={"ang_difference_adj": "value"}).assign(
            feature="ang_difference_adj"
        )[
            [
                "time_ms",
                "value",
                "feature",
                "SR_category",
                "cond0",
                "cond1",
                "speed_bin",
            ]
        ],
        toplt.rename(columns={"head_body_dist": "value"}).assign(
            feature="head_body_dist"
        )[
            [
                "time_ms",
                "value",
                "feature",
                "SR_category",
                "cond0",
                "cond1",
                "speed_bin",
            ]
        ],
        toplt.rename(columns={"propBoutAligned_speed": "value"}).assign(
            feature="speed"
        )[
            [
                "time_ms",
                "value",
                "feature",
                "SR_category",
                "cond0",
                "cond1",
                "speed_bin",
            ]
        ],
        toplt.rename(columns={"fish_length": "value"}).assign(
            feature="fish_length"
        )[
            [
                "time_ms",
                "value",
                "feature",
                "SR_category",
                "cond0",
                "cond1",
                "speed_bin",
            ]
        ],
    ],
    ignore_index=True,
)

p = sns.relplot(
    data=df_reconstruct.groupby(['time_ms','feature','cond0','cond1'], as_index=False)['value'].mean().reset_index(),
    x="time_ms",
    y="value",
    row="feature",
    col="cond0",
    hue='cond1',
    kind="line",
    aspect=2.4,
    height=2,
    facet_kws={"sharey": "row"},
    errorbar=None,
)
plt.savefig(fig_dir + f"/timeseries ctrl undulation.pdf", format="PDF")

# %%
bout_df = all_around_peak_data

what_feature = "propBoutAligned_absy"

bout_df_avg = bout_df.groupby(
    ["cond0", "cond1", "expNum", "bout_number", "traj_deviation_dir","speed_bin"]
)[["ang_difference_adj",'propBoutAligned_pitch',what_feature]].mean()

#%%
bout_df_avg['y_signAdjusted'] = bout_df_avg['propBoutAligned_absy'] * -1 / 60
sns.displot(
    data=bout_df_avg,
    x='y_signAdjusted',
    y='ang_difference_adj',
    col='cond1',
    row='traj_deviation_dir',
    facet_kws={
        'ylim': [-3,3],
        # 'xlim': [-45,55],
        },
    height=2.5,
)
# %%
