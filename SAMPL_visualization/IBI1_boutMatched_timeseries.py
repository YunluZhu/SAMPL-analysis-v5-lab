

#%%
# import sys
import os
import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.plt_tools import round_half_up 
import math
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from plot_functions.plt_tools import (set_font_type, defaultPlotting, day_night_split)
from tqdm import tqdm
from plot_functions.get_IBIangles import get_timeseriesIBIangles
from plot_functions.get_bout_features import get_aligned_bouts_wIBI
import matplotlib as mpl

##### Parameters to change #####
pick_data = 'blind' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'day' # 'day' or 'night', does not support 'all'
if_strict_DayNightSplit = True # if True, only include bouts that are fully within the day or night period. If False, include bouts that may cross the day/night boundary.
##### Parameters to change #####

#%%
# Paste root directory here
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
mpl.rc('figure', max_open_warning = 0)

#%%
# here's the new function. It recovers the exact bout-IBI correspondence and returns the raw bout data, bout time data, and raw IBI data.
# the way we matched IBI and bout is by the location of the bout in the original dataset (bout-attributes > boutNum)
# then, in IEI_attributes, we can find the corresponding boutNum and epochNum, and then match them to the bout-attributes to get the bout_id_IBImatched.

def load_matched_data(exp_path, frame_rate, total_aligned):
    """Recover exact bout-IBI correspondence before any filtering."""
    bout_file = os.path.join(exp_path, "bout_data.h5")
    ibi_file = os.path.join(exp_path, "IEI_data.h5")

    raw = pd.read_hdf(bout_file, "prop_bout_aligned").reset_index(drop=True)
    bout_time = pd.read_hdf(bout_file, "prop_bout2").reset_index(drop=True)
    ba = pd.read_hdf(bout_file, "bout_attributes").reset_index(drop=True)
    ia = pd.read_hdf(ibi_file, "IEI_attributes").reset_index(drop=True)
    ibi_raw = pd.read_hdf(
        ibi_file, "prop_bout_IEI_aligned"
    ).reset_index(drop=True)

    # boutNum restarts at zero for each original .dlm file.
    ba["file_id"] = ba["boutNum"].eq(0).cumsum() - 1
    ba["bout_id_IBImatched"] = np.arange(len(ba))

    # Aligned bouts retain the order of ba[if_align].
    aligned_ba = ba.loc[ba["if_align"]].reset_index(drop=True)

    assert len(aligned_ba) == len(bout_time)
    assert len(raw) == len(bout_time) * total_aligned

    np.testing.assert_array_equal(
        aligned_ba["boutNum"].to_numpy(),
        bout_time["propBout_matchIndex"].to_numpy(),
    )
    np.testing.assert_array_equal(
        aligned_ba["propBout_time"].to_numpy(),
        bout_time["aligned_time"].to_numpy(),
    )

    bout_time["bout_id_IBImatched"] = aligned_ba["bout_id_IBImatched"].to_numpy()
    raw["bout_id_IBImatched"] = np.repeat(
        bout_time["bout_id_IBImatched"].to_numpy(), total_aligned
    )
    raw["idx"] = np.tile(
        np.arange(total_aligned), len(bout_time)
    )

    # IEI_attributes excludes the first bout of each epoch.
    within_epoch = ba.groupby(
        ["file_id", "epochNum"], sort=False
    ).cumcount()

    following_bouts = ba.loc[within_epoch.gt(0)].reset_index(drop=True)

    cols = [
        "boutNum", "epochNum", "swim_start_idx", "swim_end_idx", "boxNum"
    ]
    pd.testing.assert_frame_equal(
        following_bouts[cols], ia[cols], check_dtype=False
    )

    ia["file_id"] = following_bouts["file_id"].to_numpy()
    ia["bout_id_IBImatched"] = following_bouts["bout_id_IBImatched"].to_numpy()
    ia["local_iei_id"] = ia.groupby("file_id", sort=False).cumcount()
    ia["ibi_id"] = np.arange(len(ia))

    # Actual saved IBI trace boundaries, inclusive.
    buf = math.ceil(0.1 * frame_rate)
    ia["n_frames"] = (
        ia["swim_start_idx"] - ia["swim_end_shift"] - 2 * buf + 1
    ).astype(int)

    saved_ia = ia.loc[ia["n_frames"].gt(1)]
    counts = saved_ia["n_frames"].to_numpy()

    assert counts.sum() == len(ibi_raw)

    np.testing.assert_array_equal(
        np.repeat(saved_ia["local_iei_id"].to_numpy(), counts),
        ibi_raw["IEI_matchIndex"].to_numpy(),
    )

    ibi_raw["ibi_id"] = np.repeat(
        saved_ia["ibi_id"].to_numpy(), counts
    )
    ibi_raw["bout_id_IBImatched"] = np.repeat(
        saved_ia["bout_id_IBImatched"].to_numpy(), counts
    )

    return raw, bout_time, ibi_raw
#%%
# %%
# this section remain the same as plt_functions > get_bout_features > get_aligned_bouts_wIBI
# I just cleaned it up a bit using AI

peak_idx, total_aligned = get_index(FRAME_RATE)

idxRANGE = [
    peak_idx - round_half_up(0.3 * FRAME_RATE),
    peak_idx + round_half_up(0.2 * FRAME_RATE),
]

SMOOTH = 11

all_full_conditions = sorted(
    entry.name
    for entry in os.scandir(root)
    if entry.is_dir() and not entry.name.startswith(".")
)
folder_paths = [os.path.join(root, name) for name in all_full_conditions]

all_cond0 = []
all_cond1 = []
bout_parts = []
ibi_parts = []

for full_condition_idx, (full_condition, folder) in enumerate(
    zip(all_full_conditions, folder_paths)
):
    cond0, cond1 = full_condition.split("_", 1)
    all_cond0.append(cond0)
    all_cond1.append(cond1)

    experiments = sorted(
        entry.name
        for entry in os.scandir(folder)
        if entry.is_dir() and not entry.name.startswith(".")
    )

    for expNum, exp in enumerate(experiments):
        exp_path = os.path.join(folder, exp)

        raw, bout_time, ibi_raw = load_matched_data(
            exp_path, FRAME_RATE, total_aligned
        )

        # Calculate velocities within each bout, avoiding differences
        # across the boundary between consecutive aligned windows.
        raw["yvel"] = (
            raw.groupby("bout_id_IBImatched", sort=False)["propBoutAligned_y"]
            .diff() * FRAME_RATE
        )
        raw["xvel"] = (
            raw.groupby("bout_id_IBImatched", sort=False)["propBoutAligned_x"]
            .diff() * FRAME_RATE
        )

        # Carry bout_id_IBImatched through your existing day/night selection.
        selected = day_night_split(
            bout_time,
            "aligned_time",
            narrow_bin=if_strict_DayNightSplit,
            ztime=which_ztime,
        )

        selection = selected[
            ["bout_id_IBImatched", "aligned_time", "ztime"]
        ].copy()

        # Selected bouts, restricted to your around-peak window.
        exp_data = raw.loc[
            raw["bout_id_IBImatched"].isin(selection["bout_id_IBImatched"])
            & raw["idx"].ge(idxRANGE[0])
            & raw["idx"].lt(idxRANGE[1])
        ].copy()

        exp_data = exp_data.merge(
            selection,
            on="bout_id_IBImatched",
            how="left",
            validate="many_to_one",
            sort=False,
        )

        exp_data["angvel_smoothed"] = (
            exp_data.groupby("bout_id_IBImatched", sort=False)[
                "propBoutAligned_angVel"
            ].transform(lambda x: savgol_filter(x.to_numpy(), SMOOTH, 3))
        )

        # Complete preceding IBI traces for those same selected bouts.
        exp_ibi = ibi_raw.loc[
            ibi_raw["bout_id_IBImatched"].isin(selection["bout_id_IBImatched"])
        ].copy()

        exp_ibi = exp_ibi.merge(
            selection,
            on="bout_id_IBImatched",
            how="left",
            validate="many_to_one",
            sort=False,
        )

        metadata = dict(
            full_condition=full_condition,
            full_condition_idx=full_condition_idx,
            cond0=cond0,
            cond1=cond1,
            expNum=expNum,
            exp_id=exp,
        )

        bout_parts.append(exp_data.assign(**metadata))
        ibi_parts.append(exp_ibi.assign(**metadata))


all_around_peak_data = pd.concat(bout_parts, ignore_index=True)
all_IBI_data = pd.concat(ibi_parts, ignore_index=True)

all_around_peak_data["time_ms"] = (
    (all_around_peak_data["idx"] - peak_idx) / FRAME_RATE * 1000
)

# IBI times relative to the following bout's peak.
all_IBI_data["time_ms"] = (
    all_IBI_data["absTime"] - all_IBI_data["aligned_time"]
).dt.total_seconds() * 1000

#%%
# okay, this should work. You can match IBI and bout using keys:
# full_condition: the experimental condition, cond0 and cond1 combined
# exp_id: full name of the experiment_id, you may also use expNum
# bout_id_IBImatched: the unique identifier for each bout (within each condition in an experimental repeat)

keys = ["full_condition", "exp_id", "bout_id_IBImatched"]
# here's an example
example = all_around_peak_data.groupby(keys).head(1).iloc[200][keys].to_dict()
example_bouts = all_around_peak_data.loc[
    all_around_peak_data[keys].eq(example).all(axis=1)
]
example_preceding_ibi = all_IBI_data.loc[
    all_IBI_data[keys].eq(example).all(axis=1)
]
# note that there are overlapped time points, this is because IBIs end 100ms before speed exceeds 5mm/s. While aligned bouts start 300ms before time of the peak speed

# %%
