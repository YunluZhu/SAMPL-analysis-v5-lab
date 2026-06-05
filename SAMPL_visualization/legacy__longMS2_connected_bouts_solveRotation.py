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
which_ztime = "night"  # 'day', 'night', or 'all'
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
    "WHM",
    "pre_IBI_time",
    "pitch_initial",
    "pitch_end",
    "rot_total",
    "y_initial",
    "y_end",
    "x_initial",
    "x_end",
]

# %% associate consecutive bouts

#####################
max_lag = 1
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

# Compare current y_initial with next bout's y_initial
sel_consecutive_bouts["bout_direction"] = sel_consecutive_bouts.apply(
    lambda row: "climb" if row["y_initial"] < row["y_end"] else "dive", axis=1
)

selected_data = (
    sel_consecutive_bouts.groupby(["cond1", "ztime", "expNum", "id"], as_index=False)
    .apply(
        lambda group: group.assign(
            preIBI_y_displ=group["y_initial"]
            - group["y_end"].shift(
                1
            ),  # preIBI_y_displ = y end from last bout - y initial from current bout
            preIBI_x_displ=np.abs(
                group["x_initial"] - group["x_end"].shift(1)
            ),  # preIBI_y_displ = y end from last bout - y initial from current bout
            # postIBI_y_displ=group["y_initial"].shift(-1) - group["y_end"],   # postIBI_y_displ = y initial from next bout - y end from current bout
            preIBI_rot=group["pitch_initial"] - group["pitch_end"].shift(1),
            # postIBI_rot=group["pitch_initial"].shift(-1) - group["pitch_end"]
        ),
        include_groups=False,
    )
    .reset_index(drop=True)  # Reset index after apply()
)

# %%
data_to_plot = selected_data.query("bouts == 2")
data_to_plot = data_to_plot.query(
    "pre_IBI_time < 20"
)  # only consider pre IBI within 20s

IBI_THRESHOLD = 1#np.quantile(data_to_plot["pre_IBI_time"], 0.1)
# data_to_plot['angAccel'] = data_to_plot['preIBI_rot'] / data_to_plot['pre_IBI_time']
data_to_plot["expNum"] = data_to_plot["exp_conduid"].apply(lambda x: x[-1])
# data_to_plot = data_to_plot.query("angAccel > -10")
data_to_plot = data_to_plot.assign(
    IBI_duration=pd.cut(
        data_to_plot["pre_IBI_time"],
        bins=[0, IBI_THRESHOLD, np.inf],
        labels=["short IBI", "long IBI"],
    )
)
# qthresh = (
#     data_to_plot
#     .groupby("cond0")["pre_IBI_time"]
#     .transform(lambda x: x.quantile(0.2))
# )

# data_to_plot["IBI_duration"] = np.where(
#     data_to_plot["pre_IBI_time"] <= qthresh,
#     "short IBI",
#     "long IBI"
# )
# %% postural compensation
data_to_plot_selected = data_to_plot.query("IBI_duration == 'long IBI'")
data_to_plot_shortIBI = data_to_plot.query("IBI_duration == 'short IBI'")

# %%
predictions = []
r2_labels = []
stats_results = []


X_VAR = "preIBI_rot"
Y_VAR = "rot_total"
for group_name, group_df in data_to_plot_selected.groupby("cond0"):
    subset = group_df[[X_VAR, Y_VAR]].dropna()
    if len(subset) > 1:
        X = sm.add_constant(subset[X_VAR])
        y = subset[Y_VAR]

        # Robust regression
        model = sm.RLM(y, X, M=norms.TukeyBiweight())
        results = model.fit()
        intercept, slope = results.params["const"], results.params[X_VAR]

        # Create prediction line
        x_min = subset[X_VAR].quantile(0.005)
        x_max = subset[X_VAR].quantile(0.995)
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = intercept + slope * x_grid

        # Compute R²
        y_fit = intercept + slope * subset[X_VAR]
        r2 = r2_score(subset[Y_VAR], y_fit)

        # Store for plotting
        predictions.append(
            pd.DataFrame({X_VAR: x_grid, Y_VAR: y_grid, "cond0": group_name})
        )

        r2_labels.append(
            {
                "IBI_group": group_name,
                "x": subset[X_VAR].quantile(0.95),
                "y": subset[Y_VAR].quantile(0.95),
                "label": f"R² = {r2:.3f}",
            }
        )

        # # Calculate p-value
        # boot_slopes = []
        # for _ in range(1000):
        #     boot_subset = subset.sample(frac=1, replace=True)
        #     X_boot = sm.add_constant(boot_subset[X_VAR])
        #     y_boot = boot_subset[Y_VAR]
        #     try:
        #         boot_model = sm.RLM(y_boot, X_boot, M=norms.TukeyBiweight())
        #         boot_results = boot_model.fit()
        #         boot_slopes.append(boot_results.params[X_VAR])
        #     except:
        #         continue

        # boot_slopes = np.array(boot_slopes)

        # # Confidence interval
        # ci_lower = np.percentile(boot_slopes, 2.5)
        # ci_upper = np.percentile(boot_slopes, 97.5)

        # # Is 0 inside the 95% CI? → not significant if so
        # zero_in_ci = ci_lower <= 0 <= ci_upper

        # # Two-tailed p-value: proportion of bootstrap slopes on the opposite side of 0
        # p_value = 2 * min(
        #     np.mean(np.array(boot_slopes) <= 0), np.mean(np.array(boot_slopes) >= 0)
        # )

        stats_results.append(
            {
                "IBI_group": group_name,
                "Slope": slope,
                "Intercept": intercept,
                "Rsquare": r2,
                # "pvalue": p_value,
            }
        )

pred_df = pd.concat(predictions, ignore_index=True)
r2_df = pd.DataFrame(r2_labels)
stats_df = pd.DataFrame(stats_results)

# %%
g = sns.relplot(
    kind="scatter",
    data=data_to_plot_selected.groupby("cond0").sample(n=2000),
    x="preIBI_rot",
    y="rot_total",
    col="cond0",
    alpha=0.12,
    hue="IBI_duration",
    palette=sns.color_palette("Set1", n_colors=2),
)
g.set(
    xlim=(-45, 10),
    ylim=(-10, 45),
)
stats_lookup = stats_df.set_index("IBI_group")[
    ["Slope", "Intercept", "pvalue"]
].to_dict("index")

for ax in g.axes.flat:
    # Extract the condition name from the facet title
    cond = ax.get_title().split(" = ")[1]

    if cond not in stats_lookup:
        continue

    m = stats_lookup[cond]["Slope"]
    b = stats_lookup[cond]["Intercept"]

    # get x-range from data in this facet
    sub = data_to_plot_selected.query("IBI_duration == 'long IBI' and cond0 == @cond")[
        X_VAR
    ].dropna()

    if len(sub) < 2:
        continue

    x_vals = np.array([sub.quantile(0.005), sub.quantile(0.995)])
    y_vals = m * x_vals + b

    ax.plot(x_vals, y_vals, color="black", linewidth=2)
plt.savefig(
    os.path.join(fig_dir, "longIBI_postural_compensation_connected_bouts.pdf"),
    format="pdf",
)

print(stats_df)

#%%
predictions = []
r2_labels = []
stats_results = []


X_VAR = "preIBI_rot"
Y_VAR = "rot_total"
for group_name, group_df in data_to_plot_shortIBI.groupby("cond0"):
    subset = group_df[[X_VAR, Y_VAR]].dropna()
    if len(subset) > 1:
        X = sm.add_constant(subset[X_VAR])
        y = subset[Y_VAR]

        # Robust regression
        model = sm.RLM(y, X, M=norms.TukeyBiweight())
        results = model.fit()
        intercept, slope = results.params["const"], results.params[X_VAR]

        # Create prediction line
        x_min = subset[X_VAR].quantile(0.005)
        x_max = subset[X_VAR].quantile(0.995)
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = intercept + slope * x_grid

        # Compute R²
        y_fit = intercept + slope * subset[X_VAR]
        r2 = r2_score(subset[Y_VAR], y_fit)

        # Store for plotting
        predictions.append(
            pd.DataFrame({X_VAR: x_grid, Y_VAR: y_grid, "cond0": group_name})
        )

        r2_labels.append(
            {
                "IBI_group": group_name,
                "x": subset[X_VAR].quantile(0.95),
                "y": subset[Y_VAR].quantile(0.95),
                "label": f"R² = {r2:.3f}",
            }
        )

        # # Calculate p-value
        # boot_slopes = []
        # for _ in range(1000):
        #     boot_subset = subset.sample(frac=1, replace=True)
        #     X_boot = sm.add_constant(boot_subset[X_VAR])
        #     y_boot = boot_subset[Y_VAR]
        #     try:
        #         boot_model = sm.RLM(y_boot, X_boot, M=norms.TukeyBiweight())
        #         boot_results = boot_model.fit()
        #         boot_slopes.append(boot_results.params[X_VAR])
        #     except:
        #         continue

        # boot_slopes = np.array(boot_slopes)

        # # Confidence interval
        # ci_lower = np.percentile(boot_slopes, 2.5)
        # ci_upper = np.percentile(boot_slopes, 97.5)

        # # Is 0 inside the 95% CI? → not significant if so
        # zero_in_ci = ci_lower <= 0 <= ci_upper

        # # Two-tailed p-value: proportion of bootstrap slopes on the opposite side of 0
        # p_value = 2 * min(
        #     np.mean(np.array(boot_slopes) <= 0), np.mean(np.array(boot_slopes) >= 0)
        # )

        stats_results.append(
            {
                "IBI_group": group_name,
                "Slope": slope,
                "Intercept": intercept,
                "Rsquare": r2,
                # "pvalue": p_value,
            }
        )
print(stats_results)
g = sns.relplot(
    kind="scatter",
    data=data_to_plot_shortIBI.groupby('cond0').sample(n=data_to_plot_shortIBI.groupby('cond0').size().min()),
    x="preIBI_rot",
    y="rot_total",
    col="cond0",
    alpha=0.12,
    hue="IBI_duration",
)
g.set(
    xlim=(-45, 10),
    ylim=(-10, 45),
)
stats_lookup = stats_df.set_index("IBI_group")[
    ["Slope", "Intercept"]
].to_dict("index")
plt.savefig(
    os.path.join(fig_dir, "shortIBI_postural_compensation_connected_bouts.pdf"),
    format="pdf",
)
# %%
# calculate slope per exp repeat
# %%
predictions = []
r2_labels = []
stats_results = []


X_VAR = "preIBI_rot"
Y_VAR = "rot_total"
for group_name, group_df in data_to_plot_selected.groupby(
    ["cond0", "expNum"], observed=True
):
    subset = group_df[[X_VAR, Y_VAR]].dropna()
    if len(subset) > 1:
        X = sm.add_constant(subset[X_VAR])
        y = subset[Y_VAR]

        # Robust regression
        model = sm.RLM(y, X, M=norms.TukeyBiweight())
        results = model.fit()
        intercept, slope = results.params["const"], results.params[X_VAR]

        # Create prediction line
        x_min = subset[X_VAR].quantile(0.005)
        x_max = subset[X_VAR].quantile(0.995)
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = intercept + slope * x_grid

        # Compute R²
        y_fit = intercept + slope * subset[X_VAR]
        r2 = r2_score(subset[Y_VAR], y_fit)

        # Store for plotting
        predictions.append(
            pd.DataFrame(
                {
                    X_VAR: x_grid,
                    Y_VAR: y_grid,
                    "cond0": group_name[0],
                    "expNum": group_name[1],
                }
            )
        )

        r2_labels.append(
            {
                "IBI_group": group_name,
                "x": subset[X_VAR].quantile(0.95),
                "y": subset[Y_VAR].quantile(0.95),
                "label": f"R² = {r2:.3f}",
                "cond0": group_name[0],
                "expNum": group_name[1],
            }
        )

        # # Calculate p-value
        # boot_slopes = []
        # for _ in range(1000):
        #     boot_subset = subset.sample(frac=1, replace=True)
        #     X_boot = sm.add_constant(boot_subset[X_VAR])
        #     y_boot = boot_subset[Y_VAR]
        #     try:
        #         boot_model = sm.RLM(y_boot, X_boot, M=norms.TukeyBiweight())
        #         boot_results = boot_model.fit()
        #         boot_slopes.append(boot_results.params[X_VAR])
        #     except:
        #         continue

        # boot_slopes = np.array(boot_slopes)

        # # Confidence interval
        # ci_lower = np.percentile(boot_slopes, 2.5)
        # ci_upper = np.percentile(boot_slopes, 97.5)

        # # Is 0 inside the 95% CI? → not significant if so
        # zero_in_ci = ci_lower <= 0 <= ci_upper

        # # Two-tailed p-value: proportion of bootstrap slopes on the opposite side of 0
        # p_value = 2 * min(
        #     np.mean(np.array(boot_slopes) <= 0), np.mean(np.array(boot_slopes) >= 0)
        # )

        stats_results.append(
            {
                "IBI_group": group_name,
                "Slope": slope,
                "Intercept": intercept,
                "Rsquare": r2,
                # "pvalue": p_value,
                "cond0": group_name[0],
                "expNum": group_name[1],
            }
        )

pred_df_byRep = pd.concat(predictions, ignore_index=True)
r2_df_byRep = pd.DataFrame(r2_labels)
stats_df_byRep = pd.DataFrame(stats_results)

# %%
plt_categorical_combined_3(
    data=stats_df_byRep,
    x="cond0",
    y="Slope",
    hue="cond0",
    units="expNum",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(
    os.path.join(fig_dir, "longIBI_postural_compensation_connected_bouts_SLOPE.pdf"),
    format="pdf",
)

plt_categorical_combined_3(
    data=stats_df_byRep,
    x="cond0",
    y="Intercept",
    hue="cond0",
    units="expNum",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(
    os.path.join(fig_dir, "longIBI_postural_compensation_connected_bouts_INTERCEPT.pdf"),
    format="pdf",
)

plt_categorical_combined_3(
    data=stats_df_byRep,
    x="cond0",
    y="Rsquare",
    hue="cond0",
    units="expNum",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(
    os.path.join(fig_dir, "longIBI_postural_compensation_connected_bouts_Rsquare.pdf"),
    format="pdf",
)


# %%
# boring
# let's calculate compensation efficacy

data_to_plot_selected = data_to_plot_selected.assign(
    compensation_residual=lambda df: df["rot_total"] + df["preIBI_rot"],
    compensation_gain=lambda df: df["rot_total"] / df["preIBI_rot"] * -1,
    residual_ratio = lambda df: (df["rot_total"] + df["preIBI_rot"]) / df["preIBI_rot"]
)

data_to_plot_shortIBI = data_to_plot_shortIBI.assign(
    compensation_residual=lambda df: df["rot_total"] + df["preIBI_rot"],
    compensation_gain=lambda df: df["rot_total"] / df["preIBI_rot"] * -1,
    residual_ratio = lambda df: (df["rot_total"] + df["preIBI_rot"]) / df["preIBI_rot"]
)


#%% skewness of IBI
data_to_plot_sel_for_hist = selected_data.query("bouts == 2")

data_to_plot_sel_for_hist = data_to_plot_sel_for_hist.assign(
    log_pre_IBI_time=np.log10(data_to_plot_sel_for_hist["pre_IBI_time"])
)
data_to_plot_sel_for_hist = data_to_plot_sel_for_hist.query("pre_IBI_time < 40 and pre_IBI_time > 0.2")

IBI_THRESHOLD = 1
# plot distribution of IBI
sns.displot(
    data=data_to_plot_sel_for_hist,
    x='log_pre_IBI_time',
    hue='cond0',
    # col='cond0',
    palette=my_palette,
    stat='probability',
    common_norm=False,
    element='poly',
    bins=50,
    fill=True,
    height=3,
)
plt.axvline(x=np.log10(IBI_THRESHOLD), color='grey', linestyle='--')
plt.savefig(
    os.path.join(fig_dir, "IBI_distribution_connected_bouts.pdf"), format="pdf"
)


# ecdf
plt.figure(figsize=(3, 3))
sns.ecdfplot(
    data=data_to_plot_sel_for_hist,
    x='log_pre_IBI_time',
    hue='cond0',
    palette=my_palette,
    # stat='probability',
    # common_norm=False,
    # element='poly',
    # bins=50,
    # fill=True,
)
# add vertical line indicating IBI_THRESHOLD
plt.axvline(x=np.log10(IBI_THRESHOLD), color='grey', linestyle='--')
plt.savefig(
    os.path.join(fig_dir, "IBI_ecdf_connected_bouts.pdf"), format="pdf"
)

# %% KDE and slope calculation for finding inflection points for IBI distribution threshold
# import numpy as np
# import pandas as pd
# from scipy.ndimage import gaussian_filter1d
# from scipy.stats import gaussian_kde

# import matplotlib.pyplot as plt

# # PARAMETERS
# n_points_grid = 100  # number of points for uniform x-grid
# smoothing_sigma = 3  # sigma for Gaussian smoothing

# sns.displot(
#     data=data_to_plot_sel_for_hist.query("pre_IBI_time < 5"),
#     x='pre_IBI_time',
#     hue='cond0',
#     # col='cond0',
#     palette=my_palette,
#     stat='probability',
#     common_norm=False,
#     element='poly',
#     bins=50,
#     fill=True,
#     height=3,
# )

# # STORE RESULTS
# pdf_res_ = []

# # LOOP OVER cond0 AND expNum
# for cond0, group in data_to_plot_sel_for_hist.query("pre_IBI_time < 5").groupby('cond0'):
    
#     data = group["pre_IBI_time"].values
    
#     # KDE estimate
#     kde = gaussian_kde(data, bw_method='scott')  # or try bw_method=0.2, etc.
#     x_grid = np.linspace(data.min(), data.max(), n_points_grid)
#     pdf = kde(x_grid)  # estimated density
    
#     # slope of the PDF
#     slope = np.diff(pdf) / np.diff(x_grid)
#     x_mid = (x_grid[:-1] + x_grid[1:]) / 2
    
#     # smooth slope
#     slope_smooth = gaussian_filter1d(slope, sigma=smoothing_sigma)
    
#     # store results
#     this_res = pd.DataFrame({
#         'cond0': cond0,
#         'x_mid': x_mid,
#         'pdf': pdf[1:],           # match slope
#         'slope': slope,
#         'slope_smooth': slope_smooth
#     })
    
#     pdf_res_.append(this_res)

# # combine all groups
# pdf_res = pd.concat(pdf_res_, ignore_index=True)

# plt.figure(figsize=(4,4))

# for cond0, df in pdf_res.groupby('cond0'):
#     plt.plot(df['x_mid'], df['slope_smooth'], label=cond0)

# # horizontal line at y=0
# plt.axhline(0, color='grey', linestyle='--')
# plt.xlabel("pre-IBI time")
# plt.ylabel("PDF slope (smoothed)")
# plt.title("Smoothed PDF slopes per condition")
# plt.legend()
# plt.tight_layout()
# plt.show()

# from scipy.signal import find_peaks

# # store results
# minima_res = []

# for cond0, df in pdf_res.groupby('cond0'):
#     # restrict to x_mid in 0-3
#     df_range = df[(df['x_mid'] >= 0) & (df['x_mid'] <= 2.5)].copy()
    
#     # Invert slope_smooth to find minima as peaks
#     inverted_slope = -df_range['slope_smooth'].values
    
#     # find peaks in inverted slope → minima in slope
#     peaks, _ = find_peaks(inverted_slope, prominence=0.01)
    
#     # store minima x positions and slope values
#     minima_df = pd.DataFrame({
#         'cond0': cond0,
#         'x_min': df_range['x_mid'].iloc[peaks].values,
#         'slope_min': df_range['slope_smooth'].iloc[peaks].values
#     })
    
#     minima_res.append(minima_df)

# # combine all cond0
# minima_res = pd.concat(minima_res, ignore_index=True)
# print(minima_res)

# #% plot histogram with vertical lines at minima
# sns.displot(
#     data=data_to_plot_sel_for_hist.query("pre_IBI_time < 10"),
#     x='pre_IBI_time',
#     hue='cond0',
#     # col='cond0',
#     palette=my_palette,
#     stat='probability',
#     common_norm=False,
#     element='poly',
#     bins='scott',
#     fill=True,
#     height=3,
#     aspect=2
# )
# for cond0, df_min in minima_res.groupby('cond0'):
#     for _, row in df_min.iterrows():
#         plt.axvline(x=row['x_min'], color='grey', linestyle='--' )

# plt.savefig(
#     os.path.join(fig_dir, "IBI_distribution_with_minima_connected_bouts.pdf"), format="pdf"
# )

# #%
# # calculate percentage per that's above minima per cond0
# minima_thresholds = minima_res.groupby('cond0')['x_min'].min().to_dict()
# data_to_plot = data_to_plot.assign(
#     IBI_above_minima=data_to_plot.apply(
#         lambda row: 'above minima' if row['pre_IBI_time'] >= minima_thresholds[row['cond0']] else 'below minima',
#         axis=1
#     )
# )
# # quantify ratio per cond0
# IBI_minima_count_df = data_to_plot.groupby(['cond0','expNum'])['IBI_above_minima'].value_counts(normalize=True).reset_index(name='ratio_IBI_above_minima').query("IBI_above_minima=='above minima'")
# plt_categorical_combined_3(
#     data=IBI_minima_count_df,
#     x="cond0",
#     y="ratio_IBI_above_minima",
#     hue="cond0",
#     units="expNum",
#     errorbar="se",
#     palette=my_palette,
# )
# plt.savefig(
#     os.path.join(fig_dir, "ratio_IBI_above_minima_connected_bouts.pdf"), format="pdf"
# )
# # pairwise comparison
# tukey = pairwise_tukeyhsd(endog=IBI_minima_count_df['ratio_IBI_above_minima'], groups=IBI_minima_count_df['cond0'], alpha=0.05)
# print(tukey)


#%%
# calculate ratio of longIBI
IBI_count_df = data_to_plot.groupby(['cond0','expNum'])['IBI_duration'].value_counts(normalize=True).reset_index(name='ratio_longIBI').query("IBI_duration=='long IBI'")

plt_categorical_combined_3(
    data=IBI_count_df,
    x="cond0",
    y="ratio_longIBI",
    hue="cond0",
    units="expNum",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(
    os.path.join(fig_dir, "ratio_longIBI_connected_bouts.pdf"), format="pdf"
)
# pairwise comparison
tukey = pairwise_tukeyhsd(endog=IBI_count_df['ratio_longIBI'], groups=IBI_count_df['cond0'], alpha=0.05)
print(tukey)


#%%
# generate histogram data for skewness calculation
# from scipy.stats.mstats import winsorize

# data_to_plot["IBI_wins"] = data_to_plot.groupby("cond0")["log_pre_IBI_time"].transform(
#     lambda x: winsorize(x, limits=[0.01, 0.01])
# )
data_to_plot_filtered = data_to_plot.query("pre_IBI_time < 8")
data_to_plot_filtered = data_to_plot_filtered.assign(
    log_pre_IBI_time=np.log10(data_to_plot_filtered["pre_IBI_time"])
)

skewness_IBI = data_to_plot_filtered.groupby(['cond0','expNum']).agg(
    skewness=('log_pre_IBI_time', lambda x: st.skew(x.dropna(), bias=False)),
    mean = ('log_pre_IBI_time', 'mean'),
    mode = ('log_pre_IBI_time', lambda x: x.mode()[0] if not x.mode().empty else np.nan),
    median = ('log_pre_IBI_time', 'median'),
).reset_index()

plt.figure(figsize=(5,4))

# scatter
sns.scatterplot(
    data=skewness_IBI,
    x="mean",
    y="skewness",
    hue="cond0",
    palette=my_palette,
    s=60
)

# connect points within each cond0 by expNum
for  exp, df_sub in skewness_IBI.groupby(["expNum"]):
    color = my_palette[all_cond0.index(cond)]
    plt.plot(
        df_sub["mean"],
        df_sub["skewness"],
        color=color,
        alpha=0.6
    )

plt.xlabel("Mean pre-IBI time")
plt.ylabel("Skewness (bias-corrected)")
plt.title("Skewness vs. Mean pre-IBI time")

plt.tight_layout()
plt.show()

#%%
# plot 4 quantiles of IBI per cond0
data_to_plot['IBI_quartile'] = (
    data_to_plot
    .groupby(['cond0'])['pre_IBI_time']
    .transform(lambda x: pd.qcut(x, q=4, labels=False))
)

quartile_stats = data_to_plot.groupby(['cond0','IBI_quartile','expNum']).agg(
    median_IBI=('pre_IBI_time', 'median'),
    mean_rot_total=('rot_total', 'median'),
).reset_index()

sns.relplot(
    data=quartile_stats,
    x='IBI_quartile',
    y='median_IBI',
    hue='cond0',
    row='expNum',
    palette=my_palette,
    kind='line',
    markers=True,   
    height=3,
)

# %%

#%%
kk = data_to_plot.groupby(['cond0','IBI_duration','expNum'])[['y_initial','y_end']].apply(
    lambda x: (x['y_end'] - x['y_initial']).median()
).reset_index(name='median_y_displ')

plt_categorical_combined_3(
    data=kk,
    x="IBI_duration",
    y="median_y_displ",
    hue="cond0",
    col='cond0',
    units="expNum",
    errorbar="se",
    palette=my_palette,
)

data_to_plot = data_to_plot.assign(
    y_displ_total=data_to_plot["y_end"] - data_to_plot["y_initial"]
)
#%% scatter with binned line, plot as a fucntion of pre_IBI time
pick_par = 'rot_total'

# ---- Binned line data ----
upper_IBI_limit = 9
bins = np.arange(0, upper_IBI_limit, 0.5)

df_binned = data_to_plot.groupby(['cond0','expNum']).apply(
    lambda group: distribution_binned_average_opt(
        df=group.query("pre_IBI_time < @upper_IBI_limit"),
        bin_col=pick_par,
        by_col="pre_IBI_time",
        method="median",
        bin=bins
    )
)
df_binned.columns = ['binned_pre_IBI_time', f'binned_{pick_par}']
df_binned = df_binned.reset_index()

# compute bin centers
df_binned["bin_center"] = df_binned["pre_IBI_time"].apply(
    lambda interval: (interval.left + interval.right) / 2
)

def add_binned_line(data, **kwargs):
    ax = plt.gca()

    # identify the cond0 corresponding to this facet
    cond = data["cond0"].iloc[0]

    # find its index in the ordered cond list
    idx = all_cond0.index(cond)

    # pick correct color from palette
    color = my_palette[idx]

    # subset df_binned for this cond
    df_sub = df_binned[df_binned["cond0"] == cond]

    # plot the line
    sns.lineplot(
        data=df_sub,
        x="bin_center",
        y=f"binned_{pick_par}",
        ax=ax,
        color=color,
        linewidth=2
    )


# scatter data
df_scatter = (
    data_to_plot
    .query("pre_IBI_time < @upper_IBI_limit")
    .groupby(['cond0','expNum'])
    .sample(n=150, replace=False)
)

# build the facet grid
g = sns.FacetGrid(
    df_scatter,
    col="cond0",
    height=3,
    sharex=True,
    sharey=True
)

# scatter layer
g.map_dataframe(
    sns.scatterplot,
    x="pre_IBI_time",
    y=pick_par,
    color="grey",
    alpha=0.3,
    s=10
)

# line layer: COLOR + ORDER MATCHED VIA all_cond0 + palette
g.map_dataframe(add_binned_line)

# y limits
g.set(
    ylim=(
        data_to_plot[pick_par].quantile(0.01),
        data_to_plot[pick_par].quantile(0.99)
    )
)

g.tight_layout()


plt.savefig(
    os.path.join(fig_dir, f"allIBI_preIBItime_vs_{pick_par}_connected_bouts.pdf"), format="pdf"
)   

#%% scatter with binned line, plot as a fucntion of rotation
# hue by IBI time

data_to_plot = data_to_plot.assign(
    log_pre_IBI_time=np.log10(data_to_plot["pre_IBI_time"])
)
pick_par = 'y_displ_total'
x_val = 'rot_total'

data_to_plot_sampled = (
    data_to_plot
    .groupby(['cond0'])
    .sample(n=2000, replace=False)
)

g = sns.relplot(
    kind='scatter',
    data=data_to_plot_sampled,
    x=x_val,
    y=pick_par,
    alpha=0.2,
    hue='log_pre_IBI_time',
    col='cond0',
    palette='mako',
    # palette=sns.choose_diverging_palette(as_cmap=True),
    size=10,
    height=3,
)
g.set(
    xlim=(data_to_plot[x_val].quantile(0.01), data_to_plot[x_val].quantile(0.99)),
    ylim=(data_to_plot[pick_par].quantile(0.01), data_to_plot[pick_par].quantile(0.99)),
)

plt.savefig(
    os.path.join(fig_dir, f"allIBI_rot_total_vs_{pick_par}_connected_bouts.pdf"), format="pdf"
)   

#%%
sns.relplot(
    data=df_binned,
    x="bin_center",
    y=f"binned_{pick_par}",
    hue="cond0",
    palette=my_palette,
    kind="line",
    height=3,
)

#%% residual vs preIBI time
plt.figure(figsize=(3, 3))
gg = sns.relplot(
    data=data_to_plot.query("pre_IBI_time < 9").groupby(['cond0','expNum']).sample(n=150),
    x="pre_IBI_time",
    y="compensation_residual",
    # hue="pre_IBI_time",
    col='cond0',
    palette="viridis",
    alpha=0.19,
    height=3,
    # row='IBI_duration',
    facet_kws={'sharey': True, 'sharex': 'row'},
)
gg.set(
    ylim=(data_to_plot["compensation_residual"].quantile(0.01), data_to_plot["compensation_residual"].quantile(0.99)),
    xlim=(data_to_plot["pre_IBI_time"].quantile(0.01), 9),
)
plt.savefig(
    os.path.join(fig_dir, f"allIBI_preIBItime_vs_compensation_residual_connected_bouts.pdf"), format="pdf"
)
#%%
pick_par = 'y_displ_total'
# plot as a fucntion of pre_IBI time

plt.figure(figsize=(3, 3))
gg = sns.relplot(
    data=data_to_plot.query("pre_IBI_time < @data_to_plot.pre_IBI_time.quantile(0.75)").groupby('cond0').sample(n=2000),
    x="rot_total",
    y=pick_par,
    hue="pre_IBI_time",
    col='cond0',
    palette="viridis",
    alpha=0.12,
    height=3,
    # row='IBI_duration',
    facet_kws={'sharey': True, 'sharex': 'row'},
)
# set limits, soft coded by percentile
gg.set(
    xlim=(data_to_plot["rot_total"].quantile(0.01), data_to_plot["rot_total"].quantile(0.99)),
    ylim=(data_to_plot[pick_par].quantile(0.01), data_to_plot[pick_par].quantile(0.99)),
)
plt.savefig(
    os.path.join(fig_dir, f"allIBI_rot_vs_{pick_par}_connected_bouts.pdf"), format="pdf"
)

#%%
# IBI distribution
sns.displot(
    data=data_to_plot.query("pre_IBI_time < 5"),
    x='pre_IBI_time',
    hue='cond0',
    col='cond0',
    palette=my_palette,
    stat='probability',
    common_norm=False,
    element='poly',
    bins=30,
    fill=True,
    height=3,
)

# %%
# plot distribution
plt.figure(figsize=(3, 3))
g = sns.histplot(
    data=data_to_plot_selected,
    x="compensation_residual",
    hue="cond0",
    stat="probability",
    element="poly",
    common_norm=False,
    bins="scott",
    palette=my_palette,
    fill=False,
)
g.set(xlim=[-20, 20])
plt.savefig(
    os.path.join(fig_dir, "longIBI_rot_compensation_residual_hist.pdf"), format="pdf"
)
#%%

plt.figure(figsize=(3, 3))
g = sns.histplot(
    data=data_to_plot_shortIBI.query("residual_ratio > -1 and residual_ratio < 1"),
    x="residual_ratio",
    hue="cond0",
    stat="probability",
    element="poly",
    common_norm=False,
    bins="scott",
    palette=my_palette,
    fill=False,
)
g.set(xlim=[-1, 1])
plt.savefig(
    os.path.join(fig_dir, "shortIBI_rot_compensation_residual_hist.pdf"), format="pdf"
)

#%%
g = sns.relplot(
    data=data_to_plot_selected.groupby('cond0').sample(n=2000),
    x='preIBI_rot',
    y='compensation_residual',
    col='cond0',
    palette=my_palette,
    alpha=0.04,
    height=3,
)
g.set(
    ylim=(-10, 10),
    xlim=(data_to_plot_selected["preIBI_rot"].quantile(0.01), data_to_plot_selected["preIBI_rot"].quantile(0.99)),
)
#%%
# #pick a range for better visualization
# data_to_plot_selected_toplt = data_to_plot_selected.query("preIBI_rot < 0")

# data_to_plot_selected_toplt = data_to_plot_selected_toplt.loc[
#     data_to_plot_selected_toplt["compensation_gain"].between(*(np.percentile(data_to_plot_selected_toplt["compensation_gain"], [2, 98])))
# ]

# plt.figure(figsize=(3,3))
# sns.histplot(
#     data=data_to_plot_selected_toplt,
#     x="compensation_gain",
#     bins='scott',
#     hue="cond0",
#     common_norm=False,
#     stat="probability",
#     element="poly",
#     palette=my_palette,
#     fill=False,
# )
# plt.savefig(
#     os.path.join(fig_dir, "longIBI_rot_compensation_gain_hist.pdf"), format="pdf"
# )

# plt_categorical_combined_3(
#     data=data_to_plot_selected,
#     x="cond0",
#     y="compensation_gain",
#     hue="cond0",
#     units="expNum",
#     errorbar="se",
#     palette=my_palette,
# )
# plt.savefig(
#     os.path.join(fig_dir, "longIBI_rot_compensation_gain.pdf"), format="pdf")

# %%
# calculate MAD per expNum
from scipy.stats import median_abs_deviation

mad_df = (
    data_to_plot_selected.groupby(["cond0", "expNum"])["compensation_residual"]
    .apply(lambda x: median_abs_deviation(x, scale=1))  # scale=1 → raw MAD
    .reset_index(name="MAD")
)
# %%
plt_categorical_combined_3(
    data=mad_df,
    x="cond0",
    y="MAD",
    hue="cond0",
    units="expNum",
    errorbar="se",
    palette=my_palette,
)
plt.savefig(
    os.path.join(fig_dir, "longIBI_rot_compensation_residual_MAD.pdf"), format="pdf"
)

# stats

df_var = mad_df[["cond0", "MAD"]].dropna()

# 1. One-way ANOVA
model = ols(f"{'MAD'} ~ C(cond0)", data=df_var).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)

# 2. Tukey’s HSD for post hoc comparison
tukey = pairwise_tukeyhsd(endog=df_var["MAD"], groups=df_var["cond0"], alpha=0.05)
print("\nTukey HSD:")
print(tukey.summary())

# %% DONE with rotation
#########################################
#########################################
#########################################
#########################################

# %% check depth control
