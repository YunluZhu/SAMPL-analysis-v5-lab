'''
For multiple comparisons across conditions and day night

'''

#%%
# import sys
import os
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_connected_bouts,get_bout_features
from plot_functions.plt_tools import set_font_type
from plot_functions.plt_functions import plt_categorical_grid2
import matplotlib as mpl
import seaborn as sns
from plot_functions.plt_functions import plt_categorical_combined_3
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# %%
##### Parameters to change #####
pick_data = 'wt_light_long' # name of your dataset to plot as defined in function get_data_dir()
which_ztime = 'all' # 'day', 'night', or 'all'
my_colors = ["#E4CB31", "#F7941D", "#E01F3E"]
my_palette = sns.color_palette(my_colors)
# %% get root directory and figure directory

root, FRAME_RATE = get_data_dir(pick_data)
folder_name = __file__.split('/')[-1].replace('.py','')
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)
try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

set_font_type()
mpl.rc('figure', max_open_warning = 0)

# %% get features
all_feature_cond_connected, all_cond0, all_cond1 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)
# %% tidy data
all_feature_cond_connected = all_feature_cond_connected.sort_values(by=['cond1','expNum']).reset_index(drop=True)

# %%
data_toplt = all_feature_cond_connected.query("cond1=='ld' and ztime=='day'")

g = sns.relplot(
    data=data_toplt.groupby('cond0').sample(n=4000),
    x='pre_IBI_time',
    y='pitch_peak',
    col='cond0',
    alpha=0.03,
    linewidth=0,
    height=2.5,
    color='gray',    
)
g.set(xlim=(0, 3),ylim=(-30,60))
plt.savefig(os.path.join(fig_dir, f"pitch_peak_vs_preIBI_scatter.pdf"), format='PDF')
# %% tidy data

all_feature_cond, all_cond0, all_cond1 = get_bout_features(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=True,)
# %% tidy data
all_feature_cond = all_feature_cond.sort_values(by=['cond1','expNum']).reset_index(drop=True)


df_to_plt = all_feature_cond.query("cond1=='ld' and ztime=='day'")

# --- Setup Variables (Using your definitions) ---
col_to_plt = 'traj_peak' 
min_val = -90
max_val = 90

N_total = len(df_to_plt)
# Calculate N_groups
N_groups = df_to_plt['cond0'].nunique()
# Calculate average sample size
n_bar = N_total / N_groups
s = np.std(df_to_plt[col_to_plt])
h = 3.49 * s * (n_bar ** (-1/3))
Range = df_to_plt[col_to_plt].max() - df_to_plt[col_to_plt].min()
# Calculate the number of bins (k) and round up
k = np.ceil(Range / h).astype(int)
# Use k to set up your bins
num_bins = k
step = Range / num_bins
bins = np.arange(min_val, max_val + step, step)

# Define Range
Range = 180.0
step = (max_val - min_val) / num_bins
bins = np.arange(min_val, max_val + step, step)

# Calculate the normalized histogram counts for each 'cond0' group
# This aggregates the counts across 'cond1' and 'ztime' for a distribution specific to 'cond0'
angle_counts_by_cond0 = df_to_plt.groupby(['cond0']).apply(
    lambda g: np.histogram(g[col_to_plt], bins)[0] / len(g)
)

# ... (rest of the code remains the same up to plotting)

bin_mid = (bins[1:] + bins[:-1]) / 2

# % --- Plotting ---
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

# 1. Set 0° (0 rad) to be at the East (3 o'clock) position:
ax.set_theta_zero_location('E') 

# 2. Set the direction to Counter-Clockwise (90° North):
ax.set_theta_direction(1)      

# 3. Restrict the plot to the range of your data: -90° to 90°
ax.set_thetamax(90)
ax.set_thetamin(-90)

# 4. Set the labels for the angular axis:
tick_angles_deg = np.linspace(-90, 90, 7)
tick_angles_rad = np.radians(tick_angles_deg)
ax.set_xticks(tick_angles_rad)
ax.set_xticklabels([f'{int(d)}°' for d in tick_angles_deg])


# --- 5. ADDING THE RADIAL SCALE (FREQUENCY) ---
# Find the overall maximum frequency value
max_freq = 0
for cond in df_to_plt.cond0.unique():
    counts = angle_counts_by_cond0.loc[cond]
    max_freq = max(max_freq, np.max(counts))

# Define clear, round ticks for the radial axis
# Use np.ceil to round up the max_freq to a nice number (e.g., 0.14 -> 0.15 or 0.2)
# We can set ticks at 0, 1/3, 2/3, and the max_r_tick
max_r_tick = np.ceil(max_freq * 20) / 20 # Rounds up to the nearest 0.05
if max_r_tick == 0: # Avoid division by zero if all counts are zero
    max_r_tick = 0.1

r_ticks = np.linspace(0, max_r_tick, 6) # e.g., 4 ticks from 0 up to max_r_tick

ax.set_rlim(0, max_r_tick) # Ensure the plot extends to the max tick value
ax.set_rticks(r_ticks)     # Set the explicit tick locations
ax.set_yticklabels([f'{r:.2f}' for r in r_ticks], backgroundcolor="white") # Set labels with 2 decimal places

# Label the radial axis clearly
ax.set_ylabel('Normalized Frequency', labelpad=25, rotation=0, 
              horizontalalignment='left', verticalalignment='top')
# --- END RADIAL SCALE ADDITION ---


# Iterate over the unique conditions for plotting
for cond in df_to_plt.cond0.unique():
    counts = angle_counts_by_cond0.loc[cond]
    ax.plot(np.radians(bin_mid), counts, label=f'Condition {cond}', linewidth=2)

ax.legend(loc='lower left', bbox_to_anchor=(1.05, 0.5), title='Condition')
ax.set_title(f'Direction Distribution of {col_to_plt} (90° North)', va='bottom')

plt.savefig(os.path.join(fig_dir, f"bout direction hist_90_North_scaled.pdf"), format='PDF')

#%%

avg_data = all_feature_cond.groupby(
    ['cond0', 'cond1', 'expNum', 'ztime'], as_index=False, observed=True
)[['pitch_initial',
       'pitch_mid_accel', 'pitch_pre_bout', 'pitch_peak', 'pitch_post_bout',
       'pitch_end', 'pitch_max_angvel', 'traj_initial', 'traj_pre_bout',
       'traj_peak', 'traj_post_bout', 'traj_end', 'spd_initial', 'spd_peak',
       'angvel_initial_phase', 'angvel_prep_phase', 'angvel_post_phase',
       'traj_initial_phase', 'spd_initial_phase', 'fish_length', 'boxNum',
       'rot_total', 'rot_bout', 'rot_pre_bout', 'rot_l_accel',
       'rot_full_accel', 'rot_full_decel', 'rot_l_decel', 'rot_early_accel',
       'rot_late_accel', 'rot_early_decel', 'rot_late_decel',
       'rot_to_max_angvel', 'bout_trajectory_Pre2Post', 'bout_displ',
       'traj_deviation', 'atk_ang', 'tsp_peak', 'angvel_chg', 'depth_chg',
       'depth_chg_fullBout', 'x_chg', 'x_chg_fullBout', 'lift_distance',
       'lift_distance_fullBout', 'additional_depth_chg', 'displ_swim',
       'ydispl_swim', 'xdispl_swim', 'y_post_swim', 'y_pre_swim',
       'x_post_swim', 'x_pre_swim', 'WHM']].median()

df_to_plt = avg_data.query("cond1=='ld' and ztime=='day'")

feature_to_plt_sel = [
    'ydispl_swim',
    'xdispl_swim',
    'atk_ang',
    'traj_peak',
    'depth_chg_fullBout'
]



x_name = 'cond0'
# gridrow = 'direction'
gridrow = 'cond1'
gridcol = None
hue = 'cond0'
units = 'expNum'
prename = ''

for feature in feature_to_plt_sel:
    plt_categorical_combined_3(
        data=df_to_plt,
        x=x_name,
        y=feature,
        hue=hue,
        units=units,
        col=gridcol,
        row=gridrow,
        # col='cond1',
        errorbar='se',
        palette=my_palette,
    )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %%

# run multiple comparison test
for param in feature_to_plt_sel:
    print(f"\n--- ANOVA for {param} ---")
    df_var = df_to_plt[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
        continue
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    
#%%

# calculate climb dive percentage by traj_peak vs 0
# based on all_feature_cond

def calc_climb_dive_percentage(df):
    total_bouts = len(df)
    climb_bouts = df[(df['traj_peak'] > 10) & (df['ydispl_swim'] > 0)]
    dive_bouts = df[df['traj_peak'] < 0]
    climb_percentage = len(climb_bouts) / total_bouts * 100
    dive_percentage = len(dive_bouts) / total_bouts * 100
    return climb_percentage, dive_percentage

climb_dive_data = []
grouped = all_feature_cond.groupby(['cond0', 'cond1', 'expNum', 'ztime'])
for name, group in grouped:
    cond0, cond1, expNum, ztime = name
    climb_perc, dive_perc = calc_climb_dive_percentage(group)
    climb_dive_data.append({
        'cond0': cond0,
        'cond1': cond1,
        'expNum': expNum,
        'ztime': ztime,
        'climb_percentage': climb_perc,
        'dive_percentage': dive_perc
    })
    
climb_dive_df = pd.DataFrame(climb_dive_data)
# %%
# plot climb percentage
df_to_plt = climb_dive_df.query("cond1=='ld' and ztime=='night'")
feature_to_plt = ['climb_percentage']

# bar plot
x_name = 'cond0'
gridrow = 'cond1'
gridcol = None
hue = 'cond0'
units = 'expNum'
prename = ''    
for feature in feature_to_plt:
    plt_categorical_combined_3(
        data=df_to_plt,
        x=x_name,
        y=feature,
        hue=hue,
        units=units,
        col=gridcol,
        row=gridrow,
        # col='cond1',
        errorbar='se',
        palette=my_palette,
    )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %
# run multiple comparison test for climb and dive percentage
for param in feature_to_plt:
    print(f"\n--- ANOVA for {param} ---")
    df_var = df_to_plt[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
        continue
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    

# %%

####################################
###### rotation!!!! ######
####################################

df_to_plt = avg_data.query("cond1=='ld' and ztime=='night'")

feature_to_plt_sel = [
    'rot_total',
    # 'angvel_chg',
    'angvel_post_phase',
    # 'atk_ang',
    # 'WHM',
    
]

# select variable
x_name = 'cond0'
# gridrow = 'direction'
gridrow = 'cond1'
gridcol = None
hue = 'cond0'
units = 'expNum'
prename = ''

for feature in feature_to_plt_sel:
    plt_categorical_combined_3(
        data=df_to_plt,
        x=x_name,
        y=feature,
        hue=hue,
        units=units,
        col=gridcol,
        row=gridrow,
        # col='cond1',
        errorbar='se',
        palette=my_palette,
    )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()

# %

# run multiple comparison test
for param in feature_to_plt_sel:
    print(f"\n--- ANOVA for {param} ---")
    df_var = df_to_plt[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
        continue
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    
#%% calculate nose up bout percentage
def calc_noseup_percentage(df):
    total_bouts = len(df)
    noseup_bouts = df[df['rot_total'] > 0]
    noseup_percentage = len(noseup_bouts) / total_bouts * 100
    return noseup_percentage
noseup_data = []
grouped = all_feature_cond.groupby(['cond0', 'cond1', 'expNum', 'ztime'])
for name, group in grouped:
    cond0, cond1, expNum, ztime = name
    noseup_perc = calc_noseup_percentage(group)
    noseup_data.append({
        'cond0': cond0,
        'cond1': cond1,
        'expNum': expNum,
        'ztime': ztime,
        'noseup_percentage': noseup_perc,
    })
noseup_df = pd.DataFrame(noseup_data)

# plot nose up rot percentage
df_to_plt = noseup_df.query("cond1=='ld' and ztime=='night'")
feature_to_plt = ['noseup_percentage']

# bar plot
x_name = 'cond0'
gridrow = 'cond1'
gridcol = None
hue = 'cond0'
units = 'expNum'
prename = ''    
for feature in feature_to_plt:
    plt_categorical_combined_3(
        data=df_to_plt,
        x=x_name,
        y=feature,
        hue=hue,
        units=units,
        col=gridcol,
        row=gridrow,
        # col='cond1',
        errorbar='se',
        palette=my_palette,
    )
    filename = os.path.join(fig_dir,f"{prename}{feature}__by{x_name}__{gridcol}.pdf")
    plt.savefig(filename,format='PDF')
    plt.show()
# %
# run multiple comparison test for climb and dive percentage
for param in feature_to_plt:
    print(f"\n--- ANOVA for {param} ---")
    df_var = df_to_plt[[x_name, param]].dropna().rename(columns={x_name: "cond0"})
    if df_var["cond0"].nunique() < 2:
        print(f"  Skipped (only one condition for {param})")
        continue
    # 1. One-way ANOVA
    model = ols(f"{param} ~ C(cond0)", data=df_var).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    print(anova_table)
    
    # 2. Tukey’s HSD for post hoc comparison
    tukey = pairwise_tukeyhsd(
        endog=df_var[param],
        groups=df_var["cond0"],
        alpha=0.05
    )
    print("\nTukey HSD:")
    print(tukey.summary())
    
#%%
# Explicit copy avoids SettingWithCopyWarning
df_to_plt = all_feature_cond.query("cond1=='ld' and ztime=='night'").copy()

# Now safe to assign
df_to_plt['q_initial'] = np.nan

for cond, sub_idx in df_to_plt.groupby('cond0').groups.items():
    sub = df_to_plt.loc[sub_idx]

    # get bin edges from INITIAL only
    _, bins = pd.qcut(
        sub['pitch_initial'],
        q=4,
        retbins=True,
        duplicates='drop'
    )

    # assign quadrants
    df_to_plt.loc[sub_idx, 'q_initial'] = pd.cut(
        sub['pitch_initial'],
        bins=bins,
        labels=[1,2,3,4],
        include_lowest=True
    ).astype(int)

metacol = ['cond0', 'cond1', 'expNum', 'ztime','q_initial']

df_long = (
    df_to_plt
    .melt(
        id_vars=metacol,
        value_vars=['pitch_initial', 'pitch_end'],
        var_name='pitch_type',
        value_name='pitch'
    )
)
# optional: rename pitch_type for clarity
df_long['pitch_type'] = df_long['pitch_type'].map({
    'pitch_initial': 'initial',
    'pitch_end': 'end'
})

#%%
df_stats = (
    df_long
    .groupby(['cond0','cond1','ztime','q_initial','pitch_type'])
    .agg(
        median=('pitch', 'median'),
        q1=('pitch', lambda x: np.percentile(x, 25)),
        q3=('pitch', lambda x: np.percentile(x, 75))
    )
    .reset_index()
)
g = sns.catplot(
    data=df_stats,
    x='pitch_type',
    y='median',
    hue='q_initial',
    col='cond0',
    kind='point',
    height=3,
    aspect=0.5,
    palette='muted',
    order=['initial','end'],
    ci=None,  # we will add IQR manually
    markers=None,
)

# Add IQR as vertical error bars
for ax, cond in zip(g.axes[0], df_stats['cond0'].unique()):
    sub = df_stats[df_stats['cond0']==cond]
    for i, row in sub.iterrows():
        x = ['initial','end'].index(row['pitch_type'])
        ax.errorbar(
            x=x,
            y=row['median'],
            yerr=[[row['median']-row['q1']], [row['q3']-row['median']]],
            fmt='none',
            c='gray',
            alpha=1,
            capsize=0
        )
plt.savefig(os.path.join(fig_dir, f"pitch_initial_vs_end_by_quadrant.pdf"), format='PDF')

#%%
# plto, by quadrant, from initial pitch to end pitch
# initial pitch vs end pitch
df_to_plt = all_feature_cond.query("cond1=='ld' and ztime=='night'")
x_val = 'pitch_initial'
y_val = 'pitch_end' 

metacol = ['cond0', 'cond1', 'expNum', 'ztime']

df_long = (
    df_to_plt
    .melt(
        id_vars=metacol,
        value_vars=[x_val, y_val],
        var_name='pitch_type',
        value_name='pitch'
    )
)
#%
sns.displot(
    data=df_long,
    kind='hist',
    element='poly',
    stat='probability',
    y='pitch',
    hue='pitch_type',
    col='cond0',
    height=2.5,
    aspect=0.8,
    common_norm=False,
)
# %%
