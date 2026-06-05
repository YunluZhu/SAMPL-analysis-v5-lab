# %%

import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_aligned_bouts_wIBI, get_bout_features, extract_bout_features_v5)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from plot_functions.plt_functions import (plt_categorical_grid, plt_network_graphs)
from plot_functions.get_bout_correlation import get_cluster_phaseSpace
from plot_functions import simfish
import math
from plot_functions.plt_tools import round_half_up 
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from scipy.stats import pearsonr
from scipy.special import exp10
import statistics as s
import networkx as nx
import random
import simpy
from networkx.algorithms.community import greedy_modularity_communities
from netgraph import Graph

# %%
set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'wt_dl'
which_ztime = 'day'
compare_which = 'cond1' # condition for separation None for treat as whole
NCLUSTER = 12
if_strict_DayNightSplit = True
sort_by_feature = 'pitch_initial' # by which parameter to sort the clusters on the figure

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} cluster{NCLUSTER}_sim_by{compare_which}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    

# %% 
all_around_peak_data, all_feature_cond, all_cond0, all_cond1, idxRANGE = get_aligned_bouts_wIBI(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=if_strict_DayNightSplit)
IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=if_strict_DayNightSplit)

# %%
nCluster = NCLUSTER
all_feature_clustered = get_cluster_phaseSpace(all_around_peak_data, all_feature_cond, idxRANGE, nCluster)
# %%
connected_bout_features = all_feature_clustered[all_feature_clustered['to_bout'].notna()]

to_cluster = connected_bout_features[['to_bout']].merge(all_feature_clustered[['cluster','bout_uid']], left_on='to_bout',right_on='bout_uid',how='left')

to_cluster.columns = ['to_bout', 'to_cluster', 'to_bout_uid']

connected_bout_features = connected_bout_features.merge(to_cluster[['to_bout','to_cluster']], on='to_bout', how='left')
connected_bout_features = connected_bout_features[connected_bout_features['to_cluster'].notna()]
connected_bout_features.reset_index(drop=True, inplace=True)
connected_bout_features = connected_bout_features.assign(
    to_cluster = connected_bout_features['to_cluster'].astype('int'),
)

# %% check graph
node_order = all_feature_clustered.groupby('cluster')[sort_by_feature].mean().sort_values().index.tolist()

if bool(compare_which):
    for this_condition in connected_bout_features[compare_which].unique():
        this_cond_features = connected_bout_features.loc[connected_bout_features[compare_which]==this_condition]
        # plot_network_graphs(extracted_features=this_cond_features, cond_sep=this_condition, sort_by_feature='ydispl_swim', total_features=connected_bout_features)
        extracted_features=this_cond_features
        cond_sep=this_condition
        total_features=connected_bout_features
        print(this_condition)
        plt_network_graphs(connected_bout_features, 
                            fig_dir = fig_dir,
                            sort_by_feature = sort_by_feature,
                            cond_sep=this_condition, 
                            node_order=node_order,
                            extracted_features=this_cond_features)
else:
    plt_network_graphs(connected_bout_features, 
                        fig_dir = fig_dir,
                        node_order=node_order,  
                        sort_by_feature = sort_by_feature)

# %%
all_graph = pd.DataFrame()
use_condition_par = True

for sel_condition in connected_bout_features[compare_which].unique():
    connected_bouts = connected_bout_features.loc[connected_bout_features[compare_which]==sel_condition,['cluster','to_cluster','expNum','cond0','cond1']]
    connected_bout_appearance = connected_bouts.groupby('cluster').size()
    graph_df = connected_bouts.groupby(['cluster','to_cluster']).size().reset_index()
    graph_df.columns = ['from_cluster','to_cluster','weight']

    # normalize by the number of bouts per cluster
    graph_df = graph_df.merge(connected_bout_appearance.to_frame(name='total_bouts').reset_index(),left_on='from_cluster', right_on='cluster')
    graph_df = graph_df.assign(
        weight_norm = graph_df['weight']/graph_df['total_bouts']
    )

    network_weights = graph_df.pivot(index='cluster', columns='to_cluster', values='weight_norm').values
    network_weights = np.nan_to_num(network_weights)
    graph_df_reconstruct = pd.DataFrame(data = {
        'source': np.repeat(np.arange(nCluster),nCluster),
        'target': [x for x in np.arange(nCluster)]*nCluster,
        'weight': network_weights.flatten(),
        })
    all_graph = pd.concat([all_graph, graph_df_reconstruct.assign(cond1=sel_condition)], ignore_index=True)
    

# %% compare graph structure
# similarity by correlation
cond_mats = {
    cond: df.pivot(index="source", columns="target", values="weight").values.flatten()
    for cond, df in all_graph.groupby("cond1")
}
_corr = 1
# Compare all pairs
for condA in cond_mats:
    for condB in cond_mats:
        if condA < condB:
            corr, _ = pearsonr(cond_mats[condA], cond_mats[condB])
            print(f"{condA} vs {condB}: correlation = {corr:.3f}")
            if corr < _corr:
                condA_sel = condA
                condB_sel = condB
                _corr = corr
# %

#%%

matA = all_graph.query("cond1 == @condA_sel").pivot(index="source", columns="target", values="weight").values
matB = all_graph.query("cond1 == @condB_sel").pivot(index="source", columns="target", values="weight").values

diff =(matA - matB)/(matA + matB + 1e-9) 

# Plot
plt.figure(figsize=(8, 6))
im = plt.imshow(
    diff,
    cmap="bwr",
    vmin=-np.max(np.abs(diff)),  # symmetric color scale
    vmax=np.max(np.abs(diff))
)
plt.colorbar(im, label=f"Weight difference ({condA_sel} – {condB_sel})")
plt.title(f"Difference heatmap: {condA_sel} – {condB_sel}")
plt.xlabel("Target cluster")
plt.ylabel("Source cluster")
plt.tight_layout()
plt.show()
plt.savefig(f"{fig_dir}/graph_c{nCluster}_diff.pdf", format='PDF')

# %% quantify column bias

# Assuming diff is your difference matrix
col_bias = diff.mean(axis=0)   # average over sources (bias per target cluster)
row_bias = diff.mean(axis=1)   # average over targets (bias per source cluster)

# Build dataframe for seaborn
bias_df = pd.DataFrame({
    "cluster": list(range(len(col_bias))) + list(range(len(row_bias))),
    "bias": np.concatenate([col_bias, row_bias]),
    "type": ["Target (col)"] * len(col_bias) + ["Source (row)"] * len(row_bias)
})

sns.catplot(data=bias_df, x="cluster", y="bias", row="type", kind="bar", height=3, aspect=1.5)
plt.savefig(f"{fig_dir}/graph_c{nCluster}_diff bias.pdf", format='PDF')

# %%
col_bias_vals = diff    # shape: (n_source, n_target)
row_bias_vals = diff.T  # shape: (n_target, n_source)

# reshape into long form
bias_df = pd.DataFrame({
    "cluster": np.tile(np.arange(diff.shape[1]), diff.shape[0]).tolist() +
               np.tile(np.arange(diff.shape[0]), diff.shape[1]).tolist(),
    "bias": np.concatenate([col_bias_vals.flatten(), row_bias_vals.flatten()]),
    "type": ["Target (col)"] * diff.size + ["Source (row)"] * diff.size
})

g = sns.catplot(
    data=bias_df,
    x="cluster", y="bias",
    row="type", kind="bar",
    height=3, aspect=1.5,
    color="gray",
    # errorbar="sd"  
)
plt.savefig(f"{fig_dir}/graph_c{nCluster}_diff bias sem.pdf", format='PDF')

# %%
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests

# --- Target cluster bias (columns) ---
pvals_col = []
means_col = []
for j in range(diff.shape[1]):
    vals = diff[:, j]  # all sources -> target j
    means_col.append(np.mean(vals))
    _, p = ttest_1samp(vals, 0, nan_policy="omit")
    pvals_col.append(p)

# --- Source cluster bias (rows) ---
pvals_row = []
means_row = []
for i in range(diff.shape[0]):
    vals = diff[i, :]  # all targets -> source i
    means_row.append(np.mean(vals))
    _, p = ttest_1samp(vals, 0, nan_policy="omit")
    pvals_row.append(p)
    
# Correct separately for rows and cols
reject_col, pvals_col_corr, _, _ = multipletests(pvals_col, method="fdr_bh")
reject_row, pvals_row_corr, _, _ = multipletests(pvals_row, method="fdr_bh")

# Put results in DataFrame
results_col = pd.DataFrame({
    "cluster": np.arange(diff.shape[1]),
    "mean_bias": means_col,
    "pval_raw": pvals_col,
    "pval_corr": pvals_col_corr,
    "significant": reject_col,
    "type": "Target"
})

results_row = pd.DataFrame({
    "cluster": np.arange(diff.shape[0]),
    "mean_bias": means_row,
    "pval_raw": pvals_row,
    "pval_corr": pvals_row_corr,
    "significant": reject_row,
    "type": "Source"
})

results_df = pd.concat([results_col, results_row], ignore_index=True)
# %%
# plot mean ± std bars first
bias_df = pd.DataFrame({
    "cluster": np.tile(np.arange(diff.shape[1]), diff.shape[0]).tolist() +
               np.tile(np.arange(diff.shape[0]), diff.shape[1]).tolist(),
    "bias": np.concatenate([diff.flatten(), diff.T.flatten()]),
    "type": ["Target"] * diff.size + ["Source"] * diff.size
})

g = sns.catplot(
    data=bias_df, x="cluster", y="bias", row="type",
    kind="bar", height=3, aspect=1.5, color="gray", errorbar="sd"
)

# overlay significance markers
for ax, typ in zip(g.axes.flatten(), ["Target", "Source"]):
    sub = results_df[results_df["type"] == typ]
    for i, row in sub.iterrows():
        if row["significant"]:
            ax.text(row["cluster"], row["mean_bias"], "*",
                    ha="center", va="bottom", color="red", fontsize=12)

plt.tight_layout()
plt.savefig(f"{fig_dir}/graph_c{nCluster}_diff bias sem sig.pdf", format='PDF')
# %%
