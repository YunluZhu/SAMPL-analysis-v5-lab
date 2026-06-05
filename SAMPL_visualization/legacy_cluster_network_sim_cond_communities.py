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

matA = all_graph.query("cond1 == @condA_sel").pivot(index="source", columns="target", values="weight").values
matB = all_graph.query("cond1 == @condB_sel").pivot(index="source", columns="target", values="weight").values

diff = matA - matB

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

# %%
G_A = nx.from_numpy_array(matA, create_using=nx.DiGraph)  # or Graph if undirected
G_B = nx.from_numpy_array(matB, create_using=nx.DiGraph)

import community as community_louvain
from networkx.algorithms.community import louvain_communities


# Louvain communities (returns list of sets of nodes)
communities_A = louvain_communities(G_A.to_undirected(), weight='weight')
communities_B = louvain_communities(G_B.to_undirected(), weight='weight')

def plot_graph_with_communities(G, communities, title="Graph"):
    pos = nx.spring_layout(G, seed=42)  # fixed layout for comparison
    plt.figure(figsize=(8, 6))

    # Draw nodes community by community
    for i, comm in enumerate(communities):
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=list(comm),
            node_color=plt.cm.tab10(i % 10),
            node_size=300,
            label=f"Community {i}"
        )

    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    nx.draw_networkx_labels(G, pos)
    plt.title(title)
    plt.axis('off')
    plt.legend()
    plt.show()


# %%
plot_graph_with_communities(G_A, communities_A, title=f"Communities in {condA_sel}")
plot_graph_with_communities(G_B, communities_B, title=f"Communities in {condB_sel}")

#%%

diff = matA - matB
G_diff = nx.from_numpy_array(diff, create_using=nx.DiGraph)

plt.figure(figsize=(8,6))
pos = nx.spring_layout(G_diff, seed=42)

# Color edges by weight difference
edges = G_diff.edges()
weights = [G_diff[u][v]['weight'] for u,v in edges]
nx.draw_networkx_nodes(G_diff, pos, node_size=300)
nx.draw_networkx_edges(G_diff, pos, edge_color=weights, edge_cmap=plt.cm.bwr, width=2)
nx.draw_networkx_labels(G_diff, pos)
plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.bwr), label="Weight difference")
plt.title(f"Edge weight differences ({condA_sel} - {condB_sel})")
plt.axis('off')
plt.show()


# %%import matplotlib.pyplot as plt
def plot_condition_with_difference(all_graph, cond_current, cond_other, node_order, fig_dir=None):
    """
    Plot a graph for cond_current with:
    - nodes ordered by node_order (e.g., pitch initial)
    - nodes colored by communities (current condition)
    - edges colored by difference vs cond_other
      Yellow = increase, Blue = decrease, White = unchanged
    """
    # Pivot adjacency matrices
    mat_current = all_graph.query("cond1 == @cond_current").pivot(index="source", columns="target", values="weight").values
    mat_other = all_graph.query("cond1 == @cond_other").pivot(index="source", columns="target", values="weight").values
    
    diff = mat_current - mat_other
    nCluster = diff.shape[0]

    # Build graph DataFrame
    graph_df = pd.DataFrame({
        'from_cluster': np.repeat(np.arange(nCluster), nCluster),
        'to_cluster': np.tile(np.arange(nCluster), nCluster),
        'weight_diff': diff.flatten()
    })
    
    G = nx.from_pandas_edgelist(graph_df, 'from_cluster', 'to_cluster', ['weight_diff'], create_using=nx.DiGraph())

    # -----------------------------
    # FIXED NODE POSITIONS BY ORDER
    # -----------------------------
    # Number of nodes
    n = len(node_order)
    # Angles around circle
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.0
    # Map node_order -> positions
    pos = {node_order[i]: np.array([r*np.cos(angles[i]), r*np.sin(angles[i])]) for i in range(n)}
    
    # -----------------------------
    # Node colors by community
    # -----------------------------
    communities = louvain_communities(nx.from_numpy_array(mat_current, create_using=nx.Graph()), weight='weight')
    cmap_nodes = sns.color_palette("tab10", n_colors=len(communities))
    color_map = {node: cmap_nodes[i] for i, comm in enumerate(communities) for node in comm}
    node_colors = [color_map[n] for n in G.nodes()]

    # -----------------------------
    # Edge attributes
    # -----------------------------
    edges = G.edges()
    weights = np.array([G[u][v]['weight_diff'] for u,v in edges])
    edge_widths = np.abs(weights) / np.max(np.abs(weights)) * 4 if np.max(np.abs(weights)) > 0 else 1
    cmap_edges = sns.diverging_palette(240, 60, as_cmap=True)  # blue → white → yellow

    # -----------------------------
    # Plot
    # -----------------------------
    plt.figure(figsize=(8,6))
    nx.draw_networkx_edges(
        G, pos=pos,
        edge_color=weights,
        edge_cmap=cmap_edges,
        width=edge_widths,
        connectionstyle='arc3, rad=0.1'
    )
    nx.draw_networkx_nodes(
        G, pos=pos,
        node_color=node_colors,
        node_size=300
    )
    nx.draw_networkx_labels(G, pos, font_color='w')

    # Colorbar for edges
    sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=plt.Normalize(vmin=-np.max(np.abs(weights)), vmax=np.max(np.abs(weights))))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.ax.set_ylabel(f"{cond_current} – {cond_other}", rotation=270)

    plt.title(f"Condition: {cond_current}")
    plt.axis('off')

    if fig_dir:
        plt.savefig(f"{fig_dir}/network_diff_{cond_current}_vs_{cond_other}.pdf", format='PDF')
    plt.show()


# %%
node_order = all_feature_clustered.groupby('cluster')[sort_by_feature].mean().sort_values().index.tolist()
# plot
plot_condition_with_difference(all_graph, cond_current=condA_sel, cond_other=condB_sel, node_order=node_order, fig_dir=fig_dir)
plot_condition_with_difference(all_graph, cond_current=condB_sel, cond_other=condA_sel, node_order=node_order, fig_dir=fig_dir)

# %%
def plot_communities_by_condition(all_graph, cond_current, cond_other, node_order, fig_dir=None):
    """
    Plot each community of cond_current separately.
    Nodes keep positions from node_order.
    Edges colored by difference vs cond_other (yellow=increase, blue=decrease).
    
    Parameters
    ----------
    all_graph : pd.DataFrame
        Must have columns ['source','target','weight','cond1']
    cond_current : str
        Condition to show
    cond_other : str
        Condition to compare against
    node_order : list
        Fixed circular order for nodes
    fig_dir : str
        Optional save directory
    """
    # Adjacency matrices
    mat_current = all_graph.query("cond1 == @cond_current").pivot(index="source", columns="target", values="weight").values
    mat_other = all_graph.query("cond1 == @cond_other").pivot(index="source", columns="target", values="weight").values
    diff = mat_current - mat_other
    nCluster = diff.shape[0]

    # Community detection in current condition
    communities = louvain_communities(nx.from_numpy_array(mat_current, create_using=nx.Graph()), weight='weight')
    cmap_nodes = sns.color_palette("tab10", n_colors=len(communities))
    node_colors_map = {node: cmap_nodes[i] for i, comm in enumerate(communities) for node in comm}

    # Circular positions by node_order
    n = len(node_order)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.0
    pos = {node_order[i]: np.array([r*np.cos(angles[i]), r*np.sin(angles[i])]) for i in range(n)}

    # Build full graph with differences
    graph_df = pd.DataFrame({
        'from_cluster': np.repeat(np.arange(nCluster), nCluster),
        'to_cluster': np.tile(np.arange(nCluster), nCluster),
        'weight_diff': diff.flatten()
    })
    G_full = nx.from_pandas_edgelist(graph_df, 'from_cluster', 'to_cluster', ['weight_diff'], create_using=nx.DiGraph())

    # --------------------------
    # Plot each community separately
    # --------------------------
    for i, comm in enumerate(communities):
        nodes_in_comm = list(comm)

        # Keep only edges present in current condition and connected to this community
        edges_in_current = [(u,v) for u,v in G_full.edges() if mat_current[u,v] > 0 and (u in nodes_in_comm or v in nodes_in_comm)]
        G_sub = G_full.edge_subgraph(edges_in_current).copy()

        # --------------------------
        # Edge widths proportional to current condition weight
        # --------------------------
        weights = np.array([mat_current[u,v] for u,v in G_sub.edges()])
        weights_adj = weights / weights.max() * 4 if np.max(weights) > 0 else 1

        # --------------------------
        # Edge color: positive differences only, subtle yellow
        # --------------------------
        diff_values = np.array([diff[u,v] for u,v in G_sub.edges()])
        diff_positive = np.clip(diff_values, 0, None)  # only positive differences
        norm = plt.Normalize(vmin=0, vmax=np.max(diff_positive)) if np.max(diff_positive) > 0 else plt.Normalize(vmin=0, vmax=1)
        
        # subtle yellow from your previous palette
        diff_positive = np.clip(diff_values, 0, None)

        # Sequential colormap from white -> subtle yellow
        cmap_edges = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)

        # Normalize edge colors
        norm = plt.Normalize(vmin=0, vmax=np.max(diff_positive)) if np.max(diff_positive) > 0 else plt.Normalize(vmin=0, vmax=1)
        edge_colors = cmap_edges(norm(diff_positive))

        # --------------------------
        # Node colors by community
        # --------------------------
        node_color = [node_colors_map[n] for n in G_sub.nodes()]

        # --------------------------
        # Plot
        # --------------------------
        plt.figure(figsize=(6,6))
        nx.draw_networkx_edges(
            G_sub, pos=pos,
            edge_color=edge_colors,
            width=weights_adj,
            connectionstyle='arc3, rad=0.1'
        )
        nx.draw_networkx_nodes(G_sub, pos=pos, node_color=node_color, node_size=300)
        nx.draw_networkx_labels(G_sub, pos=pos, font_color='w')

        # Colorbar for edges
        sm = plt.cm.ScalarMappable(cmap=cmap_edges, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.ax.set_ylabel(f"{cond_current} – {cond_other}", rotation=270)

        plt.title(f"{cond_current} - Community {i+1}")
        plt.axis('off')

        if fig_dir:
            plt.savefig(f"{fig_dir}/community_{i+1}_{cond_current}_vs_{cond_other}.pdf", format='PDF')
        plt.show()

#%%
node_order = list(range(nCluster))  # e.g., pitch initial order
plot_communities_by_condition(all_graph, cond_current=condA_sel, cond_other=condA_sel, node_order=node_order, fig_dir=fig_dir)
plot_communities_by_condition(all_graph, cond_current=condA_sel, cond_other=condB_sel, node_order=node_order, fig_dir=fig_dir)

# %%
