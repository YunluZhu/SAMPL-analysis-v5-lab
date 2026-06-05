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
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import (KMeans, SpectralClustering)
import matplotlib as mpl
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


# %%
set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'wt_dl'
which_ztime = 'day'
compare_which = 'cond1' # condition for separation None for treat as whole
if_strict_DayNightSplit = True
sort_by_feature = 'pitch_initial' # by which parameter to sort the clusters on the figure

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} bipartite_sim_by{compare_which}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')
    

# %% 
all_around_peak_data, all_feature_cond, all_cond0, all_cond1, idxRANGE = get_aligned_bouts_wIBI(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=if_strict_DayNightSplit)

all_around_peak_data = all_around_peak_data.query("cond1 in ['dd','ld']") 
all_feature_cond = all_feature_cond.query("cond1 in ['dd','ld']")

# IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime, if_strict_DayNightSplit=if_strict_DayNightSplit)
#%%

# ---- INPUT ----
# bout_features must have at least: ['bout_uid', 'to_bout', 'post_IBI', ...bout feature columns...]
# If you have multiple fish/sessions, add a 'group_id' column and run per group.
df = all_feature_cond.copy()
# remove rows without a next bout
# Identify bout feature columns (everything except linkage/meta)
meta_cols = {'bout_uid', 'to_bout', 'post_IBI'}
bout_feature_cols = [c for c in df.columns if c not in meta_cols]

# ---- 1) Build IBI table from the bout table ----
ibi_df = (
    df.loc[df['to_bout'].notna()]
      .rename(columns={'bout_uid': 'from_bout', 'post_IBI': 'ibi_duration'})
      .assign(
          to_bout=lambda x: x['to_bout'],
          ibi_uid=lambda x: x['from_bout'].astype(str) + '_' + x['to_bout'].astype(str)
      )
    [['ibi_uid', 'from_bout', 'to_bout', 'ibi_duration','cond1','cond0']]
)

def compute_ibi_features(prev_bout: pd.Series, next_bout: pd.Series) -> dict[str, float]:
    # pitch_chg = next_bout['pitch_initial'] - prev_bout['pitch_end']
    # y_chg = next_bout['y_initial'] - prev_bout['y_end']
    # x_chg = next_bout['x_initial'] - prev_bout['x_end']
    # dist = math.sqrt(y_chg**2 + x_chg**2)
    return {
        "pitch_chg": next_bout['pitch_initial'] - prev_bout['pitch_end'],
        "y_chg": next_bout['y_initial'] - prev_bout['y_end'],
        "x_chg": next_bout['x_initial'] - prev_bout['x_end'],
    }
    
df_bouts_indexed = df.set_index('bout_uid', drop=False)
extra_feature_rows = []
for _, r in ibi_df.iterrows():
    prev_b = df_bouts_indexed.loc[r['from_bout']]
    try:
        next_b = df_bouts_indexed.loc[r['to_bout']]
    except KeyError:
        print(f"Warning: to_bout {r['to_bout']} not found for IBI {r['ibi_uid']}")
        continue
    extra_feature_rows.append({'ibi_uid': r['ibi_uid'], **compute_ibi_features(prev_b, next_b)})

if extra_feature_rows:
    extra_ibi = pd.DataFrame(extra_feature_rows).set_index('ibi_uid')
    ibi_df = ibi_df.join(extra_ibi, on='ibi_uid')


# %% determine number of clusters for bouts and IBIs separately

# # --- prepare df_tpcalc as you currently do ---
# chunk_size = idxRANGE[1] - idxRANGE[0]

# df_tpcalc = (
#     all_around_peak_data[['xvel_adj','yvel','propBoutAligned_pitch','propBoutAligned_angVel']]
#     .assign(
#         grp = lambda d: np.arange(len(d)) // chunk_size,
#         pos = lambda d: np.arange(len(d)) % chunk_size
#     )
# )

# # reshape wide
# re_format = df_tpcalc.set_index(['grp','pos']).unstack('pos')
# re_format.columns = [f"{col}_{pos}" for col, pos in re_format.columns]

# # standardize
# df_std = StandardScaler().fit_transform(re_format)

# # PCA
# pca = PCA(n_components=30)
# principalComponents = pca.fit_transform(df_std)
# PCA_components = pd.DataFrame(principalComponents)

# # --- determine number of clusters ---
# k_range = range(10, 18,2)
# wcss = []
# sil_scores = []

# for k in k_range:
#     km = KMeans(n_clusters=k, random_state=42).fit(PCA_components.iloc[:,:10])
#     labels = km.labels_
#     wcss.append(km.inertia_)
#     sil_scores.append(silhouette_score(PCA_components.iloc[:,:10], labels))

# # plot elbow
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(k_range, wcss, 'o-', color='blue')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("WCSS / inertia")
# plt.title("Bout: Elbow Method")
# plt.savefig(f"{fig_dir}/bout_clusterNumber_WCSS.pdf", format='PDF')

# # plot silhouette
# plt.subplot(1,2,2)
# plt.plot(k_range, sil_scores, 'o-', color='green')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("Bout: Silhouette Analysis")
# plt.show()
# plt.savefig(f"{fig_dir}/bout_clusterNumber_Silhouette.pdf", format='PDF')

# #%% DO NOT DELETE
# # Suppose ibi_features has numeric columns describing IBIs
# ibi_feature_cols = [c for c in ibi_df.columns if c not in ['ibi_uid', 'from_bout', 'to_bout','cond1','cond0','ibi_state','state_id']]
# X_ibi = ibi_df[ibi_feature_cols].to_numpy()

# # standardize
# X_ibi_std = StandardScaler().fit_transform(X_ibi)

# # determine number of clusters
# k_range = range(2, 14, 2)
# wcss = []
# sil_scores = []

# for k in k_range:
#     km = KMeans(n_clusters=k, random_state=42).fit(X_ibi_std)
#     labels = km.labels_
#     wcss.append(km.inertia_)
#     sil_scores.append(silhouette_score(X_ibi_std, labels))

# # plot elbow and silhouette
# plt.figure(figsize=(12,5))
# plt.subplot(1,2,1)
# plt.plot(k_range, wcss, 'o-', color='blue')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("WCSS / inertia")
# plt.savefig(f"{fig_dir}/ibi_clusterNumber_WCSS.pdf", format='PDF')

# plt.subplot(1,2,2)
# plt.plot(k_range, sil_scores, 'o-', color='green')
# plt.xlabel("Number of clusters (k)")
# plt.ylabel("Silhouette Score")
# plt.title("IBI: Silhouette Analysis")
# plt.savefig(f"{fig_dir}/ibi_clusterNumber_Silhouette.pdf", format='PDF')


#%%
nCluster = 16
all_feature_clustered = get_cluster_phaseSpace(all_around_peak_data, all_feature_cond, idxRANGE, nCluster)

# rename cluster to bout_state
bout_df = all_feature_clustered.rename(columns={'cluster': 'bout_state'})
#%%
ibi_feature_cols = [c for c in ibi_df.columns if c not in ['ibi_uid', 'from_bout', 'to_bout','cond1','cond0','ibi_state','state_id']]
ibi_df = ibi_df.dropna(subset=ibi_feature_cols).reset_index(drop=True)
X_ibi = ibi_df[ibi_feature_cols].to_numpy()
X_ibi = StandardScaler().fit_transform(X_ibi)
k_ibi = 8
km_ibi = KMeans(n_clusters=k_ibi).fit(X_ibi)
ibi_df['ibi_state'] = km_ibi.labels_

#%%
bout_df['state_id'] = bout_df['bout_state'].apply(lambda s: f"B{s}")
ibi_df['state_id'] = ibi_df['ibi_state'].apply(lambda s: f"I{s}")

#%% bild bi-partite graph

graph_total = pd.DataFrame()

for sel_condition in bout_df.cond1.unique():
    sel_bout = bout_df[bout_df['cond1'] == sel_condition]
    sel_ibi = ibi_df[ibi_df['cond1'] == sel_condition]
    bout_to_ibi = (
        sel_ibi
        .merge(sel_bout[['bout_uid','bout_state']], 
            left_on='from_bout', right_on='bout_uid')
        [['bout_state','ibi_state']]
        .groupby(['bout_state','ibi_state']).size().reset_index(name='count')
    )

    # normalize by number of bouts in that bout cluster
    bout_counts = sel_bout.groupby('bout_state').size().rename('n_bouts')
    bout_to_ibi = bout_to_ibi.merge(bout_counts, on='bout_state')
    bout_to_ibi['weight_norm'] = bout_to_ibi['count'] / bout_to_ibi['n_bouts']


    # --- Step 2. IBI → Bout connections ---
    ibi_to_bout = (
        sel_ibi
        .merge(sel_bout[['bout_uid','bout_state']], 
            left_on='to_bout', right_on='bout_uid')
        [['ibi_state','bout_state']]
        .groupby(['ibi_state','bout_state']).size().reset_index(name='count')
    )

    # normalize by number of IBIs in that ibi cluster
    ibi_counts = sel_ibi.groupby('ibi_state').size().rename('n_ibis')
    ibi_to_bout = ibi_to_bout.merge(ibi_counts, on='ibi_state')
    ibi_to_bout['weight_norm'] = ibi_to_bout['count'] / ibi_to_bout['n_ibis']

    # give unique namespaces to avoid collisions
    bout_to_ibi['from_state'] = 'B' + bout_to_ibi['bout_state'].astype(str)
    bout_to_ibi['to_state']   = 'I' + bout_to_ibi['ibi_state'].astype(str)

    ibi_to_bout['from_state'] = 'I' + ibi_to_bout['ibi_state'].astype(str)
    ibi_to_bout['to_state']   = 'B' + ibi_to_bout['bout_state'].astype(str)

    graph_df = pd.concat([
        bout_to_ibi[['from_state','to_state','weight_norm']],
        ibi_to_bout[['from_state','to_state','weight_norm']]
    ])

    # --- Keep string IDs instead of converting to numeric ---
    states = sorted(set(graph_df['from_state']) | set(graph_df['to_state']))

    # adjacency matrix if you still want it
    network_weights = pd.DataFrame(
        0.0, index=states, columns=states
    )

    for _, row in graph_df.iterrows():
        network_weights.loc[row['from_state'], row['to_state']] = row['weight_norm']

    # graph_df_reconstruct in long format
    this_graph_df_reconstruct = pd.DataFrame({
        'source': graph_df['from_state'],
        'target': graph_df['to_state'],
        'weight': graph_df['weight_norm']
    }).assign(cond1=sel_condition)
    
    graph_total = pd.concat([graph_total, this_graph_df_reconstruct], ignore_index=True)
    
#%%
#%%

def plot_edges(edges_subset, G, pos, title):
    # extract raw weights
    weights = np.array([G[u][v].get('weight', 0.0) for u,v in edges_subset])

    # --- adjust weights independently for this subset ---
    if weights.max() > 0:
        weights_adj = weights / weights.max() * 4
        alphas = np.log((1 + weights / weights.max()) * (np.e/2))
    else:
        weights_adj = np.ones_like(weights)
        alphas = np.ones_like(weights)

    # normalize alphas to [0.2,1] for visibility
    alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min() + 1e-9)
    alphas = 0.2 + 0.8*alphas

    # color map
    cmap = plt.cm.bwr
    vmin, vmax = weights_adj.min(), weights_adj.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights_adj)

    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=B_nodes, node_color="skyblue")
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=I_nodes, node_color="lightgreen")
    nx.draw_networkx_labels(G, pos, font_size=10)

    for (u,v), w_adj, a in zip(edges_subset, weights_adj, alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u,v)],
            width=float(w_adj),
            edge_color=[float(w_adj)],
            edge_cmap=cmap,
            edge_vmin=vmin,
            edge_vmax=vmax,
            alpha=float(a),
            arrows=True
        )

    plt.colorbar(sm, label="weights_adj (for this edge subset)")
    plt.title(title)
    plt.axis('off')
    plt.show()


graph_df = pd.DataFrame()

for cond_sel, graph_total_sel in graph_total.groupby('cond1'):
    adj = graph_total_sel.pivot(index="source", columns="target", values="weight").fillna(0)

    # directly build graph with labels preserved
    G_sel = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)

    # plt.figure(figsize=(8,6))
    # Identify node sets
    B_nodes = [n for n in G_sel.nodes if n.startswith("B")]
    I_nodes = [n for n in G_sel.nodes if n.startswith("I")]

    # Assign "bipartite" attribute
    nx.set_node_attributes(G_sel, {n: 0 for n in B_nodes}, "bipartite")
    nx.set_node_attributes(G_sel, {n: 1 for n in I_nodes}, "bipartite")

    # Layout: nodes on two parallel lines
    pos = nx.bipartite_layout(G_sel, B_nodes)  # specify one set
    plt.figure(figsize=(8,6))

    edges = list(G_sel.edges())

    # split edges
    edges_IB = [(u,v) for (u,v) in edges if u.startswith("I") and v.startswith("B")]
    edges_BI = [(u,v) for (u,v) in edges if u.startswith("B") and v.startswith("I")]

    # plot separately with independent weight adjustment
    plot_edges(edges_IB, G_sel, pos, f"I → B | cond1={cond_sel}")
    plt.savefig(f"{fig_dir}/network_I2B_{cond_sel}.pdf", format='PDF')
    plot_edges(edges_BI, G_sel, pos, f"B → I | cond1={cond_sel}")
    plt.savefig(f"{fig_dir}/network_B2I_{cond_sel}.pdf", format='PDF')
    
    graph_df = pd.concat([graph_df, adj.assign(cond1=cond_sel)], ignore_index=True)


#%%
    
######### below works for single condition #########

# --- Step 1. Bout → IBI connections ---
bout_to_ibi = (
    ibi_df
    .merge(bout_df[['bout_uid','bout_state']], 
           left_on='from_bout', right_on='bout_uid')
    [['bout_state','ibi_state']]
    .groupby(['bout_state','ibi_state']).size().reset_index(name='count')
)

# normalize by number of bouts in that bout cluster
bout_counts = bout_df.groupby('bout_state').size().rename('n_bouts')
bout_to_ibi = bout_to_ibi.merge(bout_counts, on='bout_state')
bout_to_ibi['weight_norm'] = bout_to_ibi['count'] / bout_to_ibi['n_bouts']


# --- Step 2. IBI → Bout connections ---
ibi_to_bout = (
    ibi_df
    .merge(bout_df[['bout_uid','bout_state']], 
           left_on='to_bout', right_on='bout_uid')
    [['ibi_state','bout_state']]
    .groupby(['ibi_state','bout_state']).size().reset_index(name='count')
)

# normalize by number of IBIs in that ibi cluster
ibi_counts = ibi_df.groupby('ibi_state').size().rename('n_ibis')
ibi_to_bout = ibi_to_bout.merge(ibi_counts, on='ibi_state')
ibi_to_bout['weight_norm'] = ibi_to_bout['count'] / ibi_to_bout['n_ibis']

# give unique namespaces to avoid collisions
bout_to_ibi['from_state'] = 'B' + bout_to_ibi['bout_state'].astype(str)
bout_to_ibi['to_state']   = 'I' + bout_to_ibi['ibi_state'].astype(str)

ibi_to_bout['from_state'] = 'I' + ibi_to_bout['ibi_state'].astype(str)
ibi_to_bout['to_state']   = 'B' + ibi_to_bout['bout_state'].astype(str)

graph_df = pd.concat([
    bout_to_ibi[['from_state','to_state','weight_norm']],
    ibi_to_bout[['from_state','to_state','weight_norm']]
])

# --- Keep string IDs instead of converting to numeric ---
states = sorted(set(graph_df['from_state']) | set(graph_df['to_state']))

# adjacency matrix if you still want it
network_weights = pd.DataFrame(
    0.0, index=states, columns=states
)

for _, row in graph_df.iterrows():
    network_weights.loc[row['from_state'], row['to_state']] = row['weight_norm']

# graph_df_reconstruct in long format
graph_df_reconstruct = pd.DataFrame({
    'source': graph_df['from_state'],
    'target': graph_df['to_state'],
    'weight': graph_df['weight_norm']
})

#%%
adj = graph_df_reconstruct.pivot(index="source", columns="target", values="weight").fillna(0)

# directly build graph with labels preserved
G_total = nx.from_pandas_adjacency(adj, create_using=nx.DiGraph)

plt.figure(figsize=(8,6))
# Identify node sets
B_nodes = [n for n in G_total.nodes if n.startswith("B")]
I_nodes = [n for n in G_total.nodes if n.startswith("I")]

# Assign "bipartite" attribute
nx.set_node_attributes(G_total, {n: 0 for n in B_nodes}, "bipartite")
nx.set_node_attributes(G_total, {n: 1 for n in I_nodes}, "bipartite")

# Layout: nodes on two parallel lines
pos = nx.bipartite_layout(G_total, B_nodes)  # specify one set
plt.figure(figsize=(8,6))

edges = list(G_total.edges())

def plot_edges(edges_subset, G, pos, title):
    # extract raw weights
    weights = np.array([G[u][v].get('weight', 0.0) for u,v in edges_subset])

    # --- adjust weights independently for this subset ---
    if weights.max() > 0:
        weights_adj = weights / weights.max() * 4
        alphas = np.log((1 + weights / weights.max()) * (np.e/2))
    else:
        weights_adj = np.ones_like(weights)
        alphas = np.ones_like(weights)

    # normalize alphas to [0.2,1] for visibility
    alphas = (alphas - alphas.min()) / (alphas.max() - alphas.min() + 1e-9)
    alphas = 0.2 + 0.8*alphas

    # color map
    cmap = plt.cm.bwr
    vmin, vmax = weights_adj.min(), weights_adj.max()
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(weights_adj)

    plt.figure(figsize=(8,6))
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=B_nodes, node_color="skyblue")
    nx.draw_networkx_nodes(G, pos, node_size=500, nodelist=I_nodes, node_color="lightgreen")
    nx.draw_networkx_labels(G, pos, font_size=10)

    for (u,v), w_adj, a in zip(edges_subset, weights_adj, alphas):
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u,v)],
            width=float(w_adj),
            edge_color=[float(w_adj)],
            edge_cmap=cmap,
            edge_vmin=vmin,
            edge_vmax=vmax,
            alpha=float(a),
            arrows=True
        )

    plt.colorbar(sm, label="weights_adj (for this edge subset)")
    plt.title(title)
    plt.axis('off')
    plt.show()


# split edges
edges_IB = [(u,v) for (u,v) in edges if u.startswith("I") and v.startswith("B")]
edges_BI = [(u,v) for (u,v) in edges if u.startswith("B") and v.startswith("I")]

# plot separately with independent weight adjustment
plot_edges(edges_IB, G_total, pos, "I → B connections")
plot_edges(edges_BI, G_total, pos, "B → I connections")

# %%
