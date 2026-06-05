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
from hmmlearn import hmm
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

folder_name = f'{pick_data} hmm_sim_by{compare_which}'
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

# filter out ll conditions in all_feature_cond and all_around_peak_data
all_around_peak_data = all_around_peak_data.query('cond1 != "ll"')
all_feature_cond = all_feature_cond.query('cond1 != "ll"')
# %%
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

# %%
# join ibi to bout df
all_feature_cond = all_feature_cond.merge(ibi_df[['from_bout', 'ibi_duration','pitch_chg','y_chg','x_chg']], left_on='bout_uid', right_on='from_bout', how='left')
# %
all_around_peak_data = all_around_peak_data.reset_index(drop=True)
all_around_peak_data = all_around_peak_data.assign(bout_uid = np.repeat(all_feature_cond['bout_uid'], np.diff(idxRANGE)[0]).values)

#%%

chunk_size = idxRANGE[1] - idxRANGE[0]

df_tpcalc = (
    all_around_peak_data[['xvel_adj','yvel','propBoutAligned_pitch','propBoutAligned_angVel', 'propBoutAligned_speed']]
    .assign(
        grp = lambda d: np.arange(len(d)) // chunk_size,
        pos = lambda d: np.arange(len(d)) % chunk_size
    )
)

# reshape wide
re_format = df_tpcalc.set_index(['grp','pos']).unstack('pos')
re_format.columns = [f"{col}_{pos}" for col, pos in re_format.columns]
re_format['bout_uid'] = all_feature_cond['bout_uid'].values
re_format['cond1'] = all_feature_cond['cond1'].values

re_format = re_format.merge(ibi_df[['from_bout', 'to_bout','ibi_duration','pitch_chg','y_chg','x_chg']], left_on='bout_uid', right_on='from_bout', how='left')

#%%
re_format = re_format.dropna()  # only keep rows with a next bout
re_format_sel = re_format.query("cond1 == 'dd'")
# standardize
df_std = StandardScaler().fit_transform(re_format_sel.loc[:, ~re_format_sel.columns.isin(['bout_uid','from_bout','to_bout','cond1'])])

# PCA
pca = PCA(n_components=20)
principalComponents = pca.fit_transform(df_std)
PCA_components = pd.DataFrame(principalComponents)

sns.lineplot(pca.explained_variance_ratio_)

PCA_components.set_index(re_format_sel['bout_uid'], inplace=True)
#%% construct bout chains

# Build a mapping for easy traversal
link = dict(zip(re_format_sel['bout_uid'], re_format_sel['to_bout']))

visited = set()
chains = []

for start in re_format_sel['bout_uid']:
    if start in visited:
        continue
    chain = []
    curr = start
    while curr in link and not pd.isna(link[curr]) and curr not in visited:
        chain.append(curr)
        visited.add(curr)
        curr = link[curr]
    if chain:
        chains.append(chain)
print(f"Reconstructed {len(chains)} chains")


#%%

PCA_components_sel = PCA_components.iloc[:,:8]

X_list = []
lengths = []

for chain in chains:
    if len(chain) < 2:
        continue
    X_chain = PCA_components_sel.loc[chain]
    X_list.append(X_chain.values)
    lengths.append(len(chain))

X_concat = np.vstack(X_list)

# Sanity check
assert sum(lengths) == X_concat.shape[0], "Mismatch in data/lengths!"

print(f"Final data shape: {X_concat.shape}, sequences: {len(lengths)}")

#%%

K = 12  # choose number of hidden states
model = hmm.GaussianHMM(
    n_components=K,
    covariance_type="diag", # "full" or "diag"
    n_iter=300,
)

model.fit(X_concat, lengths)
print(model.monitor_.converged)
#%%
state_sequences = []
for chain in chains:
    if len(chain) < 2:
        continue
    X_chain = PCA_components_sel.loc[chain].values
    hidden_states = model.predict(X_chain)
    state_sequences.append(pd.DataFrame({
        "bout_uid": chain,
        "state": hidden_states
    }))

state_df = pd.concat(state_sequences, ignore_index=True)

#%%
plt.figure(figsize=(8,6))
sns.heatmap(model.transmat_, annot=True, fmt=".2f", cmap="Blues")
plt.title("HMM Transition Matrix")
plt.xlabel("To state")
plt.ylabel("From state")
plt.show()

#%% visualize on a graph

# state_df has 'bout_uid' and 'state'
# merge with to_bout info to get consecutive transitions
state_df_merged = state_df.merge(re_format_sel[['bout_uid', 'to_bout']], left_on='bout_uid', right_on='bout_uid', how='left')

# create edges: current_state -> next_state
edges_df = state_df_merged.merge(
    state_df[['bout_uid','state']],
    left_on='to_bout',
    right_on='bout_uid',
    how='left',
    suffixes=('_from','_to')
)

edges_df = edges_df.dropna(subset=['state_to'])

# Count transitions
graph_df = edges_df.groupby(['state_from','state_to']).size().reset_index(name='weight')

# Normalize by number of bouts per state
total_bouts = graph_df.groupby('state_from')['weight'].sum().reset_index(name='total_bouts')
graph_df = graph_df.merge(total_bouts, on='state_from')
graph_df['weight_norm'] = graph_df['weight'] / graph_df['total_bouts']

# ---------------------------------------------------
# 2. Create directed graph
# ---------------------------------------------------
G = nx.from_pandas_edgelist(
    graph_df,
    source='state_from',
    target='state_to',
    edge_attr=['weight'],
    create_using=nx.DiGraph()
)

# ---------------------------------------------------
# 3. Node layout and colors
# ---------------------------------------------------
nCluster = len(G.nodes)
angles = np.linspace(0, 2*np.pi, nCluster, endpoint=False)
r = 1.0
pos = {node: np.array([r*np.cos(a), r*np.sin(a)]) for node, a in zip(G.nodes, angles)}

c_list = sns.diverging_palette(250, 30, l=65, center="dark", n=nCluster)
color_map = {node: c_list[i] for i, node in enumerate(G.nodes)}
node_color = [color_map[n] for n in G.nodes()]

#%%
# ---------------------------------------------------
# 4. Edge width and alpha adjustment (your method)
# ---------------------------------------------------
edges = G.edges()
weights = np.array([G[u][v]['weight'] for u,v in edges])
weights_adj = weights / weights.max() * 4 if np.max(weights) > 0 else 1
alpha = np.log((1 + weights / np.max(weights)) * (np.e/2)) if np.max(weights) > 0 else 1

# Optional: color edges by weight
edge_cmap = sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)

# ---------------------------------------------------
# 5. Plot
# ---------------------------------------------------
fig, ax = plt.subplots(1, figsize=(8,8))
nx.draw_networkx_nodes(G, pos=pos, node_color=node_color, node_size=600, ax=ax)
nx.draw_networkx_labels(G, pos=pos, font_color='w', ax=ax)

nx.draw_networkx_edges(
    G, pos=pos,
    width=weights_adj,
    # alpha=(alpha - alpha.min()) / (1 - alpha.min()) if np.max(alpha-alpha.min())>0 else 1,
    connectionstyle='arc3, rad=0.1',
    edge_color=weights,
    edge_cmap=edge_cmap,
    arrows=True,
    arrowsize=15,
    ax=ax
)

# Edge colorbar
edges2 = nx.draw_networkx_edges(
    G, pos=pos,
    width=0,
    edge_color=weights,
    edge_cmap=edge_cmap,
    ax=ax,
    arrows=False
)
cbar = plt.colorbar(edges2, ax=ax)
cbar.ax.set_ylabel("Normalized transition weight", rotation=270)

plt.title("HMM State Transition Graph")
plt.axis('off')
plt.show()
