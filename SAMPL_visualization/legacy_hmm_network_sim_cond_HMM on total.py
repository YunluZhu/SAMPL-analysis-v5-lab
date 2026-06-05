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
from joblib import Parallel, delayed


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

def fit_hmm_for_condition(re_format, condition, n_pca=8, n_states=12, n_iter=300):
    re_format = re_format.dropna()  # only keep rows with a next bout
    # Filter condition
    re_format_sel = re_format.query("cond1 == @condition")
    
    # Standardize
    df_std = StandardScaler().fit_transform(
        re_format_sel.loc[:, ~re_format_sel.columns.isin(['bout_uid','from_bout','to_bout','cond1'])]
    )
    
    # PCA
    pca = PCA(n_components=n_pca)
    principalComponents = pca.fit_transform(df_std)
    PCA_components = pd.DataFrame(principalComponents, index=re_format_sel['bout_uid'])
    
    # Construct bout chains
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
    
    # Prepare X_concat and lengths
    PCA_components_sel = PCA_components.iloc[:, :8]  # or n_pca
    X_list = []
    lengths = []
    for chain in chains:
        if len(chain) < 2:
            continue
        X_chain = PCA_components_sel.loc[chain]
        X_list.append(X_chain.values)
        lengths.append(len(chain))
    X_concat = np.vstack(X_list)
    
    # Fit HMM
    model = hmm.GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=n_iter,
        random_state=42
    )
    model.fit(X_concat, lengths)
    
    # Predict hidden states
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
    
    return model, state_df, chains, PCA_components_sel

# ---------------------------------------------------
# Run conditions in parallel
# ---------------------------------------------------
conditions = ['ld', 'dd']

results = Parallel(n_jobs=-1)(
    delayed(fit_hmm_for_condition)(re_format, cond) for cond in conditions
)

# Unpack
models = {cond: results[i][0] for i, cond in enumerate(conditions)}
state_dfs = {cond: results[i][1] for i, cond in enumerate(conditions)}
chains_dict = {cond: results[i][2] for i, cond in enumerate(conditions)}
PCA_components_dict = {cond: results[i][3] for i, cond in enumerate(conditions)}

#%%
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def prepare_node_layout(state_dfs, all_feature_cond):
    """
    Compute global pitch_initial per state across ALL conditions,
    and return node order, positions, and color normalization.
    """
    # Merge all states across conditions
    merged_all = []
    for cond, state_df in state_dfs.items():
        merged_all.append(all_feature_cond.merge(state_df, on="bout_uid", how="inner"))
    merged_all = pd.concat(merged_all, ignore_index=True)

    # Compute average pitch_initial per state
    pitch_means = merged_all.groupby("state")["pitch_initial"].mean()
    pitch_means_sorted = pitch_means.sort_values()

    # Node order and layout
    node_order = pitch_means_sorted.index.tolist()
    n = len(node_order)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    r = 1.0
    pos_alt = {node_order[i]: np.array([r*np.cos(angles[i]), r*np.sin(angles[i])]) for i in range(n)}

    # Node colormap setup (fixed globally)
    vmin, vmax = pitch_means.min(), pitch_means.max()
    norm_nodes = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    cmap_nodes = plt.cm.seismic

    return pitch_means, node_order, pos_alt, norm_nodes, cmap_nodes


# ---------------------------------------------------
# 2. Plot one condition’s transition network
# ---------------------------------------------------
def plot_hmm_state_network_condition(cond, state_dfs, all_feature_cond,
                                     node_order, pos_alt, norm_nodes, cmap_nodes,
                                     title=None):
    """
    Plot one condition, using globally fixed node colors and positions.
    """
    state_df = state_dfs[cond]

    # Merge states with features
    merged = all_feature_cond.merge(state_df, on="bout_uid", how="inner")
    to_state = all_feature_cond[['bout_uid','to_bout']].merge(
        state_df, left_on="to_bout", right_on="bout_uid", how="left", suffixes=("","_to")
    )
    merged = merged.merge(to_state[['bout_uid','state']], on="bout_uid", how="left", suffixes=("","_to"))
    merged = merged.rename(columns={"state":"from_state", "state_to":"to_state"})

    # Transition counts
    trans_df = (
        merged.dropna(subset=["from_state","to_state"])
        .groupby(["from_state","to_state"]).size()
        .reset_index(name="weight")
    )
    if trans_df.empty:
        print(f"No transitions found for {cond}")
        return

    total_trans = trans_df.groupby("from_state")["weight"].transform("sum")
    trans_df["weight_norm"] = trans_df["weight"] / total_trans

    # Graph
    G = nx.DiGraph()
    for _, row in trans_df.iterrows():
        G.add_edge(int(row["from_state"]), int(row["to_state"]),
                   weight=row["weight"], weight_norm=row["weight_norm"])

    # Node colors (fixed by global pitch_initial norm)
    pitch_vals = merged.groupby("from_state")["pitch_initial"].mean()
    node_colors = [cmap_nodes(norm_nodes(pitch_vals.get(node, 0))) for node in G.nodes()]

    # Edge weights
    edges = G.edges()
    weights = np.array([G[u][v]['weight'] for u,v in edges])
    weights_adj = weights / weights.max() * 4 if np.max(weights) > 0 else 1
    alpha = np.log((1 + weights / np.max(weights)) * (np.e/2)) if np.max(weights) > 0 else 1

    cmap_edges = plt.cm.magma_r
    norm_edges = mcolors.Normalize(vmin=weights.min(), vmax=weights.max())

    # --- Plot
    fig, ax = plt.subplots(1, figsize=(9,6))

    # Edges
    nx.draw_networkx_edges(
        G, pos=pos_alt,
        alpha=(alpha-alpha.min())/(1-alpha.min()) if np.max(alpha-alpha.min())>0 else 1,
        width=weights_adj,
        connectionstyle='arc3, rad=0.15',
        edge_color=weights,
        edge_cmap=cmap_edges,
        edge_vmin=weights.min(),
        edge_vmax=weights.max(),
        ax=ax
    )

    # Nodes + labels
    nx.draw_networkx_nodes(G, pos=pos_alt, node_color=node_colors, ax=ax)
    nx.draw_networkx_labels(G, pos=pos_alt, font_color='w', ax=ax)

    # Edge colorbar
    sm_edges = cm.ScalarMappable(cmap=cmap_edges, norm=norm_edges)
    sm_edges.set_array([])
    cbar_edges = plt.colorbar(sm_edges, ax=ax, fraction=0.046, pad=0.04)
    cbar_edges.ax.set_ylabel("Transition probability", rotation=270)

    # Node colorbar (shared global scale)
    sm_nodes = cm.ScalarMappable(cmap=cmap_nodes, norm=norm_nodes)
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, fraction=0.046, pad=0.08)
    cbar_nodes.ax.set_ylabel("Pitch initial", rotation=270)

    if title is None:
        title = f"{cond} condition"
    plt.title(title)
    plt.axis("off")
    plt.savefig(os.path.join(fig_dir, f"{title}.pdf"), bbox_inches='tight')
# %%
# Prepare layout based on ALL conditions
pitch_means, node_order, pos_alt, norm_nodes, cmap_nodes = prepare_node_layout(state_dfs, all_feature_cond)

# Plot LD
plot_hmm_state_network_condition("ld", state_dfs, all_feature_cond,
                                 node_order, pos_alt, norm_nodes, cmap_nodes,
                                 title="LD condition")

# Plot DD
plot_hmm_state_network_condition("dd", state_dfs, all_feature_cond,
                                 node_order, pos_alt, norm_nodes, cmap_nodes,
                                 title="DD condition")










# %%
# Merge state_dfs into one big dataframe
merged_all = []
for cond, state_df in state_dfs.items():
    tmp = all_feature_cond.merge(state_df, on="bout_uid", how="inner")
    tmp["condition"] = cond
    merged_all.append(tmp)

merged_all = pd.concat(merged_all, ignore_index=True)

#%%
# Pick numeric features only
numeric_features = merged_all.select_dtypes(include=np.number).columns
# exclude_cols = ["state", "bout_uid",'cond0','cond1','ztime','condition']  # keep state separately
feature_cols = ['pitch_initial','pitch_peak','pitch_end','rot_l_accel','rot_l_decel','rot_total','traj_peak','ydispl_swim','WHM','ibi_duration','pitch_chg']#[c for c in numeric_features if c not in exclude_cols]

# Copy so we don't overwrite original
merged_scaled = merged_all.copy()

scaler = StandardScaler()
merged_scaled[feature_cols] = scaler.fit_transform(merged_scaled[feature_cols])


state_summary = merged_scaled.groupby(["state",'cond1'])[feature_cols].agg(["mean","std"])


# %%
for this_cond in ['ld','dd']:
    this_data = merged_all.query("cond1 == @this_cond")
    this_data_scaled = this_data.copy()
    this_data_scaled[feature_cols] = scaler.fit_transform(this_data[feature_cols])
    state_summary = this_data_scaled.groupby(["state"])[feature_cols].agg(["mean","std"])
    print(this_cond)
    sns.clustermap(
        state_summary.xs("mean", axis=1, level=1), 
        cmap="vlag", center=0,
        col_cluster=False, row_cluster=False,  # cluster columns only
        figsize=(6,6),
    )
    plt.suptitle("Standardized average feature values per HMM state", y=1.02)
    plt.show()
# %%
# Conclusion: features are quite variable within states between conditions. Possibly requires more states or better standardization.