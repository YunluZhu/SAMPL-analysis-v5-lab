# %%

import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_aligned_bouts, get_bout_features, extract_bout_features_v5)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from plot_functions.plt_functions import (plt_categorical_grid, plt_network_graphs)
from plot_functions.get_bout_correlation import get_cluster_phaseSpace
import math
from plot_functions.plt_tools import round_half_up 
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
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
pick_data = 'depth_7d'
which_ztime = 'day'
compare_which = None # condition for separation None for treat as whole
nCluster = 10
sort_by_feature='ydispl_swim' # by which parameter to sort the clusters on the figure

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} cluster_sim_by{compare_which}'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% 
all_around_peak_data, all_feature_cond, all_cond0, all_cond1, idxRANGE = get_aligned_bouts(root, FRAME_RATE, ztime=which_ztime)
IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)

# %%
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
if bool(compare_which):
    for this_condition in connected_bout_features[compare_which].unique():
        this_cond_features = connected_bout_features.loc[connected_bout_features[compare_which]==this_condition]
        # plot_network_graphs(extracted_features=this_cond_features, cond_sep=this_condition, sort_by_feature='ydispl_swim', total_features=connected_bout_features)
        extracted_features=this_cond_features
        cond_sep=this_condition
        total_features=connected_bout_features
        
        plt_network_graphs(connected_bout_features, 
                            fig_dir = fig_dir,
                            sort_by_feature = sort_by_feature,
                            cond_sep=this_condition, 
                            extracted_features=this_cond_features)
else:
    plt_network_graphs(connected_bout_features, 
                        fig_dir = fig_dir,
                        sort_by_feature = sort_by_feature)
    
# %% generate graph for simulation
connected_bouts = connected_bout_features.loc[:,['cluster','to_cluster','expNum','cond0','cond1']]
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


# %% I dentify highly associated nodes
# Calculate the association between nodes based on the edge weights

G = nx.from_pandas_edgelist(graph_df_reconstruct, 'source', 'target', 'weight', create_using=nx.DiGraph()) 

# %% if to calculate path probabilities between two nodes:

# Initialize a dictionary to store the probability of each path
path_probs = {}

# Iterate over all pairs of nodes
for source in G.nodes():
    for target in G.nodes():
        if source != target:
            # Find all simple paths between the source and target nodes
            paths = list(nx.all_simple_paths(G, source=source, target=target))

            # Calculate the probability of each path and sum them up
            path_prob = sum([np.prod([G[u][v]['weight'] for u, v in zip(path[:-1], path[1:])]) for path in paths])

            # Store the sum of probabilities in the dictionary
            path_probs[(source, target)] = path_prob

# %%
communities = greedy_modularity_communities(G, weight='weight')

node_communities = {}
for i, community in enumerate(communities):
    for node in community:
        node_communities[node] = i
# %% visualization 1
community_to_color = {
    0 : 'tab:blue',
    1 : 'tab:orange',
    2 : 'tab:green',
    3 : 'tab:red',
}
node_color = {node: community_to_color[community_id] for node, community_id in node_communities.items()}
edge_labels = {(edge[0], edge[1]): weight for edge, weight in path_probs.items()}

alpha = edge_labels

for node in list(G.nodes):
    alpha[(node, node)] = 0

alpha_values = np.array(list(alpha.values()))
# alpha_values = alpha_values/alpha_values.min()-1
# alpha_values = alpha_values/alpha_values.max()
# alpha_values = exp10(alpha_values)/10

alpha_adj = []
for ele in alpha_values:
    if ele < 0.15:
        ele = 0
    alpha_adj.append(ele)

alpha_values = np.array(alpha_adj)
alpha_values = alpha_values/alpha_values.max()

alpha = {edge:alpha_values[i] for i, edge in enumerate(alpha)}

Graph(G,
      node_color=node_color, node_edge_width=0, edge_alpha=alpha,
      node_layout='community', node_layout_kwargs=dict(node_to_community=node_communities),
      edge_layout='bundled', edge_layout_kwargs=dict(k=10000),
    #   edge_width=1,
      edge_cmap=sns.dark_palette("black", as_cmap=True),
      node_labels=True, arrows=True
)

plt.savefig(f"{fig_dir}/sim_c{nCluster}_community_association.pdf",format='PDF')
# %% visualization 2
# # Set up the plot
# pos = nx.spring_layout(G)
# fig, ax = plt.subplots(figsize=(10, 10))

# # Draw the nodes with color based on community

# color_map = [node_communities.get(node, 0) for node in G.nodes()]
# node_colors = [color_map[node] for node in G.nodes()]

# pos = nx.circular_layout(G)
# nx.draw_networkx_nodes(G, pos = pos, node_color=node_colors, cmap=plt.cm.tab20, node_size=500)

# # Draw the edges with weight labels
# edge_labels = {(edge[0], edge[1]): round(weight, 2) for edge, weight in path_probs.items()}
# edge_values = np.array(list(edge_labels.values()))
# alpha = edge_values/edge_values.min()-1
# alpha = alpha/alpha.max()
# nx.draw_networkx_edges(G, pos=pos, alpha=alpha,
#                        connectionstyle='arc3, rad=0.1',)
# # nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, 
# #                              label_pos=0.3, font_size=10, alpha=edge_values/edge_values.max(),
# #                              )
# nx.draw_networkx_labels(
#     G,
#     pos=pos,
#     font_color='black',
# )
# # Set the axis limits and labels
# # ax.set_xlim([-1.1, 1.1])
# # ax.set_ylim([-1.1, 1.1])
# ax.set_axis_off()

# plt.savefig(f"{fig_dir}/sim_c{nCluster}_community_association.pdf",format='PDF')

# %%
############### simulation below ################
class Node:
    def __init__(self, env, node_id, neighbors, weights):
        self.env = env
        self.node_id = node_id
        self.neighbors = neighbors
        self.weights = weights
        self.current_neighbor = None

    # def start(self):
    #     self.current_neighbor = self.choose_target()
    #     while True:
    #         yield self.env.timeout(1)
    #         self.current_neighbor = self.choose_target()

    def choose_target(self):
        neighbors = list(self.neighbors)
        weights = list(self.weights)
        return random.choices(neighbors, weights=weights)[0]

    def run_simulation(self, start_node, sim_time):
        self.current_neighbor = start_node
        simfish_history = [(0, self.current_neighbor)]
        for i in range(sim_time):
            yield self.env.timeout(1)
            self.current_neighbor = self.choose_target()
            simfish_history.append((i+1, self.current_neighbor))
        return simfish_history

class Network:
    def __init__(self, env, edges, source_col, target_col, weight_col):
        source = source_col
        target = target_col
        weight = weight_col
        self.env = env
        self.nodes = {}
        nodes = set(edges[source]).union(set(edges[target]))
        for node_id in nodes:
            neighbors = list(edges[edges[source] == node_id][target])
            weights = list(edges[edges[source] == node_id][weight])
            node = Node(self.env, node_id, neighbors, weights)
            self.nodes[node_id] = node

    # def start(self):
    #     for node_id in self.nodes:
    #         self.env.process(self.nodes[node_id].start())
            
    def run_simfish(self, start_node, sim_time):
        simfish_history = []
        node = self.nodes[start_node]
        node.current_neighbor = start_node
        simfish_history.append(node.current_neighbor)
        for i in range(sim_time):
            self.env.run(until=self.env.now + 1)
            node.current_neighbor = node.choose_target()
            simfish_history.append(node.current_neighbor)
        return simfish_history

    def check_nodes(self):
        node_info = pd.DataFrame(data={
            'node_id': self.nodes.keys(),
            'neighbors': [node.neighbors for node in self.nodes.values()],
            'weights': [node.weights for node in self.nodes.values()]
        })
        return node_info


# %%
total_bout_grouped_byCluster = all_feature_clustered.groupby('cluster')
total_bout_appearance = total_bout_grouped_byCluster.size()

x_chg_values = total_bout_grouped_byCluster['x_chg']
y_chg_values = total_bout_grouped_byCluster['ydispl_swim']
x_chg_dist = [s.NormalDist(mu=x_chg_values.mean()[cluster], sigma=x_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]
y_chg_dist = [s.NormalDist(mu=y_chg_values.mean()[cluster], sigma=y_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]

IBI_values = total_bout_grouped_byCluster['to_IBI']
IBI_dist = [s.NormalDist(mu=IBI_values.mean()[cluster], sigma=IBI_values.std()[cluster]) for cluster in np.arange(nCluster)]

IBI_y_vel_values = IBI_angles['propBoutIEI_yvel']
IBI_y_vel_dist = s.NormalDist(mu=IBI_y_vel_values.mean(), sigma=IBI_y_vel_values.std())

# %% initiate
env = simpy.Environment()
network = Network(env, graph_df_reconstruct, 'source', 'target', 'weight')
# network.check_nodes()

# %% simulate
sim_time = 100
start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
simfish_history = network.run_simfish(start_node, sim_time)

# % plot trajectory
sim_x_chg = []
sim_y_chg = []
sim_ibi_chg = []
for sim_cluster in simfish_history:
    sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
    sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
    sim_ibi_chg.append(
        IBI_dist[sim_cluster].samples(1)[0] * IBI_y_vel_dist.samples(1)[0]
    )

sim_df = pd.DataFrame(data = {
    'x_loc' : np.cumsum(sim_x_chg),
    'y_loc' : np.cumsum(sim_y_chg) + np.cumsum(sim_ibi_chg),
})

g = sns.lineplot(data=sim_df, x='x_loc', y='y_loc', sort=False)
g.set_aspect('equal', 'box')
# %% simulate multiple times and check y displ

sim_total_chg = pd.DataFrame()
run_times = 1000

for i in range(run_times):
    sim_time = 100
    start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
    simfish_history = network.run_simfish(start_node, sim_time)

    # % plot trajectory
    sim_x_chg = []
    sim_y_chg = []
    sim_ibi_chg = []
    for sim_cluster in simfish_history:
        sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
        sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
        sim_ibi_chg.append(
            IBI_dist[sim_cluster].samples(1)[0] * IBI_y_vel_dist.samples(1)[0]
        )

    sim_df = pd.DataFrame(data = {
        'x_loc' : np.cumsum(sim_x_chg),
        'y_loc' : np.cumsum(sim_y_chg) + np.cumsum(sim_ibi_chg),
    })
    sim_total_chg = pd.concat([sim_total_chg, sim_df.tail(1)], ignore_index = True)

# %%
plt.figure()
sns.kdeplot(sim_total_chg['x_loc'])
plt.savefig(f"{fig_dir}/sim_c{nCluster}_xLoc_sim{sim_time}_run{run_times}.pdf",format='PDF')


plt.figure()
sns.kdeplot(sim_total_chg['y_loc'])
plt.savefig(f"{fig_dir}/sim_c{nCluster}_yLoc_sim{sim_time}_run{run_times}.pdf",format='PDF')

# %%
