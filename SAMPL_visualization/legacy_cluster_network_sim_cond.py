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
sim_final_cond = pd.DataFrame()
sim_process_cond = pd.DataFrame()

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


    ############### simulation below ###############

    # %
    sel_cond_all_features = all_feature_clustered
    if use_condition_par:
        sel_cond_all_features = all_feature_clustered.loc[all_feature_clustered[compare_which]==sel_condition]
        sel_IBI_angles = IBI_angles.loc[IBI_angles[compare_which]==sel_condition]
    else:
        sel_cond_all_features = all_feature_clustered
        sel_IBI_angles = IBI_angles

    total_bout_grouped_byCluster = sel_cond_all_features.groupby('cluster')

    x_chg_values = total_bout_grouped_byCluster['x_chg']
    y_chg_values = total_bout_grouped_byCluster['ydispl_swim']
    x_chg_dist = [s.NormalDist(mu=x_chg_values.mean()[cluster], sigma=x_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]
    y_chg_dist = [s.NormalDist(mu=y_chg_values.mean()[cluster], sigma=y_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]

    IBI_values = total_bout_grouped_byCluster['post_IBI']
    IBI_dist = [s.NormalDist(mu=IBI_values.mean()[cluster], sigma=IBI_values.std()[cluster]) for cluster in np.arange(nCluster)]

    IBI_y_vel_values = sel_IBI_angles.loc[:,'propBoutIEI_yvel']
    IBI_y_vel_dist = s.NormalDist(mu=IBI_y_vel_values.mean(), sigma=IBI_y_vel_values.std())

    total_bout_appearance = all_feature_clustered.loc[all_feature_clustered[compare_which]==sel_condition].groupby('cluster').size()

    # % initiate
    env = simpy.Environment()
    network = simfish.Network(env, graph_df_reconstruct, 'source', 'target', 'weight')
    # network.check_nodes()

    # # % simulate
    # sim_time = 100
    # start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
    # simfish_history = network.run_simfish(start_node, sim_time)

    # # % plot trajectory
    # sim_x_chg = []
    # sim_y_chg = []
    # sim_ibi_chg = []
    # for sim_cluster in simfish_history:
    #     sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
    #     sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
    #     sim_ibi_chg.append(
    #         IBI_dist[sim_cluster].samples(1)[0] * IBI_y_vel_dist.samples(1)[0]
    #     )

    # sim_df = pd.DataFrame(data = {
    #     'x_loc' : np.cumsum(sim_x_chg),
    #     'y_loc' : np.cumsum(sim_y_chg) + np.cumsum(sim_ibi_chg),
    # })

    # g = sns.lineplot(data=sim_df, x='x_loc', y='y_loc', sort=False)
    # g.set_aspect('equal', 'box')
    # % simulate multiple times and check y displ

    sim_final = pd.DataFrame()
    sim_process = pd.DataFrame()
    
    run_times = 5000
    
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

        this_sim_final = pd.DataFrame(data = {
            'x_loc' : np.cumsum(sim_x_chg),
            'y_loc' : np.cumsum(sim_y_chg) + np.cumsum(sim_ibi_chg),
            'run_num' : np.repeat(i,sim_time+1)
        })
        
        this_sim_process = pd.DataFrame(data = {
            'swim_x_loc' : sim_x_chg,
            'swim_y_loc' : sim_y_chg,
            'IBI_y_loc' : sim_ibi_chg,
            'inst_swim_traj' : np.arctan2(sim_y_chg, sim_x_chg) * 180 / np.pi,
            'run_num' : np.repeat(i,sim_time+1)
        })
                
        sim_final = pd.concat([sim_final, this_sim_final.tail(1)], ignore_index = True)
        sim_process = pd.concat([sim_process, this_sim_process], ignore_index = True)

    sim_final_cond = pd.concat([sim_final_cond, sim_final.assign(cond1=sel_condition)], ignore_index=True)
    sim_process_cond = pd.concat([sim_process_cond, sim_process.assign(cond1=sel_condition)], ignore_index=True)

plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='y_loc',
            hue='cond1',
            hue_order=np.sort(connected_bout_features[compare_which].unique()),
            common_norm=False)
plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_yLoc_distribution_condPar-{use_condition_par}.pdf",format='PDF')

plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='x_loc',
            hue='cond1',
            hue_order=np.sort(connected_bout_features[compare_which].unique()),
            common_norm=False)
plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_xLoc_distribution_condPar-{use_condition_par}.pdf",format='PDF')

sim_final_cond = sim_final_cond.assign(
    total_traj = np.arctan2(sim_final_cond['y_loc'], sim_final_cond['x_loc']) * 180 / np.pi
)
plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='total_traj',
            hue='cond1',
            hue_order=np.sort(connected_bout_features[compare_which].unique()),
            common_norm=False)

plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_angle_distribution_condPar-{use_condition_par}.pdf",format='PDF')

# %%
std_traj = sim_process_cond.groupby(['cond1','run_num']).std().reset_index()
sns.boxplot(data=std_traj, x='cond1', y='inst_swim_traj',)
sns.pointplot(data=std_traj, x='cond1', y='inst_swim_traj',)

# %%
all_graph = pd.DataFrame()

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
    

    ############### simulation below ###############

    # %
    sel_cond_all_features = all_feature_clustered
    if use_condition_par:
        sel_cond_all_features = all_feature_clustered.loc[all_feature_clustered[compare_which]==sel_condition]
        sel_IBI_angles = IBI_angles.loc[IBI_angles[compare_which]==sel_condition]
    else:
        sel_cond_all_features = all_feature_clustered
        sel_IBI_angles = IBI_angles

    total_bout_grouped_byCluster = sel_cond_all_features.groupby('cluster')

    x_chg_values = total_bout_grouped_byCluster['x_chg']
    y_chg_values = total_bout_grouped_byCluster['ydispl_swim']
    x_chg_dist = [s.NormalDist(mu=x_chg_values.mean()[cluster], sigma=x_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]
    y_chg_dist = [s.NormalDist(mu=y_chg_values.mean()[cluster], sigma=y_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]

    IBI_values = total_bout_grouped_byCluster['post_IBI']
    IBI_dist = [s.NormalDist(mu=IBI_values.mean()[cluster], sigma=IBI_values.std()[cluster]) for cluster in np.arange(nCluster)]

    IBI_y_vel_values = sel_IBI_angles.loc[:,'propBoutIEI_yvel']
    IBI_y_vel_dist = s.NormalDist(mu=IBI_y_vel_values.mean(), sigma=IBI_y_vel_values.std())

    total_bout_appearance = all_feature_clustered.loc[all_feature_clustered[compare_which]==sel_condition].groupby('cluster').size()

    # % initiate
    env = simpy.Environment()
    network = simfish.Network(env, graph_df_reconstruct, 'source', 'target', 'weight')
    plt.figure(figsize=(6,6))
    for i in range(100):
        sim_time = 100
        start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
        simfish_history = network.run_simfish(start_node, sim_time)

        # % plot trajectory
        sim_x_chg = []
        sim_y_chg = []
        sim_ibi_chg = []
        sim_ibi_dur = []
        for sim_cluster in simfish_history:
            sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
            sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
            sim_ibi_chg.append(
                IBI_dist[sim_cluster].samples(1)[0] * IBI_y_vel_dist.samples(1)[0]
            )
            sim_ibi_dur.append(
                IBI_dist[sim_cluster].samples(1)[0]
            )

        sim_df = pd.DataFrame(data = {
            'x_loc' : np.cumsum(sim_x_chg),
            'y_loc' : np.cumsum(sim_y_chg) + np.cumsum(sim_ibi_chg),
        })
        
        g = sns.lineplot(data=sim_df, x='x_loc', y='y_loc', sort=False, alpha=0.2)
    g.set_aspect('equal', 'box')
    total_time = np.cumsum(sim_ibi_dur) + np.arange(sim_time+1)/FRAME_RATE
    print(f"Condition {sel_condition} total time simulated: {total_time[-1]/60} min")
