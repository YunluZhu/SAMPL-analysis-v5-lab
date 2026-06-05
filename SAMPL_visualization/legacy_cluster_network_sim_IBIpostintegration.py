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
pick_data = 'otog'
which_ztime = 'day'
compare_which = 'cond1' # condition for separation None for treat as whole
nCluster = 16
sort_by_feature = 'pitch_initial' # by which parameter to sort the clusters on the figure

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
all_around_peak_data, all_feature_cond, all_cond0, all_cond1, idxRANGE = get_aligned_bouts_wIBI(root, FRAME_RATE, ztime=which_ztime)
# IBI_angles, cond0, cond1 = get_IBIangles(root, FRAME_RATE, ztime=which_ztime)

# %%
all_feature_clustered = get_cluster_phaseSpace(all_around_peak_data, all_feature_cond, idxRANGE, nCluster)
# %%

all_feature_cond = all_feature_cond.assign(
    bout_uid_adj = all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + '__' + all_feature_cond['bout_uid'],
    to_bout_uid_adj = all_feature_cond['cond1'] + all_feature_cond['expNum'].astype(str) + '__' + all_feature_cond['to_bout'],
    cluster = all_feature_clustered['cluster']
)

# %%

to_cluster = all_feature_cond[['to_bout_uid_adj']].merge(all_feature_cond[['cluster','bout_uid_adj']], left_on='to_bout_uid_adj',right_on='bout_uid_adj',how='left')
to_cluster.columns = ['to_bout_ori', 'to_cluster', 'to_bout_uid_merged']
df_tomodel = all_feature_cond.assign(
    to_cluster = to_cluster['to_cluster'],
)
df_tomodel = df_tomodel.assign(
    y_IBI_displ = np.append(df_tomodel['y_initial'][1:].values - df_tomodel['y_end'][:-1].values, np.nan),
    y_swimIBI_displ = np.append(df_tomodel['y_pre_swim'][1:].values - df_tomodel['y_post_swim'][:-1].values, np.nan),
    x_swimIBI_displ = np.absolute(np.append(df_tomodel['x_pre_swim'][1:].values - df_tomodel['x_post_swim'][:-1].values, np.nan)),
    xdispl_swim = df_tomodel['xdispl_swim'].abs()
)
df_tomodel = df_tomodel[df_tomodel['to_cluster'].notna()]
df_tomodel.reset_index(drop=True, inplace=True)
df_tomodel = df_tomodel.assign(
    to_cluster = df_tomodel['to_cluster'].astype('int'),
)

# %% check graph
if bool(compare_which):
    for this_condition in df_tomodel[compare_which].unique():
        this_cond_features = df_tomodel.loc[df_tomodel[compare_which]==this_condition]
        # plot_network_graphs(extracted_features=this_cond_features, cond_sep=this_condition, sort_by_feature='ydispl_swim', total_features=connected_bout_features)
        extracted_features=this_cond_features
        cond_sep=this_condition
        total_features=df_tomodel
        print(this_condition)
        plt_network_graphs(df_tomodel, 
                            fig_dir = fig_dir,
                            sort_by_feature = sort_by_feature,
                            cond_sep=this_condition, 
                            extracted_features=this_cond_features)
else:
    plt_network_graphs(df_tomodel, 
                        fig_dir = fig_dir,
                        sort_by_feature = sort_by_feature)

# %%  simulate 5000 times for statistics

run_times = 5000

# %%  simulate 100 times for trajectories
# NOTE IBI change is calculated as a separate translocation from the actual bouts


run_times = 100
sim_final_cond = pd.DataFrame()
sim_process_cond = pd.DataFrame()

use_condition_par = True

for sel_condition in df_tomodel[compare_which].unique():
    this_cond_tomodel = df_tomodel.loc[df_tomodel[compare_which]==sel_condition,['cluster','to_cluster','expNum','cond0','cond1']]
    connected_bout_appearance = this_cond_tomodel.groupby('cluster').size()
    graph_df = this_cond_tomodel.groupby(['cluster','to_cluster']).size().reset_index()
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
    # get parameters for estimation
    # %
    sel_cond_all_features = df_tomodel
    if use_condition_par:
        sel_cond_all_features = df_tomodel.loc[df_tomodel[compare_which]==sel_condition]
    else:
        sel_cond_all_features = df_tomodel

    total_bout_grouped_byCluster = sel_cond_all_features.groupby('cluster')

    x_chg_values = total_bout_grouped_byCluster['xdispl_swim']
    y_chg_values = total_bout_grouped_byCluster['ydispl_swim']
    x_chg_dist = [s.NormalDist(mu=x_chg_values.mean()[cluster], sigma=x_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]
    y_chg_dist = [s.NormalDist(mu=y_chg_values.mean()[cluster], sigma=y_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]

    # estimate IBI drifting
    sel_cond_all_features = sel_cond_all_features.assign(
        post_IBI_cluster_id = sel_cond_all_features['cluster'].astype(str) + '_' + sel_cond_all_features['to_cluster'].astype(str)
    )
    
    all_postIBI_id = sel_cond_all_features['post_IBI_cluster_id'].unique()
    
    postIBI_distribution = pd.DataFrame(all_postIBI_id, columns=['post_IBI_id'])
    
    total_bout_grouped_byPostIBI = sel_cond_all_features.groupby('post_IBI_cluster_id')
    
    for IBI_feature in [
        'post_IBI', 
        # 'y_IBI_displ', 
        'y_swimIBI_displ',
        'x_swimIBI_displ'
        ]:
        values = total_bout_grouped_byPostIBI[IBI_feature]
        distribution = [s.NormalDist(mu=values.mean()[IBI_id], sigma=values.std()[IBI_id]) for IBI_id in all_postIBI_id]
        postIBI_distribution[IBI_feature] = distribution

    postIBI_distribution[['from_cluster', 'to_cluster']] = postIBI_distribution['post_IBI_id'].str.split('_', 1, expand=True)
    ####
    
    total_bout_appearance = df_tomodel.loc[df_tomodel[compare_which]==sel_condition].groupby('cluster').size()

    # % initiate simulation
    env = simpy.Environment()
    network = simfish.Network(env, graph_df_reconstruct, 'source', 'target', 'weight')
    # network.check_nodes()

    
    # % simulate multiple times and check y displ

    sim_final = pd.DataFrame()
    sim_process = pd.DataFrame()
    
    for i in range(run_times):
        sim_time = 100
        start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
        simfish_history = network.run_simfish(start_node, sim_time)

    
        failed_IBI_counter = 0
        # % plot trajectory
        sim_x_chg = []
        sim_y_chg = []
        sim_ibi_chg = []
        for (sim_cluster, next_cluster) in zip(simfish_history, np.append(simfish_history[1:], None)):
            sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
            sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
            if next_cluster: # for IBI y displacement. 
                this_IBI_id = str(sim_cluster) + '_' + str(int(next_cluster))
                try:
                    sim_y_chg.append(postIBI_distribution.loc[postIBI_distribution['post_IBI_id']==this_IBI_id,'y_swimIBI_displ'].values[0].samples(1)[0])
                    sim_x_chg.append(np.absolute(postIBI_distribution.loc[postIBI_distribution['post_IBI_id']==this_IBI_id,'x_swimIBI_displ'].values[0].samples(1)[0]))
                except:
                    sim_x_chg.append(0)
                    sim_y_chg.append(0)
                    failed_IBI_counter+=1
            else:
                sim_x_chg.append(0)
                sim_y_chg.append(0)
                
        this_sim_final = pd.DataFrame(data = {
            'x_loc' : np.cumsum(sim_x_chg),
            'y_loc' : np.cumsum(sim_y_chg),
            'run_num' : np.repeat(i,sim_time*2+2)
        })
        
        this_sim_process = pd.DataFrame(data = {
            'swim_x_loc' : sim_x_chg,
            'swim_y_loc' : sim_y_chg,
            # 'IBI_y_loc' : sim_ibi_chg,
            'cumsum_x_loc' : np.cumsum(sim_x_chg),
            'cumsum_y_loc' : np.cumsum(sim_y_chg),
            'run_num' : np.repeat(i,sim_time*2+2)
        })
                
        sim_final = pd.concat([sim_final, this_sim_final.tail(1)], ignore_index = True)
        sim_process = pd.concat([sim_process, this_sim_process], ignore_index = True)

    sim_final_cond = pd.concat([sim_final_cond, sim_final.assign(cond1=sel_condition)], ignore_index=True)
    sim_process_cond = pd.concat([sim_process_cond, sim_process.assign(cond1=sel_condition)], ignore_index=True)

# %%
plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='y_loc',
            hue='cond1',
            hue_order=np.sort(df_tomodel[compare_which].unique()),
            common_norm=False)
plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_yLoc_distribution_condPar-{use_condition_par}.pdf",format='PDF')

plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='x_loc',
            hue='cond1',
            hue_order=np.sort(df_tomodel[compare_which].unique()),
            common_norm=False)
plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_xLoc_distribution_condPar-{use_condition_par}.pdf",format='PDF')

sim_final_cond = sim_final_cond.assign(
    total_traj = np.arctan2(sim_final_cond['y_loc'], sim_final_cond['x_loc']) * 180 / np.pi
)
plt.figure(figsize=(4,4))
sns.kdeplot(data=sim_final_cond,
            x='total_traj',
            hue='cond1',
            hue_order=np.sort(df_tomodel[compare_which].unique()),
            common_norm=False)

plt.savefig(f"{fig_dir}/sim_c{nCluster}_{sim_time}boutsX{run_times}_angle_distribution_condPar-{use_condition_par}.pdf",format='PDF')

# # %%
# std_traj = sim_process_cond.groupby(['cond1','run_num']).std().reset_index()
# sns.boxplot(data=std_traj, x='cond1', y='inst_swim_traj',)
# plt.show()
# sns.pointplot(data=std_traj, x='cond1', y='inst_swim_traj',)
# plt.show()

# %%  simulate 100 times for trajectories
# NOTE IBI change is calculated as a separate translocation from the actual bouts


run_times = 100
sim_final_cond = pd.DataFrame()
sim_process_cond = pd.DataFrame()

use_condition_par = True

for sel_condition in df_tomodel[compare_which].unique():
    this_cond_tomodel = df_tomodel.loc[df_tomodel[compare_which]==sel_condition,['cluster','to_cluster','expNum','cond0','cond1']]
    connected_bout_appearance = this_cond_tomodel.groupby('cluster').size()
    graph_df = this_cond_tomodel.groupby(['cluster','to_cluster']).size().reset_index()
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
    # get parameters for estimation
    # %
    sel_cond_all_features = df_tomodel
    if use_condition_par:
        sel_cond_all_features = df_tomodel.loc[df_tomodel[compare_which]==sel_condition]
    else:
        sel_cond_all_features = df_tomodel

    total_bout_grouped_byCluster = sel_cond_all_features.groupby('cluster')

    x_chg_values = total_bout_grouped_byCluster['xdispl_swim']
    y_chg_values = total_bout_grouped_byCluster['ydispl_swim']
    x_chg_dist = [s.NormalDist(mu=x_chg_values.mean()[cluster], sigma=x_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]
    y_chg_dist = [s.NormalDist(mu=y_chg_values.mean()[cluster], sigma=y_chg_values.std()[cluster]) for cluster in np.arange(nCluster)]

    # estimate IBI drifting
    sel_cond_all_features = sel_cond_all_features.assign(
        post_IBI_cluster_id = sel_cond_all_features['cluster'].astype(str) + '_' + sel_cond_all_features['to_cluster'].astype(str)
    )
    
    all_postIBI_id = sel_cond_all_features['post_IBI_cluster_id'].unique()
    
    postIBI_distribution = pd.DataFrame(all_postIBI_id, columns=['post_IBI_id'])
    
    total_bout_grouped_byPostIBI = sel_cond_all_features.groupby('post_IBI_cluster_id')
    
    for IBI_feature in [
        'post_IBI', 
        # 'y_IBI_displ', 
        'y_swimIBI_displ',
        'x_swimIBI_displ'
        ]:
        values = total_bout_grouped_byPostIBI[IBI_feature]
        distribution = [s.NormalDist(mu=values.mean()[IBI_id], sigma=values.std()[IBI_id]) for IBI_id in all_postIBI_id]
        postIBI_distribution[IBI_feature] = distribution

    postIBI_distribution[['from_cluster', 'to_cluster']] = postIBI_distribution['post_IBI_id'].str.split('_', 1, expand=True)
    ####
    
    total_bout_appearance = df_tomodel.loc[df_tomodel[compare_which]==sel_condition].groupby('cluster').size()

    # % initiate simulation
    env = simpy.Environment()
    network = simfish.Network(env, graph_df_reconstruct, 'source', 'target', 'weight')
    # network.check_nodes()

    
    # % simulate multiple times and check y displ

    sim_final = pd.DataFrame()
    sim_process = pd.DataFrame()
    
    for i in range(run_times):
        sim_time = 100
        start_node = random.choices(np.arange(nCluster), weights=total_bout_appearance)[0]
        simfish_history = network.run_simfish(start_node, sim_time)

    
        failed_IBI_counter = 0
        # % plot trajectory
        sim_x_chg = []
        sim_y_chg = []
        sim_ibi_chg = []
        for (sim_cluster, next_cluster) in zip(simfish_history, np.append(simfish_history[1:], None)):
            sim_x_chg.append(x_chg_dist[sim_cluster].samples(1)[0])
            sim_y_chg.append(y_chg_dist[sim_cluster].samples(1)[0])
            if next_cluster: # for IBI y displacement. 
                this_IBI_id = str(sim_cluster) + '_' + str(int(next_cluster))
                try:
                    sim_y_chg.append(postIBI_distribution.loc[postIBI_distribution['post_IBI_id']==this_IBI_id,'y_swimIBI_displ'].values[0].samples(1)[0])
                    sim_x_chg.append(np.absolute(postIBI_distribution.loc[postIBI_distribution['post_IBI_id']==this_IBI_id,'x_swimIBI_displ'].values[0].samples(1)[0]))
                except:
                    sim_x_chg.append(0)
                    sim_y_chg.append(0)
                    failed_IBI_counter+=1
            else:
                sim_x_chg.append(0)
                sim_y_chg.append(0)
                
        this_sim_final = pd.DataFrame(data = {
            'x_loc' : np.cumsum(sim_x_chg),
            'y_loc' : np.cumsum(sim_y_chg),
            'run_num' : np.repeat(i,sim_time*2+2)
        })
        
        this_sim_process = pd.DataFrame(data = {
            'swim_x_loc' : sim_x_chg,
            'swim_y_loc' : sim_y_chg,
            # 'IBI_y_loc' : sim_ibi_chg,
            'cumsum_x_loc' : np.cumsum(sim_x_chg),
            'cumsum_y_loc' : np.cumsum(sim_y_chg),
            'run_num' : np.repeat(i,sim_time*2+2)
        })
                
        sim_final = pd.concat([sim_final, this_sim_final.tail(1)], ignore_index = True)
        sim_process = pd.concat([sim_process, this_sim_process], ignore_index = True)

    sim_final_cond = pd.concat([sim_final_cond, sim_final.assign(cond1=sel_condition)], ignore_index=True)
    sim_process_cond = pd.concat([sim_process_cond, sim_process.assign(cond1=sel_condition)], ignore_index=True)

# %
# plot
g = sns.lineplot(
                data=sim_process_cond, 
                x='cumsum_x_loc', y='cumsum_y_loc',
                hue='cond1', 
                units = 'run_num',
                estimator = None,
                alpha=0.2,
                sort=False,
                )
g.set_aspect('equal', 'box')

plt.savefig(f"{fig_dir}/sim_traj_c{nCluster}_{sim_time}boutsX{run_times} by{compare_which}.pdf",format='PDF')

# %%
