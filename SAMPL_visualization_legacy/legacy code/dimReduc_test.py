'''
using openTSNE
'''

#%%
import sys
import os,glob
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib

from openTSNE import TSNE
from examples import utils
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from sklearn.manifold import TSNE

# from collections import defaultdict
# from datetime import datetime
# from datetime import timedelta
# import math
# import scipy

# %%
# Paste root directory here
root = "/Volumes/LabData/SAMPL_data_in_use/resliced/combined_7DD_data_resliced"
fig_dir = "/Users/yunluzhu/Documents/Lab2/Data/SAMPL_ana/Figures/NT_tmp_plots_PCA_Raw"

global DayNight_select 
DayNight_select = 'all' # day or night or all

# %%
# CONSTANTS

ALIGNED_BOUT_RANGE = range(0,51)


# HEADING_LIM = 90
# HEADING_LIM = 180
# FIN_BODY_LOW_LIM = -10  # lower pitch change limit to plot fin_body sigmoid fit and binned data
# FIN_BODY_UP_LIM = 15  # 
# X_RANGE = np.arange(-10,15.01,0.01)
# BIN_WIDTH = 0.8  
# AVERAGE_BIN = np.arange(-10,15,BIN_WIDTH)

# %%
def defaultPlotting(): 
    sns.set(rc={"xtick.labelsize":'large',"ytick.labelsize":'large', "axes.labelsize":'x-large'},style="ticks")

def day_night_split(df,time_col_name):
    hour = df[time_col_name].dt.strftime('%H').astype('int')
    if DayNight_select == 'day':
        df_out = df.loc[hour[(hour>9) & (hour<23)].index, :]
    elif DayNight_select == 'night':
        df_out = df.loc[hour[(hour<9) | (hour>23)].index, :]
    elif DayNight_select == 'all':
        df_out = df
    return df_out

# def distribution_binned_average(df, condition):
#     '''
#     bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
#     '''
#     df = df.sort_values(by='pre_posture_chg')
#     bins = pd.cut(df['pre_posture_chg'], list(AVERAGE_BIN))
#     grp = df.groupby(bins)
#     df_out = grp[['pre_posture_chg','atk_ang']].mean().assign(dpf=condition[0],cond1=condition[4:])
#     return df_out
# %%
# get data 
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)
        
# jackknifed_coef = pd.DataFrame()  # coef results calculated with jackknifed pitch data
# jackknifed_y = pd.DataFrame()  # fitted y using jackknifed pitch data

all_cond_bouts = pd.DataFrame()
mean_data_cond = pd.DataFrame()

hue_order = list()

# binned_atk_angles = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_bouts_data = pd.DataFrame()
            # mean_data = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # - attack angle calculation
                exp_path = os.path.join(subpath, exp)
                angles = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned').loc[:,[
                    'propBoutAligned_heading',
                    'propBoutAligned_pitch',
                    'propBoutAligned_speed',
                    'propBoutAligned_time']]
                
                # IEIdata = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2').loc[:,[
                #     'boutNum',
                #     'epochNum',
                #     'propBoutIEI',]].set_index(['boutNum', 'epochNum'])
                # boutNum = pd.read_hdf(f"{exp_path}/bout_data.h5", key='bout_attributes').loc[:,[
                #     'boutNum',
                #     'epochNum',
                #     'if_align']]
                                
                # for_index = boutNum.set_index(['boutNum', 'epochNum'])                
                # IEIdata.loc[for_index.if_align==True,:]
                # boutNum = boutNum.loc[boutNum.if_align==True].reset_index()

                angles = angles.assign(idx=round_half_up(len(angles)/51)*list(range(0,51)))          
                angles_grp = angles.set_index('idx').groupby(np.arange(len(angles))//51)
                
                # pick the middle 0.5 sec, 21 frames
                re_format = angles_grp.apply(
                    lambda x: pd.concat([x.iloc[ALIGNED_BOUT_RANGE,0].rename(lambda i: 'heading_'+str(i)),
                                        x.iloc[ALIGNED_BOUT_RANGE,1].rename(lambda i: 'pitch_'+str(i)),
                                        x.iloc[ALIGNED_BOUT_RANGE,2].rename(lambda i: 'speed_'+str(i)),
                                        ])
                )
                
                re_format = pd.concat([re_format, 
                                       angles_grp.propBoutAligned_time.head(1).reset_index().propBoutAligned_time,
                                    #    boutNum[['boutNum','epochNum']]
                                       ],axis=1)
                
                # re_format_IEI = pd.concat([re_format.set_index(['boutNum', 'epochNum']),
                #                            IEIdata,
                #                           ],axis=1,join='inner').reset_index()
                             
                re_format_day = day_night_split(re_format,'propBoutAligned_time')
                all_bouts_data = pd.concat([all_bouts_data, re_format_day])
                
            all_cond_bouts = pd.concat([all_cond_bouts,all_bouts_data.assign(cond1=all_conditions[condition_idx])])
                                       
all_cond_bouts = all_cond_bouts.dropna().reset_index(drop=True)
data_to_ana = all_cond_bouts.drop(['cond1','propBoutAligned_time'],axis=1)
df_std = StandardScaler().fit_transform(data_to_ana)

# %%
# Create a PCA instance: pca
pca = PCA(n_components=30)
principalComponents = pca.fit_transform(df_std)

# Plot the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

# Save components to a DataFrame
PCA_components = pd.DataFrame(principalComponents)
# %% visualize clustering
plt.scatter(PCA_components[0], PCA_components[1], alpha=.002, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# %% Fiind the clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(PCA_components.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# %%
res = pd.DataFrame()
pca = PCA(n_components=2)
pca_result = pca.fit_transform(df_std)
all_cond_bouts = all_cond_bouts.assign(
    pca1 = pca_result[:,0],
    pca2 = pca_result[:,1],
)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # plot 2pc
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca1", y="pca2",
#     hue='cond1',
#     # palette=sns.color_palette("hls", 2),
#     data=all_cond_bouts,
#     legend="full",
#     alpha=0.1
# )

# %% reduce dimension for t-sne
pca_for_tsne = PCA(n_components=20)
pca_result_for_tsne = pca_for_tsne.fit_transform(df_std)
print(f"Cumulative explained variation for {pca_result_for_tsne.shape[1]} principal components: {np.sum(pca_for_tsne.explained_variance_ratio_)}")

# %% t-sne with reduced data
time_start = time.time()
    # parameters for 7dd tau Day with 20 PCA
# tsne = TSNE(n_components=2, verbose=1, perplexity=50,n_iter=1000,early_exaggeration=15,learning_rate=max(30000/15/4,200))
    # parameters for 7dd tau Night with 20 PCA
# tsne = TSNE(n_components=2, verbose=1, perplexity=50,n_iter=1000,early_exaggeration=12,learning_rate=max(10000/12/4,200))
    # test
tsne = TSNE(n_components=2, verbose=1, perplexity=50,n_iter=1000,early_exaggeration=12,learning_rate=max(30000/12/4,200))

tsne_pca_results = tsne.fit_transform(pca_result_for_tsne)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# %% Get clusters
# https://www.reneshbedre.com/blog/tsne.html

from bioinfokit.visuz import cluster
from sklearn.cluster import DBSCAN

cluster.tsneplot(score=tsne_pca_results)

# recognize the clusters using the DBSCAN algorithm

# here eps parameter is very important and optimizing eps is essential
# for well defined clusters. I have run DBSCAN with several eps values
    # parameters for 7dd tau Day
# get_clusters = DBSCAN(eps=2, min_samples=20).fit_predict(tsne_pca_results)
    # parameters for 7dd tau Night
# get_clusters = DBSCAN(eps=2.5, min_samples=20).fit_predict(tsne_pca_results)
    # parameters for 7dd tau all
# get_clusters = DBSCAN(eps=1.8, min_samples=20).fit_predict(tsne_pca_results)
    # test
get_clusters = DBSCAN(eps=1.3, min_samples=20).fit_predict(tsne_pca_results)

# get_clusters = DBSCAN(eps=2.5, min_samples=10).fit_predict(tsne_pca_results)

# check unique clusters
# -1 value represents noisy points could not assigned to any cluster

# # # get t-SNE plot with colors assigned to each cluster
# cluster.tsneplot(score=tsne_pca_results, colorlist=get_clusters, 
#     # colordot=('#92AFD7', '#edc79b', '#edc79b', '#d57a66', '#395B50', '#000000'), 
#     legendpos='upper right', legendanchor=(1.15, 1))



# %% tidy data
res_toplt = pd.DataFrame(data = tsne_pca_results,
                         columns=['TSNE1', 'TSNE2'])
res_toplt = res_toplt.assign(clusters = get_clusters,
                             time = all_cond_bouts['propBoutAligned_time'],
                             cond1 = all_cond_bouts['cond1'])

#  mark day night 
hours = res_toplt['time'].dt.strftime('%H').astype('int')
day_night = []
res_toplt = res_toplt.assign(hours = hours)
res_toplt.loc[hours[(hours>=9) & (hours<23)].index, 'day_night'] = 'day'
res_toplt.loc[hours[(hours<9) | (hours>=23)].index, 'day_night'] = 'night'

res_ctrl = res_toplt.loc[all_cond_bouts.loc[all_cond_bouts.cond1==all_conditions[0]].index,:]
res_cond = res_toplt.loc[all_cond_bouts.loc[all_cond_bouts.cond1==all_conditions[1]].index,:]

cluster_color = sns.color_palette("hls", len(set(get_clusters)))
total_clusters = list(set(res_toplt.clusters))
ctrl_cluster = list(set(res_ctrl.clusters))
cond_cluster = list(set(res_cond.clusters))

total_clusters.sort()
ctrl_cluster.sort()
cond_cluster.sort()

# %% visualize

cluster_color = pd.DataFrame(data = sns.color_palette("hls", len(set(get_clusters))),
                             index = total_clusters
                             )

ctrl_palette = list(cluster_color.loc[ctrl_cluster,:].itertuples(index=False,name=None))
cond_palette = list(cluster_color.loc[cond_cluster,:].itertuples(index=False,name=None))


figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(16,7))

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    palette = sns.color_palette(ctrl_palette),
    data=res_ctrl,
    hue = 'clusters',
    legend=False,
    alpha=0.03,
    ax=ax1
)
ax1.set_title(f"{all_conditions[0]}")

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    palette = sns.color_palette(cond_palette),
    data=res_cond,
    hue = 'clusters',
    legend='brief',
    alpha=0.05,
    ax=ax2
)
ax2.set_title(f"{all_conditions[1]}")
figure.savefig(fig_dir+"/TSNE_newfig.pdf",format='PDF')

# %% cluster size
all_size = pd.DataFrame()
for cluster_num in set(get_clusters):
    current_cluster = all_cond_bouts.loc[res_toplt.clusters==cluster_num]
    cluster_size = current_cluster.groupby('cond1').size()
    all_size = pd.concat([all_size, cluster_size],axis=1)
    all_size.columns.values[-1:]=[cluster_num]
    
total = all_cond_bouts.groupby('cond1').size()
all_size.iloc[0,:] = all_size.iloc[0,:] /total.values[0]
all_size.iloc[1,:] = all_size.iloc[1,:] /total.values[1]

all_size = all_size.T

all_size = all_size.assign(
    dir = ((all_size['7dd_1Sibs']-all_size['7dd_2Tau'])/(all_size['7dd_1Sibs']+all_size['7dd_2Tau']))
    )
all_size

# %%
figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(16,7))

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    data=res_ctrl,
    hue = 'day_night',
    legend=False,
    alpha=0.1,
    ax=ax1
)
ax1.set_title(f"{all_conditions[0]}")

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    data=res_cond,
    hue = 'day_night',
    legend='brief',
    alpha=0.1,
    ax=ax2
)
ax2.set_title(f"{all_conditions[1]}")
figure.savefig(fig_dir+"/TSNE_newfig.pdf",format='PDF')

