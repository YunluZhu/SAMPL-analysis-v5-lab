from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import get_bout_features
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,distribution_binned_average)
from plot_functions.plt_functions import plt_categorical_grid
from plot_functions.get_bout_kinetics import get_bout_kinetics
from statsmodels.stats.multicomp import MultiComparison
from sklearn import metrics
import scipy.stats as st
import time
from sklearn.manifold import TSNE


# %%
set_font_type()
# %%
# Select data and create figure folder
pick_data = 'wt_dl'
which_ztime = 'all'
spd_bins = np.arange(5,30,5)

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} feature PCA'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %% get features
all_feature_cond, all_cond1, all_cond2 = get_bout_features(root, FRAME_RATE, ztime=which_ztime)

# %%
# Create a PCA instance: pca
drop_col = ['bout_time', 'expNum', 'ztime', 'cond0', 'cond1']
data_to_ana = all_feature_cond.drop(drop_col,axis=1)
df_std = StandardScaler().fit_transform(data_to_ana)

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
# pca = PCA(n_components=3)
# pca_result = pca.fit_transform(df_std)

all_cond_bouts = all_feature_cond.assign(
    pca1 = principalComponents[:,0],
    pca2 = principalComponents[:,1],
    pca3 = principalComponents[:,2],
)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # plot 2pc
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca1", y="pca2",
    hue='cond1',
    # palette=sns.color_palette("hls", 2),
    data=all_cond_bouts,
    legend="full",
    alpha=0.01
)

# %% visualize
model = KMeans(n_clusters=3)
model.fit(PCA_components.iloc[:,:5])

all_cond_bouts = all_cond_bouts.assign(cluster = model.labels_)

# %% plot 3D scatter
fig = sns.pairplot(
    data = all_cond_bouts,
    vars = ['pca1','pca2','pca3'],
    palette=sns.color_palette("Set2", 3),
    kind='hist',
    # plot_kws=dict(size=1, alpha = 0.01, linewidth=0),
    hue = 'cluster',
    )
plt.savefig(f"{fig_dir}/PCA 3 KMeans clustered.pdf",format='PDF')
# %% see eigen vectors in the next section



# %%
################################################################################
# %%
# look at eigen vectors
# pca.explained_variance_ratio_
# pca.components_

# # of features wanted
# n_feature = 6

# # repeat with new n components
# model = PCA(n_components=5).fit(df_std)
# X_pc = model.transform(df_std)

# # number of components 
# n_pcs = model.components_.shape[0]
# # get the index of the most important feature on EACH component
# # LIST COMPREHENSION HERE
# L = np.argsort(-np.abs(model.components_), axis=1)
# initial_feature_names = list(data_to_ana.columns[0:-1])
# # get the names
# important_features = pd.DataFrame()
# for i in range(n_feature):
#     important_features = pd.concat(
#         [important_features,pd.Series([initial_feature_names[L[j,i]] for j in range(n_pcs)])],
#         axis = 1
#     )
# important_features.columns = ["feature{}".format(i+1) for i in range(n_feature)]
# print('top feature names')
# print(important_features)

# print('\n')

# # get the eigenness
# eigeness = pd.DataFrame()
# for i in range(n_pcs):
#     eigeness = pd.concat(
#         [eigeness,pd.Series([model.components_[i,L[i,j]] for j in range(n_feature)])],
#         axis = 1
#     )
# eigeness = eigeness.T
# eigeness.columns = ["feature{}".format(i+1) for i in range(n_feature)]
# eigeness.reset_index(inplace=True,drop=True)
# print('loadings')
# print(eigeness)
# %% TSNE below
# # %% reduce dimension for t-sne
# pca_for_tsne = PCA(n_components=10)
# pca_result_for_tsne = pca_for_tsne.fit_transform(df_std)
# print(f"Cumulative explained variation for {pca_result_for_tsne.shape[1]} principal components: {np.sum(pca_for_tsne.explained_variance_ratio_)}")

# # %% t-sne with reduced perp
# time_start = time.time()
#     # parameters for 7dd tau Day with 20 PCA
#     # parameters for 7dd tau Night with 20 PCA
#     # parameters for 7dd tau All with 20 PCA & 15-46 data points
# # tsne = TSNE(n_components=2, verbose=1, perplexity=400,n_iter=800,early_exaggeration=15,learning_rate=max(40000/12,200))
# # tsne = TSNE(n_components=2, verbose=1, perplexity=300,n_iter=800,early_exaggeration=15,learning_rate=max(30000/12,200))
# tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=800, early_exaggeration=15, learning_rate='auto')

# tsne_pca_results = tsne.fit_transform(pca_result_for_tsne)
# print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))


# # %% Get clusters
# # https://www.reneshbedre.com/blog/tsne.html

# from bioinfokit.visuz import cluster
# from sklearn.cluster import DBSCAN

# cluster.tsneplot(score=tsne_pca_results)

# # recognize the clusters using the DBSCAN algorithm

# # here eps parameter is very important and optimizing eps is essential
# # for well defined clusters. I have run DBSCAN with several eps values
#     # parameters for 7dd tau Day
# get_clusters = DBSCAN(eps=1.3, min_samples=20).fit_predict(tsne_pca_results)
#     # parameters for 7dd tau Night
# # get_clusters = DBSCAN(eps=2.5, min_samples=20).fit_predict(tsne_pca_results)
#     # parameters for 7dd tau all
# # get_clusters = DBSCAN(eps=1.4, min_samples=20).fit_predict(tsne_pca_results)
#     # test
# # get_clusters = DBSCAN(eps=1, min_samples=30).fit_predict(tsne_pca_results)
# perp50_cluster = get_clusters

# # check clusters
# all_feature_cond = all_feature_cond.reset_index(drop=True)
# low_perp_res = pd.DataFrame(data = tsne_pca_results,
#                          columns=['TSNE1', 'TSNE2'])
# low_perp_res = low_perp_res.assign(clusters = perp50_cluster,
#                              cond1 = all_feature_cond['cond1'],
#                              cond0 = all_feature_cond['cond0'])

# cluster_color = sns.color_palette("hls", len(set(get_clusters)))
# total_clusters = list(set(low_perp_res.clusters))

# figure, ax = plt.subplots(nrows=1, ncols=1,sharex=True, sharey=True, figsize=(8,7))

# sns.scatterplot(
#     x='TSNE1', 
#     y='TSNE2', 
#     palette = sns.color_palette(cluster_color),
#     data=low_perp_res,
#     hue = 'clusters',
#     legend=False,
#     alpha=0.03,
#     ax=ax
# )
# figure.savefig(fig_dir+"/TSNE_all_lowperp.pdf",format='PDF')

