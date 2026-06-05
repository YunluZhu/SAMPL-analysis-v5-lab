# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, get_connected_bouts)
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
import umap
from sklearn.cluster import DBSCAN
import hdbscan




# %%
set_font_type()
# %%
# Select data and create figure folder
pick_data = 'wt_bkg'
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
all_feature_cond, all_cond1, all_cond2 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime)

# %%
# Create a PCA instance: pca
drop_col = ['bout_time', 'expNum', 'ztime', 'cond0', 'cond1','exp','bout_uid','epoch_uid','to_bout']
data_to_ana = all_feature_cond.dropna()
df_std = data_to_ana.drop(drop_col,axis=1)
df_std = StandardScaler().fit_transform(df_std)

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

all_cond_bouts = data_to_ana.assign(
    pca1 = principalComponents[:,0],
    pca2 = principalComponents[:,1],
    pca3 = principalComponents[:,2],
    cond1 = data_to_ana['cond1'],
    cond0 = data_to_ana['cond0'],
    ztime = data_to_ana['ztime']
)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # plot 2pc
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca1", y="pca2",
    hue='ztime',
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
################################################################################

# %% UMAP
standard_embedding = umap.UMAP().fit_transform(df_std)

hue1 = 'cond1'
hue2 = 'ztime'
umap_toplt = pd.DataFrame(data={'umap_1':standard_embedding[:, 0], 
                   'umap_2':standard_embedding[:, 1], 
                   'hue1': data_to_ana[hue1],
                   'hue2': data_to_ana[hue2]})
g = sns.scatterplot(umap_toplt, x='umap_1', y='umap_2', hue='hue1', alpha=0.02, size=0.1)
p = sns.relplot(kind='scatter', col='hue1',hue='hue2', data= umap_toplt, x='umap_1', y='umap_2', alpha=0.02, size=0.1)

# %% re umap for clustering 
clusterable_embedding = umap.UMAP(
    n_neighbors=50,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(df_std)

hue1 = 'cond1'
hue2 = 'ztime'

umap_vis = pd.DataFrame(data={'umap_1':clusterable_embedding[:, 0], 
                   'umap_2':clusterable_embedding[:, 1], 
                   'hue1': data_to_ana[hue1],
                   'hue2': data_to_ana[hue2]})
g = sns.scatterplot(umap_vis, x='umap_1', y='umap_2', hue='hue1', alpha=0.02, size=0.1)

# %% cluste4ring
# cluster_labels = hdbscan.HDBSCAN(
#     min_samples=200,
#     min_cluster_size=5,
# ).fit_predict(clusterable_embedding)

get_clusters = DBSCAN(eps=0.8, min_samples=100).fit_predict(clusterable_embedding)

umap_toplt = umap_toplt.assign(
    cluster = get_clusters
)
p = sns.relplot(kind='scatter', col='hue1',hue='cluster', data= umap_toplt, x='umap_1', y='umap_2', alpha=0.2, s=2,
                palette = 'Spectral')


# # %%



# %%
