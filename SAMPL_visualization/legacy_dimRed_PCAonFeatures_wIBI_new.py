# %%
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
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
all_feature_cond, all_cond1, all_cond2 = get_connected_bouts(root, FRAME_RATE, ztime=which_ztime)

# %%
# 
drop_col = ['bout_time', 'expNum', 'ztime', 'cond0', 'cond1','exp','bout_uid','epoch_uid','to_bout',
            'x_initial','y_initial', 'x_end', 'y_end', 
            'x_chg','lift_distance',
            ]
data_to_ana = all_feature_cond.dropna()
data_to_ana = data_to_ana.drop(drop_col,axis=1)

#%%

embedding = PCA(n_components=8)
low_dim_res = embedding.fit_transform(data_to_ana)

# Save components to a DataFrame
low_dim_df = pd.DataFrame(low_dim_res)
low_dim_df[['expNum', 'ztime', 'cond0', 'cond1','exp']] = all_feature_cond[['expNum', 'ztime', 'cond0', 'cond1','exp']]
print('Explained variation per principal component: {}'.format(embedding.explained_variance_ratio_))

# %% visualize clustering
plt.scatter(low_dim_df[0], low_dim_df[1], alpha=.002, color='black')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

# %% Fiind the clusters
ks = range(1, 10)
inertias = []
for k in ks:
    # Create a KMeans instance with k clusters: model
    model = KMeans(n_clusters=k)
    
    # Fit model to samples
    model.fit(low_dim_df.iloc[:,:3])
    
    # Append the inertia to the list of inertias
    inertias.append(model.inertia_)
    
plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()

# %%

low_dim_df['cond_e'] = low_dim_df['cond1'] + '_' + low_dim_df['ztime'] 

# # plot 2pc
plt.figure(figsize=(16,10))
sns.scatterplot(
    x=0, y=1,
    hue='cond_e',
    # palette=sns.color_palette("hls", 2),
    data=low_dim_df.groupby('cond_e').sample(4000),
    legend="full",
    alpha=0.1
)

# %% visualize
model = KMeans(n_clusters=3)
model.fit(low_dim_df.iloc[:,:5])

low_dim_df = low_dim_df.assign(cluster = model.labels_)

# %% plot 3D scatter
fig = sns.pairplot(
    data = low_dim_df,
    vars = [0,1],
    palette=sns.color_palette("Set2", 2),
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
