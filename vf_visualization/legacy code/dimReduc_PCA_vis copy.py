# dimReduc_PCA_Kmeans
'''
run PCA and plot


'''

#%%
import sys
import os,glob
import time
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from collections import defaultdict
from datetime import datetime
from datetime import timedelta
import math
import scipy

# %%
# Paste root directory here
root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7DD_data_resliced"
fig_dir = "/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/NT_tmp_plots"

# %%
# CONSTANTS

ALIGNED_BOUT_RANGE = range(20,41)

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
    df_day = df.loc[hour[(hour>9) & (hour<23)].index, :]
    df_night = df.loc[hour[(hour<9) | (hour>23)].index, :]
    return df_day

# def distribution_binned_average(df, condition):
#     '''
#     bins raw pitch data using fixed bin width. Return binned mean of pitch and bout frequency.
#     '''
#     df = df.sort_values(by='pre_posture_chg')
#     bins = pd.cut(df['pre_posture_chg'], list(AVERAGE_BIN))
#     grp = df.groupby(bins)
#     df_out = grp[['pre_posture_chg','atk_ang']].mean().assign(dpf=condition[0],condition=condition[4:])
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

                angles = angles.assign(idx=int(len(angles)/51)*list(range(0,51)))          
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
                                       ],axis=1)
                
                re_format_day = day_night_split(re_format,'propBoutAligned_time')
                re_format_day.drop(columns=['propBoutAligned_time'],inplace=True)

                all_bouts_data = pd.concat([all_bouts_data, re_format_day])
                
            all_cond_bouts = pd.concat([all_cond_bouts,all_bouts_data.assign(condition=all_conditions[condition_idx])])
                                       
data_to_ana = all_cond_bouts.dropna().reset_index(drop=True)
df_std = StandardScaler().fit_transform(data_to_ana.iloc[:,0:-1])

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
ks = range(1, 15)
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

# %% visualize with pca 2
# res = pd.DataFrame()
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(df_std)
# tmp_plt = data_to_ana.assign(
#     pca1 = pca_result[:,0],
#     pca2 = pca_result[:,1],
# )
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # 2pc
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca1", y="pca2",
#     hue="condition",
#     # palette=sns.color_palette("hls", 2),
#     data=tmp_plt,
#     legend="full",
#     alpha=0.1
# )

# %% importance of features
# def myplot(score,coeff,labels=None):
#     xs = score[:,0]
#     ys = score[:,1]
#     n = coeff.shape[0]
#     scalex = 1.0/(xs.max() - xs.min())
#     scaley = 1.0/(ys.max() - ys.min())
#     plt.scatter(xs * scalex,ys * scaley, alpha=0.01)
#     for i in range(n):
#         plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
#         if labels is None:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
#         else:
#             plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')

# plt.xlim(-1,1)
# plt.ylim(-1,1)
# plt.xlabel("PC{}".format(1))
# plt.ylabel("PC{}".format(2))
# plt.grid()

# #Call the function. Use only the 2 PCs.
# myplot(principalComponents[:,0:2],np.transpose(pca.components_[0:2, :]))
# plt.show()

# extract features
pca.explained_variance_ratio_
pca.components_

# of features wanted
n_feature = 6

# repeat with new n components
model = PCA(n_components=5).fit(df_std)
X_pc = model.transform(df_std)

# number of components 
n_pcs = model.components_.shape[0]
# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
L = np.argsort(-np.abs(model.components_), axis=1)
initial_feature_names = list(data_to_ana.columns[0:-1])
# get the names
important_features = pd.DataFrame()
for i in range(n_feature):
    important_features = pd.concat(
        [important_features,pd.Series([initial_feature_names[L[j,i]] for j in range(n_pcs)])],
        axis = 1
    )
important_features.columns = ["feature{}".format(i+1) for i in range(n_feature)]
print('top feature names')
print(important_features)

print('\n')

# get the eigenness
eigeness = pd.DataFrame()
for i in range(n_pcs):
    eigeness = pd.concat(
        [eigeness,pd.Series([model.components_[i,L[i,j]] for j in range(n_feature)])],
        axis = 1
    )
eigeness = eigeness.T
eigeness.columns = ["feature{}".format(i+1) for i in range(n_feature)]
print('eigenvectors')
print(eigeness)
# %%

# # %% now determine the pca components and n clusters
# pca_components = 5
# n_clusters = 5

# pca = PCA(n_components = pca_components)
# pca.fit(df_std)
# scores_pca = pca.transform(df_std)

# kmeans_pca = KMeans(n_clusters=n_clusters, init='k-means++',random_state=42)
# kmeans_pca.fit(scores_pca)

# # %%
# df_pca_kmeans = pd.concat([data_to_ana.reset_index(drop=True),pd.DataFrame(scores_pca)],axis=1)
# df_pca_kmeans.columns.values[-5: ]=['C1','C2','C3','C4','C5']
# df_pca_kmeans = df_pca_kmeans.assign(Segment_Kmeans_PCA=kmeans_pca.labels_)

# # %%
# plt.figure(figsize=(10,8))
# sns.scatterplot(x='C2',y='C1',hue='Segment_Kmeans_PCA',data = df_pca_kmeans, 
#                 palette=sns.color_palette("hls", 5),
#                 alpha=0.01)

# # %%
# all_size = pd.DataFrame()
# for cluster_num in set(df_pca_kmeans.Segment_Kmeans_PCA.values):
#     current_cluster = df_pca_kmeans.loc[df_pca_kmeans.Segment_Kmeans_PCA==cluster_num]
#     cluster_size = current_cluster.groupby('condition').size()
#     all_size = pd.concat([all_size, cluster_size],axis=1)
#     all_size.columns.values[-1:]=[cluster_num]
    
# total = all_cond_bouts.groupby('condition').size()
# all_size.iloc[0,:] = all_size.iloc[0,:] /total.values[0]
# all_size.iloc[1,:] = all_size.iloc[1,:] /total.values[1]

# all_size = all_size.T

# all_size = all_size.assign(
#     dir = ((all_size['7dd_1Sibs']-all_size['7dd_2Tau'])/(all_size['7dd_1Sibs']+all_size['7dd_2Tau']))
#     )
# all_size