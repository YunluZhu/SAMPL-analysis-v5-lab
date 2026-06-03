'''
t-SNE with bout features
1. calculate bout features
2. run PCA and plot
3. run PCA to reduce dimension for t-SNE
4. run t-SNE
5. DBSCAN for recognizing clusters
6. plot DBSCAN results
7. plot t-SNE colored by conditioins
'''
# NOTE check day_night_split 

#%%
import sys
import os,glob
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np 
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

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
root = "/Volumes/LabData/ SAMPL_tmp_plotting"
fig_dir = "/Users/yunluzhu/Documents/Lab2/Data/SAMPL_ana/Figures/2021-0713 dim reduc"

global DayNight_select 
DayNight_select = 'day' # day or night or all

# %%
# CONSTANTS

ALIGNED_BOUT_RANGE = range(20,41)

HEADING_LIM = 180
FIN_BODY_LOW_LIM = -10  # lower pitch change limit to plot fin_body sigmoid fit and binned data
FIN_BODY_UP_LIM = 15  # 
X_RANGE = np.arange(-10,15.01,0.01)
BIN_WIDTH = 0.8  
AVERAGE_BIN = np.arange(-10,15,BIN_WIDTH)

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
                angles = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned').loc[:,['propBoutAligned_heading','propBoutAligned_pitch','propBoutAligned_speed']]
                angles = angles.assign(idx=round_half_up(len(angles)/51)*list(range(0,51)))
                peak_angles = angles.loc[angles['idx']==30]
                peak_angles = peak_angles.assign(
                    time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
                    )  # peak angle
                peak_angles_day = day_night_split(peak_angles, 'time')
                # peak_angles_day = peak_angles_day.dropna()
                # filter for angles meet the condition
                peak_angles_day = peak_angles_day.loc[(peak_angles_day['propBoutAligned_heading']<HEADING_LIM) & 
                                                      (peak_angles_day['propBoutAligned_heading']>-HEADING_LIM)]
                
                # calculate individual attack angles (heading - pitch)
                atk_ang = peak_angles_day['propBoutAligned_heading'] - peak_angles_day['propBoutAligned_pitch']
                
                # get indices of bout peak (for posture change calculation)
                peak_idx = peak_angles_day.index
                # calculate posture change calculation. NOTE change if frame rate changes
                pre_posture_chg = angles.loc[peak_idx-2, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                # try 100ms after peak, NOTE change if frame rate changes
                righting_rot = angles.loc[peak_idx+4, 'propBoutAligned_pitch'].values - angles.loc[peak_idx, 'propBoutAligned_pitch']
                steering_rot = angles.loc[peak_idx, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                
                output_forBout = pd.DataFrame(data={'atk_ang':atk_ang.values, 
                                             'pre_posture_chg':pre_posture_chg.values, 
                                             'pre_pitch': angles.loc[peak_idx-4,'propBoutAligned_pitch'].values, # try 100ms before peak
                                             'end_pitch': angles.loc[peak_idx+4,'propBoutAligned_pitch'].values, # try 100ms after peak
                                             'accel_rot' : steering_rot.values,
                                             'decel_rot': righting_rot.values,
                                             'heading': peak_angles_day['propBoutAligned_heading'], 
                                             'pitch': peak_angles_day['propBoutAligned_pitch'],
                                             'speed': angles.loc[peak_idx, 'propBoutAligned_speed'].values,
                                             'accel_ang': angles.loc[peak_idx-2,'propBoutAligned_pitch'].values,
                                             'decel_ang': angles.loc[peak_idx+2,'propBoutAligned_pitch'].values,  # mid bout angel decel
                                             'expNum':[expNum]*len(pre_posture_chg),
                                             'date':exp[0:6]})   
                         
                # output_forBout.drop(columns=['index'],inplace=True)
                # re_format_IEI_day = day_night_split(re_format_IEI,'propBoutAligned_time')
                # re_format_IEI_day.drop(columns=['boutNum','epochNum','propBoutAligned_time'],inplace=True)

                all_bouts_data = pd.concat([all_bouts_data, output_forBout])
                # all_bouts_IEI_data = pd.concat([all_bouts_data, re_format_IEI_day])
                
            all_cond_bouts = pd.concat([all_cond_bouts,all_bouts_data.assign(cond1=all_conditions[condition_idx])])

data_to_ana = all_cond_bouts.dropna().reset_index(drop=True)
df_std = StandardScaler().fit_transform(data_to_ana.iloc[:,0:-1])

all_conditions.sort()
# %%
# Create a PCA instance: pca
pca = PCA(n_components=13)
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
data_to_ana = data_to_ana.assign(
    pca1 = pca_result[:,0],
    pca2 = pca_result[:,1],
)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# 2pc
plt.figure(figsize=(16,10))
sns.scatterplot(
    x="pca1", y="pca2",
    hue='cond1',
    # palette=sns.color_palette("hls", 2),
    data=data_to_ana,
    legend="full",
    alpha=0.1
)

# %% reduce dimension for t-sne
pca_for_tsne = PCA(n_components=6)
pca_result_for_tsne = pca_for_tsne.fit_transform(df_std)
print(f"Cumulative explained variation for {pca_result_for_tsne.shape[1]} principal components: {np.sum(pca_for_tsne.explained_variance_ratio_)}")

# %% t-sne with reduced data
time_start = time.time()
tsne = TSNE(n_components=2, verbose=0, perplexity=300, n_iter=500)
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

# get_clusters = DBSCAN(eps=2, min_samples=10).fit_predict(tsne_pca_results) # parameters for 7dd tau Night
# get_clusters = DBSCAN(eps=1.2, min_samples=50).fit_predict(tsne_pca_results) # parameters for 7dd tau Day, with perplexity=100
get_clusters = DBSCAN(eps=0.5, min_samples=10).fit_predict(tsne_pca_results)


# check unique clusters
# -1 value represents noisy points could not assigned to any cluster

# # get t-SNE plot with colors assigned to each cluster
# cluster.tsneplot(score=tsne_pca_results, colorlist = get_clusters, 
#     legendpos='upper right', legendanchor=(1.15, 1))



# %% visualize
is_ctrl = 0
is_cond = 1

res_toplt = pd.DataFrame(data = tsne_pca_results,
                         columns=['TSNE1', 'TSNE2'])
res_toplt = res_toplt.assign(clusters = get_clusters)

res_ctrl = res_toplt.loc[data_to_ana.loc[data_to_ana.cond1==all_conditions[is_ctrl]].index,:]
res_cond = res_toplt.loc[data_to_ana.loc[data_to_ana.cond1==all_conditions[is_cond]].index,:]

cluster_color = sns.color_palette("hls", len(set(get_clusters)))
total_clusters = list(set(res_toplt.clusters))
ctrl_cluster = list(set(res_ctrl.clusters))
cond_cluster = list(set(res_cond.clusters))

total_clusters.sort()
ctrl_cluster.sort()
cond_cluster.sort()

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
    alpha=0.1,
    linewidth=0,
    ax=ax1
)
ax1.set_title(f"{all_conditions[is_ctrl]}")

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    palette = sns.color_palette(cond_palette),
    data=res_cond,
    hue = 'clusters',
    legend=False,
    alpha=0.1,
    linewidth=0,
    ax=ax2
)
ax2.set_title(f"{all_conditions[is_cond]}")
figure.savefig(fig_dir+"/TSNE_proc_newfig.pdf",format='PDF')

# %% plot on condition

res_toplt_cond = res_toplt.assign(
    cond1 = data_to_ana.condition.values
)

figure2 = plt.figure(figsize=(8,7))
sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    # palette = sns.color_palette("hls", len(all_conditions)),
    data=res_toplt_cond,
    hue = 'cond1',
    legend='full',
    alpha=0.1,
)
figure2.savefig(fig_dir+"/TSNE_proc_cond.pdf",format='PDF')

# %%
