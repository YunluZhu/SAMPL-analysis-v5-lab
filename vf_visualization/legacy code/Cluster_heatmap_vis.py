'''
t-SNE with aligned raw bout data
15-46
low perp for clustering
high perp to protect global structure
NOTE
UPDATE 211127 heading only took idx==30, added sample for plot
'''

#%%
import sys
import os,glob
import time
import pandas as pd
from plot_functions.plt_tools import round_half_up 
import numpy as np # numpy
import seaborn as sns
# import matplotlib
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

# from collections import defaultdict
# from datetime import datetime
# from datetime import timedelta
# import math
# import scipy

# %%
# Paste root directory here

pick_data = 'ori'
global DayNight_select 
DayNight_select = 'day' # day or night or all

# %%data
if pick_data == 'ori':
    root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7DD_data_resliced"
elif pick_data == 'hets':
    root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7DD_hets_resliced/combined_7DD_NTau-hets_data"
elif pick_data == 'ld':
    root = "/Volumes/LabData/VF_data_in_use/resliced/combined_7LD_resliced"
    
fig_dir = "/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/NT_tmp_plots_PCA_Raw"



# %%
# CONSTANTS

ALIGNED_BOUT_RANGE = range(15,46)


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
        df_out = df.loc[hour[(hour>=9) & (hour<23)].index, :]
    elif DayNight_select == 'night':
        df_out = df.loc[hour[(hour<9) | (hour>=23)].index, :]
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
all_cond_feature = pd.DataFrame()

hue_order = list()

# binned_atk_angles = pd.DataFrame()
# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            all_bouts_data = pd.DataFrame()
            all_feature_data = pd.DataFrame()
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
                angles = angles.assign(idx=round_half_up(len(angles)/51)*list(range(0,51)))          
            # extract features
                peak_angles = angles.loc[angles['idx']==30]
                peak_angles = peak_angles.assign(
                    time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2')['aligned_time'].values,
                    )  # peak angle
                peak_angles_day = day_night_split(peak_angles, 'time')
                # peak_angles_day = peak_angles_day.dropna()
                # # filter for angles meet the condition
                # peak_angles_day = peak_angles_day.loc[(peak_angles_day['propBoutAligned_heading']<HEADING_LIM) & 
                #                                       (peak_angles_day['propBoutAligned_heading']>-HEADING_LIM)]
                
                # calculate individual attack angles (heading - pitch)
                atk_ang = peak_angles_day['propBoutAligned_heading'] - peak_angles_day['propBoutAligned_pitch']
                
                # get indices of bout peak (for posture change calculation)
                peak_idx = peak_angles_day.index
                # calculate posture change calculation. NOTE change if frame rate changes
                pre_posture_chg = angles.loc[peak_idx-2, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                # try 100ms after peak, NOTE change if frame rate changes
                righting_rot = angles.loc[peak_idx+4, 'propBoutAligned_pitch'].values - angles.loc[peak_idx, 'propBoutAligned_pitch']
                steering_rot = angles.loc[peak_idx, 'propBoutAligned_pitch'].values - angles.loc[peak_idx-4, 'propBoutAligned_pitch']
                
                bout_features = pd.DataFrame(data={'atk_ang':atk_ang.values, 
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
                         
                # NOTE added heading adjustment
                # bout_features.loc[bout_features.heading<-90,'heading'] = -1*(180 + peak_angles_day['propBoutAligned_heading'])
                # bout_features.loc[bout_features.heading>90,'heading'] = 180 - peak_angles_day['propBoutAligned_heading']
                all_feature_data = pd.concat([all_feature_data, bout_features])

             # raw bout data    
                angles_grp = angles.set_index('idx').groupby(np.arange(len(angles))//51)
                
                re_format = angles_grp.apply(
                    lambda x: pd.concat([
                        # x.iloc[ALIGNED_BOUT_RANGE,0].rename(lambda i: 'heading_'+str(i)),
                                        x.iloc[ALIGNED_BOUT_RANGE,1].rename(lambda i: 'pitch_'+str(i)),
                                        x.iloc[ALIGNED_BOUT_RANGE,2].rename(lambda i: 'speed_'+str(i)),
                                        ])
                )
                
                re_format = pd.concat([re_format, 
                                       angles_grp.propBoutAligned_time.head(1).reset_index().propBoutAligned_time,
                                       angles.loc[angles['idx']==30].reset_index().propBoutAligned_heading,
                                    #    boutNum[['boutNum','epochNum']]
                                       ],axis=1)
                
                # re_format_IEI = pd.concat([re_format.set_index(['boutNum', 'epochNum']),
                #                            IEIdata,
                #                           ],axis=1,join='inner').reset_index()
                             
                re_format_day = day_night_split(re_format,'propBoutAligned_time')
                all_bouts_data = pd.concat([all_bouts_data, re_format_day])
                
            all_cond_feature = pd.concat([all_cond_feature,all_feature_data.assign(condition=all_conditions[condition_idx])])
            all_cond_bouts = pd.concat([all_cond_bouts,all_bouts_data.assign(condition=all_conditions[condition_idx])])
all_cond_feature.reset_index(inplace=True,drop=True)
all_cond_bouts.reset_index(inplace=True,drop=True)
combined_all = pd.concat([all_cond_feature.drop(columns='condition'),all_cond_bouts],axis=1)
combined_all = combined_all.dropna().reset_index(drop=True)    
                                  
data_to_ana = all_cond_bouts.drop(['condition','propBoutAligned_time'],axis=1)
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

# # %% Fiind the clusters
# ks = range(1, 10)
# inertias = []
# for k in ks:
#     # Create a KMeans instance with k clusters: model
#     model = KMeans(n_clusters=k)
    
#     # Fit model to samples
#     model.fit(PCA_components.iloc[:,:3])
    
#     # Append the inertia to the list of inertias
#     inertias.append(model.inertia_)
    
# plt.plot(ks, inertias, '-o', color='black')
# plt.xlabel('number of clusters, k')
# plt.ylabel('inertia')
# plt.xticks(ks)
# plt.show()

# %%
# res = pd.DataFrame()
# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(df_std)
# all_cond_bouts = all_cond_bouts.assign(
#     pca1 = pca_result[:,0],
#     pca2 = pca_result[:,1],
# )
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))

# # plot 2pc
# plt.figure(figsize=(16,10))
# sns.scatterplot(
#     x="pca1", y="pca2",
#     hue="condition",
#     # palette=sns.color_palette("hls", 2),
#     data=all_cond_bouts,
#     legend="full",
#     alpha=0.1
# )

# %% reduce dimension for t-sne
pca_for_tsne = PCA(n_components=10)
pca_result_for_tsne = pca_for_tsne.fit_transform(df_std)
print(f"Cumulative explained variation for {pca_result_for_tsne.shape[1]} principal components: {np.sum(pca_for_tsne.explained_variance_ratio_)}")

# %% t-sne with reduced perp
time_start = time.time()

# # preserve: ori data, day
# tsne = TSNE(init='pca', n_components=2, verbose=1, perplexity=30,n_iter=800,early_exaggeration=20,learning_rate=1000)
tsne = TSNE(init='pca', n_components=2, verbose=1, perplexity=30,n_iter=800,early_exaggeration=20,learning_rate=1000)

tsne_pca_results = tsne.fit_transform(pca_result_for_tsne)
print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

# %% Get clusters using DB scan
# https://www.reneshbedre.com/blog/tsne.html

from bioinfokit.visuz import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics


cluster.tsneplot(score=tsne_pca_results)

# recognize the clusters using the DBSCAN algorithm
# # preserve: original data, day
# db_clusters = DBSCAN(eps=3, min_samples=45).fit(tsne_pca_results)

db_clusters = DBSCAN(eps=3, min_samples=45).fit(tsne_pca_results)

get_clusters = db_clusters.labels_
core_samples_mask = np.zeros_like(db_clusters.labels_, dtype=bool)
core_samples_mask[db_clusters.core_sample_indices_] = True
labels = db_clusters.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


# check clusters

low_perp_res = pd.DataFrame(data = tsne_pca_results,
                         columns=['TSNE1', 'TSNE2'])
low_perp_res = low_perp_res.assign(clusters = get_clusters,
                             condition = combined_all['condition'])

cluster_color = sns.color_palette("hls", len(set(get_clusters)))
total_clusters = list(set(low_perp_res.clusters))

figure, ax = plt.subplots(nrows=1, ncols=1,sharex=True, sharey=True, figsize=(8,7))

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    palette = sns.color_palette(cluster_color),
    data=low_perp_res,
    hue = 'clusters',
    # legend=False,
    alpha=0.03,
    ax=ax
)
figure.savefig(fig_dir+"/TSNE_all_lowperp.pdf",format='PDF')



# %% tidy data
all_conditions.sort()

res_toplt = pd.DataFrame(data = tsne_pca_results,
                         columns=['TSNE1', 'TSNE2'])
res_toplt = res_toplt.assign(clusters = get_clusters,
                             time = combined_all['propBoutAligned_time'],
                             )
res_toplt.dropna(inplace=True)
#  mark day night 
hours = res_toplt['time'].dt.strftime('%H').astype('int')
day_night = []
res_toplt = res_toplt.assign(hours = hours)
res_toplt.loc[hours[(hours>=9) & (hours<23)].index, 'day_night'] = 'day'
res_toplt.loc[hours[(hours<9) | (hours>=23)].index, 'day_night'] = 'night'

# combine features
res_toplt = pd.concat([res_toplt,combined_all],axis=1)
res_toplt = res_toplt.assign(rise_dive = lambda x: pd.cut(x['pitch'], 
                                        bins=[-90,-20,0,20,90],
                                        labels=["1SD",'2D', '3R','4SR']),
                             angle_chg = res_toplt.end_pitch - res_toplt.pre_pitch)

res_ctrl = res_toplt.loc[combined_all.loc[combined_all.condition==all_conditions[0]].index,:]
res_cond = res_toplt.loc[combined_all.loc[combined_all.condition==all_conditions[1]].index,:]

cluster_color = sns.color_palette("hls", len(set(get_clusters)))
total_clusters = list(set(res_toplt.clusters))
ctrl_cluster = list(set(res_ctrl.clusters))
cond_cluster = list(set(res_cond.clusters))

total_clusters.sort()
ctrl_cluster.sort()
cond_cluster.sort()



# %% cluster size

all_size = pd.DataFrame()
for cluster_num in set(get_clusters):
    current_cluster = combined_all.loc[res_toplt.clusters==cluster_num]
    cluster_size = current_cluster.groupby('condition').size()
    all_size = pd.concat([all_size, cluster_size],axis=1)
    all_size.columns.values[-1:]=[cluster_num]
    
total = combined_all.groupby('condition').size()
all_size.iloc[0,:] = all_size.iloc[0,:] /total.values[0]
all_size.iloc[1,:] = all_size.iloc[1,:] /total.values[1]

all_size = all_size.T

all_size = all_size.assign(
    dir = ((all_size[all_conditions[0]]-all_size[all_conditions[1]])/(all_size[all_conditions[0]]+all_size[all_conditions[1]]))
    )
all_size
# %% visualize
dots_to_plt = 10000
if dots_to_plt > min(len(res_ctrl),len(res_cond)):
    dots_to_plt = min(len(res_ctrl),len(res_cond))
alpha = 2000/dots_to_plt

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
    data=res_ctrl.sample(n=dots_to_plt),
    hue = 'clusters',
    legend=False,
    alpha=alpha,
    ax=ax1
)
ax1.set_title(f"{all_conditions[0]}")

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    palette = sns.color_palette(cond_palette),
    data=res_cond.sample(n=dots_to_plt),
    hue = 'clusters',
    legend=False,
    alpha=alpha,
    ax=ax2
)
ax2.set_title(f"{all_conditions[1]}")
figure.savefig(fig_dir+"/TSNE_2cond.pdf",format='PDF')

# %% hue = day night
if DayNight_select == 'all':

    figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(16,7))
    dayNight_order = ['night','day']
    sns.scatterplot(
        x='TSNE1', 
        y='TSNE2', 
        data=res_ctrl.sample(n=dots_to_plt),
        hue = 'condition',
        palette = sns.diverging_palette(220, 20,n=2),
        hue_order = dayNight_order,
        legend=False,
        alpha=alpha,
        ax=ax1
    )
    ax1.set_title(f"{all_conditions[0]}")

    sns.scatterplot(
        x='TSNE1', 
        y='TSNE2', 
        data=res_cond.sample(n=dots_to_plt),
        hue = 'day_night',
        palette = sns.diverging_palette(220, 20,n=2),
        hue_order = dayNight_order,
        legend='brief',
        alpha=alpha,
        ax=ax2
    )
    ax2.set_title(f"{all_conditions[1]}")
    figure.savefig(fig_dir+"/TSNE_day_night.pdf",format='PDF')

# %% other features

# angle_chg
# 'atk_ang'
# 'pre_posture_chg'
# 'pre_pitch'
# 'end_pitch'
# 'accel_rot'
# 'decel_rot'
# 'heading'
# 'pitch'
# 'speed'
# 'accel_ang'
# 'decel_ang'


hue_col = 'speed' # change the hue to plot

figure, (ax1,ax2) = plt.subplots(nrows=1, ncols=2,sharex=True, sharey=True, figsize=(16,7))
hue_order = list(set(res_ctrl[hue_col])).sort()
sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    data=res_ctrl.sample(n=dots_to_plt),
    hue = hue_col,
    hue_order = hue_order,
    palette = sns.diverging_palette(220, 20,as_cmap=True),
    legend=False,
    alpha=alpha,
    linewidth=0,
    ax=ax1
)
ax1.set_title(f"{all_conditions[0]}")

sns.scatterplot(
    x='TSNE1', 
    y='TSNE2', 
    data=res_cond.sample(n=dots_to_plt),
    hue = hue_col,
    hue_order = hue_order,
    palette = sns.diverging_palette(220, 20,as_cmap=True),
    legend='brief',
    alpha=alpha,
    linewidth=0,
    ax=ax2
)
ax2.set_title(f"{all_conditions[1]}")
figure.savefig(fig_dir+"/TSNE_{}.pdf".format(hue_col),format='PDF')

# %%
all_features = ['angle_chg',
                'atk_ang',
                'pre_posture_chg',
                'pre_pitch',
                'end_pitch',
                'accel_rot',
                'decel_rot',
                'heading',
                'pitch',
                'speed',
                'accel_ang',
                'decel_ang']

# for feature in all_features:
#     plt.figure(figsize=(50,5))
#     sns.pointplot(x='clusters',y=feature, hue='condition',data=res_toplt,
#                 dodge=True, join=False) 
    
# %% Visualiza differences between control and condition by clusters
all_features = ['angle_chg',
                'atk_ang',
                'pre_posture_chg',
                'pre_pitch',
                'end_pitch',
                'accel_rot',
                'decel_rot',
                'heading',
                'pitch',
                'speed',
                'accel_ang',
                'decel_ang']

ctrl_mean = res_ctrl.groupby('clusters')[all_features].mean()
ctrl_std = res_ctrl.groupby('clusters')[all_features].std()

cond_mean = res_cond.groupby('clusters')[all_features].mean()

cond_zScore = ((cond_mean-ctrl_mean)/ctrl_std)
cond_zScore = cond_zScore.loc[cond_zScore.index>=0]
# cond_zScore[np.abs(cond_zScore)<=2]=0
from matplotlib.colors import DivergingNorm
data = cond_zScore.T
cm = sns.diverging_palette(220, 20,as_cmap=True)

fig, ax = plt.subplots(figsize=(20,10))
im = ax.imshow(data, norm=DivergingNorm(0), cmap=cm, interpolation='none',)
fig.colorbar(im)
ax.set_yticks(np.arange(len(all_features)))
ax.set_xticks(np.arange(len(cond_zScore)))

ax.set_yticklabels(list(cond_zScore.columns))
plt.show()


# %% Visualiza differences between control and condition by clusters

ctrl_mean = res_ctrl.groupby('clusters').mean()
ctrl_std = res_ctrl.groupby('clusters').std()

cond_mean = res_cond.groupby('clusters').mean()

cond_zScore = ((cond_mean-ctrl_mean)/ctrl_std)
cond_zScore.dropna(inplace=True)
cond_zScore = cond_zScore.loc[cond_zScore.index>=0]










cond_zScore[np.abs(cond_zScore)<=0.5]=0
from matplotlib.colors import DivergingNorm
data = cond_zScore.T
data[np.abs(data)<=0.5]=0

cm = sns.diverging_palette(220, 20,as_cmap=True)

fig, ax = plt.subplots(figsize=(10 ,20))
im = ax.imshow(data, norm=DivergingNorm(0), cmap=cm, interpolation='none',)
fig.colorbar(im)
ax.set_yticks(np.arange(cond_zScore.shape[1]))
ax.set_xticks(np.arange(len(cond_zScore)))

ax.set_yticklabels(list(cond_zScore.columns))
plt.show()


# %%
sns.clustermap(res_ctrl.loc[:,all_features],cmap=cm,z_score=1)
sns.clustermap(res_cond.loc[:,all_features],cmap=cm,z_score=1)

# # visualiza using heatmap

# sns.heatmap(cond_zScore.T,
#                 cmap = sns.diverging_palette(220, 20,as_cmap=True),
# )

# # plot 0 is white
# from matplotlib import colors
# cm = sns.diverging_palette(220, 20,as_cmap=True)
# def centered_gradient(s, m, M, cmap='PuBu', low=0, high=0):
#     rng = M - m
#     norm = colors.Normalize(m - (rng * low),
#                             M + (rng * high))
#     normed = norm(s.values)
#     c = [colors.rgb2hex(x) for x in plt.cm.get_cmap(cmap)(normed)]
#     return ['background-color: %s' % color for color in c]

# even_range = np.max([np.abs(cond_zScore.values.min()), np.abs(cond_zScore.values.max())])
# cond_zScore.T.style.apply(centered_gradient,
#                cmap=cm,
#                m=-even_range,
#                M=even_range).set_precision(2)

# cm = sns.diverging_palette(5, 250, as_cmap=True)
# cond_zScore.style.centered_gradient(cmap=cm).set_precision(2)
# %%
