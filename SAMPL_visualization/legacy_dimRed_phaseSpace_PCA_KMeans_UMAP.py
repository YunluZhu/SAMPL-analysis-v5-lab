'''

'''

#%%
# import sys
import os
import pandas as pd # pandas library
import numpy as np # numpy
import seaborn as sns
import matplotlib.pyplot as plt
from plot_functions.get_data_dir import (get_data_dir, get_figure_dir)
from plot_functions.get_bout_features import (get_bout_features, extract_bout_features_v5)
from plot_functions.get_bout_kinetics import get_kinetics
from plot_functions.get_IBIangles import get_IBIangles
from plot_functions.plt_tools import (set_font_type, defaultPlotting,day_night_split)
from plot_functions.plt_functions import plt_categorical_grid
from plot_functions.get_bout_kinetics import get_bout_kinetics
import math
from plot_functions.plt_tools import round_half_up 
from plot_functions.get_index import get_index
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import (KMeans, SpectralClustering)

set_font_type()
# mpl.rc('figure', max_open_warning = 0)
# os.environ['KMP_DUPLICATE_LIB_OK']='True'
# %%
# Select data and create figure folder
pick_data = 'depth_7d'
which_ztime = 'day'

root, FRAME_RATE = get_data_dir(pick_data)

folder_name = f'{pick_data} dimReduction phase space'
folder_dir = get_figure_dir(pick_data)
fig_dir = os.path.join(folder_dir, folder_name)

try:
    os.makedirs(fig_dir)
    print(f'fig folder created: {folder_name}')
except:
    print('Notes: re-writing old figures')

# %%
# Paste root directory here
peak_idx , total_aligned = get_index(FRAME_RATE)
idxRANGE = [peak_idx-round_half_up(0.2*FRAME_RATE),peak_idx+round_half_up(0.2*FRAME_RATE)]
idxRANGE_features = [peak_idx-round_half_up(0.3*FRAME_RATE),peak_idx+round_half_up(0.25*FRAME_RATE)]

# %% features for plotting
all_features = [
    'propBoutAligned_speed', 
    'propBoutAligned_accel',    # angular accel calculated using raw angular vel
    # 'linear_accel', 
    'propBoutAligned_pitch', 
    'propBoutAligned_angVel',   # smoothed angular velocity
    # # 'propBoutInflAligned_accel',
    'propBoutAligned_instHeading', 
    # 'heading_sub_pitch',
            'propBoutAligned_x',
            'propBoutAligned_y', 
    #         # 'propBoutInflAligned_angVel',
    #         # 'propBoutInflAligned_speed', 
    #         # 'propBoutAligned_angVel_hDn',
    #         # # 'propBoutAligned_speed_hDn', 
    #         # 'propBoutAligned_pitch_hDn',
    #         # # 'propBoutAligned_angVel_flat', 
    #         # # 'propBoutAligned_speed_flat',
    #         # # 'propBoutAligned_pitch_flat', 
    #         # 'propBoutAligned_angVel_hUp',
    #         # 'propBoutAligned_speed_hUp', 
    #         # 'propBoutAligned_pitch_hUp', 
    # # 'ang_speed',
    'ang_accel_of_SMangVel',    # angular accel calculated using smoothed angVel
    # 'xvel', 'yvel',
    'fish_length',
    'traj_cur',

]

# %%
# CONSTANTS
BIN_NUM = 4  # number of speed bins
SMOOTH = 11
all_conditions = []
folder_paths = []
# get the name of all folders under root
for folder in os.listdir(root):
    if folder[0] != '.':
        folder_paths.append(root+'/'+folder)
        all_conditions.append(folder)


all_around_peak_data = pd.DataFrame()
all_feature_cond = pd.DataFrame()
all_cond0 = []
all_cond1 = []

# go through each condition folders under the root
for condition_idx, folder in enumerate(folder_paths):
    # enter each condition folder (e.g. 7dd_ctrl)
    for subpath, subdir_list, subfile_list in os.walk(folder):
        # if folder is not empty
        if subdir_list:
            # reset for each condition
            around_peak_data = pd.DataFrame()
            bout_features = pd.DataFrame()
            # loop through each sub-folder (experiment) under each condition
            for expNum, exp in enumerate(subdir_list):
                # angular velocity (angVel) calculation
                rows = []
                # for each sub-folder, get the path
                exp_path = os.path.join(subpath, exp)
                # get pitch                
                raw = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout_aligned')#.loc[:,['propBoutAligned_angVel','propBoutAligned_speed','propBoutAligned_accel','propBoutAligned_heading','propBoutAligned_pitch']]
                raw = raw.assign(ang_speed=raw['propBoutAligned_angVel'].abs(),
                                            yvel = raw['propBoutAligned_y'].diff()*FRAME_RATE,
                                            xvel = raw['propBoutAligned_x'].diff()*FRAME_RATE,
                                            linear_accel = raw['propBoutAligned_speed'].diff(),
                                            ang_accel_of_SMangVel = raw['propBoutAligned_angVel'].diff(),
                                           )
                # assign frame number, total_aligned frames per bout
                raw = raw.assign(idx=round_half_up(len(raw)/total_aligned)*list(range(0,total_aligned)))
                
                # - get the index of the rows in exp_data to keep (for each bout, there are range(0:51) frames. keep range(20:41) frames)
                bout_time = pd.read_hdf(f"{exp_path}/bout_data.h5", key='prop_bout2').loc[:,['aligned_time']]
                # for i in bout_time.index:
                # # if only need day or night bouts:
                for i in day_night_split(bout_time,'aligned_time',ztime=which_ztime).index:
                    rows.extend(list(range(i*total_aligned+idxRANGE[0],i*total_aligned+idxRANGE[1])))
                exp_data = raw.loc[rows,:]
                exp_data = exp_data.assign(expNum = expNum,
                                      exp_id = condition_idx*100+expNum)
                
                grp = exp_data.groupby(np.arange(len(exp_data))//(idxRANGE[1]-idxRANGE[0]))
                
                # angvel_smoothed = grp['propBoutAligned_angVel'].apply(
                #     lambda x: savgol_filter(x, 11, 3)
                # )
                # angvel_smoothed = angvel_smoothed.explode().values
                
                xvel_adj = grp['xvel'].apply(
                    lambda x: x * (np.absolute(x.mean())/x.mean())  # adjust sign of x velocity, make it positive
                )
                
                exp_data = exp_data.assign(
                    # calculate curvature of trajectory (rad/mm) = angular velocity (rad/s) / linear speed (mm/s)
                    # traj_cur = angvel_smoothed/exp_data['propBoutAligned_speed'] * math.pi / 180,
                    xvel_adj = xvel_adj
                )
                around_peak_data = pd.concat([around_peak_data,exp_data])
                
                
                ######################
                rows_features = []
                for i in bout_time.index:
                    rows_features.extend(list(range(i*total_aligned+round_half_up(idxRANGE_features[0]),i*total_aligned+round_half_up(idxRANGE_features[1]))))
                
                # assign bout numbers
                trunc_exp_data = raw.loc[rows_features,:]
                trunc_exp_data = trunc_exp_data.assign(
                    bout_num = trunc_exp_data.groupby(np.arange(len(trunc_exp_data))//(idxRANGE_features[1]-idxRANGE_features[0])).ngroup()
                )
                this_exp_features = extract_bout_features_v5(trunc_exp_data,peak_idx,FRAME_RATE)
                this_exp_features = this_exp_features.assign(
                    bout_time = bout_time.values,
                    expNum = expNum,
                )
                # day night split. also assign ztime column
                this_ztime_exp_features = day_night_split(this_exp_features,'bout_time',ztime=which_ztime)
                
                bout_features = pd.concat([bout_features,this_ztime_exp_features])            
                
    # combine data from different conditions
    cond0 = all_conditions[condition_idx].split("_")[0]
    all_cond0.append(cond0)
    cond1 = all_conditions[condition_idx].split("_")[1]
    all_cond1.append(cond1)
    all_around_peak_data = pd.concat([all_around_peak_data, around_peak_data.assign(cond0=cond0,cond1=cond1)])
    
    all_feature_cond = pd.concat([all_feature_cond, bout_features.assign(
        cond0=cond0,
        cond1=cond1
        )])
    
all_around_peak_data = all_around_peak_data.assign(time_ms = (all_around_peak_data['idx']-peak_idx)/FRAME_RATE*1000)

# %% tidy data
all_cond0 = list(set(all_cond0))
all_cond0.sort()
all_cond1 = list(set(all_cond1))
all_cond1.sort()

all_around_peak_data = all_around_peak_data.assign(
    velocity = all_around_peak_data['propBoutAligned_speed'] * (all_around_peak_data['propBoutAligned_instHeading'].abs()/all_around_peak_data['propBoutAligned_instHeading']),
    bout_num = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0])).ngroup(),
)
# %%
df_tpcalc = all_around_peak_data[['xvel_adj','yvel','propBoutAligned_pitch','propBoutAligned_angVel','idx']]

# %% transformation NOTE slow
grouped = df_tpcalc.set_index('idx').groupby(np.arange(len(df_tpcalc))//(idxRANGE[1]-idxRANGE[0]))
all_data_grouped = all_around_peak_data.groupby(np.arange(len(all_around_peak_data))//(idxRANGE[1]-idxRANGE[0]))

ALIGNED_BOUT_RANGE = range(0, (idxRANGE[1]-idxRANGE[0]))
re_format = grouped.apply(
    lambda x: pd.concat([
                        x.iloc[ALIGNED_BOUT_RANGE,0].rename(lambda i: 'xvel_adj_'+str(i)),
                        x.iloc[ALIGNED_BOUT_RANGE,1].rename(lambda i: 'yvel_'+str(i)),
                        x.iloc[ALIGNED_BOUT_RANGE,2].rename(lambda i: 'pitch_'+str(i)),
                        x.iloc[ALIGNED_BOUT_RANGE,3].rename(lambda i: 'angvel_'+str(i)),
                        ])
)

df_std = StandardScaler().fit_transform(re_format)

re_format = pd.concat([re_format, 
                        all_data_grouped.cond0.head(1).reset_index().cond0,
                        all_data_grouped.cond1.head(1).reset_index().cond1,
                        all_data_grouped.bout_num.head(1).reset_index().bout_num,
                        ],axis=1)
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

# %%
# take a look
PCA_components = pd.DataFrame(principalComponents)
# plt.scatter(PCA_components[0], PCA_components[1], alpha=.002, color='black')
# plt.xlabel('PCA 1')
# plt.ylabel('PCA 2')

# %% reduce dimension for t-sne
pca_for_tsne = PCA(n_components=15)
pca_result_for_tsne = pca_for_tsne.fit_transform(df_std)
print(f"Cumulative explained variation for {pca_result_for_tsne.shape[1]} principal components: {np.sum(pca_for_tsne.explained_variance_ratio_)}")

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

re_format_PCA = re_format.assign(
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
    data=re_format_PCA,
    legend="full",
    alpha=0.01
)

# %% visualize
nCluster = 10
model_kmeans = KMeans(n_clusters=nCluster)
model_kmeans.fit(PCA_components.iloc[:,:10])

re_format_PCA = re_format_PCA.assign(cluster = model_kmeans.labels_)

# # %% kernelized k means
# model_spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors',
#                            assign_labels='kmeans')
# labels = model_spectral.fit_predict(PCA_components.iloc[:,:5])
# re_format_PCA = re_format_PCA.assign(spectral_cluster = model_spectral.labels_)


# %% plot 3D scatter
fig = sns.pairplot(
    data = re_format_PCA,
    vars = ['pca1','pca2','pca3'],
    palette=sns.color_palette("Set2", nCluster),
    kind='hist',
    # plot_kws=dict(size=1, alpha = 0.01, linewidth=0),
    hue = 'cluster',
    )
plt.savefig(f"{fig_dir}/PCA 3 KMeans clustered.pdf",format='PDF')
# %%
# %%
# look at eigen vectors
# pca.explained_variance_ratio_
# pca.components_

# of features wanted
n_feature = 5

# repeat with new n components
model = PCA(n_components=10).fit(df_std)
X_pc = model.transform(df_std)

# number of components 
n_pcs = model.components_.shape[0]
# get the index of the most important feature on EACH component
# LIST COMPREHENSION HERE
L = np.argsort(-np.abs(model.components_), axis=1)
initial_feature_names = list(re_format.columns[0:-3])
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
eigeness.reset_index(inplace=True,drop=True)
print('loadings')
print(eigeness)

# %%
# feed clusters back to the results
all_feature_cond = all_feature_cond.assign(
    cluster = model_kmeans.labels_
)
# all_feature_cond.groupby('cluster').mean()
# %%
# time to check clusters
y_name = 'pitch_initial'
g = plt_categorical_grid(
    data = all_feature_cond.groupby(['expNum','cluster']).mean().reset_index(),
    x_name = 'cluster',
    y_name = y_name,
    units = 'expNum',
    gridrow = None,
    gridcol = None,
    aspect = 1.5
)

plt.savefig(f"{fig_dir}/Kmeans_{y_name}.pdf",format='PDF')

# %%
############################# UMAP #######################
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap
import umap.plot
# %%
reducer = umap.UMAP(
    n_neighbors=40,
    min_dist=0,
    metric='euclidean',
    # metric='cosine',
)

# %%
embedding = reducer.fit_transform(df_std)

# %% plot
fig, ax = plt.subplots()
scatter =  ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in all_feature_cond.cluster],
    alpha=0.01,
    )
legend1 = ax.legend(*scatter.legend_elements(num=10),
                    loc="upper left", title="cluster")
ax.add_artist(legend1)

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection', fontsize=18)

# %%
mapper = reducer.fit(df_std)
umap.plot.points(mapper, labels=all_feature_cond.cluster, alpha=0.3, theme='fire')
plt.savefig(f"{fig_dir}/UMAP Kmeans.pdf",format='PDF')

# %%
