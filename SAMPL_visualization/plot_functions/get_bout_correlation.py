from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import (KMeans, SpectralClustering)
import pandas as pd
import numpy as np

def get_cluster_phaseSpace(around_peak_data, bout_feature, idxRANGE:list, nCluster:int):
    # %%
    # df_tpcalc = around_peak_data[['xvel_adj','yvel','propBoutAligned_pitch','propBoutAligned_angVel','idx']]

    chunk_size = idxRANGE[1] - idxRANGE[0]

    df_tpcalc = (
        around_peak_data[['xvel_adj','yvel','propBoutAligned_pitch','propBoutAligned_angVel']]
        .assign(
            grp = lambda d: np.arange(len(d)) // chunk_size,
            pos = lambda d: np.arange(len(d)) % chunk_size
        )
    )

    # reshape: make each variable wide with suffix = position
    re_format = (
        df_tpcalc
        .set_index(['grp','pos'])
        .unstack('pos')
    )

    # flatten MultiIndex columns
    re_format.columns = [f"{col}_{pos}" for col, pos in re_format.columns]

    df_std = StandardScaler().fit_transform(re_format)#.drop(index=bout_feature[bout_feature['to_bout'].isna()].index))

    # %%
    # PCA and clustering
    pca = PCA(n_components=30)
    principalComponents = pca.fit_transform(df_std)
    PCA_components = pd.DataFrame(principalComponents)

    model_kmeans = KMeans(n_clusters=nCluster)
    model_kmeans.fit(PCA_components.iloc[:,:10])

    # %% 
    bout_feature = bout_feature.assign(
        cluster = model_kmeans.labels_
    )
    return bout_feature


def get_cluster_featuresIBI(bout_feature, nCluster:int):
    # %%
    features_tocalc = bout_feature[['pitch_initial', 'pitch_mid_accel', 'pitch_pre_bout', 'pitch_peak',
        'pitch_post_bout', 'pitch_end', 'pitch_max_angvel', 'traj_initial',
        'traj_pre_bout', 'traj_peak', 'traj_post_bout', 'traj_end',
        'spd_initial', 'spd_peak', 'angvel_initial_phase', 'angvel_prep_phase',
        'angvel_post_phase', 'traj_initial_phase', 'spd_initial_phase',
        'rot_total', 'rot_bout', 'rot_pre_bout', 'rot_l_accel',
        'rot_full_accel', 'rot_full_decel', 'rot_l_decel', 'rot_early_accel',
        'rot_late_accel', 'rot_early_decel', 'rot_late_decel',
        'rot_to_max_angvel', 'bout_trajectory_Pre2Post', 'bout_displ',
        'traj_deviation', 'atk_ang', 'tsp_peak', 'angvel_chg', 'depth_chg',
        'x_chg', 'additional_depth_chg', 'displ_swim', 'ydispl_swim',
        'xdispl_swim', 'post_IBI_time',
        # 'pre_IBI_time'
        ]].dropna()

    df_std = StandardScaler().fit_transform(features_tocalc)

    # %%
    # PCA and clustering
    pca = PCA(n_components=30)
    principalComponents = pca.fit_transform(df_std)
    PCA_components = pd.DataFrame(principalComponents)

    model_kmeans = KMeans(n_clusters=nCluster)
    model_kmeans.fit(PCA_components.iloc[:,:10])

    # %% 
    bout_feature_out = features_tocalc.assign(
        cluster = model_kmeans.labels_
    )
    return bout_feature_out