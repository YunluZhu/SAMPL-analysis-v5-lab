

df = all_data.loc[(all_data['dpf']=='7') & (all_data['condition']=='Sibs')]

# %%
df = df.sort_values(by='posture_chg')
df_out = df.groupby(np.arange(len(df))//11)[['posture_chg','atk_ang']].mean()

# %%
df_out

# %%
sns.scatterplot()


# %%

a = pd.read_hdf(f"/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/LD_data/7LD_Sibs/191205 7dpf nefma Tau neg ctrl/bout_data.h5", key='prop_bout_aligned')

# %%
a = a.assign(idx=int(len(a)/51)*list(range(0,51)))
tmp = a.groupby('idx').mean()


# %%
sns.lineplot(y=tmp['propBoutAligned_heading'], x=tmp.index)
sns.lineplot(y=tmp['propBoutAligned_y'], x=tmp.index)
# %%
