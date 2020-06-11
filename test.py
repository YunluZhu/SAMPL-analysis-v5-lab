# %%
import pandas as pd # also require installation of pytables for hdf5 output
# %% 

exp_path = '/Users/yunluzhu/Lab/! Lab2/Python VF/script/vertical_fish_analysis/tests/test_data/4dd_Tau/200120 DD 4dpf NTau pos num1'
df = pd.read_hdf(f"{exp_path}/IEI_data.h5", key='prop_bout_IEI2').loc[:,'epochNum']


# %%
df

# %%
