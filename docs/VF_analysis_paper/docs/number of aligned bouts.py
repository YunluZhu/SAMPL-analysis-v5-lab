# %%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt

# %%
# number of aligned bouts we get from the apparatus
# data was collected during 24 hrs, in constant darkness (D/D), from 30 different boxes
# recommended fish number for regular-sized cuvettes: 
#   4dpf: 6-8 fish
#   7dpf: 6-8 fish
#   14dpf: 2-4 fish

notes = {'4dpf':'6-8 fish',
         '7dpf':'6-8 fish',
         '14dpf':'2-4 fish'}

sample_4to5dpf_DD_24hrs_6to8fish = \
    [305,334,455,409,495,467,392,301,388,587,354,690,196,193,137,433,369,615,199,267,172,268,154,268,150,200,282,354,310,311]
sample_7to8dpf_DD_24hrs_6to8fish = \
    [502,617,671,343,203,425,385,526,347,331,209,206,202,298,285,415,225,375,265,258,260,218,245,269,280,99,177,183,174,107,128,200,248,251,234,171,150]
sample_14to15dpf_DD_24hrs_2to4fish = \
    [572,371,439,547,487,363,407,988,592,357,248,444,753,464,542,241,393,592,209,422,227,209,211,146,144,124,123]

aligned_number = pd.DataFrame()
aligned_number = aligned_number.assign(
    aligned_bout_num = np.concatenate((sample_4to5dpf_DD_24hrs_6to8fish,
                               sample_7to8dpf_DD_24hrs_6to8fish,
                               sample_14to15dpf_DD_24hrs_2to4fish)),
    cond = np.concatenate(( len(sample_4to5dpf_DD_24hrs_6to8fish)*['4dpf'],
                            len(sample_7to8dpf_DD_24hrs_6to8fish)*['7dpf'],
                            len(sample_14to15dpf_DD_24hrs_2to4fish)*['14dpf'])),
)
# %%
f = sns.pointplot(y='aligned_bout_num', x='cond',data=aligned_number,
                  join=False,
                  )

for i, x in enumerate(f.get_xticklabels()):
    condition = x.get_text()
    plt.text(i+0.1,
             f.get_lines()[i].get_ydata().mean(), 
             notes[condition], 
             horizontalalignment='left', size='medium', color='black')
f.set_title("Number of aligned bouts per behavior box over 24h in D/D")

folder_dir = os.getcwd()
filename = os.path.join(folder_dir,"number of aligned bouts.pdf")
plt.savefig(filename, format='PDF')
# %%
