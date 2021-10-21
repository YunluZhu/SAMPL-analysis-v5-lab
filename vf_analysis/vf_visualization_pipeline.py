'''
VF visualization pipeline
Calls functions in the visualization folder and make plots
Compatible with Python Interactive Console. Use Shift/Ctrl+Enter to run cell by cell, or click "Run Below" to run all functions
atk_angle_fin_body_ratio plot is not included in the pipeline because it requires adjustment according to different data sets.
'''
# %%
from visualization.plt_bout_properties import bout_properties
from visualization.plt_bout_speed import bout_speed_aligned_jacknife
from visualization.plt_IEIpitch_mean import IEI_pitch_mean_jacknife
from visualization.plt_IEIpitch_aligned import aligned_IEIpitch_jackknife
from visualization.plt_parabola_sensitivity_half import sensitivity_jacknife_halfP
from visualization.plt_parabola_3free_full import sensitivity_jacknife_fullP
# %%
# First, paste your root directory below
root = "/Users/yunluzhu/Lab/Lab2/Data/VF/data_in_use/combined_7DD_NTau-hets_data"

# %%
# Mean inter event pitch (pitch angle between bouts) distribution plot and std
IEI_pitch_mean_jacknife(root)

# %%
# Inter event pitch using aligned data > includes all timepoints between bouts
aligned_IEIpitch_jackknife(root)

# %%
# sensitivity to pitch angles by fitting to half parabola
sensitivity_jacknife_fullP(root)
sensitivity_jacknife_halfP(root)

# %%
# bout speed
bout_speed_aligned_jacknife(root)
# bout properties
bout_properties(root)
