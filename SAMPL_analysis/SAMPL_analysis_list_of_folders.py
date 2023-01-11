'''
- What is this script
    This scripts allows for batch analysis of a list of folders.
- How to use
    "list_of_root_folders" contains a list of tuples containing root directory and frame rate.
'''

# %%
from SAMPL_analysis import SAMPL_analysis
from bout_analysis.logger import log_SAMPL_ana

# %%

LoF_logger = log_SAMPL_ana('SAMPL_dataList_log')
logger = log_SAMPL_ana('SAMPL_ana_log')
logger.info("Called from SAMPL_analysis_list_of_folders.py")

# %%
list_of_root_folders = [
    # ("<here's the root directory>", <here's the frame rate>),
    # see examples below:
    # ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined/7dd_bkgcombined", 166),  # 7dd combined
    # ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg", 166), # 7dd bkg
    # ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_4_7_14_dpf", 166), # 4-7-14 dpf
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/TAN_lesion", 40),   # tan
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/otog", 166),        # otog
    # ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_finless", 166), # 7dd finless
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    SAMPL_analysis(root, fr)