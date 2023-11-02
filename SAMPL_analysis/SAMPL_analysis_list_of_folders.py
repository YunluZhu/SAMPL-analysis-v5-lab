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
    ("/Volumes/LabDataPro/manuscript data/2022-11 depths exploration/behavior data/fin_amputation", 166),
    ("/Volumes/LabDataPro/manuscript data/2022-11 depths exploration/behavior data/otog", 166),
    ("/Volumes/LabDataPro/SAMPL_data_v5/HairCell/HC_highFR_V5", 166)
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    SAMPL_analysis(root, fr)