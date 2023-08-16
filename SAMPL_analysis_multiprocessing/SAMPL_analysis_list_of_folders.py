'''
- What is this script
    This scripts allows for batch analysis of a list of folders.
- How to use
    "list_of_root_folders" contains a list of tuples containing root directory and frame rate.
'''

# %%
from SAMPL_analysis import SAMPL_analysis_mp
from bout_analysis.logger import log_SAMPL_ana

# %%

LoF_logger = log_SAMPL_ana('SAMPL_dataList_log')
logger = log_SAMPL_ana('SAMPL_ana_log')
logger.info("Called from SAMPL_analysis_list_of_folders.py")

# %%
list_of_root_folders = [
    # ("<here's the root directory>", <here's the frame rate>),
    # see examples below:
    ("/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau EGFP longitudinal/new", 166),
    ("/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau TagRFP to N longitudinal/new", 166),
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    SAMPL_analysis_mp(root, fr)