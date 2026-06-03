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
    # done ("/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau background/_analyzed", 166),
    # done ("/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau EGFP 2023 Fall/_analyzed", 166),
    # done ("/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau TagRFP-EGFP longitudinal 7-21 archive/Tau EGFP longitudinal/_analyzed", 166),
    # ("/Volumes/LabDataPro/SAMPL_data_v5/otog_ld/LD_hets", 166),
    # # ("/Volumes/LabDataPro/SAMPL_data_v5/otog_ld/LD_otog", 166),
    # # ("/Volumes/LabDataPro/SAMPL_data_v5/nMLF_lesion/nMLF_axon_abla", 166),
    # # ("/Volumes/LabDataPro/SAMPL_data_v5/nMLF_lesion/nMLF_axon_abla_LD", 166),
    # # ("/Volumes/LabDataPro/SAMPL_data_v5/wt_daylight_finless", 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/data_from_others/FA_PClesions', 166),
    # ("/Volumes/LabDataPro/SAMPL_data_v5/Astrocyte to GTau/_analyzed", 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/Astrocyte to RTau/_byRepeat', 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/2024_WT_SF/7dd_wtsf', 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/data_from_others/KRH_blind', 40),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/nMLF_lesion/nMLF_small_neuron_abla', 166),
    
    # ('/Volumes/LabDataPro/SAMPL_data_v5/Nefma/N EGFP N no Tau/analyzed_all', 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau EGFP/Tau EGFP 2023 Fall/_analyzed', 166),
    # ('/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau301 EGFP/Tau301 2024/analyzed', 166),
    ('/Volumes/LabDataPro/SAMPL_data_v5/2024_Cre_SF/imaged', 166),
    
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    SAMPL_analysis(root, fr, if_reana=True)
# %%
