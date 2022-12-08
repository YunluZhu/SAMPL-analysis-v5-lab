# %%
from SAMPL_analysis_by_folder_v4 import SAMPL_analysis
from bout_analysis.logger import log_SAMPL_ana

# %%

LoF_logger = log_SAMPL_ana('SAMPL_dataList_log')
logger = log_SAMPL_ana('SAMPL_ana_log')
logger.info("Called from SAMPL_analysis_list_of_folders.py")

# %%
list_of_root_folders = [
    # ("/Volumes/LabData/SAMPL_data_in_use/wt_daylight/organized", 166),                  # WT daylight
    # ("/Volumes/LabData/SAMPL_data_in_use/wt_daylight_finless", 166),                    # WT finless
    
    # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Longitudinal/long_organized", 166),           # Tau long
    # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified all", 166),   # Tau SF
    # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified ana", 166),   # SF ana
    # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish", 166), # Tau SF combined
        # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish/ld_tau", 166)
    # ("/Volumes/LabData/SAMPL_data_in_use/NefmaV4/Tau background/_analyzed", 166),              # Tau bkg
    # ('/Volumes/LabData/SAMPL_data_in_use/40hz_data/LesionV4',40),                              # VS lesion
    
    # ("/Volumes/LabData/SAMPL_KRH_blind/reorganized", 40),               # blind - KRH
    
    # ("/Volumes/LabData/SAMPL_data_in_use/LateralLineHairCell/HC_highFR/HC_organized 220429", 166),   # Hair cell
    
    # dataset used in method paper
    ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined/7dd_bkgcombined", 166),  # 7dd combined
    ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg", 166), # 7dd bkg
    ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_4_7_14_dpf", 166), # 4-7-14 dpf
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/TAN_lesion", 40),   # tan
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/otog", 166),        # otog
    ("/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_finless", 166), # 7dd finless
    
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    SAMPL_analysis(root, fr)