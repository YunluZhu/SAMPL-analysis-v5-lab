# %%
from vf_analysis_by_folder_v4 import vf_analysis
from bout_analysis.logger import log_vf_ana

# %%

LoF_logger = log_vf_ana('vf_dataList_log')
logger = log_vf_ana('vf_ana_log')
logger.info("Called from vf_analysis_list_of_folders.py")

# %%
list_of_root_folders = [
    # ("/Volumes/LabData/VF_data_in_use/wt_daylight/organized", 166),                  # WT daylight
    # ("/Volumes/LabData/VF_data_in_use/wt_daylight_finless", 166),                    # WT finless
    
    # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Longitudinal/long_organized", 166),           # Tau long
    # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified all", 166),   # Tau SF
    # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified ana", 166),   # SF ana
    # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish", 166), # Tau SF combined
        # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish/ld_tau", 166)
    # ("/Volumes/LabData/VF_data_in_use/NefmaV4/Tau background/_analyzed", 166),              # Tau bkg
    ('/Volumes/LabData/VF_data_in_use/40hz_data/LesionV4',40),                              # VS lesion
    
    # ("/Volumes/LabData/VF_KRH_blind/reorganized", 40),               # blind - KRH
    
    # ("/Volumes/LabData/VF_data_in_use/LateralLineHairCell/HC_highFR/HC_organized 220429", 166),   # Hair cell
    
    # dataset used in method paper
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/WT_DD_7dpf", 166),  # wt
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/TAN_lesion", 40),   # tan
    # ("/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/otog", 166),        # otog
]

for (root, fr) in list_of_root_folders:
    LoF_logger.info(f"Analyzed dataset: {root}")
    vf_analysis(root, fr)