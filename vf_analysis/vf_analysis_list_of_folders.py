from vf_analysis_by_folder_v4 import vf_analysis

list_of_root_folders = [
    ("/Volumes/LabData/VF_data_in_use/wt_daylight/organized", 166),                  # WT daylight
    ("/Volumes/LabData/VF_data_in_use/wt_daylight_finless", 166),                    # WT finless
    
    ("/Volumes/LabData/VF_data_in_use/NefmaV4/Longitudinal/long_organized", 166),           # Tau long
    ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified all", 166),   # Tau SF
    ("/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish", 166), # Tau SF combined
    ("/Volumes/LabData/VF_data_in_use/NefmaV4/Tau background/_analyzed", 166),              # Tau bkg
    
    ("/Volumes/LabData/VF_FA KRH_Tan_lesion/reorganized", 40),   # Tan lesion - FA
    ("/Volumes/LabData/VF_KR_blind/reorganized", 40),            # blind - KRH
    
    ("/Volumes/LabData/VF_data_in_use/LateralLineHairCell/HC_highFR/HC_organized 220429", 166),   # Hair cell
]

for (root, fr) in list_of_root_folders:
    vf_analysis(root, fr)