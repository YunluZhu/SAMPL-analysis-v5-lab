def get_figure_dir(pick_data):
    save_figures_to = f'/Users/yunluzhu/Documents/Lab2/Data/vf_ana/Figures/{pick_data}'
    return save_figures_to

def get_data_dir(pick_data):
    data_dic = {
        "7d":       ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_07dpf",166],
        "7dall":    ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_7dall",166],
        "otog":     ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/otog",166],
        "tan":      ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/TAN_lesion",40],
        
        "lesion":   ["/Volumes/LabData/vf_data_in_use/40hz_data/LesionV4",40],
        "hc4":      ['/Volumes/LabData/vf_data_in_use/LateralLineHairCell/HC_highFR/HC_organized 220429',166],
        "tau_long": ['/Volumes/LabData/vf_data_in_use/NefmaV4/Longitudinal/long_organized',166],
        "tau_bkg":  ['/Volumes/LabData/vf_data_in_use/NefmaV4/Tau background/_analyzed',166],
        "sfld":     ['/Volumes/LabData/vf_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified',166],
        "sfld_combined":    ["/Volumes/LabData/vf_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish",166],
        "sfdd":     ["/Volumes/LabData/vf_data_in_use/NefmaV4/Single Fish/SF DD/SF quantified",166],
        
        "wt_daylight": ["/Volumes/LabData/vf_data_in_use/wt_daylight/organized",166],
        "wt_fin":   ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/fin_amputation",166],
        
        "KDmut":    ["/Volumes/LabData/vf_KD/raw data/YZ_reorg",40],
        "SD_LL":    ['/Volumes/T7_SD/LateralLine_project/Data/vf_data/LL_organized_byGroup',166],
 
        "blind":    ["/Volumes/LabData/vf_KRH_blind/reorganized",40],

        "ab_age":   ["/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_4_7_14_dpf",166],
        "wt_bkg":   ["/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg",166],
 
    }
    if pick_data == 'tmp':
        root = input("directory: ")
        fr = int(input("frame rate: "))
    else: 
        root = data_dic[pick_data][0]
        fr = data_dic[pick_data][1]
    return root, fr