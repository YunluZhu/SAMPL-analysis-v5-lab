def get_figure_dir(pick_data):
    save_figures_to = f'/Users/yunluzhu/Documents/Lab2/Data/VF_ana/Figures/{pick_data}'
    return save_figures_to

def get_data_dir(pick_data):

    if pick_data == '7dd_40':
        root = "/Volumes/LabData/VF_data_in_use/40hz_data/7dd"
        fr = 40
    # elif pick_data == 'hets':
    #     root = "/Volumes/LabData/VF_data_in_use/NefmaV3/7DD_hets_v3"
    #     fr = 40
    elif pick_data == '7ld_40':
        root = "/Volumes/LabData/VF_data_in_use/40hz_data/7LD"
        fr = 40
    elif pick_data =='4dd_40':
        root = "/Volumes/LabData/VF_data_in_use/40hz_data/4DD"
        fr = 40
    # elif pick_data == 'master':
    #     root = "/Volumes/LabData/VF_data_in_use/NefmaV3/7dd_master_v3"
    #     fr = 40
    elif pick_data == 's':
        root = "/Volumes/LabData/VF_data_in_use/40hz_data/STauV3"
        fr = 40
    elif pick_data =='hc':
        root = '/Volumes/LabData/VF_data_in_use/VF_HairCell_V3'
        fr = 40
    elif pick_data == 'lesion':
        root = "/Volumes/LabData/VF_data_in_use/40hz_data/LesionV4"
        fr = 40
    elif pick_data =='hc4':
        root = '/Volumes/LabData/VF_data_in_use/LateralLineHairCell/HC_highFR/HC_organized 220429'
        fr = 166
        
    # tau data
    elif pick_data =='tau_long':
        root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Longitudinal/long_organized'
        fr = 166
    elif pick_data == 'tau_bkg':
        root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Tau background/_analyzed'
        fr = 166
    elif pick_data == 'tau_selected':
        root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Tau selected'
        fr = 166
        
        
          
    # single fish data
    elif pick_data =='sf':
        root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified'
        fr = 166
    elif pick_data =='sf ana':
        root = '/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF quantified ana'
        fr = 166
    elif pick_data == 'sfld_combined':
        root = "/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF LD/SF_LD_combined_7-9only_100mb-plus-singlefish"
        fr = 166
    elif pick_data == 'sf DD':
        root = "/Volumes/LabData/VF_data_in_use/NefmaV4/Single Fish/SF DD/SF quantified"
        fr = 166
        
        
    elif pick_data == 'lddd':
        root = '/Volumes/LabData/VF_data_in_use/40hz_data/LD DD'
        fr = 40
        
        
    elif pick_data == 'wt_daylight':
        root = '/Volumes/LabData/VF_data_in_use/wt_daylight/organized'
        fr = 166
    elif pick_data == 'wt_fin':
        root = "/Volumes/LabData/VF_data_in_use/wt_daylight_finless"
        fr = 166

        

    # dataset for method paper
    # elif pick_data == 'for_paper_wt':
    #     root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/WT_DD_7dpf"
    #     fr = 166
    elif pick_data == 'for_paper_tan':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/TAN_lesion"
        fr = 40    
    elif pick_data == 'for_paper_otog':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method/behavior data/partial_data_for_ana/otog"
        fr = 166
    elif pick_data == '7dd_bkg':
        root = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_bkg"
        fr = 166        
    elif pick_data == '7dd_all':
        root = "/Volumes/LabData/manuscript data/2022-09 VF method v2/behavior data/DD_7dpf_combined"
        fr = 166       
        
        
    elif pick_data == 'pc_lesion':
        root = "/Volumes/LabData/VF_FA_PCLesion/YZ_code"
        fr = 40    
    # elif pick_data == 'tan_lesion':
    #     root = "/Volumes/LabData/VF_FA KRH_Tan_lesion/FA"
    #     fr = 40        
    elif pick_data == 'tan_lesion':
        root = "/Volumes/LabData/VF_FA KRH_Tan_lesion/reorganized"
        fr = 40    
    elif pick_data == 'otog':
        root = "/Volumes/LabData/manuscript data/2022-07 VF method archived for depth paper/behavior data/partial_data_for_ana/otog"
        fr = 166
    elif pick_data == 'blind':
        root = "/Volumes/LabData/VF_KRH_blind/reorganized"
        fr = 40    
        
        
    elif pick_data == 'tmp':
        root = input("Dir? ")
        fr = int(input("Frame rate? "))
        
    return root, fr