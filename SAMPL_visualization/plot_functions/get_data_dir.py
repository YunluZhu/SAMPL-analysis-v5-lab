from pathlib import Path

def get_figure_dir(pick_data):
    figures_base_path = Path.cwd() / 'figures' 
    # Construct the final path string
    save_figures_to = str(figures_base_path / pick_data)
    return save_figures_to

def get_data_dir(pick_data, which_storage='LabDataPro'):
    data_dic = {
        'long21':       [f'/Volumes/{which_storage}/SAMPL_data_v5/long_7-21/Tau EGFP longitudinal/_analyzed', 166],
        'tagRFP1':   [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/TagRFP Control no Tau/_analyzed', 166],
        'tagRFP2':  [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/TagRFP Control no Tau/_analyzed_3cond', 166],
        'gfp_ctrl': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/N EGFP N no Tau/analyzed', 166],

        "depth_7d":       [f"/Volumes/{which_storage}/manuscript data/2022-11 depths exploration/behavior data/DD_07dpf",166],
        "depth_otog":     [f"/Volumes/{which_storage}/manuscript data/2022-11 depths exploration/behavior data/otog",166],
        "depth_tan":      [f"/Volumes/{which_storage}/manuscript data/2022-11 depths exploration/behavior data/TAN_lesion",40],
        "vs": [f"/Volumes/{which_storage}/SAMPL_data_v5/Nefma/VS_lesion", 40],
        'depth_fin':  [f"/Volumes/{which_storage}/manuscript data/2024 Vertical Navigation/behavior data/fin_amputation", 166],
        'wt_dl':      [f"/Volumes/{which_storage}/SAMPL_data_v5/WT_daylight_2025/wt_2025", 166],
        
        'sampl_7d':     [f"/Volumes/{which_storage}/manuscript data/2023-01 SAMPL/original uncompressed data/behavior data/DD_7dpf", 166],
        
        'tau_bkg':      [f"/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau background/_analyzed", 166],
        'tau_long':     [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau nlsMC longitudinal/long_organized', 166],
        'rtau2023':  [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau TagRFP 2023 Fall/_analyzed', 166],
        
        "wt_age":       [f"/Volumes/{which_storage}/manuscript data/2023-01 SAMPL/original uncompressed data/behavior data/DD_4_7_14_dpf", 166],
        "wt_uv":    [f"/Volumes/{which_storage}/SAMPL_data_v5/UV", 166],
        
        # navigation paper dataset
        # 'otog_ld': [f"/Volumes/{which_storage}/manuscript data/2024 Vertical Navigation/unused behavior data/otog_ld", 166],
        'blind': [f"/Volumes/{which_storage}/SAMPL_data_v5/data_from_others/KRH_blind", 40],
        'nMLF': [f'/Volumes/{which_storage}/SAMPL_data_v5/nMLF_lesion/nMLF_abla', 166],
        'nMLF_axon': [f'/Volumes/{which_storage}/SAMPL_data_v5/nMLF_lesion/nMLF_axon_abla', 166],
        'nMLF_axon_ld': [f'/Volumes/{which_storage}/SAMPL_data_v5/nMLF_lesion/nMLF_axon_abla_LD', 166],
        'nMLF_small': [f'/Volumes/{which_storage}/SAMPL_data_v5/nMLF_lesion/nMLF_small_neuron_abla', 166],
        "otog":     [f"/Volumes/{which_storage}/SAMPL_data_v5/otog",166],
        'otog_ld': [f"/Volumes/{which_storage}/SAMPL_data_v5/otog_ld", 166],
        # "otog_reana":   [f"/Volumes/{which_storage}/manuscript data/2022-11 depths exploration/behavior data/otog_reana", 166],
        # "tan":      [f"/Volumes/{which_storage}/manuscript data/2024 Vertical Navigation/behavior data/TAN_lesion",40],
        'tan_axon': [f"/Volumes/{which_storage}/SAMPL_data_v5/TAN_lesion/axonal_lesion/_analyzed", 166],
        # 'vs':   [f"/Volumes/{which_storage}4TB/VF_data_in_use/40hz_data/LesionV4",40],
        
        
        'gfp_ctrl': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/N EGFP N no Tau/analyzed', 166],
        'rfp_ctrl': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/N TagRFP Control no Tau/_analyzed', 166],
        'tau301':  [f'/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau301 EGFP/Tau301 2024/analyzed', 166],
        'gtau': ['/Volumes/LabDataPro/SAMPL_data_v5/Nefma/Tau EGFP/Tau EGFP 2023 Fall/_analyzed', 166],
        'a_rtau': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to RTau/_byRepeat", 166],
        'a_rtau_box': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to RTau/_byBox", 166],
        'a_gtau_box': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to GTau/_byBox", 166],
        'a_gtau': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to GTau/_analyzed", 166],
        
            
        # 'a_rtau': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to RTau/_byRepeat", 166],
        # 'a_rtau_box': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to RTau/_byBox", 166],
        # 'a_gtau': [f"/Volumes/{which_storage}/SAMPL_data_v5/Astrocyte to GTau/_analyzed", 166],
        
        
        'otog_lightR': [f"/Volumes/{which_storage}/SAMPL_data_v5/otog_lightDir/side light/_analyzed", 166],
        'wt_lightR': [f"/Volumes/{which_storage}/SAMPL_data_v5/wt_lightDir/_analyzed", 166],
        
        'creTau9': [f"/Volumes/{which_storage}/SAMPL_data_v5/NefmaLRL-CreER-GTau/D7-9", 166],
        'creTau8': [f"/Volumes/{which_storage}/SAMPL_data_v5/NefmaLRL-CreER-GTau/D7-8", 166],
        'creTauSep': [f"/Volumes/{which_storage}/SAMPL_data_v5/NefmaLRL-CreER-GTau/D sep", 166],
        'creTau9only': [f"/Volumes/{which_storage}/SAMPL_data_v5/NefmaLRL-CreER-GTau/D9", 166],
        
        '24gtau1': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau 24h/Tau EGFP longitudinal/_analyzed', 166],
        '24gtau-fall': [f''],
        '24rtau-n': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau 24h/Tau TagRFP to N longitudinal/_analyzed', 166],
        '24rtau-fall': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau 24h/Tau TagRFP 2023Fall', 166],
        '24rtau-wt': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau 24h/Tau TagRFP to WT longitudinal/_analyzed', 166],
        '24oritau': [f'/Volumes/{which_storage}/SAMPL_data_v5/Nefma/Tau 24h/Tau nlsMcherry', 166],
        
        # 'sldp': [f'/Volumes/{which_storage}/SAMPL_data_v5/SlDp/240304', 166],
        
        'hc': [f'/Volumes/{which_storage}/SAMPL_data_v5/HairCell/HC_highFR_V5', 166],
        'fin_ld': [f"/Volumes/{which_storage}/SAMPL_data_v5/wt_daylight_finless", 166],
        # 'PC_LD':    [f"/Volumes/{which_storage}/SAMPL_data_v5/PClesion/14dpf_LD_FA_data", 166],
        "Sam_lesion": [f"/Volumes/{which_storage}/SAMPL_data_v5/data_from_others/SD_LesionData/5dpf_DDbehavior", 166],
        "meclizine": [f'/Volumes/{which_storage}/SAMPL_data_v5/data_from_others/SD_Meclizine/250402 data', 166],
        
        "wt_light_long": [f"/Volumes/{which_storage}/SAMPL_data_v5/WT_daylight_2025 long/organized", 166],
        'sldp2025': [f'/Volumes/{which_storage}/SAMPL_data_v5/SLDP2025/organized', 166],
        'pclesion': [f'/Volumes/{which_storage}/SAMPL_data_v5/data_from_others/FA_PClesions', 166],
    }
    

    if pick_data == 'tmp':
        root = input("directory: ")
        fr = int(input("frame rate: "))
    else: 
        root = data_dic[pick_data][0]
        fr = data_dic[pick_data][1]
    return root, fr