def get_figure_dir(pick_data):
    save_figures_to = f'<your figure folder>/{pick_data}'
    return save_figures_to

def get_data_dir(pick_data):
    data_dic = {
        "<your data name>":      ["<yout data location>",166],
        "hc": ["/Volumes/LabDataPro/SAMPL_data_v5/HairCell/HC_highFR_V5", 166],
        # see example below:
        # "7dall":    ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/DD_7dall",166],
        # "tan":      ["/Volumes/LabData/manuscript data/2022-11 depths exploration/behavior data/TAN_lesion",40],
    }
    if pick_data == 'tmp':
        root = input("directory: ")
        fr = int(input("frame rate: "))
    else: 
        root = data_dic[pick_data][0]
        fr = data_dic[pick_data][1]
    return root, fr