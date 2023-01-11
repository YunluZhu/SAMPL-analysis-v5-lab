import os

def get_figure_dir(which_figure):
    folder_dir = os.getcwd()
    fig_dir = os.path.join(folder_dir, 'Manuscript figures',which_figure)
    return fig_dir

