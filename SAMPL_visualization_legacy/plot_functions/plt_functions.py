import matplotlib as mpl
import os
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from plot_functions.plt_tools import set_font_type
import warnings

def plt_categorical_grid(
    data:pd.DataFrame, 
    x_name:str, 
    y_name:str, 
    gridrow:str, 
    gridcol:str, 
    units:str, 
    errorbar=None, 
    sharey=True, 
    sns_palette='colorblind', 
    markertype='d', 
    height=4, 
    aspect=0.8,
    ):
    
    """build upon sns.catplot(), plots mean y_name vs x_name with individual repeats. Repeats are plotted as stripplot if number of repeats are different among groups defined by x_name, otherwise, repeats are connected.
    the errorbar arg requires Seaborn v0.12 to work.

    Args:
        data (pd.DataFrame): Dataframe with one value per "unit". Can be achieved using: data.groupby([x_name, gridcol, gridrow, units]).mean().reset_index()
        x_name (str): Column to plot on the X axis.
        y_name (str): Column to plot on the Y axis. 
        gridrow (str): Categorical variables that will determine the faceting of the grid.
        gridcol (str): Categorical variables that will determine the faceting of the grid.
        units (str): Units for plotting individual repeats. see sns.lineplot() units arg.
        errorbar (optional): Defines type of error bars to plot. see seaborn.catplot errorbar arg. Defaults to None.
        sharey (bool, optional): Whether to share Y axis ticks. Defaults to True.
        sns_palette (str, optional): Color palettes. Defaults to 'colorblind'.
        height (int, optional): Height of the graph in inches. Defaults to 4.
        aspect (float, optional): aspect ratio of the graph. Defaults to 0.8.
    """
    set_font_type()
    data.sort_values(by=x_name,inplace=True)
    
    assert_repeats = len(set(data.groupby([x_name])[units].apply(lambda x: len(x.unique())).values))
        
    g = sns.catplot(
        data = data,
        col = gridcol,
        row = gridrow,
        hue = x_name,
        x = x_name,
        y = y_name,
        kind = 'point',
        sharey = sharey,
        palette = sns_palette,
        errorbar = errorbar,
        markers = [markertype]*len(set(data[x_name])),
        height = height,
        aspect = aspect,
        )
    if assert_repeats == 1:
        g.map(sns.lineplot,x_name,y_name,
            estimator=None,
            units=units,
            data = data,
            sort=False,
            color='grey',
            alpha=0.2,
            zorder=0,
            )
    else:
        g.map(sns.stripplot,x_name,y_name,
            data = data,
            color='lightgrey',
            zorder=0,
            order=data[x_name].unique().sort(),
            )
    g.add_legend()
    sns.despine(offset=10, trim=False)
    return g
