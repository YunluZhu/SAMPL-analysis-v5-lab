#%%
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import seaborn as sns


def custom_sigmoid_fit(
    xval,
    yval,
    func,
    x0,
    lower_bounds,
    upper_bounds,
    x_range=None,
    **kwargs,
):
    """fit with a curve function optimized for 4 coefs. 

    Args:
        xval (_type_): x values
        yval (_type_): y values
        x_range_to_calc (_type_): The range of x values for calculating fitted curve
        func (_type_): Function to fit the data with
        x0 (list, optional): _description_.
        lower_bounds (list, optional): Lower bounds for each coef. 
        upper_bounds (list, optional): Upper bounds for each coef. 

    Returns:
        pd.DataFrame: a dataframe of fitted coefs
        pd.DataFrame: a dataframe of y values calculated according to the fitted curve given x_range
        float: p_sigma
    """
    
    for key, value in kwargs.items():
        if key == "x_range":
            x_range = value
        elif key == "a":
            x0[0] = value
            lower_bounds[0] = value - 0.01
            upper_bounds[0] = value + 0.01
        elif key == "b":
            x0[1] = value
            lower_bounds[1] = value - 0.01
            upper_bounds[1] = value + 0.01
        elif key == "c":
            x0[2] = value
            lower_bounds[2] = value - 0.01
            upper_bounds[2] = value + 0.01
        elif key == "d":
            x0[3] = value
            lower_bounds[3] = value - 0.01
            upper_bounds[3] = value + 0.01
    if not x_range:
        x_range = [xval.min(), xval.max()]

    p0 = tuple(x0)
    popt, pcov = curve_fit(
        func,
        xval,
        yval,
        # maxfev=10000,
        p0=p0,
        bounds=(lower_bounds, upper_bounds),
    )
    _x = np.linspace(x_range[0], x_range[1], 100)
    y = func(_x, *popt)
    output_coef = pd.DataFrame(data=popt).transpose()
    output_fitted = pd.DataFrame(data=y).assign(x=_x)
    p_sigma = np.sqrt(np.diag(pcov))
    return output_coef, output_fitted, p_sigma


def sigfunc_4free(x, a, b, c, d):
    """Sigmoid function with 4 degrees of freedom
    """
    y = c + (d) / (1 + np.exp(-(a * (x + b))))
    return y

# %% construct some data
DATA_NUMBER = 500
x_imp = np.linspace(-100,100,DATA_NUMBER)
delta_x = (np.random.rand(DATA_NUMBER) - 0.5) * 10
coef_imp=[0.2, 5, -3, 1]
y_imp = sigfunc_4free(x_imp, *coef_imp)
delta_y = (np.random.rand(DATA_NUMBER) - 0.5) * 1

x_input = x_imp+delta_x
y_input = y_imp+delta_y

df = pd.DataFrame(
    data={'x':x_input,
          'y':y_input}
)
#%% fit
coef, fitted_y, sigma = custom_sigmoid_fit(
    df.x,
    df.y,
    sigfunc_4free,
    coef_imp,
    lower_bounds=[0.1, 0, -100, 1],
    upper_bounds=[10, 20, 2, 100],
    x_range=None,
)
# %%
g = sns.scatterplot(
    data=df,
    x='x',
    y='y',
    color='gray',
    alpha=0.4
)
g = sns.lineplot(
    data=fitted_y,
    x='x',
    y=0
)
# %%
