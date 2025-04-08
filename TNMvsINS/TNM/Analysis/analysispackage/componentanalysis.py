import scipy.optimize as opt
import pandas as pd
import numpy as np

def linear_combination(x, *params):
    return sum(p * xi for p, xi in zip(params, x))

def residuals(params, x, y):
    return np.average((y - linear_combination(x, *params))**2)

def find_linear_combination(x, y):
    initial_guess = [1/x.shape[0]] * x.shape[0]
    res = opt.minimize(residuals, x0=initial_guess, args=(x, y), bounds=([[0,1]]*x.shape[0]))
    return res.x

def Analyze(
    df,
    train_mask=None,
    true_c_concentrations=None,
    normalize=True,
):
    """
    Analyze the data frame and return a new data frame with the results.

    Parameters
    ----------
    df : pandas.DataFrame
        The input data frame.
    
    Returns
    -------
    pandas.DataFrame
        The new data frame with the results.
    """
    # Calculate the linear combination of the columns
    x = np.array(df.values.T)
    if train_mask is None:
        x_train = x
        true_c_train = np.array(true_c_concentrations)
    else:
        x_train = x[train_mask]
        true_c_train = np.array(true_c_concentrations)[train_mask]

    coeffss = []
    for _x in x:
        coeffs = find_linear_combination(x_train, _x)
        coeffss.append(coeffs)
    coeffs = np.array(coeffss)
    if normalize:
        coeffs = coeffs/np.sum(coeffs, axis=1)[:, np.newaxis]
    determined_c = np.sum(coeffs*true_c_train, axis=1)

    # Create a new DataFrame with the results
    fitting_df = pd.DataFrame(coeffs, index=df.columns, columns=df.columns[train_mask])
    predicted_df = pd.DataFrame(determined_c, index=df.columns, columns=['Carbon Portion'])

    return fitting_df, predicted_df

