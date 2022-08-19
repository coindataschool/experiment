"""
Fast rolling beta calculation for many stocks against the market.
Equivalent to running a simple linear regression for each stock against 
the market: stock = a + b * market + noise.
"""

import numpy as np
from numpy.lib.stride_tricks import as_strided
import pandas as pd

def calc_beta(y: np.ndarray, x: np.ndarray):
    x = np.vstack((np.ones_like(x), x)) # add a column of 1's, intercept
    b = np.linalg.pinv(x.dot(x.T)).dot(x).dot(y) 
    return b[1] # beta of 1st x, assume simple linear regression

def rolling_calc_beta(y_df: pd.DataFrame, x_df: pd.DataFrame, window: int):
    result = np.ndarray(shape=y_df.shape, dtype=float)
    l, w = y_df.shape
    ls, ws = y_df.values.strides
    result[0:window-1, :] = np.nan
    y_arr = as_strided(y_df.values, shape=(l - window + 1, window, w), strides=(ls, ls, ws))
    x_arr = as_strided(x_df.values, shape=(l - window + 1, window), strides=(ls, ls))
    for row in range(window-1, l):
        result[row, :] = calc_beta(y_arr[row - window + 1, :], x_arr[row - window + 1])
    return pd.DataFrame(data=result, index=y_df.index, columns=y_df.columns)

# # TO-Test:
# y ~ a + b1*x1 + b2*x2 + ... 
# def calc_beta_multiple_linreg(df, yvar, xvars):
#     # yvar: string
#     # xvars: must be a list of strings
    
#     # extra y 
#     Y = np.ascontiguousarray(df.loc[:, yvar]) # contiguous numpy array for speed
    
#     # extract X as a 2D matrix
#     is_X = df.columns.isin(xvars)
#     X = df.values[:, is_X]
#     # prepend a column of ones for the intercept
#     X = np.ascontiguousarray(np.vstack(np.ones_like(X), X)) # contiguous arrray for speed

#     # matrix algebra
#     b = np.linalg.pinv(X.T@X)@X.T@Y # @ is .dot()
#     return pd.Series(b[-len(xvars):], xvars, name='Beta')        