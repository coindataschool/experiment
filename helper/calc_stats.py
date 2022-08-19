from typing import Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def coint_pval(
    s1: Union[np.ndarray, np.array, pd.Series], 
    s2: Union[np.ndarray, np.array, pd.Series]): 
    _, pval, _ = coint(s1, s2)
    return pval

def calc_beta_simple_linreg(y: np.ndarray, x: np.ndarray):
    x = np.vstack((np.ones_like(x), x)) # add a column of 1's, intercept
    b = np.linalg.pinv(x.dot(x.T)).dot(x).dot(y)
    return b[1]
