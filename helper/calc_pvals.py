from typing import Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller, jarque_bera

def coint_pval(
    s1: Union[np.ndarray, np.array, pd.Series], 
    s2: Union[np.ndarray, np.array, pd.Series]): 
    """ Test if two series are cointegrated and returns the p-value. """
    return coint(s1, s2)[1]

def stationarity_pval(s: Union[np.ndarray, np.array, pd.Series]):
    """ Test if a series is stationary and returns the p-value. """
    return adfuller(s)[1]

def normality_pval(s: Union[np.ndarray, np.array, pd.Series]):
    """ Test if a series is normally distributed and returns the p-value. """
    return jarque_bera(s)[1]