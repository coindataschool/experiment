from typing import Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import pearsonr, spearmanr

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

def pearsonr_pval(
    s1: Union[np.ndarray, np.array, pd.Series], 
    s2: Union[np.ndarray, np.array, pd.Series]):
    """ Test if two series are correlated (pearson) and returns the p-value. """
    return pearsonr(s1, s2)[1] # the 0th element is the pearson correlation

def spearmanr_pval(
    s1: Union[np.ndarray, np.array, pd.Series], 
    s2: Union[np.ndarray, np.array, pd.Series]):
    """ Test if two series are correlated (spearman) and returns the p-value. """
    return spearmanr(s1, s2)[1] # the 0th element is the spearman correlation