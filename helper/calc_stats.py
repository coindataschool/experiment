from typing import Union
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

def coint_pval(
    s1: Union[np.ndarray, np.array, pd.Series], 
    s2: Union[np.ndarray, np.array, pd.Series]): 
    _, pval, _ = coint(s1, s2)
    return pval