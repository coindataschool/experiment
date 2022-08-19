import pandas as pd
import numpy as np
from typing import Union

def equal(
    a: Union[pd.DataFrame, pd.Series, np.ndarray], 
    b: Union[pd.DataFrame, pd.Series, np.ndarray]):
    """ 
    Check if the corresponding values of two data frames or series or numpy arrays are the same.
    """
    return (abs(a - b) > 1e-8).sum().sum() == 0 # 0 means same values