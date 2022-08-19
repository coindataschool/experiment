""" 
A collection of rolling-apply functions that operates on two Series instead of 
a DataFrame. The two Series must have identical index. The (aggregate) function 
to be applied should take these two Series as input. 
""" 

import numpy as np
import pandas as pd

def rolling_apply_pd(
    s1: pd.Series,
    s2: pd.Series, 
    window: int, 
    func, 
    *args):
    """
    Apply an aggregate function to each rolling subset and return the function 
    output in a series. 

    This function uses pandas, which is slow and memory-non-efficent. Not 
    suitable for large datasets.
    
    Parameters
    ----------
    s1, s2 : pandas Series
        Often are two returns series.
    window : int
        Number of periods to roll back.
    func : callable (aggregate) function to apply to each rolling subframe.
    *args 
        Additional arguments for the callable function.
    """
    
    res = pd.Series(np.nan, index=s1.index[window-1:]) # drops the initial nans from the final result
    for i in range(1, len(s1)+1):
        # get a subsample, excluding the ith index since iloc starts at 0
        ss1 = s1.iloc[max(i-window, 0):i]
        ss2 = s2.iloc[max(i-window, 0):i]
        if len(ss1) >= window:
            idx = s1.index[i-1] 
            res[idx] = func(ss1, ss2, *args)
    return res


def rolling_apply_np(
    s1: pd.Series, 
    s2: pd.Series, 
    window: int, 
    func, 
    *args):
    """
    Apply an aggregate function to each rolling subset and return the function 
    output in a series. 

    This function uses numpy `ndarray` and `as_strided`, which are faster and 
    memory-efficent. However, it can still be slow if the implementation of 
    the aggregate function is slow; the aggregate function is the bottleneck. 
    
    Parameters
    ----------
    s1, s2 : pandas Series
        Often are two returns series.
    window : int
        Number of periods to roll back.
    func : callable (aggregate) function to apply to each rolling subframe.
    *args 
        Additional arguments for the callable function.
    """
    from numpy.lib.stride_tricks import as_strided     

    result = np.ndarray(shape=s1.shape[0]-window, dtype=float) # drop init nans from the final results
    l = len(s1)
    ls = s1.values.strides[0]
    # result[0:window-1] = np.nan
    s1_arr = as_strided(s1.values, shape=(l - window + 1, window), strides=(ls, ls))
    s2_arr = as_strided(s2.values, shape=(l - window + 1, window), strides=(ls, ls))
    for row in range(window-1, l):
        result[row] = func(s1_arr[row - window+1], s2_arr[row - window+1])
    return pd.Series(data=result, index=s1.index)