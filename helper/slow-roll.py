""" A collection of slow rolling/rolling-apply functions """ 

from numpy.lib.stride_tricks import as_strided 
import numpy as np
import pandas as pd

def roll(df: pd.DataFrame, window: int, **kwargs):
    # window: number of periods to roll back

    v = df.values
    d0, d1 = v.shape
    s0, s1 = v.strides

    # memory efficient
    array3d = as_strided(v, (d0 - (window-1), window, d1), (s0, s0, s1))

    # # this is a little slower
    # rolled_df = pd.concat({
    #     row: pd.DataFrame(values, columns=df.columns)
    #     for row, values in zip(df.iloc[window-1:,].index, array3d)
    # })

    # this is a little faster
    a,b,c = array3d.shape    
    rolled_df = pd.DataFrame(
        array3d.transpose(2,0,1).reshape(c,-1).T,
        index = pd.MultiIndex.from_arrays(
            [np.repeat(df.iloc[window-1:,].index, b), 
             np.tile(np.arange(b), a)]),
        columns = df.columns
    )
    
    return rolled_df.groupby(level=0, **kwargs)

# # how to use
# roll(df, window).apply(your_function)
# roll(df, window).mean()


def groll(df: pd.DataFrame, window: int): # generator
    # window: number of periods to roll back
    for i in range(df.shape[0] - window + 1):
        yield pd.DataFrame(df.values[i:i+window, :], 
                           df.index[i:i+window], 
                           df.columns)

# # how to use
# [your_function(subdf, arg1, arg2, ...) for subdf in groll(df, window)]


def rolling_apply(df: pd.DataFrame, window: int, func, *args):
    # window: number of periods to roll back
    # func: must be an aggregate function
    # *args: arguments to func if there are any
    
    res = pd.Series(np.nan, index=df.index)
    for i in range(1, len(df)):
        subdf = df.iloc[max(i-window, 0):i, :] # get a subsample, excluding the ith index since iloc starts at 0
        idx = df.index[i] # there is no forward looking bias since the ith index does not show up in subdf
        res[idx] = func(subdf, *args)
    return res

# # how to use
# rolling_apply(df, window, your_function, arg1, arg2)

