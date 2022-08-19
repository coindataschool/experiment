""" 
A collection of rolling functions that operate on a DataFrame. They are 
designed to be chained with pandas `apply()`. Slow for large datasets.
These functions don't pad NaN rows to make the index identical with the 
DataFrame's index, so the result after `apply()` an aggregate function will 
not contain leading NaNs. This is different from `df.rolling().apply()`, which
alreays returns a series with leading NaNs so that its index and length are 
the same with the DataFrame.
""" 

from numpy.lib.stride_tricks import as_strided 
import numpy as np
import pandas as pd

def roll(df: pd.DataFrame, window: int, **kwargs):
    """
    Create all rolling subframes and group them by time index and return the 
    resulting groupby object for people to call apply() on it. 

    Parameters
    ----------
    df : DataFrame
    window : int
        Number of periods to roll back.
    **kwargs
        Additional arguments for groupby.
    """

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


def groll(df: pd.DataFrame, window: int): 
    """
    Returns a generator that yield each rolling subframe when called.

    Parameters
    ----------
    df : DataFrame
    window : int 
        Number of periods to roll back.
    """
    for i in range(df.shape[0] - window + 1):
        yield pd.DataFrame(df.values[i:i+window, :], 
                           df.index[i:i+window], 
                           df.columns)

# # how to use
# [your_function(subdf, arg1, arg2, ...) for subdf in groll(df, window)]