from statsmodels.tsa.stattools import adfuller
import pandas as pd

def check_for_stationarity(x: pd.Series, sig_level=0.01):
    """
    Check if a time series is stationary or not using adfuller.
    H0: unit root exists (non-stationary). 
    So if the p-value is less than `sig_level`, we'd reject H0 and conclude the 
    series is stationary.

    Parameters
    ----------
    x : Series
    sig_level : float
        Significance level below which we'd conclude statistical significance.
    """

    pval = adfuller(x)[1]
    if pval < sig_level:
        print('p-value = ' + str(pval) + ' The series ' + x.name +' is likely stationary.')
        return True
    else:
        print('p-value = ' + str(pval) + ' The series ' + x.name +' is likely non-stationary.')
        return False

# # test
# params = (0, 1)
# T = 100
# 
# A = pd.Series(index=range(T))
# A.name = 'A'
# 
# for t in range(T):
#       A[t] = np.random.normal(params[0], params[1])
#
# check_for_stationarity(A)