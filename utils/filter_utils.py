# https://stackoverflow.com/questions/37556487/remove-spikes-from-signal-in-python
# pyright: reportGeneralTypeIssues=false
import numpy as np
import pandas as pd
import scipy.signal as ss
from numba import jit  # Optional, see below.


def clip_data(unclipped):
    """
    Clip unclipped data over value of high_clip
    unclipped contains a single column of unclipped data.
    """

    # convert to np.array to access the np.where method
    np_unclipped = np.array(unclipped.iloc[:, 1])

    # select highest 1 percent data
    perc = int(2 / 100 * np_unclipped.shape[0])
    highest_values_10per = np.partition(np_unclipped, -perc)[-perc]

    # clip data above the lowest highest 1 percent, with a factor.
    # This should remove extreme peaks that cannot be handled by the fb_ewma routine.
    mean = np.mean(np_unclipped)
    cond_high_clip = np.logical_or((np_unclipped > highest_values_10per * 2.5),(np_unclipped < -abs(np_unclipped[2])*1.1))

    unclipped.iloc[:, 1] = np.where(cond_high_clip, np.nan, np_unclipped).flatten()
    return unclipped


@jit(nopython=True)  # Optional, see below.
def ewma_generator(vals, decay_vals):
    """Adapted ewma generator for time series"""
    result = vals[0]
    yield result
    for val, decay in zip(vals[1:], decay_vals[1:]):
        result = result * (1 - decay) + val * decay
        yield result


def ewma_time_series(s, values, span_frac):
    # Assumes time series data is now named `s`.
    dt = pd.Series(s).diff().abs()
    window = max(s.max() * span_frac, dt.min()*10)
    # window = dt.max()

    decay = 1 - (dt / -window).apply(np.exp)
    g = ewma_generator(values.values, decay.values)
    result = np.array([next(g) for _ in range(len(s))])
    return result


def ewma_fb_time_series(df, span_frac):
    """Apply forwards, backwards exponential weighted moving average (EWMA) to df_column."""
    df.iloc[:, 1] = df.iloc[:, 1].interpolate()

    # Forwards EWMA.
    fwd = ewma_time_series(df.iloc[:, 0], df.iloc[:, 1], span_frac)

    # Backwards EWMA.
    bwd = ewma_time_series(df.iloc[::-1,0], df.iloc[::-1, 1], span_frac)

    # Add and take the mean of the forwards and backwards EWMA.
    try:
        stacked_ewma = np.hstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=1)
    except:
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=0)

    return fb_ewma


def ewma_fb(df, span_frac):
    """Apply forwards, backwards exponential weighted moving average (EWMA) to df_column."""
    span = max(span_frac * df.shape[0], 10)

    # Forwards EWMA.
    fwd = pd.Series.ewm(df.iloc[:, 1], span=span, adjust=True).mean()

    # Backwards EWMA.
    bwd = pd.Series.ewm(df.iloc[::-1, 1], span=span, adjust=True).mean()

    # Add and take the mean of the forwards and backwards EWMA.
    try:
        stacked_ewma = np.hstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=1)
    except:
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=0)

    return fb_ewma

    
def ewma_fb_np(df_column, span_frac):
    """Apply forwards, backwards exponential weighted moving average (EWMA) to df_column."""
    span = max(span_frac * df_column.shape[0], 10)

    # Forwards EWMA.
    fwd = pd.Series.ewm(df_column, span=span, adjust=True).mean()

    # Backwards EWMA.
    bwd = pd.Series.ewm(df_column[::-1], span=span, adjust=True).mean()

    # Add and take the mean of the forwards and backwards EWMA.
    try:
        stacked_ewma = np.hstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=1)
    except:
        stacked_ewma = np.vstack((fwd, bwd[::-1]))
        fb_ewma = np.mean(stacked_ewma, axis=0)

    return fb_ewma


def remove_outliers(spiked_signal, fbewma, delta):
    """Remove data from df_spikey that is > delta from fbewma.
    Delta is used together with the data range of the fbewma for scaling."""

    # ensure data is in numpy formats ans creates condition for the replace
    np_spiked = np.array(spiked_signal)
    np_fbewma = np.array(fbewma)
    cond_delta = np.abs(np_spiked - np_fbewma) > delta * (np.mean(np_fbewma) * 2)

    # remove outliers and replace with nan
    np_remove_outliers = np.where(cond_delta, np.nan, np_spiked)
    return np_remove_outliers


def filter_spiked_signal(original_dataframe, span_frac, delta):
    """
    Filter data by doing a forward and backwards exponential moving average, replace outliers by nans and interpolate.
    """

    # TODO use a ewma that is corrected for time. There where we have peaks of numeric cause, we will see small timesteps as well due to CFL control.
    # Therefore, the required span changes, otherwise peaks might not be observed as peaks because a normal ewma will follow the peak.
    # clipped_data = clip_data(original_dataframe)

    fbewma = ewma_fb_time_series(original_dataframe, span_frac)
    
    no_outliers = pd.DataFrame(remove_outliers(original_dataframe.iloc[:, 1], fbewma, delta)).interpolate()

    # d_filtered = no_outliers.to_numpy().flatten()
    d_filtered = pd.DataFrame(ewma_fb_np(pd.DataFrame(no_outliers).interpolate(), span_frac*2)).interpolate().to_numpy().flatten()

    return d_filtered
