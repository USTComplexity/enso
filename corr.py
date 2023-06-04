import numpy as np


def seasonal_mean(x, period):
    
    if len(x) % period:
        raise ValueError("x has non-integer multiple of periods.")

    n = len(x) // period
    
    folded = x.reshape(n, period)
    mean = np.mean(folded, axis=0)
    mean_tiled = np.tile(mean, n)

    return mean_tiled


def cross_corr(x, y, period=None, norm=False):

    x = x.copy()
    y = y.copy()

    if period is not None:
        x -= seasonal_mean(x, period)
        y -= seasonal_mean(y, period)
    else:
        x -= np.mean(x)
        y -= np.mean(y)

    corr = np.correlate(x, y, mode="full")

    if norm:
        unity = np.ones(len(x))
        norm = np.correlate(unity, unity, mode="full")

        corr /= norm * np.std(x) * np.std(y)

    lags = np.arange(-len(x)+1, len(x))

    return lags, corr


def moving_avg(x, n):
    
    winodw = np.ones(n) / n
    ma = np.convolve(x, winodw, mode="same")

    return ma
