import numpy as np


def seasonal_mean(x, period):
    
    if len(x) % period:
        raise ValueError("x has non-integer multiple of periods.")

    n = len(x) // period
    dim = len(x.shape)

    folded = np.array(np.vsplit(x, n))
    mean = np.mean(folded, axis=0)
    mean = np.tile(mean, [n] + [1 for _ in range(dim - 1)])

    return mean


def cross_corr(x, y, period=None, norm=True):

    x = x.copy()
    y = y.copy()

    if period is not None:
        x -= seasonal_mean(x, period)
        y -= seasonal_mean(y, period)
    else:
        x -= np.mean(x)
        y -= np.mean(y)

    corr = np.correlate(x, y, mode="full")

    # Normalization
    if norm:
        unity = np.ones(len(x))
        norm = np.correlate(unity, unity, mode="full")

        corr /= norm * np.std(x) * np.std(y)

    lags = np.arange(-len(x)+1, len(x))

    return lags, corr


def cross_corr_seasonal(x, y, period=365, tmin=0, tmax=1460):
    
    if len(x) % period:
        raise ValueError("x has non-integer multiple of periods.")
    
    n = len(x) // period

    x_folded = x.reshape(n, period)
    x_folded -= np.mean(x_folded, axis=0)
    
    y_folded = y.reshape(n, period)
    y_folded -= np.mean(y_folded, axis=0)
    
    corr = np.zeros((period, tmax - tmin))

    for t in range(0, period):
        for s in range(tmin, tmax):

            x0 = - min((t+s) // period, 0)
            x1 = n - max((t+s) // period, 0)

            y0 = max((t+s) // period, 0)
            y1 = n + min((t+s) // period, 0)

            u = (t+s) % period

            x_ = x_folded[x0:x1, t]
            y_ = y_folded[y0:y1, u]
            
            corr[t, s - tmin] = np.mean(x_ * y_) / (np.std(x_) * np.std(y_))

    return corr


def moving_avg(x, n):
    
    winodw = np.ones(n) / n
    ma = np.convolve(x, winodw, mode="same")

    return ma
