import numpy as np
import numpy.typing as npt
import bottleneck as bn


def corr_2pt_avg(x: npt.ArrayLike, f: npt.ArrayLike, lag_max: int, window: int):
    """
    Compute 2-point correlation matrix averaged all time lag up to lag_max
    """
    
    n_t, n_x, n_y = f.shape
    c = np.zeros([n_t - lag_max - 1 - window, n_x, n_y])

    for s in range(lag_max + 1):
        c += corr_2pt(x, f, s, window)[lag_max + 1 + window - s:,:,:]

    c /= (lag_max + 1)

    print(np.mean(np.abs(c)))

    # Sanity check
    if np.max(c) > 1 or np.min(c) < -1:
        raise ValueError("Something wrong in the correlation matrix")

    return c


def str_2pt(x: npt.ArrayLike, f: npt.ArrayLike, lag_max: int, window: int):
    """
    Compute network strength as defined by Ludescher et al., PNAS 110, 29, 11743 (2013)
    """

    n_t, n_x, n_y = f.shape
    c = np.zeros([n_t - lag_max - 1 - window, n_x, n_y])
    c2 = np.zeros([n_t - lag_max - 1 - window, n_x, n_y])
    c_max = np.zeros([n_t - lag_max - 1 - window, n_x, n_y])
    
    for s in range(lag_max + 1):
        c_s = np.abs(cov_2pt(x, f, s, window)[lag_max + 1 + window - s:,:,:])

        c += c_s
        c2 += c_s * c_s
        c_max = np.maximum(c_max, c_s)

    c /= (lag_max + 1)
    c2 = c2 / (lag_max + 1) - c * c

    print(np.max(c), np.max(np.sqrt(c2)), np.max(c_max))

    return (c_max - c) / np.sqrt(c2)


def corr_2pt(x: npt.ArrayLike, f: npt.ArrayLike, lag: int, window: int):
    """
    Compute 2-point correlation at specific time lag by a moving average
    """
    
    x = x[lag:, np.newaxis, np.newaxis]

    # Edge case: lag == 0 then we can't and don't need to index by -lag 
    if lag != 0:
        f = f[:-lag, :, :]
    
    c = bn.move_mean(a=x*f, window=window, axis=0)
    # print("Here")

    c -= bn.move_mean(a=x, window=window, axis=0) \
       * bn.move_mean(a=f, window=window, axis=0)
    
    c /= bn.move_std(a=x, window=window, axis=0) \
       * bn.move_std(a=f, window=window, axis=0)

    return c


def cov_2pt(x: npt.ArrayLike, f: npt.ArrayLike, lag: int, window: int):
    """
    Compute 2-point covariance at specific time lag by a moving average
    """
    
    x = x[lag:, np.newaxis, np.newaxis]

    # Edge case: lag == 0 then we can't and don't need to index by -lag 
    if lag != 0:
        f = f[:-lag, :, :]
    
    c = bn.move_mean(a=x*f, window=window, axis=0)

    c -= bn.move_mean(a=x, window=window, axis=0) \
       * bn.move_mean(a=f, window=window, axis=0)
    
    return c


def add_edges(edges: list[set], source: tuple[int, int], 
              cmat: npt.ArrayLike, thres: float):
    """
    Add edge by determining whether any connection comming out of the source
    node (specified by lat, lon indices) has exceeded the specified threshold

    This will generate a list of time-dependent edges like
        time, (x_1, y_2), (x_2, y_2)
    """

    txy = np.argwhere(np.abs(cmat) >= thres)

    for point in txy:
        t, x, y = point
        # print(point)

        if (source, (x, y)) in edges[t] or ((x, y), source) in edges[t]:
            continue
        
        edges[t].add((source, (x, y)))
        print(f"added {t}, {source}, {(x, y)}")


def build_network(f: npt.ArrayLike, lag_max: int, thres: float, window: int = 365):
    """
    Build time-dependent correlation network by add edge for each pair-correlation
    that excess the threshold 
    """

    n_t, n_x, n_y = f.shape

    edges = [set() for _ in range(n_t - lag_max + 1)]
    
    for i in range(n_x):
        for j in range(n_y):
            
            print("Working on", i, j)
            cmat = corr_2pt_avg(f[:, i, j], f, lag_max, window)
            add_edges(edges, (i, j), cmat, thres)
    
    return edges


def build_network_pnas(f: npt.ArrayLike, lag_max: int, window: int = 365):
    """
    Build network based on the original methodology from 
    Ludescher et al., PNAS 110, 29, 11743 (2013)
    """

    n_t, n_x, n_y = f.shape

    s = np.zeros(n_t - lag_max - 1 - window)

    for i in range(n_x):
        for j in range(n_y):
            
            print("Working on", i, j)
            sij = str_2pt(f[:, i, j], f, lag_max, window)
            s += np.nanmean(sij, axis=(1, 2))

    s /= n_x * n_y 
    return s