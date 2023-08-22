import numpy as np
import numpy.typing as npt
import bottleneck as bn


def corr_2pt_avg(x: npt.ArrayLike, f: npt.ArrayLike, lag_max: int, window: int):
    
    n_t, n_x, n_y = f.shape
    c = np.zeros([n_t - lag_max - 1 - window, n_x, n_y])

    for s in range(lag_max + 1):
        c += corr_2pt(x, f, s, lag_max, window)

    c /= (lag_max + 1)

    return c


def corr_2pt(x: npt.ArrayLike, f: npt.ArrayLike, lag: int, lag_max: int, window: int):
    
    x = x[lag_max+1:, np.newaxis, np.newaxis]
    f = f[lag_max+1 - lag:, :, :]
    
    c = bn.move_mean(a=x*f, window=window, axis=0)
    print("Here")

    c -= bn.move_mean(a=x, window=window, axis=0) \
       * bn.move_mean(a=f, window=window, axis=0)
    
    c /= bn.move_std(a=x, window=window, axis=0) \
       * bn.move_std(a=f, window=window, axis=0)

    return c


def add_edges(edges: list[set], source: tuple[int, int], 
              cmat: npt.ArrayLike, thres: float):
    
    txy = np.argwhere(np.abs(cmat) >= thres)

    for point in txy:
        t, x, y = point
        print(point)

        if (source, (x, y)) in edges[t] or ((x, y), source) in edges[t]:
            continue
        
        edges[t].append((source, (x, y)))


def build_network(f: npt.ArrayLike, lag_max: int, thres: float, window: int = 365):

    n_t, n_x, n_y = f.shape

    edges = [set() for _ in range(n_t - lag_max + 1)]
    
    for i in range(n_x):
        for j in range(n_y):
            
            print("Working on", i, j)
            cmat = corr_2pt_avg(f[:, i, j], f, lag_max, window)
            add_edges(edges, (i, j), cmat, thres)
    
    return edges
