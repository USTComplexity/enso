import numpy as np
from statsmodels.tsa.api import VAR

from enso.utils import sort_arrays


def build_granger(data: np.array, maxlag: int, thres: float, 
                  sort_by: str = "none"):
    
    edges = []
    weights = []
    lags = []

    var = VAR(data)
    model = var.fit(maxlags=maxlag)

    params = model.params
    p = data.shape[1]

    for i in range(p):
        for j in range(p):
            
            # Skip self-links
            if i == j:
                continue

            coef = _get_coef(params, i, j, p, maxlag)

            if np.max(np.abs(coef)) >= thres:
                edges.append((j, i))

                # Get the most significant lag; 
                # +1 because smallest lag is 1 not 0
                lags.append(np.argmax(np.abs(coef)) + 1) 
                weights.append(np.max(np.abs(coef))) 
    
    edges, weights, lags = np.array(edges), np.array(weights), np.array(lags)
    
    # Sort the edges by their weight
    if sort_by != "none":
    
        if sort_by == "weights":
            weights, edges, lags = sort_arrays(weights, edges, lags)
        elif sort_by == "lags":
            lags, edges, weights = sort_arrays(lags, edges, weights)
        else:
            raise ValueError(f"Unknown sort variable {sort_by}")
        
    return edges, weights, lags, model 


def _get_coef(params, i: int, j: int, p: int, d: int):
    """
    Get all coefficients A_ij, where x_i = A_ij x_j
    
    p is the number of variables
    d is the number of lags (maxlag)
    """
    
    indices = 1 + j + np.arange(0, p*d, p)
    return params[indices, i]
